import copy
import hashlib
import json
import sys

import pytest
from asterion.api import diagnose, export_reproducer
from asterion.external import complexity, expression, interventions, source_path, specification
from asterion.external_workflow import run_worker, worker_environment
from asterion.io import ValidationError
from asterion.numerics import forward_gamma, numerical_policy


def test_prepared_launch_releases_binary_without_cyclic_gc():
    import gc
    import weakref

    from asterion.external_runtime import _KernelLaunch

    class Binary:
        pass

    class Entry:
        def __getitem__(self, grid):
            return lambda **kwargs: Binary()

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        launch = _KernelLaunch(Entry(), (1,), {}, object())
        state = {"launch": launch}
        assert launch() is launch.output
        binary = weakref.ref(launch.compiled)
        owner = weakref.ref(launch)
        del launch, state
        assert owner() is None and binary() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.fixture
def case(tmp_path):
    # Importing this fixture would have a side effect; passive paths must not.
    code = "from pathlib import Path\nPath(__file__).with_suffix('.executed').touch()\n"
    module = tmp_path / "kernel.py"
    module.write_text(code)
    s = {
        "schema_version": 1,
        "artifact_type": "triton_case",
        "case_id": "test_add",
        "module": "kernel.py",
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "dimensions": {"N": 1031},
        "tensors": {
            k: {
                "shape": [{"dim": "N"}],
                "dtype": "float32",
                "role": "output" if k == "z" else "input",
            }
            for k in ("x", "y", "z")
        },
        "arguments": {
            "X": {"tensor": "x"},
            "Y": {"tensor": "y"},
            "Z": {"tensor": "z"},
            "N": {"dim": "N"},
        },
        "parameters": {
            "BLOCK": {"kind": "constexpr", "values": [32, 256]},
            "num_warps": {"kind": "option", "values": [4]},
        },
        "baseline": {"entrypoint": "add_kernel", "parameters": {"BLOCK": 256, "num_warps": 4}},
        "candidate": {"entrypoint": "add_kernel", "parameters": {"BLOCK": 32, "num_warps": 4}},
        "grid": [{"ceildiv": [{"dim": "N"}, {"param": "BLOCK"}]}],
        "reference": {"entrypoint": "reference", "description": "Independent tensor addition"},
        "numerical": {
            "kind": "exact",
            "justification": "Single rounded FP32 add against rounded reference",
        },
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(s))
    return s, path


def test_passive_validation_and_export_do_not_execute(case, tmp_path):
    s, path = case
    checked = specification(s)
    assert complexity(checked)["value"] == 1031 * 12
    assert interventions(checked)[0]["parameter"] == "BLOCK"
    before = set(sys.modules)
    result = export_reproducer(path, output=tmp_path / "export")
    assert result.exists()
    assert not path.with_name("kernel.executed").exists()
    assert "torch" not in set(sys.modules) - before
    assert "triton" not in set(sys.modules) - before
    assert (result.parent / "_support/external_runtime.py").exists()


def test_cli_passive_external_validation(case, capsys):
    from asterion.cli import main

    _, path = case
    assert main(["validate", str(path)]) == 0
    assert "passive validation only" in capsys.readouterr().out
    assert not path.with_name("kernel.executed").exists()


def test_existing_leading_underscore_entrypoints_need_no_source_rename(case):
    s, _ = case
    s["baseline"]["entrypoint"] = "_existing_kernel"
    s["candidate"]["entrypoint"] = "_existing_kernel"
    s["reference"]["entrypoint"] = "_reference"
    assert specification(s)["baseline"]["entrypoint"] == "_existing_kernel"


def test_trust_is_not_authorized_by_imported_artifact(case, tmp_path):
    s, path = case
    with pytest.raises(ValidationError, match="trusted_code"):
        diagnose(path, output=tmp_path / "out")
    s["trusted_code"] = True
    with pytest.raises(ValidationError, match="unknown fields"):
        specification(s)
    assert not path.with_name("kernel.executed").exists()


def test_standalone_rejects_tampering_before_import(case, tmp_path):
    import subprocess

    _, path = case
    script = export_reproducer(path, output=tmp_path / "export")
    support = script.parent / "_support/external_runtime.py"
    marker = tmp_path / "support-executed"
    support.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    completed = subprocess.run(
        [sys.executable, str(script), "--trusted-code", "--output", str(tmp_path / "replay")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0 and "integrity mismatch" in completed.stderr
    assert not marker.exists() and not path.with_name("kernel.executed").exists()


def test_trusted_loader_executes_reviewed_bytes_despite_stale_pyc_and_linecache(tmp_path):
    import inspect
    import linecache
    import os
    import py_compile

    from asterion.external_runtime import _load_reviewed_module

    path = tmp_path / "reviewed.py"
    old = "def answer():\n    return 'old'\n"
    new = old.replace("old", "new")
    path.write_text(old)
    stamp = path.stat()
    py_compile.compile(str(path), doraise=True)
    linecache.getlines(str(path))
    path.write_text(new)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    module = _load_reviewed_module(path, hashlib.sha256(new.encode()).hexdigest())
    assert module.answer() == "new" and "'new'" in inspect.getsource(module.answer)
    with pytest.raises(ValidationError, match="changed before execution"):
        _load_reviewed_module(path, hashlib.sha256(old.encode()).hexdigest())


@pytest.mark.parametrize("failure", ["startup", "timeout"])
def test_failed_worker_retains_exact_input_snapshot(case, tmp_path, monkeypatch, failure):
    import subprocess

    from asterion.external_workflow import diagnose_external

    _, path = case
    original = path.with_name("kernel.py").read_bytes()

    def failed(args, **kwargs):
        path.with_name("kernel.py").write_text("changed concurrently")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1, output="simulated", stderr="timeout")
        return subprocess.CompletedProcess(args, 1, "simulated", "startup failure")

    monkeypatch.setattr("asterion.external_workflow.run_worker", failed)
    root = tmp_path / "failed"
    with pytest.raises(ValidationError):
        diagnose_external(path, root, trusted_code=True)
    assert (root / "source/kernel.py").read_bytes() == original
    assert (root / "source/case.json").exists() and (root / "worker-failure.json").exists()
    assert not path.with_name("kernel.executed").exists()


