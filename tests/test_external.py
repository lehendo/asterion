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


def test_source_hash_and_symlink_containment(case, tmp_path):
    s, path = case
    checked = specification(s)
    path.with_name("kernel.py").write_text("modified")
    with pytest.raises(ValidationError, match="hash mismatch"):
        source_path(checked, tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.py")
    outside.write_text("pass")
    link = tmp_path / "escape.py"
    link.symlink_to(outside)
    checked["module"] = "escape.py"
    with pytest.raises(ValidationError, match="escapes"):
        source_path(checked, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(module="../kernel.py"),
        lambda s: s.update(module="/tmp/kernel.py"),
        lambda s: s.update(module="reproduce.py"),
        lambda s: s.update(module="_support/kernel.py"),
        lambda s: s.update(module="worker-home/kernel.py"),
        lambda s: s["tensors"]["z"].update(dtype="float64"),
        lambda s: s["candidate"]["parameters"].update(BLOCK=512),
        lambda s: s["parameters"]["num_warps"].update(values=[3]),
        lambda s: s.update(grid=["lambda: run()"]),
        lambda s: s.update(dimensions={"N": 2**31}),
        lambda s: s.update(
            constraints=[{"left": {"dim": "N"}, "relation": "multiple_of", "right": 32}]
        ),
        lambda s: s["arguments"].update(BLOCK=32),
        lambda s: s.update(numerical={"kind": "allclose", "rtol": 0.01, "atol": 0}),
    ],
)
def test_invalid_contracts_are_rejected(case, mutation):
    s, _ = case
    broken = copy.deepcopy(s)
    mutation(broken)
    with pytest.raises(ValidationError):
        specification(broken)


def test_expression_is_bounded_arithmetic_not_python():
    assert expression({"ceildiv": [{"dim": "N"}, {"param": "B"}]}, {"N": 1031}, {"B": 256}) == 5
    assert expression({"pow2": {"dim": "N"}}, {"N": 1031}, {}) == 2048
    for x in ({"eval": "__import__('os')"}, {"ceildiv": [1, 0]}, "N + 1"):
        with pytest.raises(ValidationError):
            expression(x, {}, {})


def test_typed_constexpr_changes_are_not_pruned_as_equal(case):
    s, _ = case
    s["parameters"]["FLAG"] = {"kind": "constexpr", "values": [True, 1]}
    s["baseline"]["parameters"]["FLAG"] = True
    s["candidate"]["parameters"]["FLAG"] = 1
    plans = interventions(specification(s))
    assert any(p["parameter"] == "FLAG" for p in plans)


def test_signed_zero_constexpr_identity_is_preserved(case):
    s, _ = case
    s["parameters"]["FLAG"] = {"kind": "constexpr", "values": [-0.0]}
    s["baseline"]["parameters"]["FLAG"] = -0.0
    s["candidate"]["parameters"]["FLAG"] = 0.0
    with pytest.raises(ValidationError, match="outside its declared domain"):
        specification(s)
    s["parameters"]["FLAG"]["values"].append(0.0)
    assert any(p["parameter"] == "FLAG" for p in interventions(specification(s)))


def test_saved_source_root_cannot_escape_artifact(case, tmp_path):
    from asterion.external_workflow import load_case
    from asterion.io import digest

    s, path = case
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "source").symlink_to(path.parent, target_is_directory=True)
    saved = {"artifact_type": "external_diagnosis", "specification": specification(s)}
    saved["result_id"] = digest(saved)
    target = artifact / "results.json"
    target.write_text(json.dumps(saved))
    with pytest.raises(ValidationError, match="escapes diagnosis"):
        load_case(target)


def test_worker_environment_drops_credentials(monkeypatch, tmp_path):
    for name in (
        "AWS_SECRET_ACCESS_KEY",
        "SSH_AUTH_SOCK",
        "GITHUB_TOKEN",
        "PRIVATE_SERVICE_TOKEN",
        "SLURM_JWT",
    ):
        monkeypatch.setenv(name, "sensitive")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    e = worker_environment(tmp_path)
    assert e["HOME"] == str(tmp_path) and e["CUDA_VISIBLE_DEVICES"] == "2"
    assert e["SLURM_JOB_ID"] == "123" and e["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert not any(
        k in e
        for k in (
            "AWS_SECRET_ACCESS_KEY",
            "SSH_AUTH_SOCK",
            "GITHUB_TOKEN",
            "PRIVATE_SERVICE_TOKEN",
            "SLURM_JWT",
        )
    )


def test_worker_timeout_owns_background_children(tmp_path):
    import os
    import subprocess
    import time

    marker = tmp_path / "background-finished"
    child = (
        "import time; from pathlib import Path; time.sleep(2); Path("
        + repr(str(marker))
        + ").touch()"
    )
    command = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
        + repr(child)
        + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print('ready',flush=True); time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        run_worker([sys.executable, "-c", command], env=dict(os.environ), timeout=1)
    assert "ready" in exc.value.stdout
    time.sleep(2.1)
    assert not marker.exists()


def test_numerical_contracts_require_explicit_policy():
    assert (
        numerical_policy(
            {"kind": "allclose", "rtol": 0, "atol": 0, "justification": "strict application"}
        )["atol"]
        == 0
    )
    assert forward_gamma(64) > forward_gamma(32) > 0
    for p in (
        {"kind": "matmul_forward_bound", "justification": "model", "accumulation_bits": 11},
        {"kind": "exact", "justification": "numeric", "rtol": 1},
    ):
        with pytest.raises(ValidationError):
            numerical_policy(p)


def test_reduction_preserves_regression_and_rejects_lost_effect(case, tmp_path):
    from asterion.external_reduction import reduce_external

    s, path = case
    calls = []

    def simulated(spec, output, **kwargs):
        calls.append(copy.deepcopy(spec))
        positive = spec["dimensions"]["N"] >= 64
        return {
            "status": "REGRESSION_ESTABLISHED" if positive else "REGRESSION_INCONCLUSIVE",
            "environment": {"fixture": "simulated CPU state test, no GPU measurements"},
            "recommendations": ["undo_BLOCK"] if positive else [],
            "variants": {},
            "comparison": {"fixture": True},
        }

    result = reduce_external(
        s, tmp_path / "reduction", trusted_code=True, source_root=path.parent, runner=simulated
    )
    assert result["status"] == "CONFIRMED_SMALLER_REPRODUCER"
    assert result["reduced_specification"]["dimensions"]["N"] == 64
    assert result["storage_ratio"] == 64 / 1031
    assert not result["trials"][-1]["accepted"]
    assert calls[-1]["measurement"]["seed"] != calls[0]["measurement"]["seed"]
    assert result["properties"]["HARDWARE_MECHANISM_SUPPORTED"] == "UNAVAILABLE"
    assert not path.with_name("kernel.executed").exists()
    manifest = json.loads((tmp_path / "reduction/witness/manifest.json").read_text())
    assert {"reduction.json", "expectation.json"} <= set(manifest["files"])


def test_numeric_rejection_never_becomes_confirmed_reduction(case, tmp_path):
    from asterion.external_reduction import reduce_external

    s, path = case

    def simulated(spec, output, **kwargs):
        return {"status": "NUMERICAL_FAILURE"}

    result = reduce_external(
        s, tmp_path / "rejected", trusted_code=True, source_root=path.parent, runner=simulated
    )
    assert result["status"] == "NO_CONFIRMED_WITNESS" and not result["trials"]
    assert not (tmp_path / "rejected/witness").exists()


