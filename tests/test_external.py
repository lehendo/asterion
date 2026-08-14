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


