"""Labeled simulation tests reduction acceptance, rejection and confirmation."""

from asterion.ir import Schedule, make_program
from asterion.reduction import reduce_regression
from asterion.reproducer import export_reproducer


def specification():
    baseline = Schedule().to_dict()
    return {
        "schema_version": 1,
        "artifact_type": "regression_spec",
        "program": make_program("pointwise", (1024,)).to_dict(),
        "baseline": baseline,
        "candidate": {**baseline, "fusion": False},
    }


def test_reduction_rejects_lost_effect_and_records_actual_smaller_witness(tmp_path):
    def simulated(spec, output):
        passing = spec["program"]["shape"][0] >= 128
        return {
            "comparison": {"verdict": "regression" if passing else "inconclusive"},
            "recommendations": ["undo_fusion"] if passing else [],
            "environment": {"measurement_kind": "simulated_fixture"},
        }

    report = reduce_regression(specification(), tmp_path, max_trials=6, runner=simulated)
    assert report["status"] == "confirmed_reduced_witness"
    assert report["element_ratio"] == 0.125
    assert report["accepted_steps"] == 3
    assert report["trials"][-1]["accepted"] is False
    assert (tmp_path / "witness" / "reproduce.py").exists()


def test_confirmation_failure_cannot_export_minimized_claim(tmp_path):
    def simulated(spec, output):
        return {
            "comparison": {
                "verdict": "regression" if output.name != "confirmation" else "inconclusive"
            },
            "recommendations": ["undo_fusion"],
            "environment": {"measurement_kind": "simulated_fixture"},
        }

    report = reduce_regression(specification(), tmp_path, max_trials=1, runner=simulated)
    assert report["status"] == "unconfirmed"
    assert not (tmp_path / "witness").exists()


def test_export_uses_verified_codegen_and_standalone_harness(tmp_path):
    export_reproducer(specification(), tmp_path)
    text = (tmp_path / "reproduce.py").read_text()
    assert "from asterion" not in text
    assert "assert_close" in text
    assert "enable_fp_fusion" in text
    assert "@triton.jit" in (tmp_path / "baseline.py").read_text()
    compile(text, "reproduce.py", "exec")
    import pytest
    from asterion.io import ValidationError

    with pytest.raises(ValidationError, match="new or empty"):
        export_reproducer(specification(), tmp_path)
