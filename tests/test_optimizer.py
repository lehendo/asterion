import pytest
from asterion.io import ValidationError, digest
from asterion.ir import Schedule
from asterion.optimizer import fit_portfolio, objective, select
from asterion.runtime import validate_portfolio


def matrix():
    records = []
    for i, cols in enumerate((127, 1024, 4093)):
        for variant, values, schedule in (
            ("a", (1.0, 10.0, 100.0), Schedule(rows=1)),
            ("b", (4.0, 4.0, 4.0), Schedule(rows=4)),
        ):
            records.append(
                {
                    "case_id": str(cols),
                    "operation": "softmax",
                    "shape": [17, cols],
                    "split": "train",
                    "dtype": "float32",
                    "layout": "contiguous",
                    "weight": 1.0,
                    "variant": variant,
                    "schedule": schedule.to_dict(),
                    "status": "ok",
                    "correctness": "passed",
                    "device_us": [values[i]] * 20,
                    "generated_code_bytes": 1000,
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": "trace",
        "measurement_kind": "simulated_fixture",
        "timing_unit": "us",
        "environment": {"gpu_runtime": None},
        "records": records,
    }


def test_portfolio_constraints_and_metadata_dispatch():
    p = fit_portfolio(matrix(), max_variants=2)
    assert len(p["variants"]) <= 2
    assert p["generated_code_bytes"] <= p["max_code_bytes"]
    assert select(p, [17, 127])[0] == "a"
    assert select(p, [17, 4093])[0] == "b"
    assert select(p, [17, 8191]) == (None, "outside_training_feature_range")
    assert validate_portfolio(p)["status"] == "validated_metadata"


def test_no_test_leakage():
    t = matrix()
    t["records"][0]["split"] = "test"
    with pytest.raises(ValidationError, match="held-out"):
        fit_portfolio(t)


def test_deterministic_decisions():
    assert fit_portfolio(matrix()) == fit_portfolio(matrix())


