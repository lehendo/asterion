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


def test_equal_feature_cannot_be_split_by_oracle():
    t = matrix()
    for r in t["records"]:
        r["shape"][1] = 127
    p = fit_portfolio(t, max_variants=2)
    assert len(p["dispatcher"]["rules"]) == 1


def test_incorrect_incomplete_schedule_excluded():
    t = matrix()
    t["records"][0].update(status="error", error="incorrect", correctness="failed")
    assert select(fit_portfolio(t), [17, 127])[0] == "b"


def test_code_budget_and_sample_validity():
    with pytest.raises(ValidationError, match="code budget"):
        fit_portfolio(matrix(), max_code_bytes=10)
    t = matrix()
    for r in t["records"]:
        r.pop("generated_code_bytes")
    with pytest.raises(ValidationError, match="code_bytes"):
        fit_portfolio(t)


def test_single_variant_budget():
    p = fit_portfolio(matrix(), max_variants=1)
    assert len(p["variants"]) == 1


def test_portfolio_corruption_rejected():
    p = fit_portfolio(matrix())
    p["variants"]["a"]["warps"] = 3
    with pytest.raises(ValidationError, match="integrity"):
        validate_portfolio(p)


def test_nonmonotone_dispatch_rules_rejected_even_with_valid_checksum():
    p = fit_portfolio(matrix())
    p["dispatcher"]["rules"] = list(reversed(p["dispatcher"]["rules"]))
    p.pop("integrity_hash")
    p["integrity_hash"] = digest(p)
    with pytest.raises(ValidationError):
        validate_portfolio(p)


def test_risk_term_is_separate_from_mean():
    assert objective([1.0, 4.0], [0.9, 0.1], risk=1) > objective([1.0, 4.0], [0.9, 0.1], risk=0)


def test_malformed_portfolio_is_actionable():
    for value in (None, [], {"schema_version": 1, "artifact_type": "portfolio"}):
        with pytest.raises(ValidationError):
            validate_portfolio(value)


def test_fixture_evidence_label_survives_selection():
    assert fit_portfolio(matrix())["measurement_kind"] == "simulated_fixture"


