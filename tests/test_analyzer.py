import copy
import json

import pytest
from asterion.analyzer import analyze, compare
from asterion.cli import main
from asterion.io import ValidationError, atomic_json, digest
from asterion.report import write_report
from asterion.schema import WorkloadSpec, validate_trace
from asterion.stats import cvar, summary, weighted_quantile
from hypothesis import given
from hypothesis import strategies as st


def trace(variant="baseline", values=(10.0, 20.0)):
    return {
        "schema_version": 1,
        "artifact_type": "trace",
        "measurement_kind": "simulated_fixture",
        "timing_unit": "us",
        "environment": {"gpu_runtime": None},
        "records": [
            {
                "case_id": f"c{i}",
                "operation": "softmax",
                "shape": [i + 1, 127],
                "dtype": "float32",
                "layout": "contiguous",
                "split": "train",
                "weight": i + 1,
                "variant": variant,
                "status": "ok",
                "correctness": "passed",
                "device_us": [v] * 20,
            }
            for i, v in enumerate(values)
        ],
    }


def spec():
    return {
        "schema_version": 1,
        "operation": "softmax",
        "cases": [
            {"id": "train", "shape": [17, 127], "split": "train"},
            {"id": "test", "shape": [31, 257], "split": "test"},
        ],
    }


def test_weighted_distributions_distinguish_absolute_and_relative():
    a, b = trace(), trace("candidate", (20.0, 10.0))
    report = compare(a, b)
    row = report["groups"][0]
    assert row["p95_relative_slowdown"] == 2.0
    assert row["candidate_p95_absolute_shape_median_us"] == 20.0
    assert row["expected_runtime_ratio"] == pytest.approx(0.8)
    assert report["coverage_complete"]
    assert report["measurement_kind"] == "simulated_fixture"


def test_fixture_and_measurement_cannot_mix():
    a, b = trace(), trace()
    b["measurement_kind"] = "measured"
    with pytest.raises(ValidationError, match="mix"):
        compare(a, b)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_reject_invalid_timing(value):
    t = trace()
    t["records"][0]["device_us"] = [value]
    with pytest.raises(ValidationError):
        validate_trace(t)


def test_failed_coverage_is_visible():
    a, b = trace(), trace("candidate")
    b["records"][0].update(status="error", error="numerical mismatch")
    report = compare(a, b)
    assert not report["coverage_complete"]
    assert len(report["failures"]) == 1
    assert report["groups"][0]["shape_count"] == 1


def test_missing_cases_are_visible():
    a, b = trace(), trace()
    b["records"].pop()
    assert compare(a, b)["only_baseline"] == ["c1"]


def test_mismatched_shapes_rejected():
    a, b = trace(), trace()
    b["records"][0]["shape"] = [33, 127]
    with pytest.raises(ValidationError, match="shape"):
        compare(a, b)


def test_mismatched_targets_rejected():
    a, b = trace(), trace()
    b["environment"]["gpu_runtime"] = {"uuid": "other"}
    with pytest.raises(ValidationError, match="targets"):
        compare(a, b)


def test_multivariant_requires_explicit_selection():
    t = trace()
    other = copy.deepcopy(t["records"][0])
    other["variant"] = "other"
    t["records"].append(other)
    with pytest.raises(ValidationError, match="explicitly"):
        compare(t, t)
    assert compare(t, t, "baseline", "baseline")["coverage_complete"]


def test_cvar_fractional_quantile_mass():
    assert cvar([1.0, 10.0], [0.9, 0.1], 0.95) == pytest.approx(10.0)
    assert cvar([1.0, 10.0], [0.9, 0.1], 0.8) == pytest.approx(5.5)
    assert cvar([1.0, 10.0], [0.9, 0.1], 0) == pytest.approx(1.9)


def test_sample_dispersion():
    s = summary([1, 1, 1, 1, 100])
    assert s["median_us"] == 1 and s["p95_us"] == 100
    assert s["small_sample"]


@given(
    st.lists(st.floats(min_value=0.001, max_value=1e4, allow_nan=False), min_size=1, max_size=40)
)
def test_quantile_monotonic(values):
    qs = [weighted_quantile(values, [1] * len(values), q) for q in [0, 0.25, 0.5, 0.95, 1]]
    assert qs == sorted(qs) and qs[0] == min(values) and qs[-1] == max(values)


def test_workload_roundtrip():
    s = WorkloadSpec.from_dict(spec())
    assert WorkloadSpec.from_dict(s.to_dict()) == s
    assert analyze(spec())["measurement_kind"] == "unavailable"


def test_split_leakage_rejected():
    s = spec()
    s["cases"][1]["shape"] = s["cases"][0]["shape"]
    with pytest.raises(ValidationError, match="leak"):
        WorkloadSpec.from_dict(s)


@pytest.mark.parametrize("field,value", [("typo", 1), ("dtype", "float16"), ("seed", -1)])
def test_invalid_workload(field, value):
    s = spec()
    s[field] = value
    with pytest.raises(ValidationError):
        WorkloadSpec.from_dict(s)


def test_invalid_boolean_dimension():
    s = spec()
    s["cases"][0]["shape"] = [True, 128]
    with pytest.raises(ValidationError):
        WorkloadSpec.from_dict(s)


def test_duplicate_record_rejected():
    t = trace()
    t["records"].append(t["records"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        validate_trace(t)


def test_correctness_gate():
    t = trace()
    t["records"][0]["correctness"] = "unknown"
    with pytest.raises(ValidationError, match="correctness"):
        validate_trace(t)


def test_report_escapes_untrusted_strings_and_saves_data(tmp_path):
    data = compare(trace(), trace("candidate"))
    data["rows"][0]["case_id"] = '<script>alert("x")</script>'
    path = write_report(data, tmp_path)
    content = path.read_text()
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "SIMULATED_FIXTURE" in content
    assert json.loads((tmp_path / "results.json").read_text()) == data
    assert (tmp_path / "cases.csv").exists()


