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


