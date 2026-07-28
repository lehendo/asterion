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


