import copy
import json

import pytest
from asterion.cli import main
from asterion.diagnostic_report import write_diagnosis
from asterion.io import ValidationError, digest
from asterion.ir import Schedule, make_program


def fixture():
    schedule = Schedule().to_dict()
    return {
        "artifact_type": "diagnosis",
        "measurement_kind": "simulated_fixture",
        "comparison": {
            "verdict": "inconclusive",
            "median_paired_ratio": 1.0,
            "blocks": 16,
            "practical_threshold": 0.1,
            "limitations": "fixture only",
        },
        "variants": {
            n: {"schedule": schedule, "generated_source": "<script>alert(1)</script>"}
            for n in ("baseline", "candidate")
        },
        "blocks": [
            {"measurements": {n: {"device_graph_us": 1.0} for n in ("baseline", "candidate")}}
        ],
        "boundary": "device_graph_us",
        "specification": {"program": {"operation": "pointwise", "shape": [17]}},
        "environment": {"gpu_runtime": {"name": "<img src=x onerror=alert(1)>"}},
        "findings": [],
        "scope": "fixture only",
    }


