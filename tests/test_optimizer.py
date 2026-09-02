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


