"""Synthetic CPU fixtures test inference contracts, not GPU performance claims."""

import copy

import pytest
from asterion.diagnosis import (
    interventions,
    paired_effect,
    reshape,
    schedule_analysis,
    specification,
)
from asterion.io import ValidationError
from asterion.ir import Schedule, make_program


def spec(operation="pointwise", shape=(4097,), **changes):
    a = Schedule()
    b = {**a.to_dict(), **changes}
    return {
        "schema_version": 1,
        "artifact_type": "regression_spec",
        "program": make_program(operation, shape).to_dict(),
        "baseline": a.to_dict(),
        "candidate": b,
    }


