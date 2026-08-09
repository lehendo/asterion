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


def test_specification_is_passive_bounded_and_canonical():
    raw = spec(fusion=False)
    result = specification(raw)
    assert result["rounds"] == 16
    assert raw != result
    for key, value in (
        ("rounds", True),
        ("rounds", 10000),
        ("seed", -1),
        ("threshold", float("nan")),
        ("atol", 0.5),
        ("module", "os"),
    ):
        invalid = {**raw, key: value}
        with pytest.raises(ValidationError):
            specification(invalid)
    with pytest.raises(ValidationError):
        specification({})


def test_exact_one_factor_interventions_and_inert_fields():
    result = interventions(spec(fusion=False, block=1024, tile_m=64))
    by_field = {x["field"]: x for x in result}
    assert by_field["tile_m"]["status"] == "inert"
    for field in ("fusion", "block"):
        plan = by_field[field]
        assert plan["status"] == "executable"
        assert list(plan["trail"]["changes"]) == [field]
        assert plan["trail"]["before"][field] != plan["trail"]["after"][field]


