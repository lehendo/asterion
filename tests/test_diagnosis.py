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


def test_lowering_model_and_reduction_bounds():
    p = make_program("pointwise", (4097,))
    fused = schedule_analysis(p, Schedule())
    materialized = schedule_analysis(p, Schedule(fusion=False))
    assert fused["launch_count"] == 1
    assert materialized["launch_count"] == 3
    assert materialized["ideal_global_bytes"] == fused["ideal_global_bytes"] * 3
    assert fused["padded_work_units"] == 4352
    reduction = spec("sum", (19, 513), row_tile=2048)
    assert reshape(reduction, (9, 257))["program"]["shape"] == [9, 257]
    with pytest.raises(ValidationError):
        reshape(reduction, (9, 4097))


def test_paired_sign_test_abstains_on_noise_and_guards_multiplicity():
    baseline = [10.0] * 16
    assert paired_effect(baseline, [12.0] * 16)["verdict"] == "regression"
    assert paired_effect(baseline, [8.0] * 16)["verdict"] == "improvement"
    assert paired_effect(baseline, [10.0] * 16)["verdict"] == "inconclusive"
    assert paired_effect(baseline, [8.0, 12.0] * 8)["verdict"] == "inconclusive"
    assert paired_effect(baseline, [11.0] * 16)["verdict"] == "inconclusive"
    assert paired_effect(baseline, [12.0] * 16, comparisons=1000)["verdict"] == "inconclusive"
    for bad in ([1.0] * 11, [float("nan")] * 16, [0.0] * 16):
        with pytest.raises(ValidationError):
            paired_effect(baseline, bad)


def test_program_input_cannot_supply_executable_source():
    raw = copy.deepcopy(spec())
    raw["program"]["source"] = "raise RuntimeError()"
    with pytest.raises(ValidationError):
        specification(raw)
