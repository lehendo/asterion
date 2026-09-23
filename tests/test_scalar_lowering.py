"""Scalar hoisting and identity lowering must generate executable stages."""

from asterion.codegen import generate
from asterion.diagnosis import schedule_analysis
from asterion.ir import Node, Program, Schedule, verify


def scalar_program():
    return Program(
        "pointwise",
        (1031,),
        (
            Node("x", "input"),
            Node("a", "const", value=1.25),
            Node("b", "const", value=0.5),
            Node("scale", "add", ("a", "b")),
            Node("positive", "relu", ("scale",)),
            Node("out", "mul", ("x", "positive")),
        ),
        "out",
    )


def test_pure_scalar_ssa_is_hoisted_not_materialized():
    p = scalar_program()
    assert verify(p)["scale"] == "scalar"
    code = generate(p, Schedule(fusion=False))
    assert "def stage_scale" not in code
    assert "def stage_positive" not in code
    assert "def stage_out" in code
    assert "scale = (a + b)" in code
    assert "positive = tl.maximum(scale, 0.0)" in code
    assert schedule_analysis(p, Schedule(fusion=False))["launch_count"] == 1


def test_unfused_identity_is_one_copy_kernel():
    p = Program("pointwise", (1031,), (Node("x", "input"),), "x")
    assert "def kernel" in generate(p, Schedule(fusion=False))
    assert schedule_analysis(p, Schedule(fusion=False))["launch_count"] == 1


def test_gemm_model_accounts_for_unrolled_masked_iterations():
    from asterion.ir import make_program

    p = make_program("gemm", (17, 19, 33))
    a = schedule_analysis(p, Schedule(unroll=1))
    b = schedule_analysis(p, Schedule(unroll=4))
    assert b["padded_work_units"] == a["padded_work_units"] * 2


def test_gemm_output_must_be_the_verified_contraction():
    from dataclasses import replace

    import pytest
    from asterion.io import ValidationError
    from asterion.ir import make_program

    p = make_program("gemm", (17, 19, 23))
    for output in ("x", "y"):
        with pytest.raises(ValidationError, match="pure contraction"):
            verify(replace(p, output=output))
