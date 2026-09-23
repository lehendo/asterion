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


