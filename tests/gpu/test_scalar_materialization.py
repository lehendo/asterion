import pytest
from asterion.backend import require_gpu
from asterion.ir import Node, Program, Schedule
from asterion.validation import certify_variant


@pytest.mark.parametrize("fusion", [True, False])
@pytest.mark.parametrize("identity", [True, False])
def test_scalar_hoisting_and_identity_differential(fusion, identity):
    require_gpu()
    nodes = (
        (Node("x", "input"),)
        if identity
        else (
            Node("x", "input"),
            Node("a", "const", value=1.25),
            Node("b", "const", value=0.5),
            Node("scale", "add", ("a", "b")),
            Node("positive", "relu", ("scale",)),
            Node("out", "mul", ("x", "positive")),
        )
    )
    p = Program("pointwise", (1031,), nodes, "x" if identity else "out")
    evidence = certify_variant(p, Schedule(fusion=fusion), "test_scalar")
    assert len(evidence["checks"]) == 8
