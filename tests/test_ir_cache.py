import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from asterion.cache import Cache
from asterion.codegen import generate
from asterion.io import ValidationError, digest
from asterion.ir import (
    Node,
    Program,
    Schedule,
    candidates,
    canonicalize,
    make_program,
    transform,
    verify,
    verify_schedule,
)
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize(
    "op,shape",
    [
        ("pointwise", (257,)),
        ("sum", (17, 129)),
        ("max", (17, 127)),
        ("softmax", (17, 257)),
        ("gemm", (17, 31, 33)),
    ],
)
def test_ir_roundtrip_and_codegen(op, shape):
    p = make_program(op, shape)
    assert Program.from_dict(p.to_dict()) == p
    assert p.hash == Program.from_dict(p.to_dict()).hash
    for s in candidates(p):
        source = generate(p, s)
        ast.parse(source)
        assert "triton.autotune" not in source


def test_canonicalization_eliminates_pure_dead_and_common_expressions():
    p = Program(
        "pointwise",
        (128,),
        (
            Node("x", "input"),
            Node("a", "const", value=1.0),
            Node("b", "const", value=1.0),
            Node("dead", "mul", ("x", "a")),
            Node("y", "add", ("x", "b")),
        ),
        "y",
    )
    q = canonicalize(p)
    assert [n.name for n in q.nodes] == ["x", "b", "y"]
    assert q.nodes[-1].args == ("x", "b")


@pytest.mark.parametrize("name", ["x;print(1)", "tl", "for", "rr"])
def test_illegal_identifier_rejected(name):
    p = make_program("sum", (2, 3))
    with pytest.raises(ValidationError):
        verify(replace(p, nodes=(Node(name, "input"), p.nodes[1])))


def test_use_before_definition_rejected():
    p = make_program("sum", (2, 3))
    with pytest.raises(ValidationError, match="before"):
        verify(replace(p, nodes=tuple(reversed(p.nodes))))


