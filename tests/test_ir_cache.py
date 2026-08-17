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


