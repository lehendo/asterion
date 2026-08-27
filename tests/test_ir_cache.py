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


def test_tiling_changes_executable_indexing_and_audit():
    p = make_program("pointwise", (1025,))
    s, log = transform(p, Schedule(), block=1024, fusion=False)
    assert log["changes"] == {"block": 1024, "fusion": False}
    assert "stage_scaled" in generate(p, s)
    assert "tl.program_id(0)*1024" in generate(p, s)
    assert generate(p, s) != generate(p, Schedule())


def test_row_group_and_gemm_loop_rewrites():
    p = make_program("softmax", (17, 129))
    s, _ = transform(p, Schedule(), rows=4)
    assert "tl.arange(0, 4)" in generate(p, s)
    g = make_program("gemm", (17, 31, 33))
    gs, _ = transform(g, Schedule(), tile_m=16, unroll=2)
    assert "tl.static_range(0, 2)" in generate(g, gs)
    assert 'input_precision="ieee"' in generate(g, gs)


@pytest.mark.parametrize(
    "kwargs",
    [{"warps": 3}, {"block": 0}, {"rows": 16}, {"unroll": 8}, {"fusion": "yes"}, {"block": True}],
)
def test_illegal_schedule_rejected(kwargs):
    with pytest.raises(ValidationError):
        verify_schedule(make_program("pointwise", (17,)), Schedule(**kwargs))


def test_row_resource_bound():
    with pytest.raises(ValidationError):
        verify_schedule(make_program("softmax", (2, 32769)), Schedule())


@given(st.integers(min_value=1, max_value=10000))
def test_pointwise_remainder_mask_is_emitted(n):
    p = make_program("pointwise", (n,))
    assert "idx < N" in generate(p, Schedule())


def test_cache_roundtrip_corruption_and_invalid_path(tmp_path):
    c = Cache(tmp_path)
    key = digest({"x": 1})
    assert c.get(key) is None
    c.put(key, {"source": "safe"})
    assert c.get(key) == {"source": "safe"}
    (tmp_path / (key + ".json")).write_text('{"corrupt":true}')
    with pytest.raises(ValidationError, match="corrupt"):
        c.get(key)
    with pytest.raises(ValidationError):
        c.get("../secret")


def test_cache_atomic_concurrent_writers(tmp_path):
    c = Cache(tmp_path)
    key = digest({"entry": 1})
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda i: c.put(key, {"value": i}), range(20)))
    assert c.get(key)["value"] in range(20)
    assert c.stats()["entries"] == 1


def test_duplicate_operand_unfused_codegen_parses():
    p = Program("pointwise", (7,), (Node("x", "input"), Node("out", "mul", ("x", "x"))), "out")
    ast.parse(generate(p, Schedule(fusion=False)))


def test_live_common_subexpressions_merge():
    p = Program(
        "pointwise",
        (17,),
        (
            Node("x", "input"),
            Node("a", "const", value=1.0),
            Node("b", "const", value=1.0),
            Node("y", "add", ("x", "a")),
            Node("z", "add", ("x", "b")),
            Node("out", "add", ("y", "z")),
        ),
        "out",
    )
    q = canonicalize(p)
    assert [n.name for n in q.nodes] == ["x", "a", "y", "out"]
    assert q.nodes[-1].args == ("y", "y")


def test_fixed_schedule_source_reuses_runtime_dimensions():
    for operation, a, b in (
        ("pointwise", (127,), (2049,)),
        ("softmax", (17, 127), (33, 4093)),
        ("gemm", (17, 31, 33), (65, 63, 129)),
    ):
        schedule = Schedule(row_tile=4096)
        assert generate(make_program(operation, a), schedule) == generate(
            make_program(operation, b), schedule
        )
    with pytest.raises(ValidationError, match="exceeds fixed"):
        verify_schedule(make_program("softmax", (3, 129)), Schedule(row_tile=128))


@pytest.mark.parametrize("field,value", [("op", []), ("args", [[]])])
def test_ir_unhashable_node_fields_are_validation_errors(field, value):
    data = make_program("pointwise", (17,)).to_dict()
    data["nodes"][-1][field] = value
    with pytest.raises(ValidationError):
        Program.from_dict(data)
