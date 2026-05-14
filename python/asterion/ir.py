"""Bounded typed SSA tensor IR. Verification checks implemented invariants, not formal proof."""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import asdict, dataclass, replace

from .io import ValidationError, digest, exact_keys, integer
from .schema import OPERATIONS, shape_check

ARITY = {
    "input": 0,
    "const": 0,
    "add": 2,
    "mul": 2,
    "sub": 2,
    "div": 2,
    "relu": 1,
    "exp": 1,
    "sum": 1,
    "max": 1,
    "matmul": 2,
}
IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    args: tuple[str, ...] = ()
    value: float | None = None


@dataclass(frozen=True)
class Program:
    operation: str
    shape: tuple[int, ...]
    nodes: tuple[Node, ...]
    output: str
    dtype: str = "float32"
    layout: str = "contiguous"
    schema_version: int = 1

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "shape": list(self.shape),
            "nodes": [{**asdict(n), "args": list(n.args)} for n in self.nodes],
            "output": self.output,
            "dtype": self.dtype,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, d):
        exact_keys(
            d, {"schema_version", "operation", "shape", "nodes", "output", "dtype", "layout"}, "IR"
        )
        try:
            nodes = []
            for n in d["nodes"]:
                exact_keys(n, {"name", "op", "args", "value"}, "IR node")
                nodes.append(Node(n["name"], n["op"], tuple(n["args"]), n.get("value")))
            result = cls(
                d["operation"],
                tuple(d["shape"]),
                tuple(nodes),
                d["output"],
                d["dtype"],
                d["layout"],
                d["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"malformed IR: {exc}") from exc
        verify(result)
        return result

    @property
    def hash(self):
        verify(self)
        return digest(self.to_dict())


def verify(p):
    if p.schema_version != 1 or p.operation not in OPERATIONS:
        raise ValidationError("unsupported IR version/operation")
    shape_check(p.shape, p.operation)
    if p.dtype != "float32" or p.layout != "contiguous":
        raise ValidationError("IR supports only contiguous float32")
    if not p.nodes or len(p.nodes) > 128:
        raise ValidationError("IR requires 1..128 nodes")
    ranks, inputs = {}, 0
    for n in p.nodes:
        if (
            not isinstance(n.name, str)
            or not IDENT.fullmatch(n.name)
            or n.name in ranks
            or keyword.iskeyword(n.name)
            or n.name
            in {
                "tl",
                "triton",
                "float",
                "range",
                "rr",
                "cc",
                "mask",
                "idx",
                "acc",
                "rm",
                "rn",
                "rk",
                "kk",
                "outer",
                "inner",
            }
        ):
            raise ValidationError("SSA names must be unique safe identifiers")
        if not isinstance(n.op, str) or n.op not in ARITY or len(n.args) != ARITY[n.op]:
            raise ValidationError(f"{n.name}: invalid operation/arity")
        if any(not isinstance(a, str) or a not in ranks for a in n.args):
            raise ValidationError(f"{n.name}: use before definition")
        if n.op == "const":
            if (
                isinstance(n.value, bool)
                or not isinstance(n.value, (float, int))
                or not math.isfinite(n.value)
            ):
                raise ValidationError("constant must be finite")
            ranks[n.name] = "scalar"
        elif n.op == "input":
            inputs += 1
            ranks[n.name] = "tensor"
        elif n.op in {"sum", "max"}:
            if p.operation not in {"sum", "max", "softmax"} or ranks[n.args[0]] != "tensor":
                raise ValidationError("row reductions require a rank-2 tensor")
            ranks[n.name] = "row"
        elif n.op == "matmul":
            if p.operation != "gemm" or any(ranks[a] != "tensor" for a in n.args):
                raise ValidationError("matmul requires two tensor operands")
            ranks[n.name] = "tensor"
        else:
            kinds = {ranks[a] for a in n.args}
            ranks[n.name] = "tensor" if "tensor" in kinds else "row" if "row" in kinds else "scalar"
    if inputs != (2 if p.operation == "gemm" else 1):
        raise ValidationError("unsupported input arity")
    expected = "row" if p.operation in {"sum", "max"} else "tensor"
    if p.output not in ranks or ranks[p.output] != expected:
        raise ValidationError("output rank does not match operation")
    if p.operation == "gemm" and (
        p.output != p.nodes[-1].name
        or [(n.op, n.args) for n in p.nodes]
        != [
            ("input", ()),
            ("input", ()),
            ("matmul", (p.nodes[0].name, p.nodes[1].name)),
        ]
    ):
        raise ValidationError("GEMM currently supports only a bounded pure contraction")
    return ranks


def make_program(operation, shape):
    nodes = [Node("x", "input")]
    if operation == "pointwise":
        nodes += [
            Node("scale", "const", value=1.25),
            Node("bias", "const", value=0.5),
            Node("scaled", "mul", ("x", "scale")),
            Node("biased", "add", ("scaled", "bias")),
            Node("out", "relu", ("biased",)),
        ]
    elif operation in {"sum", "max"}:
        nodes += [Node("out", operation, ("x",))]
    elif operation == "softmax":
        nodes += [
            Node("maximum", "max", ("x",)),
            Node("shifted", "sub", ("x", "maximum")),
            Node("exponents", "exp", ("shifted",)),
            Node("denominator", "sum", ("exponents",)),
            Node("out", "div", ("exponents", "denominator")),
        ]
    elif operation == "gemm":
        nodes += [Node("y", "input"), Node("out", "matmul", ("x", "y"))]
    else:
        raise ValidationError("unsupported operation")
    p = Program(operation, tuple(shape), tuple(nodes), "out")
    verify(p)
    return p


def canonicalize(p):
    """Pure SSA DCE and CSE; constant arithmetic uses no algebraic FP rewrites."""
    verify(p)
    live = set()
    by_name = {n.name: n for n in p.nodes}

    def visit(name):
        if name in live:
            return
        live.add(name)
        for arg in by_name[name].args:
            visit(arg)

    visit(p.output)
    # Preserve input signature, even for an unused input.
    live.update(n.name for n in p.nodes if n.op == "input")
    names, seen, result = {}, {}, []
    for n in p.nodes:
        if n.name not in live:
            continue
        args = tuple(names.get(a, a) for a in n.args)
        key = (n.op, args, n.value)
        if n.op != "input" and key in seen:
            names[n.name] = seen[key]
            continue
        result.append(replace(n, args=args))
        seen[key] = n.name
        names[n.name] = n.name
    q = replace(p, nodes=tuple(result), output=names.get(p.output, p.output))
    verify(q)
    return q


@dataclass(frozen=True)
class Schedule:
    row_tile: int = 0
    block: int = 256
    warps: int = 4
    rows: int = 1
    tile_m: int = 32
    tile_n: int = 32
    tile_k: int = 32
    unroll: int = 1
    fusion: bool = True
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        exact_keys(d, cls.__dataclass_fields__, "schedule")
        try:
            s = cls(**d)
        except TypeError as exc:
            raise ValidationError(f"invalid schedule: {exc}") from exc
        verify_schedule_fields(s)
        return s

    @property
    def id(self):
        return digest(self.to_dict())[:16]


def verify_schedule_fields(s):
    if s.schema_version != 1 or type(s.fusion) is not bool:
        raise ValidationError("invalid schedule version/fusion")
    for name in ("block", "warps", "rows", "tile_m", "tile_n", "tile_k", "unroll"):
        value = getattr(s, name)
        integer(value, name)
        if value & (value - 1):
            raise ValidationError(f"{name} must be a power of two")
    if s.warps not in {1, 2, 4, 8} or s.rows not in {1, 2, 4, 8}:
        raise ValidationError("unsupported warp/row mapping")
    if (
        not 32 <= s.block <= 4096
        or not 16 <= s.tile_m <= 64
        or not 16 <= s.tile_n <= 64
        or not 16 <= s.tile_k <= 64
    ):
        raise ValidationError("schedule tile exceeds supported bounds")
    integer(s.row_tile, "row_tile", minimum=0)
    if s.row_tile and (s.row_tile > 16384 or s.row_tile & (s.row_tile - 1)):
        raise ValidationError("row_tile must be zero (adaptive) or a power of two <=16384")
    if s.unroll not in {1, 2, 4}:
        raise ValidationError("unsupported unroll")


def verify_schedule(p, s):
    verify(p)
    verify_schedule_fields(s)
    if p.operation in {"sum", "max", "softmax"}:
        padded = s.row_tile or 1 << (p.shape[1] - 1).bit_length()
        if p.shape[1] > padded:
            raise ValidationError("row width exceeds fixed schedule tile")
        if padded > 16384 or s.rows * padded > 65536:
            raise ValidationError("row tile exceeds supported 65536-element working set")
        if not s.fusion:
            raise ValidationError("unfused row reductions are not supported")
    if p.operation == "gemm" and not s.fusion:
        raise ValidationError("unfused GEMM is unsupported")
    if p.operation != "gemm" and s.unroll != 1:
        raise ValidationError("loop unroll is implemented only for GEMM")
    return s


def transform(p, s, **changes):
    """A schedule rewrite with a serialized audit trail and legality check."""
    allowed = {
        "block",
        "warps",
        "rows",
        "tile_m",
        "tile_n",
        "tile_k",
        "unroll",
        "fusion",
        "row_tile",
    }
    if set(changes) - allowed:
        raise ValidationError("unsupported schedule transformation")
    after = replace(s, **changes)
    verify_schedule(p, after)
    return after, {
        "before": s.to_dict(),
        "after": after.to_dict(),
        "changes": changes,
        "program_hash": p.hash,
        "legality": "checked",
    }


def candidates(p):
    if p.operation == "pointwise":
        variants = [
            Schedule(block=b, warps=w, fusion=f)
            for b in (128, 256, 1024)
            for w in (4, 8)
            for f in (True, False)
        ]
    elif p.operation == "gemm":
        variants = [
            Schedule(tile_m=m, tile_n=n, tile_k=k, warps=4, unroll=u)
            for m, n, k in ((16, 32, 32), (32, 32, 32), (32, 64, 32), (64, 32, 32))
            for u in (1, 2)
        ]
    else:
        variants = [
            Schedule(rows=r, warps=w, row_tile=t)
            for t in (128, 1024, 4096, 16384)
            for r in (1, 4)
            for w in (4, 8)
        ]
    legal = []
    for s in variants:
        try:
            verify_schedule(p, s)
        except ValidationError:
            continue
        legal.append(s)
    return legal
