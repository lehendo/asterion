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


