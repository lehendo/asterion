from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .io import ValidationError, exact_keys, integer, positive, read

OPERATIONS = ("pointwise", "sum", "max", "softmax", "gemm")
SPLITS = ("train", "validation", "test")


def shape_check(shape, operation):
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValidationError("shape must be a nonempty array")
    expected = 1 if operation == "pointwise" else 3 if operation == "gemm" else 2
    if len(shape) != expected:
        raise ValidationError(f"{operation} requires {expected} dimensions")
    result = tuple(integer(d, "dimension") for d in shape)
    if any(d > 2**24 for d in result):
        raise ValidationError("individual dimensions above 2**24 are unsupported")
    # All code generation currently uses signed 32-bit linear indexing.
    import math

    sizes = (
        [result[0] * result[2], result[2] * result[1], result[0] * result[1]]
        if operation == "gemm"
        else [math.prod(result)]
    )
    if max(sizes) > 2**31 - 1:
        raise ValidationError("tensor size exceeds supported int32 indexing")
    return result


@dataclass(frozen=True)
class Case:
    id: str
    shape: tuple[int, ...]
    split: str
    weight: float = 1.0


@dataclass(frozen=True)
class WorkloadSpec:
    operation: str
    cases: tuple[Case, ...]
    seed: int = 1729
    dtype: str = "float32"
    layout: str = "contiguous"
    rtol: float = 1e-4
    atol: float = 1e-5
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data):
        exact_keys(
            data,
            {"schema_version", "operation", "cases", "seed", "dtype", "layout", "rtol", "atol"},
            "workload",
        )
        if data.get("schema_version") != 1:
            raise ValidationError("unsupported workload schema_version; expected 1")
        op = data.get("operation")
        if op not in OPERATIONS:
            raise ValidationError(f"unsupported operation {op!r}")
        if (
            data.get("dtype", "float32") != "float32"
            or data.get("layout", "contiguous") != "contiguous"
        ):
            raise ValidationError("only contiguous float32 is supported")
        entries = data.get("cases")
        if not isinstance(entries, list) or not entries:
            raise ValidationError("cases must be a nonempty array")
        cases, ids, shapes = [], set(), set()
        for item in entries:
            exact_keys(item, {"id", "shape", "split", "weight"}, "case")
            name = item.get("id")
            if not isinstance(name, str) or not name or name in ids:
                raise ValidationError("case id must be a unique nonempty string")
            split = item.get("split")
            if split not in SPLITS:
                raise ValidationError("case split must be train, validation or test")
            shape = shape_check(item.get("shape"), op)
            if shape in shapes:
                raise ValidationError(
                    "duplicate shapes across cases/splits would leak evaluation data"
                )
            cases.append(Case(name, shape, split, positive(item.get("weight", 1), "weight")))
            ids.add(name)
            shapes.add(shape)
        seed = integer(data.get("seed", 1729), "seed", minimum=0)
        return cls(
            op,
            tuple(cases),
            seed,
            "float32",
            "contiguous",
            positive(data.get("rtol", 1e-4), "rtol"),
            positive(data.get("atol", 1e-5), "atol"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path):
        return cls.from_dict(read(path))

    def to_dict(self):
        result = asdict(self)
        result["cases"] = [{**asdict(c), "shape": list(c.shape)} for c in self.cases]
        return result


