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


