"""Passive contracts for a bounded, explicitly trusted external Triton adapter.

This module never imports the declared workload, torch or Triton. Parameter
legality is a user declaration, not an Asterion IR transformation proof.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path

from .io import ValidationError, artifact_path, digest, exact_keys, integer
from .numerics import numerical_policy

DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int32": 4}
OPTIONS = {"num_warps", "num_stages", "enable_fp_fusion"}
EXECUTION_FILES = (
    "external.py",
    "external_runtime.py",
    "external_report.py",
    "experiment_stats.py",
    "numerics.py",
    "io.py",
)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value):
        raise ValidationError("expected a bounded Python identifier")
    return value


def expression(value, dimensions, parameters, *, depth=0):
    if depth > 16:
        raise ValidationError("expression nesting exceeds16")
    if type(value) in {int, float} and math.isfinite(value):
        return value
    if not isinstance(value, dict) or len(value) != 1:
        raise ValidationError("expected a literal or one declared arithmetic expression")
    op, args = next(iter(value.items()))
    if op in {"dim", "param"}:
        table = dimensions if op == "dim" else parameters
        if args not in table or type(table[args]) not in {int, float}:
            raise ValidationError(f"unknown or nonnumeric {op}: {args}")
        return table[args]
    evaluate = lambda x: expression(x, dimensions, parameters, depth=depth + 1)
    if op == "pow2":
        n = evaluate(args)
        integer(n, "power-of-two operand")
        if n > 2**30:
            raise ValidationError("power-of-two operand too large")
        return 1 << (n - 1).bit_length()
    if op not in {"add", "mul", "ceildiv"} or not isinstance(args, list) or len(args) != 2:
        raise ValidationError("supported arithmetic: add, mul, ceildiv, pow2")
    a, b = map(evaluate, args)
    if op == "ceildiv":
        integer(a, "ceildiv numerator", minimum=0)
        integer(b, "ceildiv denominator")
        result = (a + b - 1) // b
    else:
        result = a + b if op == "add" else a * b
    if not math.isfinite(result) or abs(result) > 2**40:
        raise ValidationError("arithmetic expression exceeds supported bound")
    return result


