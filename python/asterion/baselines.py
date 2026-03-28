"""Isolated official-framework adapters. No imitation is labeled as an upstream tuner."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from hashlib import sha256

from .codegen import generate
from .io import ValidationError, digest
from .ir import Schedule, candidates, make_program
from .stats import weighted_quantile


def tunable_row_source(operation):
    if operation not in {"sum", "max", "softmax"}:
        raise ValidationError("official common-space adapter currently supports row operations")
    source = generate(make_program(operation, (1, 1)), Schedule(row_tile=16384, rows=1))
    source = source.replace(
        "def kernel(X, O, R, C):",
        "def kernel(X, O, R, C, ROW_TILE: tl.constexpr, ROW_GROUP: tl.constexpr):",
    )
    source = source.replace(
        "tl.program_id(0)*1 + tl.arange(0, 1)",
        "tl.program_id(0)*ROW_GROUP + tl.arange(0, ROW_GROUP)",
    )
    source = source.replace("tl.arange(0, 16384)", "tl.arange(0, ROW_TILE)")
    if source.count("ROW_GROUP") != 3 or source.count("ROW_TILE") != 2:
        raise ValidationError(
            "generated row adapter contract changed; refusing ambiguous parameterization"
        )
    return source


