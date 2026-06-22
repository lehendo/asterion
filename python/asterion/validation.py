"""Versioned empirical numerical evidence, never a universal correctness proof."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

from .codegen import generate
from .io import ValidationError, digest, integer, positive
from .ir import Schedule, canonicalize, make_program, verify_schedule
from .schema import shape_check

PATTERNS = ("normal", "zeros", "alternating", "wide_finite")
DEFAULT_SEEDS = (1729, 947)


def kernel_key(program, schedule, target):
    program = canonicalize(program)
    verify_schedule(program, schedule)
    return digest(
        {
            "ir": {k: v for k, v in program.to_dict().items() if k != "shape"},
            "rank": len(program.shape),
            "schedule": schedule.to_dict(),
            "target": target,
            "source": generate(program, schedule),
            "compiler_contract": 1,
            "math": "ieee-fp32",
            "enable_fp_fusion": False,
        }
    )


def seal(value, field="certificate_id"):
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = digest(result)
    return result


