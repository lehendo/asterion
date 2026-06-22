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


def validate_certificate(certificate, portfolio, variant):
    try:
        c = copy.deepcopy(certificate)
        supplied = c.pop("certificate_id")
        if supplied != digest(c):
            raise ValidationError("validation certificate integrity mismatch")
        if c["schema_version"] != 1 or c["artifact_type"] != "validation_certificate":
            raise ValidationError("unsupported validation certificate")
        if c["measurement_kind"] != "measured" or c["status"] != "passed":
            raise ValidationError("only passed measured certificates authorize deployment")
        if c["operation"] != portfolio["operation"] or c["target"] != portfolio["target"]:
            raise ValidationError("validation certificate operation/target mismatch")
        if c["variant"] != variant or c["schedule"] != portfolio["variants"][variant]:
            raise ValidationError("validation certificate schedule mismatch")
        if c["dtype"] != "float32" or c["layout"] != "contiguous":
            raise ValidationError("unsupported certificate tensor contract")
        if c["input_contract"] != "finite_float32_contiguous_aligned16":
            raise ValidationError("unsupported certificate input contract")
        integer(c["generated_code_bytes"], "generated_code_bytes")
        positive(c["rtol"], "rtol")
        positive(c["atol"], "atol")
        shape = shape_check(c["shape"], c["operation"])
        schedule = Schedule.from_dict(c["schedule"])
        program = make_program(c["operation"], shape)
        if c["program_hash"] != program.hash or c["kernel_key"] != kernel_key(
            program, schedule, c["target"]
        ):
            raise ValidationError("validation certificate is stale for generated code")
        if not c["compiled_binary_hashes"] or any(
            not isinstance(h, str) or len(h) != 64 for h in c["compiled_binary_hashes"]
        ):
            raise ValidationError("certificate must identify measured compiled binaries")
        seeds = c["seeds"]
        if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
            raise ValidationError("certificate needs unique deterministic seeds")
        for seed in seeds:
            integer(seed, "seed", minimum=0)
        expected = {(pattern, seed) for pattern in PATTERNS for seed in seeds}
        observed = {(r["pattern"], r["seed"]) for r in c["checks"] if r["passed"] is True}
        if observed != expected or len(c["checks"]) != len(expected):
            raise ValidationError("certificate does not cover its declared numerical suite")
        return tuple(shape)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(f"malformed validation certificate: {exc}") from exc


