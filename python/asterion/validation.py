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


def certificate_index(portfolio):
    index = {}
    binaries = {}
    for c in portfolio.get("validation_certificates", []):
        variant = c.get("variant")
        if variant not in portfolio["variants"]:
            raise ValidationError("certificate references an unknown variant")
        shape = validate_certificate(c, portfolio, variant)
        key = (variant, shape)
        if key in index:
            raise ValidationError("duplicate validation certificate for variant/shape")
        identity = (c["kernel_key"], tuple(c["compiled_binary_hashes"]), c["generated_code_bytes"])
        if variant in binaries and binaries[variant] != identity:
            raise ValidationError("one portfolio variant requires inconsistent compiled binaries")
        binaries[variant] = identity
        index[key] = c
    if sum(identity[2] for identity in binaries.values()) > portfolio["max_code_bytes"]:
        raise ValidationError("validated compiled binaries exceed code budget")
    return index


def pattern_inputs(program, pattern, seed):
    from .backend import make_inputs, require_gpu

    torch, _ = require_gpu()
    values = make_inputs(program, seed)
    if pattern == "normal":
        return values
    if pattern == "zeros":
        return tuple(torch.zeros_like(x) for x in values)
    outputs = []
    for x in values:
        idx = torch.arange(x.numel(), device=x.device).reshape(x.shape)
        if pattern == "alternating":
            outputs.append(torch.where(idx % 2 == 0, 1.0, -1.0).to(torch.float32))
        elif pattern == "wide_finite":
            scale = 80.0 if program.operation == "softmax" else 4.0
            outputs.append(
                torch.linspace(-scale, scale, x.numel(), device=x.device).reshape(x.shape)
            )
        else:
            raise ValidationError("unknown numerical test pattern")
    return tuple(outputs)


def certify_variant(program, schedule, variant, *, rtol=1e-4, atol=1e-5, seeds=DEFAULT_SEEDS):
    from .backend import Kernel, binary_metadata, reference, require_gpu

    torch, _ = require_gpu()
    positive(rtol, "rtol")
    positive(atol, "atol")
    for seed in seeds:
        integer(seed, "seed", minimum=0)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValidationError("certification needs unique seeds")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    kernel = Kernel(program, schedule)
    checks, signature = [], None
    for pattern in PATTERNS:
        for seed in seeds:
            inputs = pattern_inputs(program, pattern, seed)
            expected = reference(program, *(x.double() for x in inputs)).float()
            bound = kernel.prepare(*inputs)
            actual = bound.launch()
            torch.cuda.current_stream().synchronize()
            try:
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            except AssertionError as exc:
                raise ValidationError(
                    f"numerical validation failed for {pattern}/{seed}: {exc}"
                ) from exc
            metadata = binary_metadata(bound)
            current = metadata["compiled_binary_hashes"]
            if signature is not None and signature != current:
                raise ValidationError("numerical suite compiled inconsistent binaries")
            signature = current
            difference = (actual - expected).abs()
            checks.append(
                {
                    "pattern": pattern,
                    "seed": seed,
                    "passed": True,
                    "max_absolute_error": difference.max().item(),
                }
            )
    return seal(
        {
            "schema_version": 1,
            "artifact_type": "validation_certificate",
            "measurement_kind": "measured",
            "status": "passed",
            "operation": program.operation,
            "shape": list(program.shape),
            "dtype": "float32",
            "layout": "contiguous",
            "input_contract": "finite_float32_contiguous_aligned16",
            "variant": variant,
            "schedule": schedule.to_dict(),
            "target": kernel.target,
            "program_hash": program.hash,
            "kernel_key": kernel.key,
            "compiled_binary_hashes": signature,
            "generated_code_bytes": metadata["generated_code_bytes"],
            "reference": "operation IR evaluated in torch float64, cast to float32",
            "rtol": rtol,
            "atol": atol,
            "seeds": list(seeds),
            "checks": checks,
            "checked_at": datetime.now(UTC).isoformat(),
            "guarantee": "empirical checks for this exact shape, implementation and target; not proof for all finite input values",
        }
    )


