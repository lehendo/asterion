"""Explicit empirical numerical policies; CPU parsing does not import torch."""

import math

from .io import ValidationError, exact_keys


def numerical_policy(value):
    exact_keys(
        value,
        {"kind", "rtol", "atol", "justification", "accumulation_bits", "model_version"},
        "numerical policy",
    )
    if value.get("kind") not in {"exact", "allclose", "matmul_forward_bound"}:
        raise ValidationError("numerical kind must be exact, allclose or matmul_forward_bound")
    if (
        not isinstance(value.get("justification"), str)
        or not 1 <= len(value["justification"]) <= 2048
    ):
        raise ValidationError("numerical policy requires an application justification")
    result = dict(value)
    if value["kind"] == "allclose":
        for name in ("rtol", "atol"):
            x = value.get(name)
            if type(x) not in {int, float} or not math.isfinite(x) or x < 0:
                raise ValidationError("explicit finite nonnegative rtol and atol required")
    elif "rtol" in value or "atol" in value:
        raise ValidationError("rtol/atol only apply to allclose")
    if value["kind"] == "matmul_forward_bound":
        if type(value.get("model_version")) is not int or value["model_version"] != 2:
            raise ValidationError(
                "forward policy requires explicit model_version=2; use the original vendored reproducer for older development artifacts rather than silently reinterpreting their contract"
            )
        if value.get("accumulation_bits") != 24:
            raise ValidationError("forward model currently supports24-bit FP32 significands only")
    elif "accumulation_bits" in value:
        raise ValidationError("accumulation_bits only applies to the forward model")
    elif "model_version" in value:
        raise ValidationError("model_version only applies to the forward model")
    return result


def forward_gamma(length, significand_bits=24):
    if type(length) is not int or length < 1:
        raise ValidationError("reference must declare a positive accumulation length")
    nu = 2 * length * 2.0 ** (-significand_bits)
    if nu >= 0.5:
        raise ValidationError("accumulation exceeds the bounded forward-error model")
    return nu / (1 - nu)


