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


def check_output(actual, reference, policy):
    """Check the declared application contract and retain informative residuals.

    A forward envelope is conditional on the standard IEEE rounding model and a
    trusted reference's absolute-product sum. It does not prove compiler behavior.
    """
    import torch

    policy = numerical_policy(policy)
    if not isinstance(reference, dict):
        reference = {"value": reference}
    expected = reference.get("value")
    if not isinstance(expected, torch.Tensor) or actual.shape != expected.shape:
        raise ValidationError("reference must return a matching tensor or value record")
    if actual.device != expected.device:
        raise ValidationError("reference must remain on the same CUDA device")
    x, y = actual.double(), expected.double()
    finite = bool(torch.isfinite(x).all().item() and torch.isfinite(y).all().item())
    if not finite:
        return {
            "passed": False,
            "status": "NONFINITE_RESULT",
            "policy": policy,
            "elements": actual.numel(),
        }
    error = (x - y).abs()
    nonzero = y != 0
    relative = (error[nonzero] / y[nonzero].abs()).max().item() if nonzero.any().item() else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    info = {
        "policy": policy,
        "elements": actual.numel(),
        "max_absolute_error": error.max().item(),
        "max_relative_error_nonzero": relative,
        "zero_reference_nonzero_outputs": int(((y == 0) & (x != 0)).sum().item()),
        "reference_kind": "trusted application reference; not authenticated",
        "sample_actual": x.reshape(-1)[:8].tolist(),
        "sample_reference": y.reshape(-1)[:8].tolist(),
        "sample_scope": "First up to8 flattened output elements, not necessarily the largest mismatch",
    }
    if policy["kind"] == "exact":
        permitted = torch.zeros_like(error)
    elif policy["kind"] == "allclose":
        permitted = policy["atol"] + policy["rtol"] * y.abs()
    else:
        scale = reference.get("absolute_product_sum")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.shape != actual.shape
            or scale.device != actual.device
            or scale.dtype != torch.float64
            or expected.dtype != torch.float64
        ):
            raise ValidationError(
                "forward policy needs matching FP64 reference and absolute_product_sum tensors"
            )
        scale = scale.double()
        if not torch.isfinite(scale).all().item() or (scale < 0).any().item():
            raise ValidationError("invalid forward-error scale")
        gamma = forward_gamma(reference.get("accumulation_length"))
        reference_gamma = forward_gamma(reference["accumulation_length"], 53)
        if not actual.is_floating_point():
            raise ValidationError("floating-point forward model is not an integer contract")
        f = torch.finfo(actual.dtype)
        unit = f.eps / 2
        accumulation = (gamma + reference_gamma) * scale / (1 - reference_gamma)
        permitted = (1 + unit) * accumulation + unit * y.abs() + f.tiny * f.eps / 2
        info["forward_model"] = {
            "model_version": 2,
            "gamma_2k": gamma,
            "reference_gamma_2k": reference_gamma,
            "output_unit_roundoff": unit,
            "accumulation_length": reference["accumulation_length"],
            "assumptions": "IEEE FP32 products/accumulation and FP64 reference/absolute-product sums, round-to-nearest, normal intermediate arithmetic without underflow/overflow or lossy FTZ; final output rounding/underflow allowance. Reference/scale algorithms are trusted. Not a universal proof.",
            "envelope": "(1+u_out)*(gamma_2k_fp32+gamma_2k_fp64)*scale/(1-gamma_2k_fp64) + u_out*abs(reference) + half_output_min_subnormal",
        }
        normalized = (error / permitted.clamp_min(torch.finfo(torch.float64).tiny)).max().item()
        info["max_model_normalized_error"] = normalized if math.isfinite(normalized) else None
    accepted = error <= permitted
    info.update(
        passed=bool(accepted.all().item()),
        mismatched_elements=int((~accepted).sum().item()),
        status="ACCEPTED" if bool(accepted.all().item()) else "APPLICATION_CONTRACT_NOT_SATISFIED",
        interpretation="Finite empirical check; residual alone does not identify an implementation defect or numerical instability",
    )
    if actual.dtype == torch.float32:
        # Numeric ULP distance treats positive and negative zero as equal.
        def ordered(t):
            bits = t.float().contiguous().view(torch.int32).long()
            return torch.where(bits < 0, -(2**31) - bits, bits)

        info["max_fp32_ulp_from_rounded_reference"] = (
            (ordered(actual) - ordered(expected)).abs().max().item()
        )
    return info
