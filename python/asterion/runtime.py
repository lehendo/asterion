from __future__ import annotations

import copy
import time
from dataclasses import dataclass

from .io import ValidationError, digest, integer
from .ir import Schedule, make_program, verify_schedule
from .optimizer import select
from .schema import shape_check
from .validation import certificate_index


def _validate_portfolio(p):
    if p.get("schema_version") != 1 or p.get("artifact_type") != "portfolio":
        raise ValidationError("expected version 1 portfolio")
    checked = copy.deepcopy(p)
    supplied = checked.pop("integrity_hash", None)
    if supplied != digest(checked):
        raise ValidationError("portfolio integrity hash mismatch")
    if (
        p.get("dtype") != "float32"
        or p.get("layout") != "contiguous"
        or p.get("fallback") != "pytorch"
    ):
        raise ValidationError("unsupported portfolio tensor/fallback contract")
    integer(p["max_variants"], "max_variants")
    if not isinstance(p.get("variants"), dict) or not 0 < len(p["variants"]) <= p["max_variants"]:
        raise ValidationError("variant budget violated")
    if p["generated_code_bytes"] > p["max_code_bytes"] or p["generated_code_bytes"] <= 0:
        raise ValidationError("code budget violated")
    d = p["dispatcher"]
    if d.get("feature") != "last_dimension" or d["minimum"] > d["maximum"]:
        raise ValidationError("invalid dispatch feature/range")
    last = d["minimum"] - 1
    for rule in d["rules"]:
        if rule["variant"] not in p["variants"] or rule["upper_inclusive"] <= last:
            raise ValidationError("invalid or non-monotone dispatch rules")
        last = rule["upper_inclusive"]
    if not d["rules"] or last != d["maximum"]:
        raise ValidationError("dispatcher does not cover its declared feature range")
    if not p.get("train_shapes"):
        raise ValidationError("portfolio requires training shape provenance")
    for shape in p["train_shapes"]:
        program = make_program(p["operation"], shape_check(shape, p["operation"]))
        variant, _ = select(p, shape)
        if variant is None or variant not in p["variants"]:
            raise ValidationError("training shape lacks a dispatch rule")
        verify_schedule(program, Schedule.from_dict(p["variants"][variant]))
    certificates = certificate_index(p)
    return {
        "status": "validated_metadata",
        "variants": len(p["variants"]),
        "certificate_count": len(certificates),
        "gpu_execution": "not_performed_by_metadata_validation",
        "integrity_hash": supplied,
    }


def validate_portfolio(p):
    try:
        return _validate_portfolio(p)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValidationError(f"malformed portfolio: {exc}") from exc


def input_shape(operation, inputs, torch):
    if len(inputs) != (2 if operation == "gemm" else 1):
        raise ValidationError("wrong number of runtime inputs")
    for x in inputs:
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype != torch.float32:
            raise ValidationError("runtime requires CUDA float32 tensors")
        if not x.is_contiguous() or x.requires_grad:
            raise ValidationError("runtime requires contiguous inference tensors")
        if x.device.index != torch.cuda.current_device():
            raise ValidationError("all inputs must be on the current CUDA device")
    if operation == "gemm":
        if inputs[0].ndim != 2 or inputs[1].ndim != 2 or inputs[0].shape[1] != inputs[1].shape[0]:
            raise ValidationError("incompatible GEMM inputs")
        shape = (inputs[0].shape[0], inputs[1].shape[1], inputs[0].shape[1])
    else:
        shape = tuple(inputs[0].shape)
    return shape_check(shape, operation)


@dataclass
class PreparedInvocation:
    """Owns buffers and borrowed tensor owners; launch reuses its output intentionally."""

    selection: dict
    bound: object = None
    fallback: object = None
    preparation_wall_s: float = 0.0

    def launch(self):
        return self.bound.launch() if self.bound is not None else self.fallback()

    @property
    def output(self):
        return self.bound.output if self.bound is not None else None


