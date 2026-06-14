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


class Runtime:
    """Inference runtime. Caller supplies finite data and explicit producer stream dependencies."""

    def __init__(self, portfolio, *, check_finite=False, launch_backend="triton"):
        validate_portfolio(portfolio)
        if type(check_finite) is not bool:
            raise ValidationError("check_finite must be boolean")
        self.portfolio = copy.deepcopy(portfolio)
        self.certificates = certificate_index(self.portfolio)
        self.kernels = {}
        self.plans = {}
        self.last_selection = None
        self.check_finite = check_finite
        if launch_backend not in {"triton", "native"}:
            raise ValidationError("launch_backend must be triton or native")
        self.launch_backend = launch_backend
        self.native_modules = {}
        self.closed = False

    def _plan(self, shape):
        from .backend import Kernel, target_manifest

        key = (tuple(shape), target_manifest()["uuid"])
        if key in self.plans:
            return self.plans[key]
        variant, reason = select(self.portfolio, shape)
        if self.portfolio.get("measurement_kind") != "measured":
            variant, reason = None, "unverified_portfolio_provenance"
        if self.portfolio.get("target") != target_manifest():
            variant, reason = None, "target_mismatch"
        certificate = self.certificates.get((variant, tuple(shape)))
        if variant is not None and certificate is None:
            variant, reason = None, "validation_certificate_missing"
        kernel = None
        if variant is not None:
            program = make_program(self.portfolio["operation"], shape)
            schedule = Schedule.from_dict(self.portfolio["variants"][variant])
            verify_schedule(program, schedule)
            cache_key = (variant, tuple(shape))
            kernel = self.kernels.setdefault(cache_key, None)
            if kernel is None:
                kernel = self.kernels[cache_key] = Kernel(program, schedule)
        plan = {
            "variant": variant,
            "reason": reason,
            "kernel": kernel,
            "certificate": certificate,
            "binary_verified": False,
        }
        self.plans[key] = plan
        return plan

    def prepare(self, *inputs):
        from .backend import binary_metadata, require_gpu
        from .gpu import operation_reference

        if self.closed:
            raise ValidationError("runtime is closed")
        started = time.perf_counter()
        torch, _ = require_gpu()
        op = self.portfolio["operation"]
        shape = input_shape(op, inputs, torch)
        if self.check_finite and any(not torch.isfinite(x).all().item() for x in inputs):
            raise ValidationError("nonfinite input violates the finite numerical contract")
        plan = self._plan(shape)
        reason = plan["reason"]
        kernel = plan["kernel"]
        if kernel is not None and any(x.data_ptr() % 16 for x in inputs):
            kernel, reason = None, "input_alignment_outside_certificate"
        selection = {
            "variant": plan["variant"] if kernel is not None else None,
            "reason": reason,
            "shape": list(shape),
            "certificate_id": plan["certificate"]["certificate_id"] if kernel is not None else None,
        }
        if kernel is None:
            baseline = operation_reference(op)
            invocation = PreparedInvocation(selection, fallback=lambda: baseline(*inputs))
        else:
            bound = kernel.prepare(*inputs)
            if not plan["binary_verified"]:
                bound.launch()
                actual = binary_metadata(bound)["compiled_binary_hashes"]
                if actual != plan["certificate"]["compiled_binary_hashes"]:
                    raise ValidationError("compiled binary differs from validated certificate")
                plan["binary_verified"] = True
            if self.launch_backend == "native":
                from .driver import DriverModule

                module = self.native_modules.get(kernel.key)
                if module is None:
                    # Initialize Triton metadata on each shape-specific bound object.
                    if not hasattr(bound, "last_compiled"):
                        bound.launch()
                    module = DriverModule(bound.last_compiled, kernel)
                    self.native_modules[kernel.key] = module
                bound = module.bind(bound)
            selection["launch_backend"] = self.launch_backend
            invocation = PreparedInvocation(selection, bound=bound)
        invocation.preparation_wall_s = time.perf_counter() - started
        self.last_selection = selection
        return invocation

    def close(self):
        """Wait for launches and unload owned modules; invalidates prepared native invocations.

        Retain the runtime and prepared invocations until all CUDA Graph replays finish.
        Never replay a graph after close(). Close uses a context-wide barrier if
        graphs were captured; ordinary launches use per-stream completion events.
        The caller owns producer dependencies.
        """
        self.closed = True
        for module in self.native_modules.values():
            module.close()
        self.native_modules.clear()

    def __call__(self, *inputs):
        # Each invocation gets fresh output ownership. prepare() explicitly opts into reuse.
        return self.prepare(*inputs).launch()
