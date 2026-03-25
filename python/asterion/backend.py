from __future__ import annotations

import importlib.util
import os
import tempfile

from .cache import Cache
from .codegen import generate
from .environment import command
from .io import ValidationError
from .ir import canonicalize, verify, verify_schedule


def require_gpu():
    scheduler_guard = os.environ.get("ASTERION_REQUIRE_SLURM", "0")
    if scheduler_guard not in {"0", "1"}:
        raise ValidationError("ASTERION_REQUIRE_SLURM must be 0 or 1")
    if scheduler_guard == "1" and not os.environ.get("SLURM_JOB_ID"):
        raise ValidationError("cluster policy requires an authorized Slurm GPU allocation")
    try:
        import torch
        import triton
    except ImportError as exc:
        raise ValidationError(
            "install the optional GPU dependencies in the intended environment"
        ) from exc
    if not torch.cuda.is_available():
        raise ValidationError(
            "CUDA unavailable: use a compatible CUDA device; cluster runs require a Slurm GPU allocation"
        )
    return torch, triton


_TARGET_CACHE = {}


def target_manifest():
    torch, triton = require_gpu()
    device = torch.cuda.current_device()
    if device in _TARGET_CACHE:
        return dict(_TARGET_CACHE[device])
    props = torch.cuda.get_device_properties(device)
    result = {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "total_memory": props.total_memory,
        "uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_build": torch.version.cuda,
        "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "mig_inventory": command(["nvidia-smi", "-L"]),
    }
    capability = tuple(result["capability"])
    if triton.__version__ == "3.3.1":
        if capability < (7, 5):
            raise ValidationError("legacy backend requires compute capability 7.5 or newer")
    elif capability < (8, 0):
        raise ValidationError(
            "modern backend requires compute capability 8.0 or newer; SM75 needs the validated legacy stack"
        )
    _TARGET_CACHE[device] = result
    return dict(result)


