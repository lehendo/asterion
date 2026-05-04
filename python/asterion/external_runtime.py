"""Trusted single-module Triton experiments; also exported as standalone support.

Only explicit execution enters run(). A Python process is not a security sandbox.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import linecache
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .experiment_stats import paired_effect
from .external import (
    EXECUTION_FILES,
    capabilities,
    complexity,
    expression,
    interventions,
    source_path,
    specification,
)
from .io import ValidationError, atomic_json, digest, read
from .numerics import check_output


def _load_reviewed_module(path, expected_sha256):
    """Execute exact reviewed bytes; never substitute a timestamp-based pyc.

    This intentional execution helper is only for explicitly trusted workflows.
    Passive contract parsing/export never calls it.
    """
    path = Path(path).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValidationError("reviewed module changed before execution")
    name = "trusted_kernel_" + expected_sha256
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    linecache.cache.pop(str(path), None)
    # Trusted execution uses verified source bytes instead of timestamp-based pyc.
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def static_ptx_observations(source):
    """Summarize selected line-leading PTX opcodes, never dynamic instructions."""
    if not isinstance(source, str):
        return {"category": "UNAVAILABLE", "reason": "Textual PTX unavailable"}
    text = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    prefixes = (
        "ldmatrix",
        "mma.sync",
        "wgmma.mma_async",
        "ld.global",
        "st.global",
        "ld.shared",
        "st.shared",
        "bar.sync",
    )
    counts = dict.fromkeys(prefixes, 0)
    for line in text.splitlines():
        line = line.split("//", 1)[0]
        match = re.match(r"^\s*(?:@!?%[A-Za-z0-9_]+\s+)?([a-z][a-z0-9_.]*)\s", line)
        if match:
            opcode = match.group(1)
            for prefix in prefixes:
                if opcode == prefix or opcode.startswith(prefix + "."):
                    counts[prefix] += 1
    return {
        "category": "COMPILER_EVIDENCE",
        "static_opcode_prefix_counts": counts,
        "interpretation": "Heuristic line-leading opcode statements in emitted PTX; inline blocks may be omitted. Not dynamic instruction counts, occupancy or hardware bottleneck evidence. PTX can differ from final SASS",
    }


def _target(torch, triton):
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return {
        "name": p.name,
        "capability": [p.major, p.minor],
        "uuid": str(getattr(p, "uuid", "unavailable")),
        "total_memory": p.total_memory,
        "multiprocessors": p.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_build": torch.version.cuda,
        "driver": driver,
        "tf32_reference": False,
        "compiler": "Triton; no independent native GPU lowering",
    }


def _inputs(spec, torch, seed, pattern):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = {}
    for name, tensor in spec["tensors"].items():
        if tensor["role"] != "input":
            continue
        shape = [expression(x, spec["dimensions"], {}) for x in tensor["shape"]]
        dtype = getattr(torch, tensor["dtype"])
        if tensor.get("generator") == "ones":
            result[name] = torch.ones(shape, device="cuda", dtype=dtype)
            continue
        if dtype in {torch.int8, torch.int32}:
            x = torch.randint(-3, 4, shape, device="cuda", dtype=dtype, generator=generator)
            if pattern == "zeros":
                x.zero_()
            elif pattern == "alternating":
                x = (torch.arange(x.numel(), device="cuda").reshape(shape) % 2 * 2 - 1).to(dtype)
        else:
            x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
            if pattern == "zeros":
                x.zero_()
            elif pattern == "alternating":
                x = (torch.arange(x.numel(), device="cuda").reshape(shape) % 2 * 2 - 1).float()
            elif pattern == "wide_finite":
                x *= 8
            x = x.to(dtype)
        result[name] = x.contiguous()
    return result


class _KernelLaunch:
    """Own tensors and the loaded binary without a state/closure reference cycle."""

    def __init__(self, entry, grid, kwargs, output):
        self.entry = entry
        self.grid = grid
        self.kwargs = kwargs
        self.output = output
        self.compiled = None

    def __call__(self):
        self.compiled = self.entry[self.grid](**self.kwargs)
        return self.output


def _prepare(spec, variant, inputs, module, torch, triton):
    entry = getattr(module, variant["entrypoint"], None)
    if not isinstance(entry, triton.runtime.jit.JITFunction):
        raise ValidationError(
            "entrypoint must be a plain triton.jit function; hidden autotuning/wrappers unsupported"
        )
    output_name = next(k for k, t in spec["tensors"].items() if t["role"] == "output")
    decl = spec["tensors"][output_name]
    shape = [expression(x, spec["dimensions"], {}) for x in decl["shape"]]
    output = torch.empty(shape, device="cuda", dtype=getattr(torch, decl["dtype"]))
    output.fill_(float("nan") if output.is_floating_point() else torch.iinfo(output.dtype).min)
    tensors = {**inputs, output_name: output}
    values = variant["parameters"]
    kwargs = {
        name: tensors[value["tensor"]]
        if isinstance(value, dict) and set(value) == {"tensor"}
        else expression(value, spec["dimensions"], {})
        for name, value in spec["arguments"].items()
    }
    kwargs.update(values)
    grid = tuple(expression(x, spec["dimensions"], values) for x in spec["grid"])
    state = {
        "launch": _KernelLaunch(entry, grid, kwargs, output),
        "output": output,
        "grid": list(grid),
        "tensor_metadata": {
            name: {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "pointer_alignment_mod16": value.data_ptr() % 16,
            }
            for name, value in tensors.items()
        },
    }

    return state


