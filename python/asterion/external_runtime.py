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


