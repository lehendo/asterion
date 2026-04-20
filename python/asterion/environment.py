from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture():
    from . import __version__

    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob("*.py")):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    packages = {}
    for name in ("asterion-gpu", "torch", "triton", "numpy", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu = command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    dirty = command(["git", "status", "--porcelain"])
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or None,
        "packages": packages,
        "source_version": __version__,
        "implementation_sha256": source_hash.hexdigest(),
        "git_sha": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if dirty is None else bool(dirty),
        "gpu_inventory": gpu,
        "gpu_runtime": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "caveats": [
            "GPU inventory unavailable outside an allocation; null means unavailable.",
            "Clock locking and exclusive GPU ownership are not assumed.",
        ],
    }
