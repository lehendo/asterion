"""CPU orchestration, explicit execution boundary and standalone export."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .external import EXECUTION_FILES, source_path, specification
from .io import ValidationError, atomic_json, digest, read

SUPPORT = EXECUTION_FILES


def run_worker(args, *, env, cwd=None, timeout):
    """Own a process group and terminate it on timeout/interruption, no shell.

    This is cleanup, not sandboxing. Trusted code can deliberately escape a group.
    Nested orchestration should convert SIGTERM to an exception for finally cleanup.
    """
    if os.name != "posix":
        raise ValidationError("trusted external execution currently requires a POSIX worker host")
    process = subprocess.Popen(
        args,
        env=env,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            exc.output, exc.stderr = stdout, stderr
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    finally:
        # The owned group may outlive its main process through background children.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.communicate()


def load_case(value, source_root=None):
    root = Path(source_root or ".").resolve()
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        root = path.parent
        value = read(path)
    if not isinstance(value, dict):
        raise ValidationError("external input must be a case or diagnosis object")
    if value.get("artifact_type") == "external_diagnosis":
        if value.get("result_id") != digest({k: v for k, v in value.items() if k != "result_id"}):
            raise ValidationError("external diagnosis integrity mismatch")
        value = value["specification"]
        source = (root / "source").resolve()
        if not source.is_relative_to(root):
            raise ValidationError("saved source symlink escapes diagnosis artifact")
        root = source
    spec = specification(value)
    source_path(spec, root)
    return spec, root


def _copy_input(spec, source, destination):
    path = source_path(spec, source)
    target = destination / spec["module"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    atomic_json(destination / "case.json", spec)
    source_path(spec, destination)


def worker_environment(home):
    """Remove credential variables/agents; this is not filesystem isolation."""
    allowed = {
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUDA_HOME",
        "CUDA_PATH",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "OMP_NUM_THREADS",
        "ASTERION_REQUIRE_SLURM",
        "SLURM_JOB_ID",
        "SLURM_CPUS_PER_TASK",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env.update(HOME=str(home), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    return env


