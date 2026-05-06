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


def diagnose_external(value, output, *, trusted_code=False, source_root=None, timeout_seconds=900):
    spec, source = load_case(value, source_root)
    if trusted_code is not True:
        raise ValidationError(
            "external Python execution requires trusted_code=True / --trusted-code; passive input cannot authorize itself"
        )
    if type(timeout_seconds) not in {int, float} or not 1 <= timeout_seconds <= 1800:
        raise ValidationError("worker timeout must be1..1800seconds")
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValidationError("external output must be new or empty")
    root.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="external-input-", dir=root.parent) as temporary:
        stage = Path(temporary)
        _copy_input(spec, source, stage)
        home = stage / "worker-home"
        home.mkdir()
        args = [
            sys.executable,
            "-m",
            "asterion.external_runtime",
            str(stage / "case.json"),
            "--output",
            str(root),
            "--trusted-code",
        ]
        try:
            completed = run_worker(
                args,
                env=worker_environment(home),
                cwd=stage,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            root.mkdir(parents=True, exist_ok=True)
            if not (root / "source/case.json").is_file():
                _copy_input(spec, stage, root / "source")
            (root / "worker.log").write_text((exc.stdout or "") + (exc.stderr or ""))
            atomic_json(
                root / "worker-failure.json",
                {
                    "status": "TIMEOUT",
                    "timeout_seconds": timeout_seconds,
                    "partial_evidence": "partial.json when available",
                },
            )
            raise ValidationError(
                "external worker timed out; partial artifacts are preserved"
            ) from exc
        if completed.returncode and not (root / "source/case.json").is_file():
            root.mkdir(parents=True, exist_ok=True)
            _copy_input(spec, stage, root / "source")
    root.mkdir(parents=True, exist_ok=True)
    (root / "worker.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode or not (root / "results.json").exists():
        atomic_json(
            root / "worker-failure.json",
            {
                "status": "EXECUTION_FAILURE",
                "exit_code": completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise ValidationError("external worker failed; inspect worker.log and partial.json")
    result = read(root / "results.json")
    if result.get("result_id") != digest({k: v for k, v in result.items() if k != "result_id"}):
        raise ValidationError("external worker returned inconsistent evidence")
    worker_seconds = time.perf_counter() - started
    export_started = time.perf_counter()
    export_external(root / "results.json", root / "standalone")
    export_seconds = time.perf_counter() - export_started
    atomic_json(
        root / "workflow-cost.json",
        {
            "worker_process_seconds": worker_seconds,
            "export_seconds": export_seconds,
            "total_workflow_seconds": time.perf_counter() - started,
            "experiment_seconds": result["elapsed_seconds"],
            "human_effort": "unmeasured",
            "scope": "instrumented workflow cost; not human productivity or allocated GPU accounting",
        },
    )
    return result


