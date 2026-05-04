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


