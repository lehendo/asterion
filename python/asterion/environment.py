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


