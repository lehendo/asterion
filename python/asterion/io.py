from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


class ValidationError(ValueError):
    """Invalid or unsupported public input."""


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidationError(f"expected finite JSON-compatible values: {exc}") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


