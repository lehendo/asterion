"""Bounded greedy reduction with fresh measurement-gated witness confirmation."""

from __future__ import annotations

import math
import time
from pathlib import Path

from .diagnosis import reshape, specification
from .io import ValidationError, atomic_json, integer, positive, read
from .reproducer import export_reproducer


def smaller_shapes(shape):
    """Declared finite neighborhood: halve one dimension, preserving rank."""
    for axis, size in enumerate(shape):
        if size > 1:
            candidate = list(shape)
            candidate[axis] = max(1, size // 2)
            yield candidate


