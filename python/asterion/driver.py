"""Optional bounded CUDA Driver launcher for trusted Triton-generated CUBINs.

This is native host execution, not native GPU compilation. The adapter is pinned
to Triton 3.3.1, validates its observed entry ABI and refuses scratch/cluster modes.
"""

from __future__ import annotations

import ctypes as c
import os
import re
import threading

from .io import ValidationError


def entry_contract(ptx, pointer_count, scalar_count):
    match = re.search(
        r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)\s*\.reqntid\s+(\d+),\s*1,\s*1", ptx, re.DOTALL
    )
    if not match:
        raise ValidationError("unsupported generated PTX entry contract")
    widths = tuple(int(x) for x in re.findall(r"\.param\s+\.u(32|64)\b", match[2]))
    expected = (64,) * pointer_count + (32,) * scalar_count + (64,)
    if widths != expected:
        raise ValidationError(f"unsupported kernel ABI: observed {widths}, expected {expected}")
    return match[1], int(match[3]), len(widths)


