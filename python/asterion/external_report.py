"""Passive self-contained reports for external-kernel experiment records."""

import html
import json
from pathlib import Path

from .io import ValidationError, canonical, digest


def _bounded_report(value):
    if not isinstance(value, dict):
        raise ValidationError("report artifact must be an object")
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 64 or count > 100000:
            raise ValidationError("external report structure exceeds supported nesting/node limits")
        if isinstance(item, (dict, list)) and count + len(pending) + len(item) > 100000:
            raise ValidationError("external report structure exceeds supported nesting/node limits")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
    if len(canonical(value)) > 8 * 1024 * 1024:
        raise ValidationError("external report exceeds supported8MiB limit")


