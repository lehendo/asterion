"""Deterministic descriptive statistics, with explicit quantile definitions."""

from __future__ import annotations

import math
import random
import statistics

from .io import ValidationError, positive


def weighted_quantile(values, weights, q):
    if not 0 <= q <= 1 or len(values) != len(weights) or not values:
        raise ValidationError("invalid weighted quantile inputs")
    if any(not math.isfinite(v) for v in values):
        raise ValidationError("quantile values must be finite")
    total = sum(positive(w, "weight") for w in weights)
    cumulative = 0.0
    pairs = sorted(zip(values, weights))
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= q * total:
            return value
    return pairs[-1][0]


