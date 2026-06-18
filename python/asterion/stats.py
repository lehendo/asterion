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


def summary(samples):
    values = [positive(v, "sample") for v in samples]
    if not values:
        raise ValidationError("samples cannot be empty")
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return {
        "count": len(values),
        "median_us": median,
        "p95_us": weighted_quantile(values, [1] * len(values), 0.95),
        "mad_us": mad,
        "relative_mad": mad / median,
        "noisy": mad / median > 0.10,
        "small_sample": len(values) < 20,
    }


