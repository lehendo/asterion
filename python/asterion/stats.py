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


def ratio_interval(baseline, candidate, seed=1729, draws=500):
    # Independent nonparametric bootstrap of the ratio of sample medians.
    # Descriptive only; independence/stationarity are not proven by this interval.
    if min(len(baseline), len(candidate)) < 5:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        a = statistics.median(rng.choices(baseline, k=len(baseline)))
        b = statistics.median(rng.choices(candidate, k=len(candidate)))
        values.append(b / a)
    return [
        weighted_quantile(values, [1] * draws, 0.025),
        weighted_quantile(values, [1] * draws, 0.975),
    ]


def cvar(values, weights, q=0.95):
    """Upper-tail weighted mean, including fractional mass at the quantile."""
    if not 0 <= q < 1 or not values or len(values) != len(weights):
        raise ValidationError("invalid CVaR inputs")
    if any(not math.isfinite(v) for v in values):
        raise ValidationError("CVaR values must be finite")
    total = sum(positive(w, "weight") for w in weights)
    mass = (1 - q) * total
    remaining, cost = mass, 0.0
    for value, weight in sorted(zip(values, weights), reverse=True):
        take = min(weight, remaining)
        cost += value * take
        remaining -= take
        if remaining <= total * 1e-14:
            break
    return cost / mass
