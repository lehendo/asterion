"""Training-only bounded beam search over metadata decision partitions.

No approximation/novelty guarantee is claimed. Every partition is deployable from
shape metadata; evaluation never chooses a variant with a test-time oracle.
"""

from __future__ import annotations

import math
from itertools import pairwise

from .io import ValidationError, digest, integer, positive
from .schema import validate_trace
from .stats import cvar, summary

FEATURE = "last_dimension"


def feature(shape):
    return shape[-1]


def objective(ratios, weights, risk=0.5, q=0.9):
    if not ratios or len(ratios) != len(weights) or risk < 0 or not 0 <= q < 1:
        raise ValidationError("invalid objective")
    values = [math.log(positive(r, "latency ratio")) for r in ratios]
    total = sum(positive(w, "weight") for w in weights)
    return sum(v * w for v, w in zip(values, weights)) / total + risk * cvar(values, weights, q)


