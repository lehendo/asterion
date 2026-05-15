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


