"""Counter-free runtime cost measurements with explicit timing boundaries."""

from __future__ import annotations

import time

from .backend import require_gpu
from .io import ValidationError
from .optimizer import select
from .runtime import input_shape
from .stats import summary


def host_samples(fn, samples=30, batch=100):
    if samples < 5 or batch < 1:
        raise ValidationError("host timing requires >=5 samples and positive batch size")
    values = []
    for _ in range(samples):
        before = time.perf_counter_ns()
        for _ in range(batch):
            fn()
        values.append((time.perf_counter_ns() - before) / 1000 / batch)
    return {
        "raw_us": values,
        **summary(values),
        "boundary": "CPU wall time per call, batched; no GPU synchronization",
    }


