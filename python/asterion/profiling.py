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


def isolated_end_to_end(fn, samples=30):
    torch, _ = require_gpu()
    values = []
    for _ in range(samples):
        torch.cuda.current_stream().synchronize()
        before = time.perf_counter_ns()
        _result = fn()
        torch.cuda.current_stream().synchronize()
        values.append((time.perf_counter_ns() - before) / 1000)
    return {
        "raw_us": values,
        **summary(values),
        "boundary": "single invocation through current-stream completion, including host dispatch and sync",
    }


def runtime_components(runtime, inputs):
    torch, _ = require_gpu()
    shape = input_shape(runtime.portfolio["operation"], inputs, torch)
    runtime.prepare(*inputs)
    return {
        "metadata": host_samples(
            lambda: input_shape(runtime.portfolio["operation"], inputs, torch)
        ),
        "policy_selection": host_samples(lambda: select(runtime.portfolio, shape)),
        "plan_lookup_with_target_check": host_samples(lambda: runtime._plan(shape)),
        "warm_preparation": host_samples(lambda: runtime.prepare(*inputs), batch=20),
        "caveat": "Component timings are separate interventions; their medians are not additive.",
    }
