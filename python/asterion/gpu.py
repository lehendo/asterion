from __future__ import annotations

import random
import time
from pathlib import Path

from .analyzer import analyze
from .backend import Kernel, make_inputs, require_gpu, target_manifest
from .environment import capture
from .io import ValidationError, atomic_json, digest
from .ir import Schedule, make_program
from .report import write_report
from .schema import WorkloadSpec, validate_trace


def measure(fn, samples=30, warmup=10, launches=50):
    torch, _ = require_gpu()
    if samples < 5 or warmup < 1 or launches < 1:
        raise ValidationError("measurement requires >=5 samples, >=1 warmup and >=1 launch")
    for _ in range(warmup):
        fn()
    torch.cuda.current_stream().synchronize()
    # Graph replay separates device execution from Python launch starvation.
    # Eager launch-loop event samples remain available as an explicit secondary boundary.
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(launches):
            _retained_output = fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    torch.cuda.current_stream().synchronize()
    device, host, launch_loop = [], [], []
    for _ in range(samples):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end) * 1000 / launches)
        before = time.perf_counter()
        start.record()
        for _ in range(launches):
            _retained_output = fn()
        end.record()
        end.synchronize()
        host.append((time.perf_counter() - before) * 1e6 / launches)
        launch_loop.append(start.elapsed_time(end) * 1000 / launches)
    return {
        "device_us": device,
        "host_us": host,
        "launch_loop_device_us": launch_loop,
        "timing": {
            "method": "cuda_graph_events",
            "stream": "pytorch_current",
            "samples": samples,
            "warmup": warmup,
            "launches_per_sample": launches,
            "host_boundary": "eager launch loop plus end-event synchronization / launches",
            "device_boundary": "CUDA graph replay of launch batch / launches",
            "secondary_device_boundary": "eager event interval, includes host submission gaps",
            "clock_lock": "unavailable",
            "exclusive_gpu": "not_assumed",
        },
    }


def operation_reference(operation):
    import torch

    # Framework baselines bypass the IR interpreter so Dynamo sees ordinary tensor operations.
    if operation == "pointwise":
        return lambda x: torch.relu(x * 1.25 + 0.5)
    if operation == "sum":
        return lambda x: torch.sum(x, dim=1)
    if operation == "max":
        return lambda x: torch.amax(x, dim=1)
    if operation == "softmax":
        return lambda x: torch.softmax(x, dim=1)
    if operation == "gemm":
        return lambda x, y: torch.matmul(x, y)
    raise ValidationError("unsupported baseline operation")


