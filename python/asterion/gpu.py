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


def new_trace(spec):
    env = capture()
    env["gpu_runtime"] = target_manifest()
    return {
        "schema_version": 1,
        "artifact_type": "trace",
        "measurement_kind": "measured",
        "timing_unit": "us",
        "environment": env,
        "workload": spec.to_dict(),
        "spec_hash": digest(spec.to_dict()),
        "records": [],
        "boundary": "FP32 operation; primary device samples use CUDA Graph batches; eager host/launch-loop samples are separate. Each method records its allocation/launch call boundary.",
        "caveats": [
            "Prepared methods reuse outputs; eager/public runtime methods allocate. Consult per-method boundaries.",
            "Events are steady-state launch-batched timings; compilation reported separately.",
            "No counter evidence, causal mechanism or exclusive-node claim.",
        ],
    }


def benchmark(
    workload,
    output,
    variants=("pytorch", "fixed"),
    split="train",
    samples=30,
    warmup=10,
    portfolio=None,
    official=None,
):
    torch, _ = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    spec = workload if isinstance(workload, WorkloadSpec) else WorkloadSpec.from_yaml(workload)
    allowed = {
        "pytorch",
        "asterion",
        "asterion_prepared",
        "inductor",
        "fixed",
        "official_triton",
        "asterion_native",
        "asterion_native_prepared",
    }
    if not variants or set(variants) - allowed:
        raise ValidationError(
            "variants: pytorch, fixed, inductor, asterion, asterion_prepared, asterion_native, asterion_native_prepared, official_triton"
        )
    if any(v.startswith("asterion") for v in variants) and portfolio is None:
        raise ValidationError(
            "supply a validated portfolio for Asterion runtime benchmarking; fixed is untuned codegen"
        )
    if "official_triton" in variants and official is None:
        raise ValidationError("official_triton requires an actual trained upstream adapter")
    from .io import read
    from .profiling import isolated_end_to_end, runtime_components
    from .runtime import Runtime

    runtime = (
        Runtime(read(portfolio) if isinstance(portfolio, (str, Path)) else portfolio)
        if portfolio
        else None
    )
    native_runtime = (
        Runtime(runtime.portfolio, launch_backend="native")
        if runtime and any("native" in v for v in variants)
        else None
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = new_trace(spec)
    trace["precision"] = {
        "dtype": "float32",
        "tf32": False,
        "matmul_precision": "highest",
        "asterion_fp_fusion": False,
    }
    cases = [c for c in spec.cases if split == "all" or c.split == split]
    if not cases:
        raise ValidationError("no cases for selected split")
    rng = random.Random(spec.seed)
    overhead = {}
    for case in cases:
        program = make_program(spec.operation, case.shape)
        inputs = make_inputs(program, spec.seed)
        baseline = operation_reference(spec.operation)
        expected = baseline(*inputs)
        order = list(variants)
        rng.shuffle(order)
        for variant in order:
            r = {
                "case_id": case.id,
                "operation": spec.operation,
                "shape": list(case.shape),
                "dtype": spec.dtype,
                "layout": spec.layout,
                "split": case.split,
                "weight": case.weight,
                "variant": variant,
                "status": "error",
                "correctness": "unavailable",
            }
            try:
                start = time.perf_counter()
                if variant == "pytorch":
                    fn = lambda baseline=baseline, inputs=inputs: baseline(*inputs)
                    r["call_boundary"] = "framework eager invocation with output allocation"
                elif variant == "inductor":
                    compiled = torch.compile(baseline, fullgraph=True, dynamic=False)
                    fn = lambda compiled=compiled, inputs=inputs: compiled(*inputs)
                    r["call_boundary"] = "torch.compile invocation with output allocation"
                elif variant.startswith("asterion"):
                    selected_runtime = native_runtime if "native" in variant else runtime
                    prepared = selected_runtime.prepare(*inputs)
                    r["runtime_selection"] = prepared.selection
                    r["preparation_wall_s"] = prepared.preparation_wall_s
                    if variant.endswith("_prepared"):
                        fn = prepared.launch
                        r["call_boundary"] = "prepared reusable output launch"
                    else:
                        fn = lambda runtime=selected_runtime, inputs=inputs: runtime(*inputs)
                        r["call_boundary"] = (
                            "public runtime invocation with metadata, policy, allocation and launch"
                        )
                elif variant == "official_triton":
                    fn, selection = official.prepare(inputs)
                    r["runtime_selection"] = selection
                    r["call_boundary"] = (
                        "upstream-selected static configuration with prepared output"
                    )
                else:
                    kernel = Kernel(program, Schedule())
                    bound = kernel.prepare(*inputs)
                    fn = bound.launch
                    r.update(
                        schedule=kernel.schedule.to_dict(),
                        ir_hash=program.hash,
                        generated_source=kernel.source_path,
                        codegen_hash=kernel.key,
                        call_boundary="untuned generated kernel with prepared output",
                    )
                actual = fn()
                torch.cuda.current_stream().synchronize()
                r["first_invocation_wall_s"] = time.perf_counter() - start
                torch.testing.assert_close(actual, expected, rtol=spec.rtol, atol=spec.atol)
                r.update(status="ok", correctness="passed", **measure(fn, samples, warmup))
                r["isolated_end_to_end"] = isolated_end_to_end(fn, samples)
            except Exception as exc:  # noqa: BLE001 - preserve failed framework/compiler evidence
                r["error"] = f"{type(exc).__name__}: {exc}"
            trace["records"].append(r)
            atomic_json(output / "trace.json", trace)
        if runtime:
            overhead[case.id] = runtime_components(runtime, inputs)
            atomic_json(output / "runtime-overhead.json", overhead)
    if native_runtime:
        native_runtime.close()
    validate_trace(trace)
    write_report(analyze(trace), output / "analysis")
    return trace
