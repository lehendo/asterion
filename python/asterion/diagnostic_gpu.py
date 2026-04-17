"""Trusted execution of verified bounded IR experiments on a CUDA allocation."""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

from .diagnosis import assemble, schedule_analysis, selected_plans, specification
from .io import ValidationError, atomic_json, digest, read
from .ir import Program, Schedule


def diagnose(value, output):
    started = time.perf_counter()
    from .backend import Kernel, binary_metadata, make_inputs, require_gpu, target_manifest
    from .environment import capture
    from .validation import certify_variant

    spec = specification(read(value) if isinstance(value, (str, Path)) else value)
    torch, _ = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValidationError(
            "diagnose output must be a new or empty directory; prior evidence is preserved"
        )
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "reproducer.json", spec)
    from .reproducer import export_reproducer

    export_reproducer(spec, out / "standalone")
    p = Program.from_dict(spec["program"])
    schedules = {name: spec[name] for name in ("baseline", "candidate")}
    schedules.update({x["id"]: x["schedule"] for x in selected_plans(spec)})
    inputs = make_inputs(p, spec["seed"])
    variants, bound, graphs = {}, {}, {}
    environment = capture()
    environment["gpu_runtime"] = target_manifest()
    atomic_json(out / "standalone" / "environment.json", environment)
    partial = {
        "artifact_type": "diagnosis_partial",
        "specification": spec,
        "environment": environment,
        "variants": variants,
        "blocks": [],
        "status": "validating",
    }
    atomic_json(out / "partial.json", partial)
    for name, raw in schedules.items():
        s = Schedule.from_dict(raw)
        try:
            certificate = certify_variant(
                p,
                s,
                name,
                rtol=spec["rtol"],
                atol=spec["atol"],
                seeds=(spec["seed"], spec["seed"] + 1),
            )
            kernel = Kernel(p, s)
            prepared = kernel.prepare(*inputs)
            prepared.launch()
            torch.cuda.synchronize()
            destination = out / "code" / name
            destination.mkdir(parents=True, exist_ok=True)
            source = Path(kernel.source_path).read_text()
            (destination / "kernel.py").write_text(source)
            compilation = []
            for index, compiled in enumerate(prepared.compiled_kernels.values()):
                assembly = {}
                for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
                    code = compiled.asm.get(kind)
                    if isinstance(code, (str, bytes)):
                        data = code.encode() if isinstance(code, str) else code
                        path = destination / f"stage-{index}.{kind}"
                        path.write_bytes(data)
                        assembly[kind] = {
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data),
                            "path": str(path.relative_to(out)),
                        }
                compilation.append(
                    {
                        "assembly": assembly,
                        "registers": getattr(compiled, "n_regs", None),
                        "spills": getattr(compiled, "n_spills", None),
                        "shared_bytes": getattr(compiled.metadata, "shared", None),
                    }
                )
            variants[name] = {
                "status": "passed",
                "schedule": s.to_dict(),
                "program_hash": p.hash,
                "kernel_key": kernel.key,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generated_source": source,
                "source": str((destination / "kernel.py").relative_to(out)),
                "certificate": certificate,
                "lowering": compilation,
                "analysis": schedule_analysis(p, s),
                **binary_metadata(prepared),
                "compile_options": {"num_warps": s.warps, "enable_fp_fusion": False},
                "preparation_seconds": time.perf_counter() - started,
            }
            bound[name] = prepared
            for _ in range(5):
                prepared.launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(spec["launches"]):
                    prepared.launch()
            graph.replay()
            torch.cuda.synchronize()
            graphs[name] = graph
        except Exception as exc:
            variants[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            partial["status"] = "failed_correctness_or_preparation"
            atomic_json(out / "partial.json", partial)
            raise
        atomic_json(out / "partial.json", partial)
    rng = random.Random(spec["seed"])
    partial["status"] = "measuring"
    for index in range(spec["rounds"]):
        order = list(schedules)
        rng.shuffle(order)
        measurements = {}
        for name in order:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            device = start.elapsed_time(end) * 1000 / spec["launches"]
            before = time.perf_counter()
            bound[name].launch()
            torch.cuda.current_stream().synchronize()
            host = (time.perf_counter() - before) * 1e6
            measurements[name] = {"device_graph_us": device, "prepared_completion_us": host}
        partial["blocks"].append({"index": index, "order": order, "measurements": measurements})
        atomic_json(out / "partial.json", partial)
    result = assemble(spec, environment, variants, partial["blocks"])
    result["prepared_comparison"] = assemble(
        spec, environment, variants, partial["blocks"], boundary="prepared_completion_us"
    )["comparison"]
    result["measurement_protocol"] = {
        "randomization": "all variants once per seeded shuffled block",
        "device_graph_us": "CUDA events around graph of repeated prepared launches, divided by launch count",
        "prepared_completion_us": "one checked reusable-output launch plus current-stream completion, wall clock",
        "warmup": 5,
        "clock_lock": "unavailable",
        "exclusive_gpu": "not_assumed",
        "correctness": "two deterministic seeds, four patterns, float64 IR reference cast FP32 before all timing",
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    result["reproduction"] = {
        "specification": "reproducer.json",
        "command": "asterion reproduce reproducer.json --output rerun",
        "requirements": "CUDA GPU and compatible optional torch/Triton dependencies; trusted verified IR execution",
    }
    if result["recommendations"]:
        import statistics

        best = min(
            result["recommendations"],
            key=lambda name: statistics.median(
                b["measurements"][name]["device_graph_us"] for b in partial["blocks"]
            ),
        )
        recommendation = specification(
            {
                **spec,
                "baseline": spec["candidate"],
                "candidate": variants[best]["schedule"],
                "intervention_policy": "none",
            }
        )
        atomic_json(out / "recommendation.json", recommendation)
        result["recommended_variant"] = best
        result["recommendation_scope"] = (
            "Helpful at the tested shape, target and graph-device boundary; no future-distribution guarantee or automatic deployment. recommendation.json reruns candidate versus the isolated reversion."
        )
    atomic_json(
        out / "standalone" / "expectation.json",
        {
            "verdict": result["comparison"]["verdict"],
            "median_paired_ratio": result["comparison"]["median_paired_ratio"],
            "boundary": result["boundary"],
        },
    )
    result["result_id"] = digest(result)
    atomic_json(out / "results.json", result)
    from .diagnostic_report import write_diagnosis

    write_diagnosis(result, out)
    partial["status"] = "complete"
    atomic_json(out / "partial.json", partial)
    return result
