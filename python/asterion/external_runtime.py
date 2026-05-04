"""Trusted single-module Triton experiments; also exported as standalone support.

Only explicit execution enters run(). A Python process is not a security sandbox.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import linecache
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .experiment_stats import paired_effect
from .external import (
    EXECUTION_FILES,
    capabilities,
    complexity,
    expression,
    interventions,
    source_path,
    specification,
)
from .io import ValidationError, atomic_json, digest, read
from .numerics import check_output


def _load_reviewed_module(path, expected_sha256):
    """Execute exact reviewed bytes; never substitute a timestamp-based pyc.

    This intentional execution helper is only for explicitly trusted workflows.
    Passive contract parsing/export never calls it.
    """
    path = Path(path).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValidationError("reviewed module changed before execution")
    name = "trusted_kernel_" + expected_sha256
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    linecache.cache.pop(str(path), None)
    # Trusted execution uses verified source bytes instead of timestamp-based pyc.
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def static_ptx_observations(source):
    """Summarize selected line-leading PTX opcodes, never dynamic instructions."""
    if not isinstance(source, str):
        return {"category": "UNAVAILABLE", "reason": "Textual PTX unavailable"}
    text = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    prefixes = (
        "ldmatrix",
        "mma.sync",
        "wgmma.mma_async",
        "ld.global",
        "st.global",
        "ld.shared",
        "st.shared",
        "bar.sync",
    )
    counts = dict.fromkeys(prefixes, 0)
    for line in text.splitlines():
        line = line.split("//", 1)[0]
        match = re.match(r"^\s*(?:@!?%[A-Za-z0-9_]+\s+)?([a-z][a-z0-9_.]*)\s", line)
        if match:
            opcode = match.group(1)
            for prefix in prefixes:
                if opcode == prefix or opcode.startswith(prefix + "."):
                    counts[prefix] += 1
    return {
        "category": "COMPILER_EVIDENCE",
        "static_opcode_prefix_counts": counts,
        "interpretation": "Heuristic line-leading opcode statements in emitted PTX; inline blocks may be omitted. Not dynamic instruction counts, occupancy or hardware bottleneck evidence. PTX can differ from final SASS",
    }


def _target(torch, triton):
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    return {
        "name": p.name,
        "capability": [p.major, p.minor],
        "uuid": str(getattr(p, "uuid", "unavailable")),
        "total_memory": p.total_memory,
        "multiprocessors": p.multi_processor_count,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_build": torch.version.cuda,
        "driver": driver,
        "tf32_reference": False,
        "compiler": "Triton; no independent native GPU lowering",
    }


def _inputs(spec, torch, seed, pattern):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = {}
    for name, tensor in spec["tensors"].items():
        if tensor["role"] != "input":
            continue
        shape = [expression(x, spec["dimensions"], {}) for x in tensor["shape"]]
        dtype = getattr(torch, tensor["dtype"])
        if tensor.get("generator") == "ones":
            result[name] = torch.ones(shape, device="cuda", dtype=dtype)
            continue
        if dtype in {torch.int8, torch.int32}:
            x = torch.randint(-3, 4, shape, device="cuda", dtype=dtype, generator=generator)
            if pattern == "zeros":
                x.zero_()
            elif pattern == "alternating":
                x = (torch.arange(x.numel(), device="cuda").reshape(shape) % 2 * 2 - 1).to(dtype)
        else:
            x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
            if pattern == "zeros":
                x.zero_()
            elif pattern == "alternating":
                x = (torch.arange(x.numel(), device="cuda").reshape(shape) % 2 * 2 - 1).float()
            elif pattern == "wide_finite":
                x *= 8
            x = x.to(dtype)
        result[name] = x.contiguous()
    return result


class _KernelLaunch:
    """Own tensors and the loaded binary without a state/closure reference cycle."""

    def __init__(self, entry, grid, kwargs, output):
        self.entry = entry
        self.grid = grid
        self.kwargs = kwargs
        self.output = output
        self.compiled = None

    def __call__(self):
        self.compiled = self.entry[self.grid](**self.kwargs)
        return self.output


def _prepare(spec, variant, inputs, module, torch, triton):
    entry = getattr(module, variant["entrypoint"], None)
    if not isinstance(entry, triton.runtime.jit.JITFunction):
        raise ValidationError(
            "entrypoint must be a plain triton.jit function; hidden autotuning/wrappers unsupported"
        )
    output_name = next(k for k, t in spec["tensors"].items() if t["role"] == "output")
    decl = spec["tensors"][output_name]
    shape = [expression(x, spec["dimensions"], {}) for x in decl["shape"]]
    output = torch.empty(shape, device="cuda", dtype=getattr(torch, decl["dtype"]))
    output.fill_(float("nan") if output.is_floating_point() else torch.iinfo(output.dtype).min)
    tensors = {**inputs, output_name: output}
    values = variant["parameters"]
    kwargs = {
        name: tensors[value["tensor"]]
        if isinstance(value, dict) and set(value) == {"tensor"}
        else expression(value, spec["dimensions"], {})
        for name, value in spec["arguments"].items()
    }
    kwargs.update(values)
    grid = tuple(expression(x, spec["dimensions"], values) for x in spec["grid"])
    state = {
        "launch": _KernelLaunch(entry, grid, kwargs, output),
        "output": output,
        "grid": list(grid),
        "tensor_metadata": {
            name: {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "pointer_alignment_mod16": value.data_ptr() % 16,
            }
            for name, value in tensors.items()
        },
    }

    return state


def _artifacts(state, root, name):
    compiled = state["launch"].compiled
    folder = root / "code" / name
    folder.mkdir(parents=True)
    artifacts = {}
    for kind in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        content = compiled.asm.get(kind)
        if not isinstance(content, (str, bytes)):
            continue
        data = content.encode() if isinstance(content, str) else content
        path = folder / ("kernel." + kind)
        path.write_bytes(data)
        artifacts[kind] = {
            "path": str(path.relative_to(root)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {
        "assembly": artifacts,
        "compiled_hash": str(getattr(compiled, "hash", "unavailable")),
        "compiler_options": {
            k: getattr(compiled.metadata, k, None)
            for k in ("num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion")
        },
        "registers": getattr(compiled, "n_regs", None),
        "spills": getattr(compiled, "n_spills", None),
        "shared_bytes": getattr(compiled.metadata, "shared", None),
        "ptx_static": static_ptx_observations(compiled.asm.get("ptx")),
        "category": "COMPILER_EVIDENCE",
        "interpretation": "Compiler metadata, not measured hardware occupancy or bandwidth",
    }


def run(value, source_root, output):
    import os

    if os.environ.get("ASTERION_REQUIRE_SLURM") == "1" and not os.environ.get("SLURM_JOB_ID"):
        raise ValidationError("this deployment requires an allocated Slurm GPU job")
    import torch
    import triton

    started = time.perf_counter()
    spec = specification(value)
    original = source_path(spec, source_root)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValidationError("external output must be new or empty; old evidence is preserved")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "source" / spec["module"]
    destination.parent.mkdir(parents=True)
    shutil.copyfile(original, destination)
    source_path(spec, root / "source")
    atomic_json(root / "source" / "case.json", spec)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    target = _target(torch, triton)
    if target["capability"][0] < 8:
        raise ValidationError(
            "external adapter requires the tested modern Triton CUDA target SM80 or newer"
        )
    result = {
        "schema_version": 1,
        "artifact_type": "external_diagnosis",
        "specification": spec,
        "capabilities": capabilities("triton_case"),
        "environment": target,
        "source_identity": {"module": spec["module"], "sha256": spec["source_sha256"]},
        "engine_sources": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in EXECUTION_FILES
        },
        "variants": {},
        "detection_blocks": [],
        "intervention_blocks": [],
        "findings": [],
        "measurement_seconds": {},
        "recommendations": [],
        "status": "PREPARING",
        "timing_status": "UNAVAILABLE",
        "numerical_status": "PENDING",
        "complexity": complexity(spec),
        "trust": "Explicitly trusted Python executes with user authority; a worker process is not a security sandbox",
        "hardware_mechanism": {
            "category": "UNAVAILABLE",
            "reason": "No hardware counters collected",
        },
        "limitations": [
            "Finite numerical validation, not universal correctness",
            "One declared parameter can alter many downstream compiler decisions",
            "Output addresses, cache state, clocks and shared-device interference remain alternatives",
            "Inconclusive does not establish equivalence; adaptive selection has no global error guarantee",
        ],
    }

    def save():
        atomic_json(root / "partial.json", result)

    save()
    import_started = time.perf_counter()
    module = _load_reviewed_module(destination, spec["source_sha256"])
    result["module_import_seconds"] = time.perf_counter() - import_started
    reference = getattr(module, spec["reference"]["entrypoint"], None)
    if not callable(reference):
        raise ValidationError("trusted numerical reference entrypoint is not callable")
    config = spec["measurement"]
    inputs = _inputs(spec, torch, config["seed"], "normal")
    states = {}

    def prepare(name, variant):
        check_started = time.perf_counter()
        record = {
            "configuration": variant,
            "checks": [],
            "status": "VALIDATING",
            "phase_seconds": {
                "input_preparation": 0.0,
                "reference": 0.0,
                "validation_checks": 0.0,
                "launch_completion": 0.0,
            },
        }
        result["variants"][name] = record
        save()
        try:
            for seed in (config["seed"], (config["seed"] + 1) % 2**31):
                for pattern in ("normal", "zeros", "alternating", "wide_finite"):
                    phase_started = time.perf_counter()
                    values = _inputs(spec, torch, seed, pattern)
                    before = {k: v.clone() for k, v in values.items()}
                    torch.cuda.synchronize()
                    record["phase_seconds"]["input_preparation"] += (
                        time.perf_counter() - phase_started
                    )
                    # Reference receives detached double copies, never output buffers.
                    phase_started = time.perf_counter()
                    expected = reference(
                        {k: v.double().clone() for k, v in values.items()}, dict(spec["dimensions"])
                    )
                    torch.cuda.synchronize()
                    record["phase_seconds"]["reference"] += time.perf_counter() - phase_started
                    state = _prepare(spec, variant, values, module, torch, triton)
                    torch.cuda.synchronize()
                    launch_started = time.perf_counter()
                    actual = state["launch"]()
                    torch.cuda.synchronize()
                    record["phase_seconds"]["launch_completion"] += (
                        time.perf_counter() - launch_started
                    )
                    if "first_launch_completion_seconds" not in record:
                        record["first_launch_completion_seconds"] = (
                            time.perf_counter() - launch_started
                        )
                    phase_started = time.perf_counter()
                    check = check_output(actual, expected, spec["numerical"])
                    check.update(seed=seed, pattern=pattern)
                    check["inputs_unchanged"] = all(
                        torch.equal(values[k], before[k]) for k in values
                    )
                    check["passed"] = check["passed"] and check["inputs_unchanged"]
                    record["phase_seconds"]["validation_checks"] += (
                        time.perf_counter() - phase_started
                    )
                    record["checks"].append(check)
                    if not check["passed"]:
                        record["status"] = "NUMERICAL_FAILURE"
                        record["validation_seconds"] = time.perf_counter() - check_started
                        record["compiler"] = _artifacts(state, root, name)
                        save()
                        return False
            state = _prepare(spec, variant, inputs, module, torch, triton)
            state["launch"]()
            torch.cuda.synchronize()
            record.update(
                status="ACCEPTED",
                validation_seconds=time.perf_counter() - check_started,
                grid=state["grid"],
                tensor_metadata=state["tensor_metadata"],
                compiler=_artifacts(state, root, name),
            )
            for _ in range(config["warmup"]):
                state["launch"]()
            torch.cuda.synchronize()
            # Previous reviewed module instances can own unreachable JIT/cache
            # cycles. Their CUDA destructors must run outside graph capture.
            gc.collect()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(config["launches"]):
                    state["launch"]()
            graph.replay()
            torch.cuda.synchronize()
            state["graph"] = graph
            states[name] = state
            save()
            return True
        except Exception as exc:  # noqa: BLE001 - retain distinct third-party compiler/runtime failures per variant
            record.update(status="EXECUTION_FAILURE", error=f"{type(exc).__name__}: {exc}")
            save()
            return False

    def blocks(names, field, seed):
        phase_started = time.perf_counter()
        rng = random.Random(seed)
        for index in range(config["rounds"]):
            order = list(names)
            rng.shuffle(order)
            block = {"index": index, "order": order, "measurements": {}}
            for name in order:
                state = states[name]
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                state["graph"].replay()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) * 1000 / config["launches"]
                begin = time.perf_counter()
                state["launch"]()
                torch.cuda.current_stream().synchronize()
                block["measurements"][name] = {
                    "device_graph_us": elapsed,
                    "prepared_completion_us": (time.perf_counter() - begin) * 1e6,
                }
            result[field].append(block)
            save()
        result["measurement_seconds"][field] = time.perf_counter() - phase_started

    def compare(field, left, right, count=1):
        values = result[field]
        return paired_effect(
            [b["measurements"][left]["device_graph_us"] for b in values],
            [b["measurements"][right]["device_graph_us"] for b in values],
            threshold=config["threshold"],
            comparisons=count,
        )

    a = prepare("baseline", spec["baseline"])
    b = prepare("candidate", spec["candidate"])
    if not a or not b:
        result["status"] = (
            "NUMERICAL_FAILURE"
            if any(v["status"] == "NUMERICAL_FAILURE" for v in result["variants"].values())
            else "EXECUTION_FAILURE"
        )
        result["numerical_status"] = "NOT_ADMITTED"
    else:
        result["numerical_status"] = "ACCEPTED_TESTED_DOMAIN"
        blocks(["baseline", "candidate"], "detection_blocks", config["seed"])
        result["comparison"] = compare("detection_blocks", "baseline", "candidate")
        verdict = result["comparison"]["verdict"]
        result["status"] = (
            "REGRESSION_ESTABLISHED"
            if verdict == "regression"
            else "NO_REGRESSION_DETECTED"
            if verdict == "improvement"
            else "REGRESSION_INCONCLUSIVE"
        )
        result["timing_status"] = "MEASURED"
        result["time_to_first_comparison_seconds"] = time.perf_counter() - started
        if verdict == "regression" and config["interventions"] == "changed":
            plans = interventions(spec)
            result["intervention_plans"] = plans
            valid = [
                p
                for p in plans
                if p["status"] == "declared_legal" and prepare(p["id"], p["variant"])
            ]
            if digest(spec["baseline"]) != digest(spec["candidate"]):
                blocks(
                    ["candidate", "baseline"] + [p["id"] for p in valid],
                    "intervention_blocks",
                    config["seed"] + 7001,
                )
                result["full_revert_control"] = compare(
                    "intervention_blocks", "candidate", "baseline", 1 + len(valid)
                )
                for plan in valid:
                    effect = compare("intervention_blocks", "candidate", plan["id"], 1 + len(valid))
                    helpful = effect["verdict"] == "improvement"
                    result["findings"].append(
                        {
                            "parameter": plan["parameter"],
                            "intervention": plan["id"],
                            "category": "CONTROLLED_ABLATION" if helpful else "HYPOTHESIS",
                            "supported": helpful,
                            "effect": effect,
                            "scope": "Declared one-parameter change at this tested boundary; hardware mechanism unestablished",
                        }
                    )
                    if helpful:
                        result["recommendations"].append(plan["id"])
                if not result["recommendations"]:
                    effect = result["full_revert_control"]
                    helpful = effect["verdict"] == "improvement"
                    result["findings"].append(
                        {
                            "parameter": None,
                            "intervention": "baseline",
                            "kind": "full_variant_revert",
                            "isolated_parameter": False,
                            "category": "CONTROLLED_ABLATION" if helpful else "HYPOTHESIS",
                            "supported": helpful,
                            "effect": effect,
                            "scope": "Revert the complete declared source/configuration package; does not isolate one compiler or hardware mechanism",
                        }
                    )
                    if helpful:
                        result["recommendations"].append("baseline")
        result["noise"] = {}
        for name in ("baseline", "candidate"):
            samples = [
                b["measurements"][name]["device_graph_us"] for b in result["detection_blocks"]
            ]
            median = statistics.median(samples)
            mad = statistics.median(abs(x - median) for x in samples)
            result["noise"][name] = {
                "median_us": median,
                "relative_mad": mad / median,
                "flagged": mad / median > 0.10,
            }
    source_path(spec, root / "source")
    result["elapsed_seconds"] = time.perf_counter() - started
    result["time_to_actionable_result_seconds"] = (
        result["elapsed_seconds"] if result["recommendations"] else None
    )
    result["experiment_counts"] = {
        "validated_configurations": len(result["variants"]),
        "accepted_configurations": len(states),
        "timed_variant_blocks": sum(
            len(b["order"])
            for field in ("detection_blocks", "intervention_blocks")
            for b in result[field]
        ),
    }
    result["measurement_protocol"] = {
        "boundary": "CUDA events around a fixed-buffer graph batch divided by launches",
        "secondary": "one trusted wrapper launch plus current-stream completion; not application throughput",
        "clocks_locked": False,
        "exclusive_gpu_assumed": False,
        "cache_state": "existing caches; first-launch completion reported, not labeled cold compilation",
        "cost_accounting": "Validation input/reference/check phases synchronize for accounting; first-launch completion combines JIT/cache lookup and execution, not isolated compilation. Worker startup/import and export costs recorded separately by orchestration.",
    }
    result["result_id"] = digest(result)
    atomic_json(root / "results.json", result)
    save()
    from .external_report import write_external_report

    write_external_report(result, root)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute reviewed trusted external Triton code; not a sandbox"
    )
    parser.add_argument("specification")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-code", action="store_true")
    args = parser.parse_args()
    if not args.trusted_code:
        parser.error("--trusted-code is required; source and reference are executable Python")
    spec = Path(args.specification).resolve()
    result = run(read(spec), spec.parent, args.output)
    print(result["status"])


if __name__ == "__main__":
    main()
