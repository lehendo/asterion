from __future__ import annotations

import random
import time
from pathlib import Path

from .backend import Kernel, binary_metadata, make_inputs, reference, require_gpu
from .gpu import measure, new_trace
from .io import ValidationError, atomic_json, integer, positive
from .ir import candidates, make_program
from .optimizer import fit_portfolio
from .schema import WorkloadSpec


def optimize(workload, output, budget_seconds=300, max_variants=4, objective="robust"):
    torch, _ = require_gpu()
    positive(budget_seconds, "budget_seconds")
    integer(max_variants, "max_variants")
    if objective not in {"mean", "robust"}:
        raise ValidationError("invalid objective")
    spec = workload if isinstance(workload, WorkloadSpec) else WorkloadSpec.from_yaml(workload)
    train = [c for c in spec.cases if c.split == "train"]
    if not train:
        raise ValidationError("optimizer requires training cases")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = new_trace(spec)
    # Redact held-out shapes from training artifact and all objective-building inputs.
    trace["workload"]["cases"] = [c for c in trace["workload"]["cases"] if c["split"] == "train"]
    options = candidates(make_program(spec.operation, train[0].shape))
    random.Random(spec.seed).shuffle(options)
    started = time.perf_counter()
    exhausted = False
    for s in options:
        batch = []
        for case in train:
            if time.perf_counter() - started >= budget_seconds:
                exhausted = True
                break
            p = make_program(spec.operation, case.shape)
            r = {
                "case_id": case.id,
                "operation": spec.operation,
                "shape": list(case.shape),
                "split": "train",
                "dtype": "float32",
                "layout": "contiguous",
                "weight": case.weight,
                "variant": s.id,
                "schedule": s.to_dict(),
                "status": "error",
                "correctness": "unavailable",
            }
            try:
                inputs = make_inputs(p, spec.seed)
                expected = reference(p, *inputs)
                before = time.perf_counter()
                kernel = Kernel(p, s)
                bound = kernel.prepare(*inputs)
                actual = bound.launch()
                torch.cuda.current_stream().synchronize()
                r["first_invocation_wall_s"] = time.perf_counter() - before
                torch.testing.assert_close(actual, expected, rtol=spec.rtol, atol=spec.atol)
                r["generated_source_bytes"] = Path(kernel.source_path).stat().st_size
                r.update(binary_metadata(bound))
                r.update(status="ok", correctness="passed", **measure(bound.launch))
            except Exception as exc:  # noqa: BLE001 - preserve candidate failures without falsifying success
                r["error"] = f"{type(exc).__name__}: {exc}"
            batch.append(r)
            trace["records"].append(r)
            trace["tuning"] = {
                "budget_wall_s": budget_seconds,
                "elapsed_wall_s": time.perf_counter() - started,
                "budget_overrun_s": max(0.0, time.perf_counter() - started - budget_seconds),
                "budget_checks": "between compile/measure units, one unit may overrun",
                "candidate_evaluations": len(trace["records"]),
                "held_out_evaluated": False,
            }
            atomic_json(output / "training-trace.json", trace)
        if exhausted:
            break
    portfolio = fit_portfolio(trace, max_variants, objective)
    portfolio["tuning"] = trace["tuning"]
    from .io import digest

    portfolio.pop("integrity_hash")
    portfolio["integrity_hash"] = digest(portfolio)
    atomic_json(output / "portfolio.json", portfolio)
    return portfolio
