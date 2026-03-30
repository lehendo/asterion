"""Isolated official-framework adapters. No imitation is labeled as an upstream tuner."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from hashlib import sha256

from .codegen import generate
from .io import ValidationError, digest
from .ir import Schedule, candidates, make_program
from .stats import weighted_quantile


def tunable_row_source(operation):
    if operation not in {"sum", "max", "softmax"}:
        raise ValidationError("official common-space adapter currently supports row operations")
    source = generate(make_program(operation, (1, 1)), Schedule(row_tile=16384, rows=1))
    source = source.replace(
        "def kernel(X, O, R, C):",
        "def kernel(X, O, R, C, ROW_TILE: tl.constexpr, ROW_GROUP: tl.constexpr):",
    )
    source = source.replace(
        "tl.program_id(0)*1 + tl.arange(0, 1)",
        "tl.program_id(0)*ROW_GROUP + tl.arange(0, ROW_GROUP)",
    )
    source = source.replace("tl.arange(0, 16384)", "tl.arange(0, ROW_TILE)")
    if source.count("ROW_GROUP") != 3 or source.count("ROW_TILE") != 2:
        raise ValidationError(
            "generated row adapter contract changed; refusing ambiguous parameterization"
        )
    return source


class OfficialTriton:
    """Actual triton.autotune on a common Asterion-generated schedule space.

    Train each explicit shape with upstream selection. Deployment uses the stored
    upstream best config from the nearest legal training shape; no test-time tuning.
    """

    def __init__(self, operation):
        from .backend import require_gpu
        from .cache import Cache

        self.torch, self.triton = require_gpu()
        if self.triton.__version__ != "3.3.1":
            raise ValidationError("official adapter is validated against Triton 3.3.1")
        source = tunable_row_source(operation)
        key = digest({"adapter": source, "triton": self.triton.__version__})
        path = Cache.default().root / ("official-" + key + ".py")
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".py")
        with os.fdopen(fd, "w") as out:
            out.write(source)
        os.replace(tmp, path)
        spec = importlib.util.spec_from_file_location("asterion_official_" + key, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.base = module.kernel
        self.operation = operation
        self.training = []
        self.evidence = []

    def train(self, cases, *, budget_seconds=120, samples=30):
        from .backend import make_inputs
        from .gpu import measure, operation_reference

        started = time.perf_counter()
        for case in cases:
            if time.perf_counter() - started >= budget_seconds:
                break
            shape = case.shape
            program = make_program(self.operation, shape)
            options = candidates(program)
            configs = [
                self.triton.Config(
                    {"ROW_TILE": s.row_tile, "ROW_GROUP": s.rows}, num_warps=s.warps, num_stages=3
                )
                for s in options
            ]
            inputs = make_inputs(program, 1729)
            output = (
                self.torch.empty_like(inputs[0])
                if self.operation == "softmax"
                else self.torch.empty(shape[0], device="cuda", dtype=self.torch.float32)
            )
            expected = operation_reference(self.operation)(*inputs)
            trials = []

            def bench(fn, quantiles, trials=trials, output=output, expected=expected):
                fn()
                self.torch.cuda.current_stream().synchronize()
                self.torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
                values = measure(fn, samples=samples, warmup=10, launches=50)["device_us"]
                trials.append(values)
                return [weighted_quantile(values, [1] * len(values), q) / 1000 for q in quantiles]

            tuner = self.triton.autotune(configs=configs, key=["R", "C"], do_bench=bench)(self.base)
            grid = lambda meta, shape=shape: (self.triton.cdiv(shape[0], meta["ROW_GROUP"]),)
            before = time.perf_counter()
            tuner[grid](*inputs, output, *shape, enable_fp_fusion=False)
            self.torch.cuda.current_stream().synchronize()
            self.torch.testing.assert_close(
                output, operation_reference(self.operation)(*inputs), rtol=1e-4, atol=1e-5
            )
            best = tuner.best_config
            schedule = Schedule(
                row_tile=best.kwargs["ROW_TILE"],
                rows=best.kwargs["ROW_GROUP"],
                warps=best.num_warps,
            )
            compiled = self.base[grid](
                *inputs,
                output,
                *shape,
                ROW_TILE=schedule.row_tile,
                ROW_GROUP=schedule.rows,
                num_warps=schedule.warps,
                enable_fp_fusion=False,
            )
            cubin = compiled.asm["cubin"]
            self.training.append((tuple(shape), schedule))
            self.evidence.append(
                {
                    "case_id": case.id,
                    "shape": list(shape),
                    "selected_schedule": schedule.to_dict(),
                    "compiled_binary_hash": sha256(cubin).hexdigest(),
                    "generated_code_bytes": len(cubin),
                    "upstream": "triton.autotune",
                    "candidate_evaluations": len(trials),
                    "attempted_configs": len(configs),
                    "raw_trial_device_us": trials,
                    "candidate_correctness": "all measured configs checked against PyTorch",
                    "sampling": {"samples": samples, "warmup": 10, "launches": 50},
                    "tuning_wall_s": time.perf_counter() - before,
                }
            )
        self.elapsed_wall_s = time.perf_counter() - started
        self.budget_wall_s = budget_seconds
        unique = {row["compiled_binary_hash"]: row["generated_code_bytes"] for row in self.evidence}
        self.generated_code_bytes = sum(unique.values())
        if len(unique) > 4 or self.generated_code_bytes > 100000:
            raise ValidationError("official deployed configurations exceed common K/code budget")
        if not self.training:
            raise ValidationError("official Triton baseline has no successful training shape")
        return self

    def prepare(self, inputs):
        from .gpu import operation_reference

        shape = tuple(inputs[0].shape)
        legal = [(s, cfg) for s, cfg in self.training if cfg.row_tile >= shape[1]]
        if not legal:
            return lambda: operation_reference(self.operation)(*inputs), {
                "reason": "no_legal_trained_upstream_config"
            }
        _, schedule = min(legal, key=lambda item: (abs(item[0][1] - shape[1]), item[0]))
        output = (
            self.torch.empty_like(inputs[0])
            if self.operation == "softmax"
            else self.torch.empty(shape[0], device=inputs[0].device, dtype=self.torch.float32)
        )
        for tensor in (*inputs, output):
            tensor.record_stream(self.torch.cuda.current_stream())
        grid = (self.triton.cdiv(shape[0], schedule.rows),)

        def launch():
            self.base[grid](
                *inputs,
                output,
                *shape,
                ROW_TILE=schedule.row_tile,
                ROW_GROUP=schedule.rows,
                num_warps=schedule.warps,
                enable_fp_fusion=False,
            )
            for tensor in (*inputs, output):
                tensor.record_stream(self.torch.cuda.current_stream())
            return output

        return launch, {
            "reason": "nearest_legal_upstream_training_config",
            "schedule": schedule.to_dict(),
        }
