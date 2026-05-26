"""Bounded greedy reduction with fresh measurement-gated witness confirmation."""

from __future__ import annotations

import math
import time
from pathlib import Path

from .diagnosis import reshape, specification
from .io import ValidationError, atomic_json, integer, positive, read
from .reproducer import export_reproducer


def smaller_shapes(shape):
    """Declared finite neighborhood: halve one dimension, preserving rank."""
    for axis, size in enumerate(shape):
        if size > 1:
            candidate = list(shape)
            candidate[axis] = max(1, size // 2)
            yield candidate


def reduce_regression(value, output, *, max_trials=6, budget_seconds=120, runner=None):
    """No global minimality claim; every retained step must preserve evidence.

    The runner injection enables CPU contract tests with labeled simulated data.
    Public CLI always uses real correctness-gated CUDA execution.
    """
    integer(max_trials, "max_trials")
    if max_trials > 16:
        raise ValidationError("reduction supports at most 16 trials")
    positive(budget_seconds, "budget_seconds")
    if budget_seconds > 1800:
        raise ValidationError("reduction budget exceeds 1800 seconds")
    if runner is None:
        from .diagnostic_gpu import diagnose

        runner = diagnose
    spec = specification(read(value) if isinstance(value, (str, Path)) else value)
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValidationError("reduction output must be new or empty; prior trials are preserved")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    original = runner(spec, root / "original")
    report = {
        "schema_version": 1,
        "artifact_type": "regression_reduction",
        "original_specification": spec,
        "trials": [],
        "status": "not_reducible",
        "strategy": "greedy one-axis halving; no global or exhaustive minimum claim",
        "budget_seconds": budget_seconds,
        "max_trials": max_trials,
        "budget_semantics": "checked between trials; in-flight compilation/numerical checks finish; confirmation reserved separately",
        "limitations": [
            "Adaptive selection can overfit noise; final fresh-seed confirmation is required",
            "Only shape reduction; no graph rewriting or arbitrary Triton reduction",
            "Finite numerical checks do not prove universal correctness",
        ],
    }

    def save():
        atomic_json(root / "reduction.json", report)

    if original["comparison"]["verdict"] != "regression" or not original["recommendations"]:
        report["reason"] = (
            "No repeatable regression with a helpful isolated intervention to preserve"
        )
        save()
        return report
    required = original["recommendations"][0]
    report["preserved_intervention"] = required
    current = spec
    attempts = 0
    accepted = 0
    while attempts < max_trials and time.perf_counter() - started < budget_seconds:
        changed = False
        for shape in smaller_shapes(current["program"]["shape"]):
            if attempts >= max_trials or time.perf_counter() - started >= budget_seconds:
                break
            attempts += 1
            trial = {"index": attempts, "shape": shape, "accepted": False}
            try:
                candidate = reshape(current, shape)
                result = runner(candidate, root / f"trial-{attempts:02d}")
                trial["comparison"] = result["comparison"]
                trial["recommendations"] = result["recommendations"]
                keep = (
                    result["comparison"]["verdict"] == "regression"
                    and required in result["recommendations"]
                )
                if keep:
                    current = candidate
                    trial["accepted"] = True
                    accepted += 1
                    changed = True
            except (ValidationError, RuntimeError) as exc:
                trial["error"] = f"{type(exc).__name__}: {exc}"
            report["trials"].append(trial)
            save()
            if changed:
                break
        if not changed:
            break
    confirm_spec = dict(current)
    confirm_spec["seed"] = (current["seed"] + 100003) % (2**31)
    confirmation = runner(confirm_spec, root / "confirmation")
    confirmed = (
        confirmation["comparison"]["verdict"] == "regression"
        and required in confirmation["recommendations"]
    )
    report.update(
        status="confirmed_reduced_witness"
        if confirmed and accepted
        else "confirmed_unchanged_capture"
        if confirmed
        else "unconfirmed",
        reduced_specification=confirm_spec,
        accepted_steps=accepted,
        element_ratio=math.prod(confirm_spec["program"]["shape"])
        / math.prod(spec["program"]["shape"]),
        confirmation={
            "comparison": confirmation["comparison"],
            "recommendations": confirmation["recommendations"],
        },
        elapsed_seconds=time.perf_counter() - started,
        stopping_reason="trial_or_time_limit"
        if attempts == max_trials or time.perf_counter() - started >= budget_seconds
        else "no_accepted_neighbor",
    )
    if confirmed:
        export_reproducer(confirm_spec, root / "witness")
        atomic_json(root / "witness" / "reduction.json", report)
        atomic_json(root / "witness" / "environment.json", confirmation["environment"])
        atomic_json(
            root / "witness" / "expectation.json",
            {"verdict": "regression", "comparison": confirmation["comparison"]},
        )
    save()
    return report
