"""Auditable common-measurement baselines; no evaluation/test measurements enter fitting."""

from __future__ import annotations

import math
import random

from .io import ValidationError, digest
from .optimizer import fit_portfolio
from .stats import summary, weighted_quantile
from .validation import seal


def fit_baseline(trace, kind, *, max_variants=4, max_code_bytes=100000, seed=1729):
    if kind in {"mean_portfolio", "robust_portfolio"}:
        return fit_portfolio(
            trace,
            max_variants,
            "mean" if kind == "mean_portfolio" else "robust",
            max_code_bytes=max_code_bytes,
        )
    if kind not in {"fixed", "random_search_single", "representative", "nearest_bucket"}:
        raise ValidationError("unsupported baseline policy")
    result = fit_portfolio(trace, max_variants, "mean", max_code_bytes=max_code_bytes)
    schedules = {}
    rows = {}
    costs = {}
    for record in trace["records"]:
        row = rows.setdefault(
            record["case_id"], {"shape": record["shape"], "weight": record["weight"], "values": {}}
        )
        if record["status"] == "ok" and "schedule" in record:
            variant = record["variant"]
            row["values"][variant] = summary(record["device_us"])["median_us"]
            schedules[variant] = record["schedule"]
            costs[variant] = max(costs.get(variant, 0), record["generated_code_bytes"])
    ordered = sorted(rows.values(), key=lambda row: (row["shape"][-1], row["shape"]))
    global_choices = [v for v in schedules if all(v in row["values"] for row in ordered)]
    if not global_choices:
        raise ValidationError("single-schedule baselines require a globally legal candidate")
    if kind == "fixed":
        # Pure metadata choice: smallest covering row tile and conventional row/warp mapping.
        selected = min(
            global_choices,
            key=lambda v: (
                schedules[v]["row_tile"],
                schedules[v]["rows"],
                schedules[v]["warps"],
                v,
            ),
        )
    elif kind == "random_search_single":
        trials = list(global_choices)
        random.Random(seed).shuffle(trials)
        # Same complete legal measurement pool, random exploration order, best mean log latency.
        selected = min(
            trials, key=lambda v: sum(math.log(row["values"][v]) * row["weight"] for row in ordered)
        )
    elif kind == "representative":
        center = weighted_quantile(
            [row["shape"][-1] for row in ordered], [row["weight"] for row in ordered], 0.5
        )
        representative = min(
            ordered, key=lambda row: (abs(row["shape"][-1] - center), row["shape"])
        )
        selected = min(global_choices, key=lambda v: (representative["values"][v], v))
    if kind != "nearest_bucket":
        rules = [{"upper_inclusive": ordered[-1]["shape"][-1], "variant": selected}]
    else:
        groups = {}
        for row in ordered:
            groups.setdefault(row["shape"][-1], []).append(row)
        keys = sorted(groups)
        rules = []
        for i, key in enumerate(keys):
            group = groups[key]
            available = [v for v in schedules if all(v in row["values"] for row in group)]
            selected = min(
                available,
                key=lambda v: (sum(math.log(row["values"][v]) * row["weight"] for row in group), v),
            )
            upper = key if i == len(keys) - 1 else (key + keys[i + 1]) // 2
            rules.append({"upper_inclusive": upper, "variant": selected})
    chosen = {r["variant"] for r in rules}
    if len(chosen) > max_variants or sum(costs[v] for v in chosen) > max_code_bytes:
        raise ValidationError(f"{kind} exceeds the declared deployment resource budget")
    if trace["records"][0]["operation"] in {"sum", "max", "softmax"}:
        for rule in rules:
            tile = schedules[rule["variant"]].get("row_tile", 0)
            if tile:
                rule["upper_inclusive"] = min(rule["upper_inclusive"], tile)
    result["variants"] = {v: schedules[v] for v in sorted(chosen)}
    result["dispatcher"]["rules"] = rules
    result["generated_code_bytes"] = sum(costs[v] for v in chosen)
    result["objective"] = "baseline"
    result["training_objective"] = None
    result["algorithm"] = {
        "name": kind,
        "seed": seed,
        "measurement_pool": "shared correctness-gated training trace",
        "candidate_evaluations": len(trace["records"]),
        "budget_normalization": "same legal candidate measurements; policy fitting time is separate",
    }
    result["training_trace_hash"] = digest(trace)
    return seal(result, "integrity_hash")
