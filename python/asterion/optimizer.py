"""Training-only bounded beam search over metadata decision partitions.

No approximation/novelty guarantee is claimed. Every partition is deployable from
shape metadata; evaluation never chooses a variant with a test-time oracle.
"""

from __future__ import annotations

import math
from itertools import pairwise

from .io import ValidationError, digest, integer, positive
from .schema import validate_trace
from .stats import cvar, summary

FEATURE = "last_dimension"


def feature(shape):
    return shape[-1]


def objective(ratios, weights, risk=0.5, q=0.9):
    if not ratios or len(ratios) != len(weights) or risk < 0 or not 0 <= q < 1:
        raise ValidationError("invalid objective")
    values = [math.log(positive(r, "latency ratio")) for r in ratios]
    total = sum(positive(w, "weight") for w in weights)
    return sum(v * w for v, w in zip(values, weights)) / total + risk * cvar(values, weights, q)


def fit_portfolio(
    trace,
    max_variants=4,
    objective_name="robust",
    beam_width=32,
    max_leaves=4,
    max_code_bytes=100000,
):
    validate_trace(trace)
    integer(max_variants, "max_variants")
    integer(beam_width, "beam_width")
    integer(max_leaves, "max_leaves")
    integer(max_code_bytes, "max_code_bytes")
    if objective_name not in {"mean", "robust"}:
        raise ValidationError("objective must be mean or robust")
    if any(r["split"] != "train" for r in trace["records"]):
        raise ValidationError(
            "optimizer accepts training-only measurements; held-out data is forbidden"
        )
    if any(r["operation"] != trace["records"][0]["operation"] for r in trace["records"]):
        raise ValidationError("one operation per portfolio")
    by_case = {}
    variants = {}
    for r in trace["records"]:
        group = by_case.setdefault(
            r["case_id"], {"shape": r["shape"], "weight": r["weight"], "values": {}}
        )
        if group["shape"] != r["shape"] or group["weight"] != r["weight"]:
            raise ValidationError("inconsistent training case metadata")
        if r["status"] == "ok" and "schedule" in r:
            group["values"][r["variant"]] = summary(r["device_us"])["median_us"]
            if r["variant"] in variants and variants[r["variant"]] != r["schedule"]:
                raise ValidationError("variant has inconsistent schedule definitions")
            variants[r["variant"]] = r["schedule"]
    # Require complete correctness-gated training coverage, not a missing-data winner.
    legal = sorted(variants)
    if not legal or any(not row["values"] for row in by_case.values()):
        raise ValidationError("no generated schedule has complete correct training coverage")
    rows = sorted(by_case.values(), key=lambda r: (feature(r["shape"]), tuple(r["shape"])))
    weights = [r["weight"] for r in rows]
    reference = [min(r["values"].values()) for r in rows]
    risk = 0.5 if objective_name == "robust" else 0.0
    # Source-size proxy must be recorded at each measured candidate; enforce an explicit budget.
    if trace["measurement_kind"] == "measured":
        for variant in legal:
            signatures = {
                tuple(r.get("compiled_binary_hashes", []))
                for r in trace["records"]
                if r["variant"] == variant and r["status"] == "ok"
            }
            if any(not signature for signature in signatures) or len(signatures) != 1:
                raise ValidationError(
                    "variant lacks one consistent observed compiled-binary signature"
                )

    def covers(variant, lo, hi):
        return all(variant in rows[i]["values"] for i in range(lo, hi))

    costs = {
        v: max(
            (r.get("generated_code_bytes", 0) for r in trace["records"] if r["variant"] == v),
            default=0,
        )
        for v in legal
    }
    if any(costs[v] <= 0 for v in legal):
        raise ValidationError("generated_code_bytes must be recorded for each schedule")

    def score(leaves):
        chosen = []
        for lo, hi, v in leaves:
            chosen.extend(rows[i]["values"][v] / reference[i] for i in range(lo, hi))
        return objective(chosen, weights, risk) + 1e-9 * sum(
            costs[v] for v in {leaf[2] for leaf in leaves}
        )

    # Prefix beam search permits region-specific schedules even when no single
    # schedule covers the whole family. Bounded pruning has no optimality guarantee.
    def prefix_score(leaves, end):
        ratios = []
        for lo, hi, v in leaves:
            ratios.extend(rows[i]["values"][v] / reference[i] for i in range(lo, hi))
        return objective(ratios, weights[:end], risk) + 1e-9 * sum(
            costs[v] for v in {leaf[2] for leaf in leaves}
        )

    frontier = {0: [()]}
    complete = []
    for _depth in range(1, min(max_leaves, len(rows)) + 1):
        next_frontier = {}
        for end in range(1, len(rows) + 1):
            proposals = set()
            for start, prefixes in frontier.items():
                if start >= end:
                    continue
                if start and feature(rows[start - 1]["shape"]) == feature(rows[start]["shape"]):
                    continue
                for variant in legal:
                    if not covers(variant, start, end):
                        continue
                    for prefix in prefixes:
                        candidate = prefix + ((start, end, variant),)
                        chosen = {v for _, _, v in candidate}
                        if (
                            len(chosen) <= max_variants
                            and sum(costs[v] for v in chosen) <= max_code_bytes
                        ):
                            proposals.add(candidate)
            if proposals:
                next_frontier[end] = sorted(
                    proposals, key=lambda leaves: (prefix_score(leaves, end), leaves)
                )[:beam_width]
        complete.extend(next_frontier.get(len(rows), []))
        frontier = next_frontier
    if not complete:
        raise ValidationError("no candidate partition fits generated-code budget and coverage")
    best = min(complete, key=lambda leaves: (score(leaves), len(leaves), leaves))
    unique = sorted({v for _, _, v in best})
    leaves = []
    for i, (lo, hi, v) in enumerate(best):
        # Last boundary includes the observed maximum; other boundaries bisect distinct values.
        upper = (
            feature(rows[-1]["shape"])
            if i == len(best) - 1
            else (feature(rows[hi - 1]["shape"]) + feature(rows[hi]["shape"])) // 2
        )
        leaves.append({"upper_inclusive": upper, "variant": v})
    # Clamp interpolated boundaries to verified static tile coverage. The next
    # selected rule handles the remaining interval; this preserves every training
    # assignment while avoiding illegal gaps between sampled column counts.
    if trace["records"][0]["operation"] in {"sum", "max", "softmax"}:
        for leaf in leaves:
            tile = variants[leaf["variant"]].get("row_tile", 0)
            if tile:
                leaf["upper_inclusive"] = min(leaf["upper_inclusive"], tile)
        if any(a["upper_inclusive"] >= b["upper_inclusive"] for a, b in pairwise(leaves)):
            raise ValidationError("selected dispatch partition has an unsupported tile boundary")
    result = {
        "schema_version": 1,
        "artifact_type": "portfolio",
        "measurement_kind": trace["measurement_kind"],
        "operation": trace["records"][0]["operation"],
        "dtype": "float32",
        "layout": "contiguous",
        "target": trace["environment"].get("gpu_runtime"),
        "variants": {v: variants[v] for v in unique},
        "dispatcher": {
            "feature": FEATURE,
            "minimum": feature(rows[0]["shape"]),
            "maximum": feature(rows[-1]["shape"]),
            "rules": leaves,
        },
        "fallback": "pytorch",
        "objective": objective_name,
        "training_objective": score(best),
        "max_variants": max_variants,
        "max_code_bytes": max_code_bytes,
        "generated_code_bytes": sum(costs[v] for v in unique),
        "code_budget_unit": "cubin_bytes"
        if trace["measurement_kind"] == "measured"
        else "simulated_fixture_bytes",
        "train_shapes": [r["shape"] for r in rows],
        "training_trace_hash": digest(trace),
        "algorithm": {
            "name": "bounded prefix-partition beam search",
            "beam_width": beam_width,
            "max_leaves": max_leaves,
            "risk_weight": risk,
            "cvar_q": 0.9,
            "approximation_guarantee": None,
        },
        "correctness_scope": "validated training cases; other legal shapes use verified codegen and carry no measured performance guarantee",
    }
    result["integrity_hash"] = digest(result)
    return result


