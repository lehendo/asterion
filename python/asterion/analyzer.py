from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from .io import ValidationError, digest, read
from .schema import WorkloadSpec, validate_trace
from .stats import ratio_interval, summary, weighted_quantile


def analyze(source):
    data = read(source) if isinstance(source, (str, Path)) else source
    if data.get("artifact_type") != "trace":
        spec = WorkloadSpec.from_dict(data)
        return {
            "schema_version": 1,
            "artifact_type": "workload_analysis",
            "measurement_kind": "unavailable",
            "spec_hash": digest(spec.to_dict()),
            "operation": spec.operation,
            "case_count": len(spec.cases),
            "splits": {
                s: sum(c.split == s for c in spec.cases) for s in ("train", "validation", "test")
            },
            "message": "Validated workload only. No GPU measurements were executed.",
        }
    validate_trace(data)
    groups = defaultdict(list)
    failed = []
    for r in data["records"]:
        if r["status"] == "ok":
            groups[(r["variant"], r["split"])].append(r)
        else:
            failed.append(
                {
                    "case_id": r["case_id"],
                    "variant": r["variant"],
                    "status": r["status"],
                    "error": r["error"],
                }
            )
    results = []
    for (variant, split), records in sorted(groups.items()):
        medians = [summary(r["device_us"])["median_us"] for r in records]
        weights = [r["weight"] for r in records]
        results.append(
            {
                "variant": variant,
                "split": split,
                "shape_count": len(records),
                "expected_shape_median_us": sum(v * w for v, w in zip(medians, weights))
                / sum(weights),
                "p95_shape_median_us": weighted_quantile(medians, weights, 0.95),
                "max_shape_median_us": max(medians),
                "cases": [
                    {"case_id": r["case_id"], "shape": r["shape"], **summary(r["device_us"])}
                    for r in records
                ],
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "analysis",
        "measurement_kind": data["measurement_kind"],
        "source_hash": digest(data),
        "groups": results,
        "failures": failed,
        "caveats": [
            "Aggregate latencies summarize the declared distribution of shape medians.",
            "They are not the p95 of pooled invocation runtimes.",
            "No occupancy, coalescing, or causal performance claims are inferred.",
        ],
    }


def _index(trace, variant):
    validate_trace(trace)
    variants = sorted({r["variant"] for r in trace["records"]})
    if variant is None:
        if len(variants) != 1:
            raise ValidationError(f"choose a variant explicitly from {variants}")
        variant = variants[0]
    if variant not in variants:
        raise ValidationError(f"unknown variant {variant!r}")
    return {r["case_id"]: r for r in trace["records"] if r["variant"] == variant}


def compare(baseline, candidate, baseline_variant=None, candidate_variant=None):
    a = read(baseline) if isinstance(baseline, (str, Path)) else baseline
    b = read(candidate) if isinstance(candidate, (str, Path)) else candidate
    ai, bi = _index(a, baseline_variant), _index(b, candidate_variant)
    if a["measurement_kind"] != b["measurement_kind"]:
        raise ValidationError("cannot mix measured and simulated fixture traces")
    if a["environment"].get("gpu_runtime") != b["environment"].get("gpu_runtime"):
        raise ValidationError("GPU runtime targets differ; compare only matching device manifests")
    rows, failures = [], []
    for key in sorted(set(ai) & set(bi)):
        ar, br = ai[key], bi[key]
        for field in ("operation", "shape", "dtype", "layout", "split", "weight"):
            if ar[field] != br[field]:
                raise ValidationError(f"case {key}: incompatible {field}")
        if ar["status"] != "ok" or br["status"] != "ok":
            failures.append(
                {"case_id": key, "baseline_status": ar["status"], "candidate_status": br["status"]}
            )
            continue
        av, bv = summary(ar["device_us"]), summary(br["device_us"])
        rows.append(
            {
                "case_id": key,
                "shape": ar["shape"],
                "split": ar["split"],
                "weight": ar["weight"],
                "baseline_median_us": av["median_us"],
                "candidate_median_us": bv["median_us"],
                "relative_slowdown": bv["median_us"] / av["median_us"],
                "speedup": av["median_us"] / bv["median_us"],
                "ratio_bootstrap_95_interval": ratio_interval(ar["device_us"], br["device_us"]),
                "noisy": av["noisy"] or bv["noisy"],
                "small_sample": av["small_sample"] or bv["small_sample"],
            }
        )
    groups = []
    for split in ("train", "validation", "test"):
        selected = [r for r in rows if r["split"] == split]
        if not selected:
            continue
        weights = [r["weight"] for r in selected]
        ratios = [r["relative_slowdown"] for r in selected]
        am = sum(r["baseline_median_us"] * r["weight"] for r in selected) / sum(weights)
        bm = sum(r["candidate_median_us"] * r["weight"] for r in selected) / sum(weights)
        groups.append(
            {
                "split": split,
                "shape_count": len(selected),
                "baseline_expected_shape_median_us": am,
                "candidate_expected_shape_median_us": bm,
                "expected_runtime_ratio": bm / am,
                "geometric_mean_relative_slowdown": math.exp(
                    sum(math.log(v) * w for v, w in zip(ratios, weights)) / sum(weights)
                ),
                "p95_relative_slowdown": weighted_quantile(ratios, weights, 0.95),
                "max_relative_slowdown": max(ratios),
                "candidate_p95_absolute_shape_median_us": weighted_quantile(
                    [r["candidate_median_us"] for r in selected], weights, 0.95
                ),
            }
        )
    missing_a, missing_b = sorted(set(bi) - set(ai)), sorted(set(ai) - set(bi))
    return {
        "schema_version": 1,
        "artifact_type": "comparison",
        "measurement_kind": a["measurement_kind"],
        "baseline_hash": digest(a),
        "candidate_hash": digest(b),
        "rows": rows,
        "groups": groups,
        "only_candidate": missing_a,
        "only_baseline": missing_b,
        "failures": failures,
        "coverage_complete": not (missing_a or missing_b or failures),
        "caveats": [
            "Ratios compare candidate/baseline shape medians, not oracle regret.",
            "Aggregates are conditional on successful matched cases; consult coverage.",
            "Bootstrap intervals assume independent stationary timing samples.",
            "A sampled maximum is not a guarantee for unseen workloads.",
        ],
    }
