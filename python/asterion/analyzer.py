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


