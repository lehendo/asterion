"""Bounded compiler experiments and conservative schedule-level attribution.

This module is CPU-only. Imported specifications are data, never Python programs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

from .codegen import generate
from .io import ValidationError, digest, exact_keys, integer, positive
from .ir import Program, Schedule, canonicalize, transform, verify, verify_schedule

EVIDENCE = {"MEASURED", "CONTROLLED_ABLATION", "MODEL_ESTIMATE", "HYPOTHESIS", "UNAVAILABLE"}
FIELDS = tuple(k for k in Schedule.__dataclass_fields__ if k != "schema_version")


def specification(value):
    """Validate a bounded IR experiment; no paths, modules or executable input."""
    exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "program",
            "baseline",
            "candidate",
            "seed",
            "rounds",
            "launches",
            "threshold",
            "rtol",
            "atol",
            "intervention_policy",
        },
        "experiment",
    )
    try:
        if value["schema_version"] != 1 or value["artifact_type"] != "regression_spec":
            raise ValidationError("expected regression_spec version 1")
        p = canonicalize(Program.from_dict(value["program"]))
        a, b = (Schedule.from_dict(value[k]) for k in ("baseline", "candidate"))
        verify_schedule(p, a)
        verify_schedule(p, b)
        result = {
            "schema_version": 1,
            "artifact_type": "regression_spec",
            "program": p.to_dict(),
            "baseline": a.to_dict(),
            "candidate": b.to_dict(),
            "intervention_policy": value.get("intervention_policy", "effective"),
        }
        if result["intervention_policy"] not in {"effective", "all_changed", "none"}:
            raise ValidationError("intervention_policy must be effective, all_changed or none")
        for key, default, low, high in (
            ("seed", 1729, 0, 2**31 - 1),
            ("rounds", 16, 12, 100),
            ("launches", 100, 10, 1000),
        ):
            number = integer(value.get(key, default), key, minimum=low)
            if number > high:
                raise ValidationError(f"{key} exceeds bounded experiment limit")
            result[key] = number
        for key, default in (("threshold", 0.10), ("rtol", 1e-4), ("atol", 1e-5)):
            result[key] = positive(value.get(key, default), key)
        if result["threshold"] > 1 or max(result["rtol"], result["atol"]) > 0.01:
            raise ValidationError("threshold/tolerance exceeds supported bound")
        if math.prod(p.shape) > 64 * 1024 * 1024:
            raise ValidationError("diagnostic workload exceeds bounded allocation limit")
        return result
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(f"malformed regression specification: {exc}") from exc


