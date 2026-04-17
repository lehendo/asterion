"""Self-contained passive report. No imported source or artifact paths execute."""

from __future__ import annotations

import difflib
import html
import json
import statistics
from pathlib import Path

from .io import ValidationError, atomic_json, digest, positive


def write_diagnosis(result, output):
    if not isinstance(result, dict) or result.get("artifact_type") != "diagnosis":
        raise ValidationError("expected saved diagnosis")
    if (
        "result_id" in result
        and digest({k: v for k, v in result.items() if k != "result_id"}) != result["result_id"]
    ):
        raise ValidationError("saved diagnosis integrity mismatch")
    try:
        page = render(result)
    except (KeyError, TypeError, ValueError, AttributeError, statistics.StatisticsError) as exc:
        raise ValidationError(f"malformed saved diagnosis: {exc}") from exc
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", result)
    (output / "report.html").write_text(page)
    return output / "report.html"


