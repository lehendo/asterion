"""Passive self-contained reports for external-kernel experiment records."""

import html
import json
from pathlib import Path

from .io import ValidationError, canonical, digest


def _bounded_report(value):
    if not isinstance(value, dict):
        raise ValidationError("report artifact must be an object")
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 64 or count > 100000:
            raise ValidationError("external report structure exceeds supported nesting/node limits")
        if isinstance(item, (dict, list)) and count + len(pending) + len(item) > 100000:
            raise ValidationError("external report structure exceeds supported nesting/node limits")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
    if len(canonical(value)) > 8 * 1024 * 1024:
        raise ValidationError("external report exceeds supported8MiB limit")


def write_external_reduction_report(result, output):
    _bounded_report(result)
    if result.get("artifact_type") != "external_reduction" or result.get("result_id") != digest(
        {k: v for k, v in result.items() if k != "result_id"}
    ):
        raise ValidationError("external reduction integrity mismatch")
    if (
        not isinstance(result.get("original_specification"), dict)
        or not isinstance(result["original_specification"].get("dimensions"), dict)
        or not isinstance(result.get("original_complexity"), dict)
        or not isinstance(result.get("trials"), list)
        or not isinstance(result.get("reduced_specification", {}), dict)
    ):
        raise ValidationError("malformed external reduction report")
    escape = lambda value: html.escape(str(value))
    status = escape(result.get("status"))
    ratio = result.get("storage_ratio")
    storage = (
        f"{ratio:.2%} of original declared storage"
        if isinstance(ratio, (int, float))
        else "No confirmed reduction"
    )
    sections = [
        (
            "Witness export",
            {
                "status": result.get("export_status", "not attempted"),
                "error": result.get("export_error"),
            },
        ),
        ("What was preserved", result.get("properties", {})),
        ("Confirmed comparison", result.get("confirmation", {})),
        (
            "Original operation and complexity",
            {
                "dimensions": result["original_specification"]["dimensions"],
                "complexity": result["original_complexity"],
            },
        ),
        (
            "Reduced operation and complexity",
            {
                "dimensions": result.get("reduced_specification", {}).get("dimensions"),
                "complexity": result.get("reduced_complexity"),
            },
        ),
        ("Every attempted reduction", result["trials"]),
        (
            "Budget and limitations",
            {
                k: result.get(k)
                for k in (
                    "strategy",
                    "budget_semantics",
                    "max_trials",
                    "budget_seconds",
                    "elapsed_seconds",
                    "stopping_reason",
                    "generator_semantics",
                    "reason",
                    "original_status",
                )
            },
        ),
    ]
    body = "".join(
        f"<details open><summary>{escape(title)}</summary><pre>{html.escape(json.dumps(value, indent=2))}</pre></details>"
        for title, value in sections
    )
    replay = (
        "Review executable source, dependencies and manifest first. In a compatible trusted GPU environment run <code>python witness/reproduce.py --trusted-code --output replay</code>. Inspect the new results; performance may change across environments. The bundle needs torch and Triton, but no installed Asterion package."
        if result.get("export_status") == "EXPORTED"
        else "No confirmed replayable witness was exported. Inspect the retained trials and failure reason before attempting a new experiment."
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Asterion reduced witness</title><style>body{{margin:0;background:#101827;color:#e7edf6;font:16px/1.6 system-ui}}main{{max-width:1100px;margin:auto;padding:32px}}small{{color:#83dfca}}details{{padding:20px;background:#1b283b;border:1px solid #34445b;border-radius:12px;margin:16px 0}}summary{{font-weight:650}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}</style><main><small>ASTERION · BOUNDED REDUCTION</small><h1>{status}</h1><p>{escape(storage)}. Helpful intervention required: {escape(result.get("preserved_intervention", "unavailable"))}.</p><p>Confirmation preserves only the reported measured properties. It does not establish global minimality or preservation of a hardware bottleneck. A confirmed unchanged capture is not a smaller reproducer.</p>{body}<h2>Reproduction</h2><p>{replay}</p><small>Evidence identity {escape(result["result_id"])}</small></main></html>"""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.html"
    path.write_text(page)
    return path


