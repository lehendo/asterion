from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .io import atomic_json


def write_report(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "results.json", data)
    rows = data.get("rows", [])
    if rows:
        with (output / "cases.csv").open("w", newline="") as stream:
            fields = [
                "case_id",
                "shape",
                "split",
                "weight",
                "baseline_median_us",
                "candidate_median_us",
                "relative_slowdown",
                "speedup",
                "noisy",
                "small_sample",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def esc(v):
        return html.escape(str(v), quote=True)

    plotted = sorted(rows, key=lambda r: r["relative_slowdown"], reverse=True)
    svg = []
    if plotted:
        maximum = max(1.0, max(r["relative_slowdown"] for r in plotted))
        for i, r in enumerate(plotted):
            width = 420 * r["relative_slowdown"] / maximum
            color = "#c84038" if r["relative_slowdown"] > 1 else "#23846c"
            svg.append(
                f'<text x="8" y="{i * 28 + 18}">{esc(r["case_id"])}</text>'
                f'<rect x="170" y="{i * 28 + 4}" width="{width:.2f}" height="19" fill="{color}"/>'
                f'<text x="{180 + width:.2f}" y="{i * 28 + 18}">{r["relative_slowdown"]:.3f}×</text>'
            )
        marker = 170 + 420 / maximum
        svg.append(
            f'<line x1="{marker}" x2="{marker}" y1="0" y2="{len(plotted) * 28}" stroke="#162331" stroke-dasharray="4"/>'
        )
    badge = data.get("measurement_kind", "unavailable").upper()
    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(r[k])}</td>"
            for k in (
                "case_id",
                "shape",
                "split",
                "baseline_median_us",
                "candidate_median_us",
                "relative_slowdown",
            )
        )
        + "</tr>"
        for r in plotted
    )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Asterion performance report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#162331}}h1{{margin-bottom:8px}}.badge{{background:#e9edf1;padding:6px 12px;display:inline-block}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d5dce2;text-align:left}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:18px;overflow-wrap:anywhere}}svg{{width:100%;max-width:850px}}a{{color:#126482}}</style>
<h1>Asterion</h1><p>GPU performance, without the guesswork.</p>
<p class="badge">{esc(badge)}</p><p>{esc(data.get("message", "Candidate / baseline latency: values above 1 indicate slowdown."))}</p>
<p><a href="results.json">Machine-readable results</a> · <a href="cases.csv">Per-shape CSV (comparison reports)</a></p>
<svg role="img" aria-label="Relative slowdown by shape; dashed line is parity" viewBox="0 0 760 {max(35, len(plotted) * 28)}">{"".join(svg)}</svg>
<table><thead><tr><th>Case</th><th>Shape</th><th>Split</th><th>Baseline µs</th><th>Candidate µs</th><th>Slowdown</th></tr></thead><tbody>{table}</tbody></table>
<h2>Distribution summaries</h2><pre>{esc(json.dumps(data.get("groups", data), indent=2))}</pre>
<h2>Coverage and caveats</h2><pre>{esc(json.dumps({k: v for k, v in data.items() if k in ("coverage_complete", "only_baseline", "only_candidate", "failures", "caveats")}, indent=2))}</pre>
</html>"""
    # HTML contains only escaped text and generated inline SVG; no network, scripts or telemetry.
    (output / "report.html").write_text(body)
    (output / "report.md").write_text(
        "# Asterion report\n\nEvidence: "
        + badge
        + "\n\n"
        + json.dumps(data.get("groups", []), indent=2)
        + "\n"
    )
    return output / "report.html"
