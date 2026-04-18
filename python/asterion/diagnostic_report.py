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


def render(result):
    esc = lambda value: html.escape(str(value))
    comparison = result["comparison"]
    medians = {
        name: statistics.median(
            positive(b["measurements"][name][result["boundary"]], "timing")
            for b in result["blocks"]
        )
        for name in result["variants"]
    }
    scale = max(medians.values())
    bars = []
    for i, (name, value) in enumerate(medians.items()):
        y = 18 + i * 45
        color = "#f4a58a" if name == "candidate" else "#80d9cd"
        bars.append(
            f'<text x="0" y="{y + 15}" fill="#e6edf6">{esc(name)}</text><rect x="190" y="{y}" width="{480 * value / scale:.2f}" height="24" rx="4" fill="{color}"/><text x="700" y="{y + 16}" fill="#e6edf6">{value:.3f} µs</text>'
        )
    chart = f'<svg viewBox="0 0 850 {len(bars) * 45 + 30}" role="img" aria-label="Measured median latency by variant"><title>Saved block medians at the declared boundary</title>{"".join(bars)}</svg>'
    baseline = result["variants"]["baseline"]["schedule"]
    rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{esc({k: value for k, value in v['schedule'].items() if baseline.get(k) != value} or 'baseline schedule')}</td><td>{esc(v.get('analysis'))}</td><td>{esc([{k: stage.get(k) for k in ('registers', 'spills', 'shared_bytes')} for stage in v.get('lowering', [])])}</td></tr>"
        for name, v in result["variants"].items()
    )
    findings = "".join(
        f"<article><h3>{esc(f['field'])} · {esc(f['category'])}</h3><p>{esc(f['explanation'])}</p><p>Intervention/candidate ratio: {f['result']['median_paired_ratio']:.3f}; {esc(f['result']['verdict'])}.</p><p>Hypothesis: {esc(f.get('mechanism_hypothesis', {}).get('statement', 'unavailable'))}</p><p>Lowering model changes: {esc(f.get('lowering_model_changes', {}).get('changes', {}))}</p><p>{esc(f['hardware_cause'])}</p><p>Alternatives: {esc('; '.join(f['alternative_explanations']))}</p></article>"
        for f in result["findings"]
    )
    raw = esc(json.dumps(result, indent=2))
    source_a, source_b = (
        result["variants"][name].get("generated_source", "") for name in ("baseline", "candidate")
    )
    difference = (
        "\n".join(
            difflib.unified_diff(
                source_a.splitlines(),
                source_b.splitlines(),
                fromfile="baseline",
                tofile="candidate",
                lineterm="",
            )
        )
        or "No saved source difference. Launch options can differ independently; inspect exact schedules."
    )
    target = result["environment"].get("gpu_runtime") or {}
    prepared = result.get("prepared_comparison")
    host = (
        f"<p>Prepared launch plus completion: candidate/baseline {prepared['median_paired_ratio']:.3f}× ({esc(prepared['verdict'])}). Includes host checks and synchronization.</p>"
        if prepared
        else ""
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Asterion regression experiment</title><style>
body{{font:16px system-ui;margin:0;background:#101623;color:#e6edf6}}main{{max-width:1100px;margin:auto;padding:40px}}h1{{font-size:36px}}.lead,article{{padding:22px;background:#1b2639;border-radius:12px;margin:20px 0}}small{{color:#a9bad1}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;vertical-align:top;padding:14px;border-bottom:1px solid #39445a;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.metric{{font-size:32px;color:#80d9cd}}details{{margin-top:24px}}code{{color:#80d9cd}}svg{{width:100%;max-height:650px}}.scroll{{overflow-x:auto}}@media(max-width:600px){{main{{padding:18px}}table{{min-width:750px}}h1{{font-size:29px}}}}</style><main>
<small>ASTERION · CONTROLLED COMPILER EXPERIMENT · {esc(result.get("measurement_kind", "unavailable")).upper()}</small><h1>What changed, and which intervention helped?</h1>
<section class="lead"><span class="metric">{esc(comparison["verdict"])} · {comparison["median_paired_ratio"]:.3f}×</span><p>Candidate/baseline paired latency ratio at <code>{esc(result["boundary"])}</code>. {comparison["blocks"]} randomized blocks; practical threshold {comparison["practical_threshold"]:.0%}.</p>{host}<p>{esc(comparison["limitations"])}</p></section>
<h2>Measured latency</h2>{chart}<p><small>Bars show per-variant sample medians; the headline is the median of paired ratios. Device graph timing excludes compilation and Python dispatch.</small></p>
<h2>Supported workload</h2><p>{esc(result["specification"]["program"]["operation"])} · shape {esc(result["specification"]["program"]["shape"])} · contiguous FP32</p><p>{esc(result["scope"])}</p>
<p>{esc(target.get("name", "unavailable"))} · capability {esc(target.get("capability", "unavailable"))} · torch {esc(target.get("torch", "unavailable"))} · Triton {esc(target.get("triton", "unavailable"))}</p>
<h2>Controlled interventions</h2>{findings or "<p>No effective changed scheduling fields to intervene on.</p>"}
<h2>Compiler structure</h2><div class="scroll"><table><tr><th>Variant</th><th>Changes from baseline</th><th>MODEL_ESTIMATE</th><th>Compiler resource metadata</th></tr>{rows}</table></div><p><small>Modeled counts ignore cache behavior and spills. Compiler register/shared-memory metadata does not establish hardware occupancy or a cause.</small></p><details><summary>Generated source difference</summary><pre>{esc(difference)}</pre></details>
<h2>Reproduce</h2><p><code>asterion reproduce reproducer.json --output rerun</code></p><p>Generated source, available intermediate representations, binary hashes, numerical checks and raw blocks accompany the saved results. Reproduction on different hardware may change the outcome.</p>
<details><summary>Inspect complete saved evidence and environment</summary><pre>{raw}</pre></details></main></html>"""
    return page
