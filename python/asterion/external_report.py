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


def write_external_report(result, output):
    _bounded_report(result)
    if result.get("artifact_type") != "external_diagnosis" or result.get("result_id") != digest(
        {k: v for k, v in result.items() if k != "result_id"}
    ):
        raise ValidationError("external diagnosis integrity mismatch")
    try:
        status = result["status"]
        spec = result["specification"]
        comparison = result.get("comparison", {})
        cards = [
            ("Regression", status),
            ("Numerical admission", result["numerical_status"]),
            (
                "Candidate / baseline",
                f"{comparison['median_paired_ratio']:.3f}x" if comparison else "unavailable",
            ),
            (
                "Graph device baseline → candidate",
                f"{comparison['baseline_median_us']:.3f} → {comparison['candidate_median_us']:.3f} µs"
                if comparison
                else "unavailable",
            ),
            ("Helpful interventions", ", ".join(result["recommendations"]) or "none established"),
        ]
        escape = lambda value: html.escape(str(value))
        card_html = "".join(
            f"<article><small>{escape(k)}</small><h2>{escape(v)}</h2></article>" for k, v in cards
        )
        rows = []
        for name, variant in result["variants"].items():
            compiler = variant.get("compiler", {})
            values = [
                name,
                variant.get("status", "unavailable"),
                variant.get("configuration", {}).get("entrypoint", "unavailable"),
                json.dumps(variant.get("configuration", {}).get("parameters", {}), sort_keys=True),
                variant.get("grid", "unavailable"),
                compiler.get("registers", "unavailable"),
                compiler.get("spills", "unavailable"),
                compiler.get("shared_bytes", "unavailable"),
            ]
            rows.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
        table = (
            "<div class='table'><table><thead><tr>"
            + "".join(
                f"<th>{x}</th>"
                for x in (
                    "Variant",
                    "Admission",
                    "Entrypoint",
                    "Declared parameters",
                    "Grid",
                    "Registers",
                    "Spills",
                    "Shared bytes",
                )
            )
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
        if result["recommendations"]:
            actions = []
            for finding in result["findings"]:
                if not finding.get("supported"):
                    continue
                label = (
                    "Revert the complete source/configuration package"
                    if finding.get("kind") == "full_variant_revert"
                    else "Test the validated reversion of " + str(finding.get("parameter"))
                )
                actions.append(
                    f"<li>{escape(label)}: <code>{escape(finding['intervention'])}</code>. {escape(finding['scope'])}.</li>"
                )
            action = "<h2>Action supported at this boundary</h2><ul>" + "".join(actions) + "</ul>"
        else:
            action = "<h2>Conclusion</h2><p>No helpful intervention is established. Review numerical admission and comparison uncertainty before drawing a performance conclusion.</p>"
        sections = [
            ("Workload and declared configurations", spec),
            (
                "Environment and source identity",
                {
                    "environment": result["environment"],
                    "source": result["source_identity"],
                    "engine_sources": result.get("engine_sources", {}),
                },
            ),
            (
                "LEVEL1 · Measured comparison and noise",
                {
                    "comparison": comparison,
                    "noise": result.get("noise"),
                    "protocol": result["measurement_protocol"],
                },
            ),
            ("LEVEL2 · Controlled interventions", result["findings"]),
            ("LEVEL3 · Numerical records and compiler evidence", result["variants"]),
            ("LEVEL4 · Hardware mechanism", result["hardware_mechanism"]),
            (
                "Costs, capabilities and limitations",
                {
                    k: result.get(k)
                    for k in (
                        "elapsed_seconds",
                        "time_to_first_comparison_seconds",
                        "time_to_actionable_result_seconds",
                        "measurement_seconds",
                        "module_import_seconds",
                        "experiment_counts",
                        "capabilities",
                        "limitations",
                        "trust",
                    )
                },
            ),
        ]
        body = "".join(
            f"<details {'open' if i < 3 else ''}><summary>{escape(title)}</summary><pre>{html.escape(json.dumps(value, indent=2))}</pre></details>"
            for i, (title, value) in enumerate(sections)
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidationError(f"malformed external diagnosis: {exc}") from exc
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><title>Asterion regression laboratory</title><style>body{{margin:0;background:#101827;color:#e7edf6;font:16px/1.6 system-ui}}main{{max-width:1150px;margin:auto;padding:32px}}h1{{font-size:36px;margin-bottom:8px}}small{{color:#83dfca}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}}article,details{{background:#1b283b;padding:20px;border:1px solid #34445b;border-radius:12px;margin:16px 0}}h2{{font-size:19px;overflow-wrap:anywhere}}summary{{font-weight:650;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 ui-monospace,monospace}}code{{overflow-wrap:anywhere}}a{{color:#9ceada}}.table{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #34445b;vertical-align:top;max-width:300px;overflow-wrap:anywhere}}th{{color:#83dfca}}</style><main><small>ASTERION · EXTERNAL KERNEL EXPERIMENT</small><h1>{escape(spec["case_id"])}</h1><p>Measured counterfactuals for explicitly trusted Triton source. Numerical admission covers the declared finite fixtures. Compiler metadata is not measured hardware occupancy or bandwidth.</p><section class="cards">{card_html}</section>{action}<h2>Executed variants</h2>{table}{body}<details><summary>Reproduce and interpret</summary><p>Use <code>asterion reproduce results.json --trusted-code --output replay</code> in a compatible trusted GPU environment. Exported Python is executable code. A numerically rejected configuration is never recommended. An inconclusive comparison does not establish equivalence. Inspect raw blocks in results.json.</p></details><small>Evidence identity {escape(result["result_id"])}</small></main></html>"""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.html"
    path.write_text(page)
    return path
