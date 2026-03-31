from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import analyze, compare
from .environment import capture
from .io import ValidationError, read
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="asterion",
        description="Controlled GPU regression experiments and saved performance evidence",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    for command in ("diagnose", "reproduce"):
        d = sub.add_parser(
            command, help="diagnose native IR or explicitly trusted declared Triton source"
        )
        d.add_argument("specification")
        d.add_argument("--output", required=True)
        d.add_argument(
            "--trusted-code",
            action="store_true",
            help="authorize execution of reviewed external Python; no security sandbox",
        )
        if command == "reproduce":
            d.add_argument(
                "--export-only",
                action="store_true",
                help="generate a standalone trusted bundle without executing GPU code",
            )
    red = sub.add_parser(
        "reduce",
        help="measure and confirm a smaller native IR or trusted Triton regression witness",
    )
    red.add_argument("specification")
    red.add_argument("--output", required=True)
    red.add_argument("--max-trials", type=int, default=6)
    red.add_argument("--budget-seconds", type=float, default=120)
    red.add_argument(
        "--trusted-code",
        action="store_true",
        help="authorize reviewed external Python reduction experiments",
    )
    a = sub.add_parser("analyze")
    a.add_argument("source")
    a.add_argument("--output", required=True)
    c = sub.add_parser("compare", help="compare saved performance traces or analyzer measurements")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--baseline-variant")
    c.add_argument("--candidate-variant")
    c.add_argument("--output", required=True)
    e = sub.add_parser("explain")
    e.add_argument("source")
    e.add_argument("--output", required=True)
    b = sub.add_parser("benchmark")
    b.add_argument("workload")
    b.add_argument("--output", required=True)
    b.add_argument("--compare", default="pytorch,fixed")
    b.add_argument("--portfolio")
    b.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    b.add_argument("--samples", type=int, default=30)
    b.add_argument("--warmup", type=int, default=10)
    o = sub.add_parser("optimize")
    o.add_argument("workload")
    o.add_argument("--output", required=True)
    o.add_argument("--budget-seconds", type=float, default=300)
    o.add_argument("--max-variants", type=int, default=4)
    o.add_argument("--objective", choices=["mean", "robust"], default="robust")
    v = sub.add_parser(
        "validate",
        help="passively validate a portfolio, native regression case or declared Triton case",
    )
    v.add_argument("source")
    cert = sub.add_parser(
        "certify", help="empirically validate explicit selected shapes, without performance tuning"
    )
    cert.add_argument("portfolio")
    cert.add_argument("workload")
    cert.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    cert.add_argument("--output", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("action", choices=["stats"])
    args = parser.parse_args(argv)
    try:
        if args.command in {"diagnose", "reproduce"}:
            if getattr(args, "export_only", False):
                from .api import export_reproducer

                print(export_reproducer(args.specification, output=args.output))
                return 0
            from .api import diagnose

            result = diagnose(
                args.specification, output=args.output, trusted_code=args.trusted_code
            )
            print(Path(args.output) / "report.html")
            if result["artifact_type"] == "external_diagnosis":
                print(result["status"])
                return 3 if result["status"] in {"NUMERICAL_FAILURE", "EXECUTION_FAILURE"} else 0
            if result["comparison"]["verdict"] == "inconclusive":
                print("asterion: no repeatable practical effect established", file=sys.stderr)
        elif args.command == "reduce":
            from .api import reduce_regression

            result = reduce_regression(
                args.specification,
                output=args.output,
                max_trials=args.max_trials,
                budget_seconds=args.budget_seconds,
                trusted_code=args.trusted_code,
            )
            print(result["status"])
            if result["status"] in {
                "unconfirmed",
                "CONFIRMATION_FAILED",
                "NO_CONFIRMED_WITNESS",
                "ORIGINAL_EXECUTION_FAILURE",
                "REPRODUCER_EXPORT_FAILED",
            }:
                return 3
        elif args.command == "doctor":
            print(json.dumps(capture(), indent=2))
        elif args.command == "analyze":
            print(write_report(analyze(args.source), args.output))
        elif args.command == "compare":
            data = compare(
                args.baseline, args.candidate, args.baseline_variant, args.candidate_variant
            )
            print(write_report(data, args.output))
        elif args.command == "explain":
            source = read(args.source)
            target = Path(args.output)
            if target.suffix == ".html":
                if target.name != "report.html":
                    raise ValidationError("HTML output filename must be report.html")
                target = target.parent
            if source.get("artifact_type") == "diagnosis":
                from .diagnostic_report import write_diagnosis

                print(write_diagnosis(source, target))
                return 0
            if source.get("artifact_type") == "external_diagnosis":
                from .external_report import write_external_report

                print(write_external_report(source, target))
                return 0
            if source.get("artifact_type") == "external_reduction":
                from .external_report import write_external_reduction_report

                print(write_external_reduction_report(source, target))
                return 0
            if source.get("artifact_type") == "trace":
                source = analyze(source)
            elif source.get("artifact_type") not in {"analysis", "comparison", "workload_analysis"}:
                raise ValidationError("explain requires a trace or analyzer result")
            print(write_report(source, target))
        elif args.command == "benchmark":
            from .gpu import benchmark

            trace = benchmark(
                args.workload,
                args.output,
                args.compare.split(","),
                args.split,
                args.samples,
                args.warmup,
                portfolio=args.portfolio,
            )
            if any(r["status"] != "ok" for r in trace["records"]):
                print(
                    "asterion: benchmark preserved failed cases; inspect trace.json",
                    file=sys.stderr,
                )
                return 3
        elif args.command == "optimize":
            from .tuning import optimize

            optimize(
                args.workload, args.output, args.budget_seconds, args.max_variants, args.objective
            )
        elif args.command == "validate":
            value = read(args.source)
            if value.get("artifact_type") in {"triton_case", "external_diagnosis"}:
                from .external import capabilities
                from .external_workflow import load_case

                spec, _ = load_case(args.source)
                print(
                    json.dumps(
                        {
                            "artifact_type": "triton_case",
                            "case_id": spec["case_id"],
                            "source_sha256": spec["source_sha256"],
                            "capabilities": capabilities("triton_case"),
                            "execution": "not performed; passive validation only",
                        },
                        indent=2,
                    )
                )
                return 0
            if value.get("artifact_type") == "regression_spec":
                from .diagnosis import specification

                print(json.dumps(specification(value), indent=2))
                return 0
            from .runtime import validate_portfolio

            print(json.dumps(validate_portfolio(value), indent=2))
        elif args.command == "certify":
            from .io import atomic_json
            from .schema import WorkloadSpec
            from .validation import certify_portfolio

            spec = WorkloadSpec.from_yaml(args.workload)
            portfolio = read(args.portfolio)
            if portfolio["operation"] != spec.operation:
                raise ValidationError("workload/portfolio operations differ")
            shapes = [c.shape for c in spec.cases if c.split == args.split]
            if not shapes:
                raise ValidationError("no shapes in requested certification split")
            result = certify_portfolio(portfolio, shapes, rtol=spec.rtol, atol=spec.atol)
            atomic_json(args.output, result)
            print(args.output)
            if any(a["status"] == "failed" for a in result["validation_attempts"]):
                return 3
        elif args.command == "cache":
            from .cache import Cache

            print(json.dumps(Cache.default().stats(), indent=2))
        return 0
    except (ValidationError, OSError, json.JSONDecodeError, ImportError) as exc:
        print(f"asterion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
