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


def schedule_analysis(program, schedule):
    """Executable lowering structure; traffic is an ideal model, not counters."""
    p = canonicalize(program)
    verify_schedule(p, schedule)
    s = schedule
    if p.operation == "pointwise":
        ranks = verify(p)
        stages = [n for n in p.nodes if n.op != "input" and ranks[n.name] == "tensor"]
        launches = 1 if s.fusion else max(1, len(stages))
        padded = math.ceil(p.shape[0] / s.block) * s.block
        # Fused: input read and output write. Unfused: count tensor dependencies.
        tensor = {n.name for n in p.nodes if ranks[n.name] == "tensor"}
        transfers = (
            2
            if s.fusion or not stages
            else sum(1 + len({a for a in n.args if a in tensor}) for n in stages)
        )
        traffic = 4 * p.shape[0] * transfers
        useful = p.shape[0]
    elif p.operation == "gemm":
        m, n, k = p.shape
        launches = 1
        padded = (
            math.ceil(m / s.tile_m)
            * s.tile_m
            * math.ceil(n / s.tile_n)
            * s.tile_n
            * math.ceil(k / (s.tile_k * s.unroll))
            * s.tile_k
            * s.unroll
        )
        useful = m * n * k
        traffic = 4 * (m * k + k * n + m * n)
    else:
        rows, width = p.shape
        tile = s.row_tile or 1 << (width - 1).bit_length()
        launches = 1
        padded = math.ceil(rows / s.rows) * s.rows * tile
        useful = rows * width
        traffic = 4 * (rows * width + (rows * width if p.operation == "softmax" else rows))
    return {
        "category": "MODEL_ESTIMATE",
        "launch_count": launches,
        "padded_work_units": padded,
        "useful_work_units": useful,
        "padding_ratio": padded / useful,
        "ideal_global_bytes": traffic,
        "limitations": "Lowering-derived counts; ignores caches, spills, compiler elimination and hardware occupancy. GEMM traffic is a compulsory lower bound.",
    }


def interventions(spec):
    """Undo each effective changed field independently; retain full revert control."""
    spec = specification(spec)
    p = Program.from_dict(spec["program"])
    a, b = (Schedule.from_dict(spec[k]) for k in ("baseline", "candidate"))
    source = generate(p, b)
    results = []
    for field in FIELDS:
        if getattr(a, field) == getattr(b, field):
            continue
        try:
            s, trail = transform(p, b, **{field: getattr(a, field)})
            effective = generate(p, s) != source or s.warps != b.warps
            results.append(
                {
                    "id": "undo_" + field,
                    "field": field,
                    "schedule": s.to_dict(),
                    "trail": trail,
                    "status": "executable" if effective else "inert",
                }
            )
        except ValidationError as exc:
            results.append(
                {"id": "undo_" + field, "field": field, "status": "illegal", "reason": str(exc)}
            )
    return results


def paired_effect(baseline, candidate, *, threshold=0.10, comparisons=1):
    """Exact one-sided sign tests on randomized blocks, with multiplicity guard.

    Tests exceedance of a practical effect, not equality of means. Ties count
    against a claim. Independence/stationarity remain experimental assumptions.
    """
    if (
        not isinstance(baseline, list)
        or not isinstance(candidate, list)
        or len(baseline) != len(candidate)
        or not 12 <= len(baseline) <= 100
    ):
        raise ValidationError("paired comparison needs 12..100 aligned blocks")
    positive(threshold, "threshold")
    integer(comparisons, "comparisons")
    ratios = [
        positive(b, "candidate timing") / positive(a, "baseline timing")
        for a, b in zip(baseline, candidate)
    ]
    n = len(ratios)

    def tail(wins):
        return sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n

    worse = tail(sum(r > 1 + threshold for r in ratios))
    better = tail(sum(r < 1 / (1 + threshold) for r in ratios))
    alpha = 0.01 / (2 * comparisons)
    verdict = (
        "regression" if worse <= alpha else "improvement" if better <= alpha else "inconclusive"
    )
    return {
        "category": "MEASURED",
        "verdict": verdict,
        "median_paired_ratio": statistics.median(ratios),
        "paired_ratios": ratios,
        "baseline_median_us": statistics.median(baseline),
        "candidate_median_us": statistics.median(candidate),
        "regression_p": worse,
        "improvement_p": better,
        "per_tail_alpha": alpha,
        "practical_threshold": threshold,
        "blocks": n,
        "method": "one-sided exact sign tests, two directions and declared comparison count Bonferroni guarded",
        "limitations": "Assumes independent stationary blocks; inconclusive does not mean equivalent. Controls declared comparisons, not adaptive experiment selection.",
    }


def selected_plans(spec):
    spec = specification(spec)
    if spec["intervention_policy"] == "none":
        return []
    statuses = (
        {"executable", "inert"} if spec["intervention_policy"] == "all_changed" else {"executable"}
    )
    return [p for p in interventions(spec) if p["status"] in statuses]


def assemble(spec, environment, variants, blocks, *, boundary="device_graph_us"):
    spec = specification(spec)
    plans = interventions(spec)
    usable = selected_plans(spec)
    names = ["baseline", "candidate"] + [x["id"] for x in usable]
    if any(name not in variants for name in names):
        raise ValidationError("missing experiment variant")
    p = Program.from_dict(spec["program"])
    expected_schedules = {"baseline": spec["baseline"], "candidate": spec["candidate"]}
    expected_schedules.update({x["id"]: x["schedule"] for x in usable})
    try:
        for name in names:
            variant = variants[name]
            certificate = dict(variant["certificate"])
            identity = certificate.pop("certificate_id")
            if (
                variant["status"] != "passed"
                or certificate["status"] != "passed"
                or identity != digest(certificate)
                or certificate["program_hash"] != p.hash
                or variant["program_hash"] != p.hash
                or certificate["schedule"] != expected_schedules[name]
                or variant["schedule"] != expected_schedules[name]
                or certificate["target"] != environment["gpu_runtime"]
                or certificate["kernel_key"] != variant["kernel_key"]
                or certificate["compiled_binary_hashes"] != variant["compiled_binary_hashes"]
            ):
                raise ValidationError("variant numerical evidence/identity mismatch")
            expected_checks = {
                (pattern, seed)
                for pattern in ("normal", "zeros", "alternating", "wide_finite")
                for seed in (spec["seed"], spec["seed"] + 1)
            }
            observed_checks = {
                (check["pattern"], check["seed"])
                for check in certificate["checks"]
                if check["passed"] is True
            }
            if observed_checks != expected_checks or len(certificate["checks"]) != len(
                expected_checks
            ):
                raise ValidationError("variant numerical suite is incomplete")
        if len(blocks) != spec["rounds"]:
            raise ValidationError("incomplete randomized block count")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValidationError(f"missing numerical evidence: {exc}") from exc
    for block in blocks:
        if set(block["order"]) != set(names) or len(block["order"]) != len(names):
            raise ValidationError("each randomized block must contain every variant once")

    def samples(name):
        return [b["measurements"][name][boundary] for b in blocks]

    main = paired_effect(
        samples("baseline"),
        samples("candidate"),
        threshold=spec["threshold"],
        comparisons=1 + len(usable),
    )
    findings = []
    candidate_model = schedule_analysis(p, Schedule.from_dict(spec["candidate"]))
    for plan in usable:
        effect = paired_effect(
            samples("candidate"),
            samples(plan["id"]),
            threshold=spec["threshold"],
            comparisons=1 + len(usable),
        )
        supported = main["verdict"] == "regression" and effect["verdict"] == "improvement"
        after_model = schedule_analysis(p, Schedule.from_dict(plan["schedule"]))
        changes = {
            key: {"before": candidate_model[key], "after": after_model[key]}
            for key in ("launch_count", "padding_ratio", "ideal_global_bytes")
            if candidate_model[key] != after_model[key]
        }
        mechanism = "The schedule can change downstream register allocation, instruction selection or mapping; the physical mechanism is unresolved."
        if after_model["launch_count"] < candidate_model["launch_count"]:
            mechanism = "Fewer generated launches and modeled intermediate global-memory transfers may contribute; their separate contributions have not been isolated."
        elif after_model["padding_ratio"] < candidate_model["padding_ratio"]:
            mechanism = "Reduced padded work may contribute; tile-dependent register use and downstream compilation remain alternative explanations."
        findings.append(
            {
                "field": plan["field"],
                "intervention": plan,
                "result": effect,
                "category": "CONTROLLED_ABLATION" if supported else "HYPOTHESIS",
                "supported": supported,
                "hypothesis": f"The change to {plan['field']} contributes to the observed regression under this fixed workload.",
                "evidence_quality": "repeated controlled schedule effect"
                if supported
                else "unresolved",
                "mechanism_hypothesis": {"category": "HYPOTHESIS", "statement": mechanism},
                "lowering_model_changes": {"category": "MODEL_ESTIMATE", "changes": changes},
                "supporting_evidence": [
                    "Correctness-gated one-field reversion",
                    "Randomized paired block timings",
                ]
                if supported
                else [],
                "contradicting_evidence": []
                if supported
                else [
                    "No joint evidence of baseline regression and helpful reversion at the declared threshold"
                ],
                "explanation": f"Reverting {plan['field']} {'improved' if supported else 'did not establish improvement in'} this fixed workload and execution boundary.",
                "alternative_explanations": [
                    "Compiler changes induced by the schedule parameter",
                    "Device interference or time-varying clocks",
                    "Interactions with other schedule parameters",
                    "Distinct output/intermediate buffer addresses and cache state",
                ],
                "hardware_cause": "UNAVAILABLE: no privileged counters; no occupancy or bandwidth causal claim",
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "diagnosis",
        "measurement_kind": "measured",
        "specification": spec,
        "workload_id": digest(spec["program"]),
        "environment": environment,
        "environment_id": digest(environment),
        "boundary": boundary,
        "comparison": main,
        "findings": findings,
        "intervention_plans": plans,
        "variants": variants,
        "blocks": blocks,
        "recommendations": [f["intervention"]["id"] for f in findings if f["supported"]],
        "scope": "Supported contiguous FP32 Asterion IR only; controlled schedule effects, not compiler commit bisection or hardware root-cause proof.",
    }


def reshape(spec, shape):
    result = dict(specification(spec))
    p = Program.from_dict(result["program"])
    result["program"] = replace(p, shape=tuple(shape)).to_dict()
    return specification(result)
