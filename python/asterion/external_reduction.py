"""Bounded storage reduction for declared external-kernel dimensions."""

import copy
import hashlib
import time
from pathlib import Path

from .external import complexity, specification
from .external_workflow import diagnose_external, export_external, load_case
from .io import ValidationError, atomic_json, digest, integer, positive, read


def _compiler_similarity(original, confirmation):
    records = {}
    for name in ("baseline", "candidate"):
        left = original.get("variants", {}).get(name, {}).get("compiler", {}).get("assembly", {})
        right = (
            confirmation.get("variants", {}).get(name, {}).get("compiler", {}).get("assembly", {})
        )
        records[name] = {
            kind: left[kind]["sha256"] == right[kind]["sha256"]
            for kind in sorted(set(left) & set(right))
        }
    return {
        "exact_artifact_hash_equality": records,
        "interpretation": "Equality is byte identity, not hardware mechanism preservation; different shapes may change specialization, residency and execution regime",
    }


def reduce_external(
    value,
    output,
    *,
    trusted_code=False,
    max_trials=6,
    budget_seconds=120,
    source_root=None,
    runner=None,
):
    if trusted_code is not True:
        raise ValidationError(
            "external reduction executes reviewed Python and requires --trusted-code"
        )
    integer(max_trials, "max_trials")
    positive(budget_seconds, "budget_seconds")
    if max_trials > 16 or not 1 <= budget_seconds <= 1800:
        raise ValidationError(
            "external reduction supports at most16trials and1..1800budget seconds plus confirmation"
        )
    spec, source = load_case(value, source_root)
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValidationError("reduction output must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    execute = diagnose_external if runner is None else runner
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "artifact_type": "external_reduction",
        "status": "NO_CONFIRMED_WITNESS",
        "original_specification": spec,
        "original_complexity": complexity(spec),
        "trials": [],
        "max_trials": max_trials,
        "budget_seconds": budget_seconds,
        "budget_semantics": "original/proposal workers receive remaining-budget timeouts; fresh confirmation has a separate min(900,max(30,budget_seconds/2)) second allowance. Filesystem/report overhead is additional",
        "strategy": "deterministic dimension halving with minimum declared tensor storage first; no source rewriting or global minimum",
        "hardware_mechanism": "UNAVAILABLE",
        "generator_semantics": "deterministic inputs regenerated at each shape, not necessarily an exact subset of original values",
    }

    def save():
        report.pop("result_id", None)
        report["result_id"] = digest(report)
        atomic_json(root / "reduction.json", report)
        from .external_report import write_external_reduction_report

        write_external_reduction_report(report, root)

    try:
        original = execute(
            spec,
            root / "original",
            trusted_code=True,
            source_root=source,
            timeout_seconds=min(900, budget_seconds),
        )
    except ValidationError as exc:
        report.update(
            status="ORIGINAL_EXECUTION_FAILURE",
            reason=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )
        save()
        return report
    if original["status"] != "REGRESSION_ESTABLISHED":
        report.update(
            reason="Original regression not established or numerical admission failed",
            original_status=original["status"],
            elapsed_seconds=time.perf_counter() - started,
        )
        save()
        return report
    required = original["recommendations"][0] if original["recommendations"] else None
    report["preserved_intervention"] = required
    current = spec
    accepted = 0
    while len(report["trials"]) < max_trials and time.perf_counter() - started < budget_seconds:
        candidates = []
        for dim, size in sorted(current["dimensions"].items()):
            if size <= 1:
                continue
            candidate = copy.deepcopy(current)
            candidate["dimensions"][dim] = size // 2
            try:
                candidate = specification(candidate)
                candidates.append((complexity(candidate)["value"], dim, candidate, None))
            except ValidationError as exc:
                candidates.append((float("inf"), dim, candidate, str(exc)))
        candidates.sort(key=lambda c: (c[0], c[1]))
        changed = False
        for size, dim, candidate, error in candidates:
            if (
                len(report["trials"]) >= max_trials
                or time.perf_counter() - started >= budget_seconds
            ):
                break
            index = len(report["trials"]) + 1
            trial = {
                "index": index,
                "dimension": dim,
                "dimensions": candidate["dimensions"],
                "accepted": False,
            }
            if error:
                trial.update(status="UNSUPPORTED_CONSTRAINT", reason=error)
            elif size >= complexity(current)["value"]:
                trial.update(status="NO_STORAGE_REDUCTION")
            else:
                try:
                    result = execute(
                        candidate,
                        root / f"trial-{index:02d}",
                        trusted_code=True,
                        source_root=source,
                        timeout_seconds=max(
                            1, min(900, budget_seconds - (time.perf_counter() - started))
                        ),
                    )
                    same_target = result["environment"] == original["environment"]
                    keep = (
                        result["status"] == "REGRESSION_ESTABLISHED"
                        and same_target
                        and (required is None or required in result["recommendations"])
                    )
                    trial.update(
                        status=result["status"],
                        accepted=keep,
                        complexity=complexity(candidate),
                        comparison=result.get("comparison"),
                        recommendations=result["recommendations"],
                        same_target=same_target,
                    )
                    if keep:
                        current = candidate
                        accepted += 1
                        changed = True
                except ValidationError as exc:
                    trial.update(status="EXECUTION_FAILURE", reason=str(exc))
            report["trials"].append(trial)
            save()
            if changed:
                break
        if not changed:
            break
    confirmation_spec = copy.deepcopy(current)
    confirmation_spec["measurement"]["seed"] = (current["measurement"]["seed"] + 100003) % 2**31
    report["confirmation_specification"] = confirmation_spec
    save()
    try:
        confirmation = execute(
            confirmation_spec,
            root / "confirmation",
            trusted_code=True,
            source_root=source,
            timeout_seconds=min(900, max(30, budget_seconds / 2)),
        )
    except ValidationError as exc:
        report.update(
            status="CONFIRMATION_FAILED",
            reason=str(exc),
            accepted_steps=accepted,
            reduced_specification=current,
            elapsed_seconds=time.perf_counter() - started,
        )
        save()
        return report
    confirmed = (
        confirmation["status"] == "REGRESSION_ESTABLISHED"
        and confirmation["environment"] == original["environment"]
        and (required is None or required in confirmation["recommendations"])
    )
    final_size = complexity(current)
    report.update(
        status="CONFIRMED_SMALLER_REPRODUCER"
        if confirmed and accepted
        else "CONFIRMED_UNCHANGED_CAPTURE"
        if confirmed
        else "CONFIRMATION_FAILED",
        accepted_steps=accepted,
        reduced_specification=confirmation_spec,
        reduced_complexity=final_size,
        storage_ratio=final_size["value"] / report["original_complexity"]["value"],
        properties={
            "REGRESSION_PRESERVED": confirmed,
            "INTERVENTION_EFFECT_PRESERVED": confirmed if required else "UNAVAILABLE",
            "COMPILER_ARTIFACT_SIMILARITY": _compiler_similarity(original, confirmation),
            "HARDWARE_MECHANISM_SUPPORTED": "UNAVAILABLE",
        },
        confirmation={
            "status": confirmation["status"],
            "comparison": confirmation.get("comparison"),
            "recommendations": confirmation["recommendations"],
        },
        stopping_reason="trial_or_time_limit"
        if len(report["trials"]) >= max_trials or time.perf_counter() - started >= budget_seconds
        else "no_accepted_neighbor",
        elapsed_seconds=time.perf_counter() - started,
    )
    report["export_status"] = "PENDING" if confirmed else "NOT_ATTEMPTED"
    save()
    if confirmed:
        try:
            if runner is None:
                export_external(root / "confirmation/results.json", root / "witness")
            else:
                export_external(confirmation_spec, root / "witness", source_root=source)
            report["export_status"] = "EXPORTED"
            save()
            atomic_json(root / "witness/reduction.json", report)
            atomic_json(root / "witness/expectation.json", report["confirmation"])
            manifest = read(root / "witness/manifest.json")
            for name in ("reduction.json", "expectation.json"):
                manifest["files"][name] = hashlib.sha256(
                    (root / "witness" / name).read_bytes()
                ).hexdigest()
            atomic_json(root / "witness/manifest.json", manifest)
        except (ValidationError, OSError) as exc:
            report.update(
                status="REPRODUCER_EXPORT_FAILED", export_status="FAILED", export_error=str(exc)
            )
            save()
    return report
