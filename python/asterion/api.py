"""Small public API; GPU imports remain lazy for analyzer-only installations."""

from pathlib import Path

from .schema import WorkloadSpec


def optimize(
    spec: WorkloadSpec | str | Path,
    *,
    output: str | Path,
    max_variants: int = 4,
    tuning_budget_s: float = 300,
    objective: str = "robust",
):
    """Optimize training cases on an allocated server GPU; persist all evidence to output."""
    from .tuning import optimize as run

    return run(spec, output, tuning_budget_s, max_variants, objective)


def diagnose(spec, *, output: str | Path, trusted_code=False):
    """Diagnose native IR or explicitly trusted external Triton source."""
    from .io import ValidationError, read

    value = read(spec) if isinstance(spec, (str, Path)) else spec
    if not isinstance(value, dict):
        raise ValidationError("workload input must be a specification or saved diagnosis object")
    if value.get("artifact_type") in {"triton_case", "external_diagnosis"}:
        from .external_workflow import diagnose_external

        return diagnose_external(spec, output, trusted_code=trusted_code)
    from .diagnostic_gpu import diagnose as run

    return run(spec, output)


def reduce_regression(
    spec, *, output: str | Path, max_trials=6, budget_seconds=120, trusted_code=False
):
    """Try bounded shape reductions and require fresh confirmation before export."""
    from .io import ValidationError, read

    value = read(spec) if isinstance(spec, (str, Path)) else spec
    if not isinstance(value, dict):
        raise ValidationError("workload input must be a specification or saved diagnosis object")
    if value.get("artifact_type") in {"triton_case", "external_diagnosis"}:
        from .external_reduction import reduce_external

        return reduce_external(
            spec,
            output,
            max_trials=max_trials,
            budget_seconds=budget_seconds,
            trusted_code=trusted_code,
        )
    from .reduction import reduce_regression as run

    return run(spec, output, max_trials=max_trials, budget_seconds=budget_seconds)


def export_reproducer(spec, *, output: str | Path):
    """CPU-only native IR regeneration or passive reviewed-source bundle export."""
    from .io import ValidationError, read

    value = read(spec) if isinstance(spec, (str, Path)) else spec
    if not isinstance(value, dict):
        raise ValidationError("workload input must be a specification or saved diagnosis object")
    if value.get("artifact_type") in {"triton_case", "external_diagnosis"}:
        from .external_workflow import export_external

        return export_external(spec, output)
    from .reproducer import export_reproducer as run

    return run(value, output)
