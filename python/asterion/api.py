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


