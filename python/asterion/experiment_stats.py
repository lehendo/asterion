"""Reusable bounded paired statistics for native and external experiments."""

import math
import statistics

from .io import ValidationError, integer, positive


def paired_effect(baseline, candidate, *, threshold=0.10, comparisons=1):
    if (
        not isinstance(baseline, list)
        or not isinstance(candidate, list)
        or len(baseline) != len(candidate)
        or not 12 <= len(baseline) <= 100
    ):
        raise ValidationError("paired comparison needs12..100 aligned blocks")
    positive(threshold, "threshold")
    integer(comparisons, "comparisons")
    ratios = [
        positive(b, "candidate timing") / positive(a, "baseline timing")
        for a, b in zip(baseline, candidate)
    ]
    n = len(ratios)
    tail = lambda wins: sum(math.comb(n, i) for i in range(wins, n + 1)) / 2**n
    worse = tail(sum(r > 1 + threshold for r in ratios))
    better = tail(sum(r < 1 / (1 + threshold) for r in ratios))
    alpha = 0.01 / (2 * comparisons)
    return {
        "category": "MEASURED",
        "verdict": "regression"
        if worse <= alpha
        else "improvement"
        if better <= alpha
        else "inconclusive",
        "median_paired_ratio": statistics.median(ratios),
        "paired_ratios": ratios,
        "baseline_median_us": statistics.median(baseline),
        "candidate_median_us": statistics.median(candidate),
        "regression_p": worse,
        "improvement_p": better,
        "per_tail_alpha": alpha,
        "practical_threshold": threshold,
        "blocks": n,
        "method": "exact paired sign exceedance tests; two directions and declared comparison count Bonferroni guarded",
        "limitations": "Independent stationary blocks assumed; inconclusive is not equivalence; adaptive selection not globally controlled",
    }
