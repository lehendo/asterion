import pytest
from asterion.diagnosis import paired_effect as legacy_effect
from asterion.experiment_stats import paired_effect
from asterion.io import ValidationError
from asterion.numerics import check_output, forward_gamma


@pytest.mark.parametrize("candidate", [[2.0] * 20, [0.5] * 20, [1.0] * 20, [1.05, 1.5] * 10])
def test_external_statistics_preserve_existing_decisions(candidate):
    a = paired_effect([1.0] * 20, candidate, comparisons=3)
    b = legacy_effect([1.0] * 20, candidate, comparisons=3)
    for key in ("verdict", "regression_p", "improvement_p", "per_tail_alpha", "paired_ratios"):
        assert a[key] == b[key]


