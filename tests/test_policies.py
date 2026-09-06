import pytest
from asterion.io import ValidationError
from asterion.policies import fit_baseline
from asterion.runtime import validate_portfolio
from test_optimizer import matrix


@pytest.mark.parametrize(
    "kind",
    [
        "fixed",
        "random_search_single",
        "representative",
        "nearest_bucket",
        "mean_portfolio",
        "robust_portfolio",
    ],
)
def test_comparison_policies_are_valid_and_bounded(kind):
    p = fit_baseline(matrix(), kind, max_variants=2)
    assert validate_portfolio(p)["variants"] <= 2
    assert p["generated_code_bytes"] <= p["max_code_bytes"]


@pytest.mark.parametrize(
    "kind", ["fixed", "representative", "nearest_bucket", "random_search_single"]
)
def test_comparison_policies_reject_test_data(kind):
    t = matrix()
    t["records"][0]["split"] = "test"
    with pytest.raises(ValidationError, match="held-out"):
        fit_baseline(t, kind)


def test_policy_fitting_is_deterministic():
    assert fit_baseline(matrix(), "nearest_bucket") == fit_baseline(matrix(), "nearest_bucket")
