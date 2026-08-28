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


def test_exact_allclose_and_model_have_different_meanings():
    torch = pytest.importorskip("torch")
    actual = torch.tensor([[1.0000001192092896, 0.0]])
    expected = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    exact = check_output(
        actual, expected, {"kind": "exact", "justification": "single rounded copy"}
    )
    assert not exact["passed"] and exact["max_fp32_ulp_from_rounded_reference"] == 1
    close = check_output(
        actual,
        expected,
        {
            "kind": "allclose",
            "rtol": 1e-6,
            "atol": 0,
            "justification": "application relative error requirement",
        },
    )
    assert close["passed"]
    model = check_output(
        actual,
        {
            "value": expected,
            "absolute_product_sum": torch.ones_like(expected),
            "accumulation_length": 64,
        },
        {
            "kind": "matmul_forward_bound",
            "model_version": 2,
            "accumulation_bits": 24,
            "justification": "standard FP32 accumulation model",
        },
    )
    assert model["passed"] and "assumptions" in model["forward_model"]


