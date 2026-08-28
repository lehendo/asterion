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


def test_numerical_invalid_is_not_a_timing_negative():
    torch = pytest.importorskip("torch")
    value = check_output(
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
        {"kind": "exact", "justification": "must be finite exact zero"},
    )
    assert not value["passed"] and value["status"] == "NONFINITE_RESULT"


def test_forward_policy_rejects_large_residual_and_unbounded_model():
    torch = pytest.importorskip("torch")
    policy = {
        "kind": "matmul_forward_bound",
        "model_version": 2,
        "accumulation_bits": 24,
        "justification": "bounded normal arithmetic fixture",
    }
    expected = {
        "value": torch.tensor([1.0], dtype=torch.float64),
        "absolute_product_sum": torch.tensor([1.0], dtype=torch.float64),
        "accumulation_length": 64,
    }
    result = check_output(torch.tensor([1.01]), expected, policy)
    assert not result["passed"] and result["max_model_normalized_error"] > 1
    with pytest.raises(ValidationError, match="bounded forward"):
        forward_gamma(2**22)


def test_forward_v2_includes_final_rounding_of_accumulation_error():
    torch = pytest.importorskip("torch")
    policy = {
        "kind": "matmul_forward_bound",
        "model_version": 2,
        "accumulation_bits": 24,
        "justification": "Synthetic algebraic rounding-bound fixture, not a GPU correctness claim",
    }
    gamma, ref_gamma = forward_gamma(64), forward_gamma(64, 53)
    # A value .9998 within the accumulation envelope can round to FP16 1.
    # With a zero exact reference the final rounding still magnifies that error.
    scale = 0.9998 * (1 - ref_gamma) / (gamma + ref_gamma)
    reference = {
        "value": torch.tensor([0.0], dtype=torch.float64),
        "absolute_product_sum": torch.tensor([scale], dtype=torch.float64),
        "accumulation_length": 64,
    }
    result = check_output(torch.tensor([1.0], dtype=torch.float16), reference, policy)
    assert result["passed"] and result["forward_model"]["reference_gamma_2k"] > 0
    reference["value"] = reference["value"].float()
    with pytest.raises(ValidationError, match="FP64"):
        check_output(torch.tensor([1.0], dtype=torch.float16), reference, policy)
    policy.pop("model_version")
    with pytest.raises(ValidationError, match="explicit model_version"):
        check_output(torch.tensor([1.0], dtype=torch.float16), reference, policy)
