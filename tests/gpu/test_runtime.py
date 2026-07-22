import copy

import pytest
from asterion.backend import Kernel, binary_metadata, make_inputs, require_gpu, target_manifest
from asterion.gpu import measure
from asterion.ir import Schedule, make_program
from asterion.optimizer import fit_portfolio
from asterion.runtime import Runtime

pytestmark = pytest.mark.gpu


def test_real_measurement_portfolio_dispatch_and_safe_fallback(monkeypatch):
    torch, _ = require_gpu()
    rows = []
    for i, shape in enumerate(((3, 17), (4, 33))):
        program = make_program("softmax", shape)
        schedule = Schedule(row_tile=64)
        kernel = Kernel(program, schedule)
        inputs = make_inputs(program, 1729)
        bound = kernel.prepare(*inputs)
        actual = bound.launch()
        torch.testing.assert_close(actual, torch.softmax(inputs[0], dim=1), rtol=1e-4, atol=1e-5)
        rows.append(
            {
                "case_id": str(i),
                "operation": "softmax",
                "shape": list(shape),
                "split": "train",
                "dtype": "float32",
                "layout": "contiguous",
                "weight": 1.0,
                "variant": schedule.id,
                "schedule": schedule.to_dict(),
                **binary_metadata(bound),
                "status": "ok",
                "correctness": "passed",
                **measure(bound.launch, samples=5, warmup=2, launches=5),
            }
        )
    trace = {
        "schema_version": 1,
        "artifact_type": "trace",
        "measurement_kind": "measured",
        "timing_unit": "us",
        "environment": {"gpu_runtime": target_manifest()},
        "records": rows,
    }
    portfolio = fit_portfolio(trace, max_variants=1)
    from asterion.validation import certify_portfolio

    portfolio = certify_portfolio(portfolio, [(3, 17), (4, 33), (5, 23)], seeds=(1729,))
    runtime = Runtime(portfolio)
    x = make_inputs(make_program("softmax", (3, 17)), 7)[0]
    torch.testing.assert_close(runtime(x), torch.softmax(x, dim=1), rtol=1e-4, atol=1e-5)
    assert runtime.last_selection["reason"] == "metadata_rule"
    # An unseen shape within the feature range must still fall back until numerically validated.
    unseen = make_inputs(make_program("softmax", (5, 23)), 8)[0]
    torch.testing.assert_close(runtime(unseen), torch.softmax(unseen, dim=1), rtol=1e-4, atol=1e-5)
    assert runtime.last_selection["reason"] == "metadata_rule"
    other_target = copy.deepcopy(target_manifest())
    other_target["name"] = "different device"
    monkeypatch.setattr("asterion.backend.target_manifest", lambda: other_target)
    wrong = Runtime(portfolio)
    torch.testing.assert_close(wrong(x), torch.softmax(x, dim=1), rtol=1e-4, atol=1e-5)
    assert wrong.last_selection["reason"] == "target_mismatch"


def test_prepared_metadata_mutation_is_rejected():
    from asterion.io import ValidationError

    _torch, _ = require_gpu()
    p = make_program("pointwise", (17,))
    x = make_inputs(p, 1729)[0]
    bound = Kernel(p, Schedule()).prepare(x)
    x.resize_(1)
    with pytest.raises(ValidationError, match="metadata changed"):
        bound.launch()


def test_generic_row_reduction_masks_actual_expression():
    from asterion.ir import Node, Program

    torch, _ = require_gpu()
    x = -torch.arange(1, 18, device="cuda", dtype=torch.float32).reshape(1, 17)
    p = Program("sum", (1, 17), (Node("x", "input"), Node("out", "max", ("x",))), "out")
    torch.testing.assert_close(Kernel(p, Schedule())(x), x.amax(dim=1))
    p = Program(
        "sum",
        (1, 17),
        (Node("x", "input"), Node("e", "exp", ("x",)), Node("out", "sum", ("e",))),
        "out",
    )
    torch.testing.assert_close(Kernel(p, Schedule())(x), torch.exp(x).sum(dim=1))


@pytest.mark.parametrize("mutation", ["storage", "autograd"])
def test_prepared_inference_contract_cannot_change(mutation):
    from asterion.io import ValidationError

    torch, _ = require_gpu()
    p = make_program("pointwise", (17,))
    bound = Kernel(p, Schedule()).prepare(*make_inputs(p, 83))
    if mutation == "storage":
        bound.inputs[0].set_(torch.empty_like(bound.inputs[0]))
    else:
        bound.inputs[0].requires_grad_(True)
    with pytest.raises(ValidationError, match=mutation):
        bound.launch()
