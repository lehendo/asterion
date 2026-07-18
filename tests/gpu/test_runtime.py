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


