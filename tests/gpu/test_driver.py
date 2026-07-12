"""Native host launcher parity; no claim of native GPU compilation."""

import os

import pytest
from asterion.backend import (
    Kernel,
    binary_metadata,
    make_inputs,
    reference,
    require_gpu,
    target_manifest,
)
from asterion.driver import DriverModule
from asterion.io import ValidationError
from asterion.ir import Schedule, make_program
from asterion.optimizer import fit_portfolio
from asterion.runtime import Runtime
from asterion.validation import certify_portfolio

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("ASTERION_DRIVER_LIBRARY"),
        reason="optional driver library not configured",
    ),
]


@pytest.mark.parametrize(
    "operation,shape",
    [
        ("pointwise", (4097,)),
        ("sum", (17, 129)),
        ("max", (19, 1025)),
        ("softmax", (33, 4093)),
        ("gemm", (17, 31, 33)),
    ],
)
def test_native_matches_generated_binary_on_current_stream(operation, shape):
    torch, _ = require_gpu()
    p = make_program(operation, shape)
    bound = Kernel(p, Schedule()).prepare(*make_inputs(p, 73))
    bound.launch()
    module = DriverModule(bound.last_compiled, bound.kernel)
    native = module.bind(bound)
    producer = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        actual = native.launch()
        expected = reference(p, *bound.inputs)
    producer.wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-5)
    module.close()
    with pytest.raises(ValidationError, match="closed"):
        native.launch()


def test_graph_replay_and_borrowed_tensor_lifetime():
    torch, _ = require_gpu()
    p = make_program("softmax", (17, 129))
    bound = Kernel(p, Schedule()).prepare(*make_inputs(p, 74))
    bound.launch()
    module = DriverModule(bound.last_compiled, bound.kernel)
    native = module.bind(bound)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native.launch()  # create per-stream completion event outside capture
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        native.launch()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        native.output, torch.softmax(bound.inputs[0], 1), rtol=1e-4, atol=1e-5
    )
    del graph  # prepared invocation/module must outlive graph use
    module.close()


def test_native_rejects_rebound_storage():
    torch, _ = require_gpu()
    p = make_program("pointwise", (17,))
    bound = Kernel(p, Schedule()).prepare(*make_inputs(p, 75))
    bound.launch()
    module = DriverModule(bound.last_compiled, bound.kernel)
    native = module.bind(bound)
    bound.inputs[0].set_(torch.empty_like(bound.inputs[0]))
    with pytest.raises(ValidationError, match="storage changed"):
        native.launch()
    module.close()


def test_native_runtime_certificate_and_module_reuse():
    torch, _ = require_gpu()
    shapes = [(3, 17), (4, 33)]
    s = Schedule(row_tile=64)
    records = []
    for shape in shapes:
        p = make_program("softmax", shape)
        bound = Kernel(p, s).prepare(*make_inputs(p, 76))
        bound.launch()
        records.append(
            {
                "case_id": str(shape),
                "operation": "softmax",
                "shape": shape,
                "split": "train",
                "dtype": "float32",
                "layout": "contiguous",
                "weight": 1.0,
                "variant": s.id,
                "schedule": s.to_dict(),
                **binary_metadata(bound),
                "status": "ok",
                "correctness": "passed",
                "device_us": [1.0] * 5,
            }
        )
    trace = {
        "schema_version": 1,
        "artifact_type": "trace",
        "measurement_kind": "measured",
        "timing_unit": "us",
        "environment": {"gpu_runtime": target_manifest()},
        "records": records,
    }
    portfolio = certify_portfolio(fit_portfolio(trace, max_variants=1), shapes, seeds=(1729,))
    runtime = Runtime(portfolio, launch_backend="native")
    prepared = []
    for shape in shapes:
        x = make_inputs(make_program("softmax", shape), 77)[0]
        invocation = runtime.prepare(x)
        assert invocation.selection["launch_backend"] == "native"
        torch.testing.assert_close(invocation.launch(), torch.softmax(x, 1), rtol=1e-4, atol=1e-5)
        prepared.append(invocation)
    assert len(runtime.native_modules) == 1
    runtime.close()
    with pytest.raises(ValidationError, match="closed"):
        prepared[0].launch()
    with pytest.raises(ValidationError, match="closed"):
        runtime.prepare(x)


def test_unfused_native_program_is_explicitly_unsupported():
    p = make_program("pointwise", (17,))
    bound = Kernel(p, Schedule(fusion=False)).prepare(*make_inputs(p, 78))
    bound.launch()
    # Module may represent only the final stage; never expose it as the full DAG.
    module = DriverModule(bound.last_compiled, bound.kernel)
    with pytest.raises(ValidationError, match="single-kernel"):
        module.bind(bound)
    module.close()


def test_shared_module_concurrent_bound_arguments():
    from concurrent.futures import ThreadPoolExecutor

    torch, _ = require_gpu()
    schedule = Schedule(row_tile=64)
    bounds = []
    for shape in ((3, 17), (4, 33)):
        program = make_program("softmax", shape)
        bound = Kernel(program, schedule).prepare(*make_inputs(program, 79))
        bound.launch()
        bounds.append(bound)
    torch.cuda.synchronize()
    module = DriverModule(bounds[0].last_compiled, bounds[0].kernel)
    invocations = [module.bind(b) for b in bounds]

    def launch(invocation):
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(10):
                invocation.launch()
        stream.synchronize()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(launch, invocations))
    for bound in bounds:
        torch.testing.assert_close(
            bound.output, torch.softmax(bound.inputs[0], 1), rtol=1e-4, atol=1e-5
        )
    module.close()


