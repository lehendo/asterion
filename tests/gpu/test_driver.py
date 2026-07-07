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


