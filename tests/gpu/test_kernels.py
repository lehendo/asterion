import pytest
from asterion.backend import Kernel, make_inputs, reference, require_gpu
from asterion.io import ValidationError
from asterion.ir import Node, Program, Schedule, make_program

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def torch():
    torch, _ = require_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


CASES = [
    ("pointwise", (1,)),
    ("pointwise", (257,)),
    ("pointwise", (65539,)),
    ("sum", (1, 1)),
    ("sum", (17, 129)),
    ("sum", (33, 4093)),
    ("max", (1, 1)),
    ("max", (17, 127)),
    ("max", (9, 1025)),
    ("softmax", (1, 1)),
    ("softmax", (17, 127)),
    ("softmax", (33, 1025)),
    ("softmax", (7, 8191)),
    ("gemm", (1, 1, 1)),
    ("gemm", (17, 31, 33)),
    ("gemm", (65, 63, 129)),
]


@pytest.mark.parametrize("op,shape", CASES)
@pytest.mark.parametrize("choice", [0, 1])
def test_generated_schedules_against_torch_and_float64(torch, op, shape, choice):
    p = make_program(op, shape)
    s = (
        Schedule()
        if not choice
        else Schedule(
            block=1024,
            rows=4,
            tile_m=16,
            tile_n=32,
            tile_k=16,
            unroll=2 if op == "gemm" else 1,
            fusion=op != "pointwise",
        )
    )
    kernel = Kernel(p, s)
    xs = make_inputs(p, 1729)
    out = kernel(*xs)
    if op == "softmax":
        expected = torch.softmax(xs[0], dim=1)
    else:
        expected = reference(p, *xs)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=5e-5)
    # Independent higher-precision numerical reference, rather than matching one implementation.
    high = reference(p, *(x.double() for x in xs)).float()
    torch.testing.assert_close(out, high, rtol=2e-4, atol=5e-5)
    for _ in range(3):
        torch.testing.assert_close(kernel(*xs), expected, rtol=1e-4, atol=5e-5)


