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


def test_extreme_finite_softmax(torch):
    p = make_program("softmax", (3, 257))
    x = torch.linspace(-1e4, 1e4, 771, device="cuda").reshape(3, 257)
    torch.testing.assert_close(
        Kernel(p, Schedule(rows=4))(x), torch.softmax(x, 1), rtol=1e-4, atol=1e-5
    )


def test_nondefault_stream_and_borrowed_lifetime(torch):
    p = make_program("pointwise", (4097,))
    kernel = Kernel(p, Schedule(fusion=False))
    producer = torch.cuda.current_stream()
    x = make_inputs(p, 42)[0]
    stream = torch.cuda.Stream()
    stream.wait_stream(producer)
    with torch.cuda.stream(stream):
        bound = kernel.prepare(x)
        actual = bound.launch()
        expected = reference(p, x)
    producer.wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_runtime_rejects_layout_dtype_and_grad(torch):
    p = make_program("pointwise", (17,))
    k = Kernel(p, Schedule())
    with pytest.raises(ValidationError):
        k(torch.randn(34, device="cuda")[::2])
    with pytest.raises(ValidationError):
        k(torch.randn(17, device="cuda", dtype=torch.float64))
    with pytest.raises(ValidationError, match="autograd"):
        k(torch.randn(17, device="cuda", requires_grad=True))


def test_generic_ir_expression_and_cache_reuse(torch):
    p = Program(
        "pointwise",
        (1027,),
        (
            Node("x", "input"),
            Node("squared", "mul", ("x", "x")),
            Node("out", "add", ("squared", "x")),
        ),
        "out",
    )
    x = make_inputs(p, 7)[0]
    for fusion in (False, True):
        k = Kernel(p, Schedule(fusion=fusion))
        torch.testing.assert_close(k(x), x * x + x)
        assert k.key == Kernel(p, Schedule(fusion=fusion)).key


@pytest.mark.parametrize(
    "operation,shapes",
    [
        ("pointwise", [(127,), (4097,)]),
        ("softmax", [(3, 127), (17, 1009)]),
        ("gemm", [(17, 31, 33), (32, 48, 65)]),
    ],
)
def test_dynamic_shapes_reuse_exact_compiled_binary(torch, operation, shapes):
    from asterion.backend import binary_metadata

    signatures = []
    keys = []
    for shape in shapes:
        program = make_program(operation, shape)
        kernel = Kernel(program, Schedule(row_tile=1024))
        inputs = make_inputs(program, 1729)
        bound = kernel.prepare(*inputs)
        output = bound.launch()
        torch.testing.assert_close(output, reference(program, *inputs), rtol=1e-4, atol=5e-5)
        signatures.append(binary_metadata(bound)["compiled_binary_hashes"])
        keys.append(kernel.key)
    assert keys[0] == keys[1]
    assert signatures[0] == signatures[1]
