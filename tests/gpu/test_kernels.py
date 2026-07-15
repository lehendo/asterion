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


