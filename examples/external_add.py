"""Independently written literal Triton example; no Asterion IR or code generator.

The small-block candidate is an injected launch-configuration example, not a
reported production incident. Import/execute only as explicitly trusted code.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + index, index < N, other=0)
    b = tl.load(Y + index, index < N, other=0)
    tl.store(Z + index, a + b, index < N)


@triton.jit
def wrong_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Z + index, tl.full((BLOCK,), 1.0, tl.float32), index < N)


def reference(inputs, dimensions):
    return (inputs["x"] + inputs["y"]).to(torch.float32)


def reference_float16(inputs, dimensions):
    return (inputs["x"] + inputs["y"]).half()


def reference_bfloat16(inputs, dimensions):
    return (inputs["x"] + inputs["y"]).bfloat16()
