"""Optional bounded CUDA Driver launcher for trusted Triton-generated CUBINs.

This is native host execution, not native GPU compilation. The adapter is pinned
to Triton 3.3.1, validates its observed entry ABI and refuses scratch/cluster modes.
"""

from __future__ import annotations

import ctypes as c
import os
import re
import threading

from .io import ValidationError


def entry_contract(ptx, pointer_count, scalar_count):
    match = re.search(
        r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)\s*\.reqntid\s+(\d+),\s*1,\s*1", ptx, re.DOTALL
    )
    if not match:
        raise ValidationError("unsupported generated PTX entry contract")
    widths = tuple(int(x) for x in re.findall(r"\.param\s+\.u(32|64)\b", match[2]))
    expected = (64,) * pointer_count + (32,) * scalar_count + (64,)
    if widths != expected:
        raise ValidationError(f"unsupported kernel ABI: observed {widths}, expected {expected}")
    return match[1], int(match[3]), len(widths)


class DriverModule:
    def __init__(self, compiled, kernel, *, library=None):
        from .backend import require_gpu

        self.torch, triton = require_gpu()
        program = kernel.program
        self.kernel_key = kernel.key
        if triton.__version__ != "3.3.1":
            raise ValidationError("native driver adapter requires validated Triton 3.3.1 ABI")
        metadata = compiled.metadata
        if metadata.num_ctas != 1 or metadata.global_scratch_size or metadata.tmem_size:
            raise ValidationError(
                "native driver does not support clusters, global scratch or tensor memory"
            )
        pointers = 3 if program.operation == "gemm" else 2
        name, threads, argc = entry_contract(compiled.asm["ptx"], pointers, len(program.shape))
        library = library or os.environ.get("ASTERION_DRIVER_LIBRARY")
        if not library:
            raise ValidationError("set ASTERION_DRIVER_LIBRARY to the optional CUDA driver library")
        self.lib = c.CDLL(library)
        self.lib.asterion_driver_abi_version.restype = c.c_uint32
        if self.lib.asterion_driver_abi_version() != 1:
            raise ValidationError("native driver ABI mismatch")
        self.lib.asterion_driver_open.argtypes = [
            c.c_void_p,
            c.c_size_t,
            c.c_char_p,
            c.c_uint32,
            c.c_uint32,
            c.c_uint32,
            c.POINTER(c.c_void_p),
            c.c_void_p,
            c.c_size_t,
        ]
        self.lib.asterion_driver_launch.argtypes = [
            c.c_void_p,
            c.POINTER(c.c_uint64),
            c.c_size_t,
            c.c_uint32,
            c.c_uint32,
            c.c_void_p,
            c.c_void_p,
            c.c_size_t,
        ]
        self.lib.asterion_driver_close.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
        self.handle = c.c_void_p()
        self.error = c.create_string_buffer(2048)
        self.lock = threading.Lock()
        self.closed = False
        image = compiled.asm["cubin"]
        image_buffer = c.create_string_buffer(image)
        self._check(
            self.lib.asterion_driver_open(
                image_buffer,
                len(image),
                name.encode(),
                metadata.shared,
                threads,
                argc,
                c.byref(self.handle),
                self.error,
                len(self.error),
            )
        )

    def _check(self, status):
        if status:
            raise ValidationError(self.error.value.decode(errors="replace"))

    def bind(self, bound):
        if bound.kernel.key != self.kernel_key:
            raise ValidationError("native module and bound kernel identity differ")
        if bound.stages:
            raise ValidationError("native launcher currently supports single-kernel programs")
        return DriverInvocation(self, bound)

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self._check(
                    self.lib.asterion_driver_close(self.handle, self.error, len(self.error))
                )
                self.handle = c.c_void_p()

    def __del__(self):
        if getattr(self, "handle", None) and not getattr(self, "closed", True):
            try:
                self.close()
            except (ValidationError, OSError):
                # Interpreter/context teardown is not a supported explicit-close operation.
                return


class DriverInvocation:
    def __init__(self, module, bound):
        self.module, self.bound = module, bound
        self.output = bound.output
        self.pointers = tuple(x.data_ptr() for x in (*bound.inputs, bound.output))
        self.parameters = (
            c.c_uint64 * (len(bound.inputs) + 1 + len(bound.kernel.program.shape) + 1)
        )()

    def launch(self):
        module, bound = self.module, self.bound
        torch = module.torch
        p, s = bound.kernel.program, bound.kernel.schedule
        tensors = (*bound.inputs, bound.output)
        for tensor, shape in zip(tensors, (*bound.input_shapes, bound.output_shape)):
            if (
                tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.dtype != torch.float32
            ):
                raise ValidationError("prepared native tensor metadata changed")
            if tensor.device.index != torch.cuda.current_device():
                raise ValidationError("prepared native tensor is not on current CUDA device")
        if any(x.requires_grad for x in tensors):
            raise ValidationError("prepared native inference tensors now require autograd")
        stream = torch.cuda.current_stream()
        for tensor in tensors:
            tensor.record_stream(stream)
        pointers = tuple(x.data_ptr() for x in tensors)
        if pointers != self.pointers:
            raise ValidationError("prepared native tensor storage changed")
        values = pointers + tuple(p.shape) + (0,)
        if p.operation == "gemm":
            gx, gy = (
                (p.shape[0] + s.tile_m - 1) // s.tile_m,
                (p.shape[1] + s.tile_n - 1) // s.tile_n,
            )
        else:
            block = s.block if p.operation == "pointwise" else s.rows
            gx, gy = (p.shape[0] + block - 1) // block, 1
        with module.lock:
            if module.closed:
                raise ValidationError("native driver module is closed")
            for i, value in enumerate(values):
                self.parameters[i] = value
            module._check(
                module.lib.asterion_driver_launch(
                    module.handle,
                    self.parameters,
                    len(values),
                    gx,
                    gy,
                    c.c_void_p(stream.cuda_stream),
                    module.error,
                    len(module.error),
                )
            )
        return self.output
