"""Optional pure host C++20 semantics, loaded via versioned ownership-free C ABI."""

import ctypes as c
import os

from .io import ValidationError
from .ir import verify_schedule_fields


class NativeCore:
    def __init__(self, path=None):
        path = path or os.environ.get("ASTERION_NATIVE_LIBRARY")
        if not path:
            raise ValidationError("set ASTERION_NATIVE_LIBRARY to the built host library")
        self.lib = c.CDLL(path)
        self.lib.asterion_abi_version.restype = c.c_uint32
        if self.lib.asterion_abi_version() != 2:
            raise ValidationError("incompatible native ABI")
        self.lib.asterion_verify_schedule.argtypes = [
            c.c_uint32,
            c.POINTER(c.c_uint64),
            c.c_size_t,
            c.POINTER(c.c_uint64),
            c.c_size_t,
        ]
        self.lib.asterion_cvar.argtypes = [
            c.POINTER(c.c_double),
            c.POINTER(c.c_double),
            c.c_size_t,
            c.c_double,
            c.POINTER(c.c_double),
        ]
        self.lib.asterion_objective.argtypes = [
            c.POINTER(c.c_double),
            c.POINTER(c.c_double),
            c.c_size_t,
            c.c_double,
            c.c_double,
            c.POINTER(c.c_double),
        ]

    def schedule_is_legal(self, operation, shape, schedule):
        kind = {"pointwise": 0, "sum": 1, "max": 2, "softmax": 3, "gemm": 4}.get(operation)
        if kind is None or any(type(x) is not int or x < 0 or x > 2**64 - 1 for x in shape):
            return False
        # Struct packing cannot preserve Python bool-vs-int; validate field types before ABI conversion.
        try:
            verify_schedule_fields(schedule)
        except ValidationError:
            return False
        values = [
            schedule.block,
            schedule.warps,
            schedule.rows,
            schedule.tile_m,
            schedule.tile_n,
            schedule.tile_k,
            schedule.unroll,
            int(schedule.fusion),
            schedule.schema_version,
            schedule.row_tile,
        ]
        return (
            self.lib.asterion_verify_schedule(
                kind, (c.c_uint64 * len(shape))(*shape), len(shape), (c.c_uint64 * 10)(*values), 10
            )
            == 0
        )

    def cvar(self, values, weights, q=0.95):
        if len(values) != len(weights) or not values:
            raise ValidationError("invalid native CVaR arrays")
        out = c.c_double()
        status = self.lib.asterion_cvar(
            (c.c_double * len(values))(*values),
            (c.c_double * len(weights))(*weights),
            len(values),
            q,
            c.byref(out),
        )
        if status:
            raise ValidationError("native CVaR rejected invalid input")
        return out.value

    def objective(self, ratios, weights, risk=0.5, q=0.9):
        if len(ratios) != len(weights) or not ratios:
            raise ValidationError("invalid native objective arrays")
        out = c.c_double()
        status = self.lib.asterion_objective(
            (c.c_double * len(ratios))(*ratios),
            (c.c_double * len(weights))(*weights),
            len(ratios),
            risk,
            q,
            c.byref(out),
        )
        if status:
            raise ValidationError("native objective rejected invalid input")
        return out.value
