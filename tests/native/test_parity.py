import random

import pytest
from asterion.io import ValidationError
from asterion.ir import Schedule, make_program, verify_schedule
from asterion.native import NativeCore
from asterion.optimizer import objective
from asterion.stats import cvar


def test_fixed_seed_native_statistical_parity():
    core = NativeCore()
    rng = random.Random(1729)
    for count in range(1, 100):
        values = [rng.uniform(0.2, 20.0) for _ in range(count)]
        weights = [rng.uniform(0.01, 2.0) for _ in range(count)]
        for q in (0.0, 0.5, 0.9, 0.95):
            assert core.cvar(values, weights, q) == pytest.approx(
                cvar(values, weights, q), rel=1e-13
            )
            assert core.objective(values, weights, 0.5, q) == pytest.approx(
                objective(values, weights, 0.5, q), rel=1e-13, abs=1e-14
            )


def test_native_legality_matches_python():
    core = NativeCore()
    rng = random.Random(42)
    for _ in range(1000):
        op = rng.choice(["pointwise", "sum", "max", "softmax", "gemm"])
        shape = tuple(
            rng.randint(1, 40000)
            for _ in range(1 if op == "pointwise" else 3 if op == "gemm" else 2)
        )
        s = Schedule(
            block=rng.choice([0, 32, 256, 4096, 8192]),
            warps=rng.choice([1, 2, 3, 4, 8]),
            rows=rng.choice([1, 4, 8]),
            fusion=rng.choice([True, False]),
            row_tile=rng.choice([0, 128, 4096, 16384]),
        )
        try:
            verify_schedule(make_program(op, shape), s)
            valid = True
        except ValidationError:
            valid = False
        assert core.schedule_is_legal(op, shape, s) == valid


@pytest.mark.parametrize(
    "values,weights,q", [([1.0], [0.0], 0.95), ([float("inf")], [1.0], 0.95), ([1.0], [1.0], 1.0)]
)
def test_native_errors(values, weights, q):
    with pytest.raises(ValidationError):
        NativeCore().cvar(values, weights, q)
