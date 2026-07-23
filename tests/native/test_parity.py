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


