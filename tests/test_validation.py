"""Synthetic certificates test metadata rules only; GPU tests produce real evidence."""

import copy

import pytest
from asterion.io import ValidationError
from asterion.ir import Schedule, make_program
from asterion.optimizer import fit_portfolio
from asterion.runtime import validate_portfolio
from asterion.validation import PATTERNS, certificate_index, kernel_key, seal, validate_certificate
from test_optimizer import matrix


def packet():
    p = fit_portfolio(matrix(), max_variants=1)
    v = next(iter(p["variants"]))
    shape = p["train_shapes"][0]
    program = make_program(p["operation"], shape)
    schedule = Schedule.from_dict(p["variants"][v])
    c = seal(
        {
            "schema_version": 1,
            "artifact_type": "validation_certificate",
            "measurement_kind": "measured",
            "status": "passed",
            "operation": p["operation"],
            "shape": shape,
            "dtype": "float32",
            "layout": "contiguous",
            "input_contract": "finite_float32_contiguous_aligned16",
            "variant": v,
            "schedule": schedule.to_dict(),
            "target": p["target"],
            "program_hash": program.hash,
            "kernel_key": kernel_key(program, schedule, p["target"]),
            "compiled_binary_hashes": ["a" * 64],
            "generated_code_bytes": 1234,
            "rtol": 1e-4,
            "atol": 1e-5,
            "seeds": [1729],
            "checks": [{"pattern": pattern, "seed": 1729, "passed": True} for pattern in PATTERNS],
        }
    )
    return p, v, c


