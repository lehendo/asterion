"""Passive contracts for a bounded, explicitly trusted external Triton adapter.

This module never imports the declared workload, torch or Triton. Parameter
legality is a user declaration, not an Asterion IR transformation proof.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path

from .io import ValidationError, artifact_path, digest, exact_keys, integer
from .numerics import numerical_policy

DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int32": 4}
OPTIONS = {"num_warps", "num_stages", "enable_fp_fusion"}
EXECUTION_FILES = (
    "external.py",
    "external_runtime.py",
    "external_report.py",
    "experiment_stats.py",
    "numerics.py",
    "io.py",
)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value):
        raise ValidationError("expected a bounded Python identifier")
    return value


def expression(value, dimensions, parameters, *, depth=0):
    if depth > 16:
        raise ValidationError("expression nesting exceeds16")
    if type(value) in {int, float} and math.isfinite(value):
        return value
    if not isinstance(value, dict) or len(value) != 1:
        raise ValidationError("expected a literal or one declared arithmetic expression")
    op, args = next(iter(value.items()))
    if op in {"dim", "param"}:
        table = dimensions if op == "dim" else parameters
        if args not in table or type(table[args]) not in {int, float}:
            raise ValidationError(f"unknown or nonnumeric {op}: {args}")
        return table[args]
    evaluate = lambda x: expression(x, dimensions, parameters, depth=depth + 1)
    if op == "pow2":
        n = evaluate(args)
        integer(n, "power-of-two operand")
        if n > 2**30:
            raise ValidationError("power-of-two operand too large")
        return 1 << (n - 1).bit_length()
    if op not in {"add", "mul", "ceildiv"} or not isinstance(args, list) or len(args) != 2:
        raise ValidationError("supported arithmetic: add, mul, ceildiv, pow2")
    a, b = map(evaluate, args)
    if op == "ceildiv":
        integer(a, "ceildiv numerator", minimum=0)
        integer(b, "ceildiv denominator")
        result = (a + b - 1) // b
    else:
        result = a + b if op == "add" else a * b
    if not math.isfinite(result) or abs(result) > 2**40:
        raise ValidationError("arithmetic expression exceeds supported bound")
    return result


def configuration(spec, variant):
    exact_keys(variant, {"entrypoint", "parameters"}, "variant")
    identifier(variant["entrypoint"])
    values = variant["parameters"]
    if not isinstance(values, dict) or set(values) != set(spec["parameters"]):
        raise ValidationError("configuration must supply every declared parameter")
    for name, value in values.items():
        domain = spec["parameters"][name]["values"]
        if not any(digest(value) == digest(v) for v in domain):
            raise ValidationError(f"{name} is outside its declared domain")
    for constraint in spec["constraints"]:
        exact_keys(constraint, {"left", "relation", "right"}, "constraint")
        a = expression(constraint["left"], spec["dimensions"], values)
        b = expression(constraint["right"], spec["dimensions"], values)
        relation = constraint["relation"]
        valid = (
            a == b
            if relation == "equal"
            else a >= b
            if relation == "at_least"
            else a <= b
            if relation == "at_most"
            else None
        )
        if relation == "multiple_of":
            integer(a, "multiple numerator")
            integer(b, "multiple denominator")
            valid = a % b == 0
        if valid is not True:
            raise ValidationError("configuration violates a declared constraint")
    grid = [expression(x, spec["dimensions"], values) for x in spec["grid"]]
    if any(type(x) is not int or not 1 <= x < 2**31 for x in grid) or math.prod(grid) >= 2**31:
        raise ValidationError("grid must contain bounded positive integer dimensions")
    return copy.deepcopy(variant)


def specification(value):
    exact_keys(
        value,
        {
            "schema_version",
            "artifact_type",
            "case_id",
            "module",
            "source_sha256",
            "dimensions",
            "tensors",
            "arguments",
            "parameters",
            "baseline",
            "candidate",
            "grid",
            "constraints",
            "reference",
            "numerical",
            "measurement",
            "origin",
        },
        "external case",
    )
    try:
        if value["schema_version"] != 1 or value["artifact_type"] != "triton_case":
            raise ValidationError("expected triton_case version1")
        s = copy.deepcopy(value)
        identifier(s["case_id"])
        module = Path(s["module"])
        if module.is_absolute() or ".." in module.parts or module.suffix != ".py":
            raise ValidationError("module must be an artifact-relative Python file")
        if module.parts[0] in {
            "_support",
            "_support.py",
            "reproduce.py",
            "case.json",
            "manifest.json",
            "worker-home",
            "SUPPORT-LICENSE.txt",
            "USAGE.txt",
        }:
            raise ValidationError("module collides with a reserved reproducer or worker path")
        if not re.fullmatch(r"[0-9a-f]{64}", s["source_sha256"]):
            raise ValidationError("source_sha256 must identify reviewed source")
        dimensions = s["dimensions"]
        if not isinstance(dimensions, dict) or not 1 <= len(dimensions) <= 8:
            raise ValidationError("declare1..8 dimensions")
        for name, n in dimensions.items():
            identifier(name)
            integer(n, name)
            if n > 2**30:
                raise ValidationError("dimension exceeds supported index bound")
        tensors = s["tensors"]
        if not isinstance(tensors, dict) or not 2 <= len(tensors) <= 8:
            raise ValidationError("declare2..8 nonaliasing contiguous tensors")
        allocated = 0
        outputs = []
        for name, tensor in tensors.items():
            identifier(name)
            exact_keys(tensor, {"shape", "dtype", "role", "generator"}, "tensor")
            if tensor["dtype"] not in DTYPES or tensor["role"] not in {"input", "output"}:
                raise ValidationError("unsupported tensor dtype or role")
            if tensor.get("generator", "normal") not in {"normal", "ones"}:
                raise ValidationError(
                    "supported deterministic generators are normal and constant ones"
                )
            if tensor["role"] == "output" and "generator" in tensor:
                raise ValidationError("only inputs declare generators")
            if not isinstance(tensor["shape"], list) or not 1 <= len(tensor["shape"]) <= 4:
                raise ValidationError("tensor rank must be1..4")
            shape = [expression(x, dimensions, {}) for x in tensor["shape"]]
            if any(type(n) is not int or n < 1 for n in shape) or math.prod(shape) >= 2**31:
                raise ValidationError("invalid tensor shape/index range")
            allocated += math.prod(shape) * DTYPES[tensor["dtype"]]
            if tensor["role"] == "output":
                outputs.append(name)
        if len(outputs) != 1 or allocated > 2 * 1024 * 1024 * 1024:
            raise ValidationError(
                "one output and at most2GiB declared tensor storage supported; reference workspace is additional"
            )
        params = s["parameters"]
        if not isinstance(params, dict) or len(params) > 12:
            raise ValidationError("at most12 exposed parameters supported")
        for name, parameter in params.items():
            identifier(name)
            exact_keys(parameter, {"kind", "values"}, "parameter")
            if parameter["kind"] not in {"constexpr", "option"}:
                raise ValidationError("parameter kind must be constexpr or option")
            vals = parameter["values"]
            if (
                not isinstance(vals, list)
                or not 1 <= len(vals) <= 16
                or len({digest(v) for v in vals}) != len(vals)
            ):
                raise ValidationError("parameter domain must contain1..16 unique JSON scalars")
            if any(
                type(v) not in {int, float, bool, str} or (isinstance(v, str) and len(v) > 128)
                for v in vals
            ):
                raise ValidationError("parameter domain contains unsupported values")
            if parameter["kind"] == "option":
                if name not in OPTIONS:
                    raise ValidationError("unsupported compiler option")
                if name == "num_warps" and any(
                    type(v) is not int or v not in {1, 2, 4, 8, 16, 32} for v in vals
                ):
                    raise ValidationError("num_warps must be an exposed power of two1..32")
                if name == "num_stages" and any(
                    type(v) is not int or not 1 <= v <= 8 for v in vals
                ):
                    raise ValidationError("num_stages must be1..8")
                if name == "enable_fp_fusion" and any(type(v) is not bool for v in vals):
                    raise ValidationError("enable_fp_fusion must be boolean")
        if not isinstance(s["arguments"], dict) or not 1 <= len(s["arguments"]) <= 32:
            raise ValidationError("declare bounded named kernel arguments")
        for name, binding in s["arguments"].items():
            identifier(name)
            if name in params:
                raise ValidationError("runtime argument collides with exposed parameter")
            if isinstance(binding, dict) and set(binding) == {"tensor"}:
                if binding["tensor"] not in tensors:
                    raise ValidationError("argument references unknown tensor")
            else:
                expression(binding, dimensions, {})
        if not isinstance(s["grid"], list) or not 1 <= len(s["grid"]) <= 3:
            raise ValidationError("grid rank must be1..3")
        s.setdefault("constraints", [])
        if not isinstance(s["constraints"], list) or len(s["constraints"]) > 32:
            raise ValidationError("at most32 declarative constraints supported")
        reference = s["reference"]
        exact_keys(reference, {"entrypoint", "description"}, "trusted reference")
        identifier(reference["entrypoint"])
        if (
            not isinstance(reference["description"], str)
            or not 1 <= len(reference["description"]) <= 2048
        ):
            raise ValidationError("describe the independent application reference")
        s["numerical"] = numerical_policy(s["numerical"])
        measurement = {
            "rounds": 20,
            "launches": 100,
            "warmup": 5,
            "seed": 1729,
            "threshold": 0.10,
            "interventions": "changed",
        }
        exact_keys(s.get("measurement", {}), set(measurement), "measurement")
        measurement.update(s.get("measurement", {}))
        for key, low, high in (
            ("rounds", 12, 100),
            ("launches", 1, 1000),
            ("warmup", 1, 100),
            ("seed", 0, 2**31 - 1),
        ):
            integer(measurement[key], key, minimum=low)
            if measurement[key] > high:
                raise ValidationError(f"{key} exceeds bounded experiment limit")
        if (
            type(measurement["threshold"]) not in {int, float}
            or not 0 < measurement["threshold"] <= 1
        ):
            raise ValidationError("threshold must be in(0,1]")
        if measurement["interventions"] not in {"changed", "none"}:
            raise ValidationError("interventions must be changed or none")
        s["measurement"] = measurement
        s.setdefault("origin", {})
        exact_keys(s["origin"], {"url", "incident", "adaptations", "license"}, "origin")
        if any(not isinstance(v, str) or len(v) > 4096 for v in s["origin"].values()):
            raise ValidationError("origin values must be bounded descriptive strings")
        s["baseline"] = configuration(s, s["baseline"])
        s["candidate"] = configuration(s, s["candidate"])
        return s
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValidationError(f"malformed external case: {exc}") from exc


def source_path(spec, root):
    path = artifact_path(root, spec["module"])
    if path.stat().st_size > 1024 * 1024:
        raise ValidationError("external module exceeds1MiB supported source bound")
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["source_sha256"]:
        raise ValidationError("reviewed module source hash mismatch")
    return path


def interventions(spec):
    plans = []
    for name, value in spec["baseline"]["parameters"].items():
        candidate_value = spec["candidate"]["parameters"][name]
        if digest(value) == digest(candidate_value):
            continue
        variant = copy.deepcopy(spec["candidate"])
        variant["parameters"][name] = value
        try:
            configuration(spec, variant)
            plans.append(
                {
                    "id": "undo_" + name,
                    "parameter": name,
                    "variant": variant,
                    "status": "declared_legal",
                }
            )
        except ValidationError as exc:
            plans.append(
                {
                    "id": "undo_" + name,
                    "parameter": name,
                    "status": "unsupported",
                    "reason": str(exc),
                }
            )
    return plans


def complexity(spec):
    total = 0
    for tensor in spec["tensors"].values():
        shape = [expression(x, spec["dimensions"], {}) for x in tensor["shape"]]
        total += math.prod(shape) * DTYPES[tensor["dtype"]]
    return {
        "measure": "declared_input_plus_output_bytes",
        "value": total,
        "excludes": "reference, validation, graph and compiler storage",
    }


def capabilities(kind):
    if kind == "triton_case":
        return {
            "execute": True,
            "validate": True,
            "benchmark": True,
            "declared_parameter_interventions": True,
            "compiler_provenance": True,
            "shape_reduction": True,
            "standalone_export": True,
            "arbitrary_source_rewriting": False,
            "compiler_version_comparison": False,
            "legality_guarantee": "user declaration plus empirical checks; no IR proof",
        }
    if kind == "regression_spec":
        return {
            "execute": True,
            "validate": True,
            "benchmark": True,
            "legality_guarantee": "bounded Asterion IR verifier",
            "compiler_version_comparison": False,
        }
    if kind == "trace":
        return {
            "execute": False,
            "benchmark": False,
            "standalone_export": False,
            "passive_analysis": True,
        }
    raise ValidationError("unsupported workload adapter")
