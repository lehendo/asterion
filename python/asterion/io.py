from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


class ValidationError(ValueError):
    """Invalid or unsupported public input."""


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidationError(f"expected finite JSON-compatible values: {exc}") from exc


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValidationError("artifact exceeds the supported 64 MiB document limit")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        try:
            with path.open() as stream:
                value = yaml.safe_load(stream)
        except (yaml.YAMLError, RecursionError) as exc:
            raise ValidationError(f"invalid YAML: {exc}") from exc
    else:
        try:
            value = json.loads(path.read_text())
        except RecursionError as exc:
            raise ValidationError("document nesting exceeds supported parser limits") from exc
    if not isinstance(value, dict):
        raise ValidationError("document must be an object")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def artifact_path(root, relative):
    """Resolve an untrusted artifact-relative filename inside its declared root."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValidationError("artifact path must remain inside its declared root")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValidationError("artifact symlink escapes its declared root")
    return result


def positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{field} must be a positive finite number")
    return float(value)


def integer(value, field, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{field} must be an integer >= {minimum}")
    return value


def exact_keys(obj, allowed, context):
    if not isinstance(obj, dict):
        raise ValidationError(f"{context} must be an object")
    unknown = set(obj) - set(allowed)
    if unknown:
        raise ValidationError(f"{context}: unknown fields {sorted(unknown)}")
