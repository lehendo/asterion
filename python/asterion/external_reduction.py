"""Bounded storage reduction for declared external-kernel dimensions."""

import copy
import hashlib
import time
from pathlib import Path

from .external import complexity, specification
from .external_workflow import diagnose_external, export_external, load_case
from .io import ValidationError, atomic_json, digest, integer, positive, read


def _compiler_similarity(original, confirmation):
    records = {}
    for name in ("baseline", "candidate"):
        left = original.get("variants", {}).get(name, {}).get("compiler", {}).get("assembly", {})
        right = (
            confirmation.get("variants", {}).get(name, {}).get("compiler", {}).get("assembly", {})
        )
        records[name] = {
            kind: left[kind]["sha256"] == right[kind]["sha256"]
            for kind in sorted(set(left) & set(right))
        }
    return {
        "exact_artifact_hash_equality": records,
        "interpretation": "Equality is byte identity, not hardware mechanism preservation; different shapes may change specialization, residency and execution regime",
    }


