"""Source-only execution identity, including required distribution licensing."""

import hashlib
from pathlib import Path

from .source_identity import source_manifest


def execution_sources(root):
    result = source_manifest(root)
    license_path = Path(root) / "LICENSE"
    if license_path.exists():
        result["LICENSE"] = hashlib.sha256(license_path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))
