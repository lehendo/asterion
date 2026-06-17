"""Deterministic source identity without experiment protocols or datasets."""

import hashlib
from pathlib import Path


def source_manifest(root, *, include_generated=False):
    root = Path(root)
    files = set()
    for base in (
        "python",
        "cpp",
        "scripts",
        "cluster",
        "tests",
        "benchmarks",
        "examples",
        "schemas",
        ".github",
    ):
        folder = root / base
        if folder.exists():
            files.update(
                p
                for p in folder.rglob("*")
                if p.is_file()
                and not any(part in {"__pycache__", ".pytest_cache"} for part in p.parts)
                and p.suffix not in {".md", ".MD", ".pyc"}
            )
    files.update(
        root / name
        for name in ("CMakeLists.txt", "pyproject.toml", ".gitignore")
        if (root / name).exists()
    )
    if not include_generated:
        files = {
            p
            for p in files
            if not any(
                part.endswith(".egg-info") or part in {"build", "dist", ".ruff_cache"}
                for part in p.relative_to(root).parts
            )
        }
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
    }
