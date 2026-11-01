"""Validate/export the explicit public file set and inspect built distributions.

This tool only copies ordinary files and reads archives; it never changes Git.
It is an integrity/accidental-exposure check, not a complete secret scanner.
"""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def files(root):
    names = (root / "public-files.txt").read_text().splitlines()
    if len(names) != len(set(names)) or not names:
        raise ValueError("Public allowlist must be nonempty and contain no duplicates")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Noncanonical public path")
        if path.parts[0] in {".git", ".local-notes", "cluster", "scripts", "benchmarks"}:
            raise ValueError("Private category in public allowlist")
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("Public file is missing or a symlink: " + name)
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError("Public file escapes source root")
    return set(names)


def check(root):
    names = files(root)
    for name in names:
        path = root / name
        text = path.read_text()
        # Reports list only file names, never matching credential contents.
        private = r"/(?:Users|u|scratch)/[A-Za-z0-9]|campuscluster\.illinois\.edu"
        if re.search(private, text):
            raise ValueError("Private location in public file: " + name)
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or re.match(r"^[A-Za-z]+:", target):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError("Documentation link escapes root: " + name)
                if str(resolved.relative_to(root.resolve())) not in names:
                    raise ValueError("Documentation links outside public set: " + name)
    # Packaging must select exactly these files, not rely on ignore rules or Git.
    expected = "global-exclude *\n" + "".join("include " + n + "\n" for n in sorted(names))
    if (root / "MANIFEST.in").read_text() != expected:
        raise ValueError("Source distribution manifest differs from public allowlist")
    return names


