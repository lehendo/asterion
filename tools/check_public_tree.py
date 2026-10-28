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


