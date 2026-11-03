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


def archives(root, folder):
    names = check(root)
    packages = sorted(folder.glob("*.whl")) + sorted(folder.glob("*.tar.gz"))
    if not any(p.suffix == ".whl" for p in packages) or not any(
        p.name.endswith(".tar.gz") for p in packages
    ):
        raise ValueError("Both wheel and source distribution are required")
    outcomes = []
    metadata = {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    }
    wheel_metadata = {
        "METADATA",
        "WHEEL",
        "RECORD",
        "entry_points.txt",
        "top_level.txt",
        "licenses/LICENSE",
    }
    for package in packages:
        actual = {}
        if package.suffix == ".whl":
            with zipfile.ZipFile(package) as archive:
                actual = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if not name.endswith("/")
                }
            expected_python = {
                n.removeprefix("python/") for n in names if n.startswith("python/asterion/")
            }
            if {n for n in actual if n.startswith("asterion/")} != expected_python:
                raise ValueError("Wheel package file set mismatch")
            for name, data in actual.items():
                if name.startswith("asterion/"):
                    if data != (root / "python" / name).read_bytes():
                        raise ValueError("Wheel source differs: " + name)
                elif (
                    ".dist-info/" not in name
                    or name.split(".dist-info/", 1)[1] not in wheel_metadata
                ):
                    raise ValueError("Unexpected wheel member: " + name)
            if not any(
                n.endswith("/licenses/LICENSE") and d == (root / "LICENSE").read_bytes()
                for n, d in actual.items()
            ):
                raise ValueError("Wheel license missing")
        else:
            with tarfile.open(package) as archive:
                for member in archive.getmembers():
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("Source archive has a nonregular member")
                    parts = PurePosixPath(member.name).parts
                    if len(parts) < 2 or ".." in parts or member.name.startswith("/"):
                        raise ValueError("Source archive path is unsafe")
                    actual[str(PurePosixPath(*parts[1:]))] = archive.extractfile(member).read()
            if not names <= set(actual):
                raise ValueError("Source archive missing public files")
            for name, data in actual.items():
                if name in names:
                    if data != (root / name).read_bytes():
                        raise ValueError("Source archive content mismatch: " + name)
                elif name not in {"PKG-INFO", "setup.cfg"} and not (
                    name.startswith("python/asterion_gpu.egg-info/") and Path(name).name in metadata
                ):
                    raise ValueError("Unexpected source archive member: " + name)
        outcomes.append(
            {
                "package": package.name,
                "members": len(actual),
                "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
            }
        )
    return outcomes


