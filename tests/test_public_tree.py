"""Keep accidental private files out of the public export and distributions."""

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).parents[1] / "tools/check_public_tree.py"
    spec = importlib.util.spec_from_file_location("public_tree", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_manifest_is_exact_and_documents_do_not_link_private_notes(checker):
    root = Path(__file__).parents[1]
    assert "README.md" in checker.check(root)


def test_public_allowlist_rejects_private_or_escaping_paths(checker, tmp_path):
    for name in ("../outside", "cluster/env.sh", ".local-notes/notes.md"):
        (tmp_path / "public-files.txt").write_text(name + "\n")
        with pytest.raises(ValueError):
            checker.files(tmp_path)


def test_archive_scanner_rejects_an_extra_private_member(checker, tmp_path):
    # Build a small real tar with a disallowed member; no setup/build invocation.
    from unittest.mock import patch

    folder = tmp_path / "dist"
    folder.mkdir()
    import zipfile

    with zipfile.ZipFile(folder / "fixture.whl", "w") as wheel:
        wheel.writestr("fixture.dist-info/licenses/LICENSE", "fixture license")
    (tmp_path / "LICENSE").write_text("fixture license")
    with tarfile.open(folder / "fixture.tar.gz", "w:gz") as archive:
        entry = tarfile.TarInfo("fixture/cluster/private.sbatch")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    with (
        patch.object(checker, "check", return_value=set()),
        pytest.raises(ValueError, match="Unexpected source archive member"),
    ):
        checker.archives(tmp_path, folder)
