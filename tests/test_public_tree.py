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


