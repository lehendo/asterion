import pytest
from asterion.io import ValidationError, artifact_path, read


def test_oversized_passive_document_is_rejected_before_decode(tmp_path):
    path = tmp_path / "oversized.json"
    with path.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValidationError, match="64 MiB"):
        read(path)


def test_excessively_nested_document_has_a_controlled_error(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text("[" * 10000 + "]" * 10000)
    with pytest.raises(ValidationError, match="nesting"):
        read(path)


def test_artifact_relative_paths_cannot_escape_through_names_or_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not part of this artifact")
    (root / "alias").symlink_to(outside)
    for relative in ("../outside", str(outside), "alias"):
        with pytest.raises(ValidationError):
            artifact_path(root, relative)
    assert artifact_path(root, "safe.json") == root / "safe.json"
