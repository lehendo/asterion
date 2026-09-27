"""Public provenance must work without private experiment modules."""

import hashlib

from asterion.provenance import execution_sources
from asterion.source_identity import source_manifest


def test_identity_hashes_sources_and_excludes_generated_metadata(tmp_path):
    package = tmp_path / "python/sample"
    package.mkdir(parents=True)
    source = package / "kernel.py"
    source.write_bytes(b"value = 1\n")
    generated = package / "cache.egg-info"
    generated.mkdir()
    (generated / "PKG-INFO").write_text("generated")
    manifest = source_manifest(tmp_path)
    assert manifest == {"python/sample/kernel.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    source.write_bytes(b"value = 2\n")
    assert source_manifest(tmp_path) != manifest
    assert "python/sample/cache.egg-info/PKG-INFO" in source_manifest(
        tmp_path, include_generated=True
    )


def test_execution_identity_retains_full_license(tmp_path):
    (tmp_path / "LICENSE").write_bytes(b"fixture license\n")
    assert execution_sources(tmp_path) == {
        "LICENSE": hashlib.sha256(b"fixture license\n").hexdigest()
    }
