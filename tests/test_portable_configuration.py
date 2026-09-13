"""Simulated dependency probes; these CPU tests do not execute GPU work."""

import sys
from types import SimpleNamespace

import pytest
from asterion.backend import require_gpu
from asterion.cache import Cache
from asterion.driver import DriverModule
from asterion.io import ValidationError
from asterion.ir import make_program


def test_cluster_guard_is_preserved_and_rejects_ambiguous_configuration(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("ASTERION_REQUIRE_SLURM", "1")
    with pytest.raises(ValidationError, match="Slurm"):
        require_gpu()
    monkeypatch.setenv("ASTERION_REQUIRE_SLURM", "true")
    with pytest.raises(ValidationError, match="0 or 1"):
        require_gpu()


def test_ordinary_device_probe_does_not_require_institution_scheduler(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("ASTERION_REQUIRE_SLURM", raising=False)
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    compiler = SimpleNamespace(__version__="fixture")
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "triton", compiler)
    assert require_gpu() == (fake, compiler)


def test_cache_uses_standard_user_location_without_internal_project_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("ASTERION_CACHE_DIR", raising=False)
    monkeypatch.delenv("ASTERION_SCRATCH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert Cache.default().root == tmp_path / "asterion"


def test_native_driver_rejects_unvalidated_modern_compiler_abi(monkeypatch):
    monkeypatch.setattr(
        "asterion.backend.require_gpu", lambda: (None, SimpleNamespace(__version__="3.8.0"))
    )
    kernel = SimpleNamespace(program=make_program("pointwise", (17,)), key="0" * 64)
    with pytest.raises(ValidationError, match="validated Triton 3.3.1"):
        DriverModule(None, kernel)


def test_unsupported_device_gets_an_error_before_compilation(monkeypatch):
    from asterion import backend

    fake = SimpleNamespace(
        __version__="fixture",
        version=SimpleNamespace(cuda="fixture"),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_capability=lambda: (7, 5),
            get_device_properties=lambda device: SimpleNamespace(
                name="fixture", total_memory=1, uuid="fixture"
            ),
        ),
    )
    monkeypatch.setattr(backend, "_TARGET_CACHE", {})
    monkeypatch.setattr(backend, "command", lambda args: None)
    monkeypatch.setattr(
        backend, "require_gpu", lambda: (fake, SimpleNamespace(__version__="3.8.0"))
    )
    with pytest.raises(ValidationError, match="8.0 or newer"):
        backend.target_manifest()
