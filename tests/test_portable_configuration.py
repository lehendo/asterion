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


