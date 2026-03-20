"""Asterion: trace analysis is usable without importing GPU dependencies."""

from .analyzer import analyze, compare
from .schema import WorkloadSpec

__version__ = "0.3.0rc1"
__all__ = [
    "WorkloadSpec",
    "analyze",
    "compare",
    "diagnose",
    "export_reproducer",
    "optimize",
    "reduce_regression",
]

from .api import diagnose, export_reproducer, optimize, reduce_regression
