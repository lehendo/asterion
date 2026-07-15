"""External literal Triton source, independent of Asterion IR/code generation."""

from pathlib import Path

import pytest
from asterion.external_runtime import run
from asterion.io import read

pytestmark = pytest.mark.gpu


def test_repeated_external_capture_with_aggressive_collection(tmp_path):
    import gc

    source = Path(__file__).parents[2] / "examples"
    spec = read(source / "external_add.json")
    spec["dimensions"]["N"] = 33
    spec["measurement"].update(rounds=12, launches=1, interventions="none")
    threshold = gc.get_threshold()
    try:
        # Replacing the same reviewed module leaves ordinary Python function/
        # globals cycles. Stress collection during later launch allocations.
        for index in range(3):
            gc.set_threshold(25, 2, 2)
            result = run(spec, source, tmp_path / ("capture-" + str(index)))
            assert result["numerical_status"] == "ACCEPTED_TESTED_DOMAIN"
            assert result["timing_status"] == "MEASURED"
    finally:
        gc.set_threshold(*threshold)


def test_external_literal_kernel_validation_and_provenance(tmp_path):
    source = Path(__file__).parents[2] / "examples"
    spec = read(source / "external_add.json")
    spec["measurement"].update(rounds=12, launches=10)
    result = run(spec, source, tmp_path / "literal")
    assert result["numerical_status"] == "ACCEPTED_TESTED_DOMAIN"
    assert result["timing_status"] == "MEASURED"
    assert len(result["detection_blocks"]) == 12
    for name in ("baseline", "candidate"):
        record = result["variants"][name]
        assert len(record["checks"]) == 8 and all(c["passed"] for c in record["checks"])
        assert record["compiler"]["assembly"]["cubin"]["sha256"]
    if result["status"] == "REGRESSION_ESTABLISHED":
        assert result["variants"]["undo_BLOCK"]["status"] == "ACCEPTED"
        assert result["intervention_blocks"]
    assert (tmp_path / "literal/report.html").exists()


