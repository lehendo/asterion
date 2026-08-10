import copy
import json

import pytest
from asterion.cli import main
from asterion.diagnostic_report import write_diagnosis
from asterion.io import ValidationError, digest
from asterion.ir import Schedule, make_program


def fixture():
    schedule = Schedule().to_dict()
    return {
        "artifact_type": "diagnosis",
        "measurement_kind": "simulated_fixture",
        "comparison": {
            "verdict": "inconclusive",
            "median_paired_ratio": 1.0,
            "blocks": 16,
            "practical_threshold": 0.1,
            "limitations": "fixture only",
        },
        "variants": {
            n: {"schedule": schedule, "generated_source": "<script>alert(1)</script>"}
            for n in ("baseline", "candidate")
        },
        "blocks": [
            {"measurements": {n: {"device_graph_us": 1.0} for n in ("baseline", "candidate")}}
        ],
        "boundary": "device_graph_us",
        "specification": {"program": {"operation": "pointwise", "shape": [17]}},
        "environment": {"gpu_runtime": {"name": "<img src=x onerror=alert(1)>"}},
        "findings": [],
        "scope": "fixture only",
    }


def test_report_escapes_untrusted_strings_and_has_no_active_code(tmp_path):
    result = fixture()
    path = write_diagnosis(result, tmp_path)
    text = path.read_text()
    assert "SIMULATED_FIXTURE" in text
    assert "<script>" not in text
    assert "<img src=x" not in text
    assert "&lt;img" in text
    assert "default-src 'none'" in text
    assert "<svg" in text
    assert "file://" not in text


def test_corrupt_or_malformed_saved_results_fail_actionably(tmp_path):
    result = fixture()
    result["result_id"] = digest(result)
    bad = copy.deepcopy(result)
    bad["scope"] = "modified"
    with pytest.raises(ValidationError, match="integrity"):
        write_diagnosis(bad, tmp_path)
    with pytest.raises(ValidationError, match="malformed"):
        write_diagnosis({"artifact_type": "diagnosis"}, tmp_path)


def test_cpu_cli_export_does_not_require_gpu(tmp_path):
    schedule = Schedule().to_dict()
    spec = {
        "schema_version": 1,
        "artifact_type": "regression_spec",
        "program": make_program("pointwise", (19,)).to_dict(),
        "baseline": schedule,
        "candidate": {**schedule, "fusion": False},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert (
        main(["reproduce", str(path), "--export-only", "--output", str(tmp_path / "export")]) == 0
    )
    assert (tmp_path / "export/reproduce.py").exists()


def test_explain_preserves_existing_html_filename_contract(tmp_path):
    source = tmp_path / "saved.json"
    source.write_text(json.dumps(fixture()))
    target = tmp_path / "view/report.html"
    assert main(["explain", str(source), "--output", str(target)]) == 0
    assert target.is_file()
