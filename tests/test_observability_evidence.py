import json
from pathlib import Path

import pytest

from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.observability_config import METRIC_NAMES, SPAN_NAMES

pytestmark = pytest.mark.integration


def test_native_mps_observability_evidence_is_hash_bound_and_passed() -> None:
    report = json.loads(
        Path("reports/observability/module15-native-mps-observability.json").read_text(
            encoding="utf-8"
        )
    )
    report_hash = report.pop("report_sha256")
    assert stable_json_sha256(report) == report_hash
    implementation_paths = {
        "observability.yaml": Path("configs/observability.yaml"),
        "observability.py": Path("src/governed_banking/observability.py"),
        "observability_config.py": Path("src/governed_banking/observability_config.py"),
        "observed_deployment_service.py": Path(
            "src/governed_banking/observed_deployment_service.py"
        ),
        "run_observability_smoke.py": Path("scripts/run_observability_smoke.py"),
    }
    assert report["implementation_sha256"] == {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    assert report["all_checks_passed"] is True
    assert report["runtime"]["selected"] == "mps"
    assert report["runtime"]["real_hardware_observed"] is True
    assert report["contains_message_text"] is False
    assert report["contains_redacted_text"] is False
    assert report["contains_request_identifiers"] is False
    assert report["contains_message_hash"] is False
    assert report["observations"]["metric_names"] == sorted(METRIC_NAMES)
    assert report["observations"]["span_names"] == sorted(SPAN_NAMES)
