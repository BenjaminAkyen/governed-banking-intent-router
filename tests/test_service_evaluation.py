from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.service_evaluation import (
    ServiceEvaluationConfig,
    validate_service_evaluation_report,
)

pytestmark = pytest.mark.integration


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "api.py": Path("src/governed_banking/api.py"),
        "audit.py": Path("src/governed_banking/audit.py"),
        "inference.py": Path("src/governed_banking/inference.py"),
        "policy.py": Path("src/governed_banking/policy.py"),
        "privacy.py": Path("src/governed_banking/privacy.py"),
        "run_service_evaluation.py": Path("scripts/run_service_evaluation.py"),
        "service_evaluation.py": Path("src/governed_banking/service_evaluation.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _report() -> dict[str, object]:
    return json.loads(
        Path("reports/service/module10-api-evaluation.json").read_text(encoding="utf-8")
    )


def test_module10_report_matches_registered_config_and_implementation() -> None:
    config_path = Path("configs/service_evaluation.yaml")
    config = ServiceEvaluationConfig.from_yaml(config_path)
    report = _report()

    validate_service_evaluation_report(
        report,
        config=config,
        config_sha256=sha256_file(config_path),
        implementation_sha256=_implementation_hashes(),
    )


def test_module10_report_is_self_hashing_and_retains_shadow_boundary() -> None:
    report = _report()
    expected_hash = report.pop("report_sha256")

    assert stable_json_sha256(report) == expected_hash
    assert report["acceptance_gate"]["all_passed"] is True
    assert report["routing"]["suggest_queue_count"] == 0
    assert report["routing"]["uncertainty_authorized_suggestion"] is False
    assert report["data_boundary"]["production_validation"] is False
    assert report["data_boundary"]["official_test_access"] is False


def test_module10_report_contains_no_fixture_or_redacted_message_values() -> None:
    report_text = json.dumps(_report(), sort_keys=True)
    privacy_config = PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))
    cases = [
        json.loads(line)
        for line in Path("data/fixtures/api-shadow-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    for case in cases:
        redacted = redact_pii(privacy_config, case["message"]).redacted_text
        assert case["message"] not in report_text
        assert redacted not in report_text
