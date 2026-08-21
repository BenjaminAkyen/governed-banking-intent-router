from __future__ import annotations

import json
from pathlib import Path

from governed_banking.data import stable_json_sha256


def _report() -> dict[str, object]:
    return json.loads(Path("reports/governance/module9-controls.json").read_text(encoding="utf-8"))


def test_module9_report_is_self_hashing_and_text_free() -> None:
    report = _report()
    expected_hash = report.pop("report_sha256")

    assert stable_json_sha256(report) == expected_hash
    assert report["contains_message_text"] is False
    assert report["contains_redacted_text"] is False
    assert report["contains_message_hash"] is False
    assert report["model_inference_performed"] is False
    assert report["official_test_metrics_computed"] is False


def test_module9_registered_acceptance_checks_pass() -> None:
    report = _report()

    assert report["acceptance_gate"]["all_passed"] is True
    assert all(report["acceptance_gate"].values())
    assert report["routing"]["suggest_queue_count"] == 0
    assert report["routing"]["uncertainty_evidence_status"] == "failed_registered_gates"
    assert report["routing"]["uncertainty_use"] == "review_signal_only"
    assert report["routing"]["uncertainty_authorized_suggestion"] is False


def test_report_contains_no_fixture_message_or_redacted_output() -> None:
    serialized = json.dumps(_report(), sort_keys=True)
    cases = [
        json.loads(line)
        for line in Path("data/fixtures/pii-redaction-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]

    assert all(case["input"] not in serialized for case in cases)
    assert all(case["expected_redacted"] not in serialized for case in cases)
