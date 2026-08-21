#!/usr/bin/env python3
"""Run Module 9 synthetic PII, routing and metadata-only audit evidence checks."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from governed_banking.audit import (
    AuditConfig,
    AuditSink,
    ModelAuditContext,
    PrivacyAuditContext,
    build_audit_event,
)
from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.privacy import PrivacyConfig, redact_pii

REPORT_SCHEMA_VERSION = 1
MODEL_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    parser.add_argument("--routing-config", type=Path, default=Path("configs/routing_policy.yaml"))
    parser.add_argument("--audit-config", type=Path, default=Path("configs/audit.yaml"))
    parser.add_argument(
        "--pii-fixture", type=Path, default=Path("data/fixtures/pii-redaction-cases.jsonl")
    )
    parser.add_argument(
        "--routing-fixture", type=Path, default=Path("data/fixtures/routing-safety-cases.jsonl")
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=Path("reports/calibration/seed-42-temperature-scaling.json"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/governance/module9-controls.json")
    )
    return parser.parse_args()


def load_jsonl(path: Path, *, maximum_records: int) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) > maximum_records:
        raise ValueError(f"fixture must contain between 1 and {maximum_records} records")
    records = [json.loads(line) for line in lines]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("fixture rows must be JSON objects")
    return records


def main() -> None:
    args = parse_args()
    privacy_config = PrivacyConfig.from_yaml(args.privacy_config)
    routing_config = RoutingPolicyConfig.from_yaml(args.routing_config)
    audit_config = AuditConfig.from_yaml(args.audit_config)
    pii_cases = load_jsonl(args.pii_fixture, maximum_records=500)
    routing_cases = load_jsonl(args.routing_fixture, maximum_records=500)
    calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
    model_sha256 = calibration_report["extraction"]["checkpoint_files_sha256"][
        "adapter_model.safetensors"
    ]
    model = ModelAuditContext(model_sha256)

    detector_case_counts: Counter[str] = Counter()
    detection_outcomes: Counter[str] = Counter()
    redactions = []
    for case in pii_cases:
        result = redact_pii(privacy_config, case["input"])
        if result.redacted_text != case["expected_redacted"]:
            raise AssertionError(f"redacted output differs for {case['case_id']}")
        if dict(result.pii_type_counts) != case["expected_counts"]:
            raise AssertionError(f"PII counts differ for {case['case_id']}")
        if result.redaction_applied is not case["expected_detection"]:
            raise AssertionError(f"detection outcome differs for {case['case_id']}")
        detector_case_counts.update(result.pii_type_counts)
        detection_outcomes["positive" if result.redaction_applied else "negative"] += 1
        redactions.append((case, result))
    missing_detectors = sorted(set(privacy_config.detector_order) - set(detector_case_counts))
    if missing_detectors:
        raise AssertionError(f"PII fixture does not cover detectors: {missing_detectors}")

    routing_action_counts: Counter[str] = Counter()
    routing_reason_counts: Counter[str] = Counter()
    routing_decisions = []
    for case in routing_cases:
        routing_input = RoutingInput(
            predicted_intent=case["predicted_intent"],
            model_seed=case["model_seed"],
            uncertainty_signal=case["uncertainty_signal"],
            uncertainty_score=case["uncertainty_score"],
            pii_type_counts=case["pii_type_counts"],
            redaction_succeeded=case["redaction_succeeded"],
        )
        decision = route_request(routing_config, routing_input)
        if decision.action != case["expected_action"] or decision.queue != case["expected_queue"]:
            raise AssertionError(f"routing decision differs for {case['case_id']}")
        routing_action_counts[decision.action] += 1
        routing_reason_counts.update(decision.reason_codes)
        routing_decisions.append((routing_input, decision))
    if routing_action_counts["suggest_queue"]:
        raise AssertionError("Module 9 shadow evaluation produced a suggestion action")

    emitted_events = []
    prohibited_value_matches = 0
    with tempfile.TemporaryDirectory(prefix="module9-audit-") as temporary:
        sink = AuditSink(Path(temporary), audit_config)
        for case, redaction in redactions:
            routing_input = RoutingInput(
                predicted_intent="card_arrival",
                model_seed=MODEL_SEED,
                uncertainty_signal="max_probability",
                uncertainty_score=0.99,
                pii_type_counts=redaction.pii_type_counts,
                redaction_succeeded=True,
            )
            decision = route_request(routing_config, routing_input)
            event = build_audit_event(
                audit_config, routing_config, model, routing_input, redaction, decision
            )
            serialized = json.dumps(event, sort_keys=True)
            if case["input"] in serialized or redaction.redacted_text in serialized:
                prohibited_value_matches += 1
            sink.append(event)
            emitted_events.append(event)
        failed_input, failed_decision = next(
            pair for pair in routing_decisions if pair[0].redaction_succeeded is False
        )
        failed_privacy = PrivacyAuditContext.failed(
            privacy_config,
            input_character_count=privacy_config.maximum_input_characters + 1,
        )
        sink.append(
            build_audit_event(
                audit_config,
                routing_config,
                model,
                failed_input,
                failed_privacy,
                failed_decision,
            )
        )
        emitted_events.append(sink.read_validated()[-1])
        persisted_events = sink.read_validated()
        directory_mode = f"{stat.S_IMODE(sink.path.parent.stat().st_mode):04o}"
        file_mode = f"{stat.S_IMODE(sink.path.stat().st_mode):04o}"
    if persisted_events != emitted_events:
        raise AssertionError("validated audit round trip differs from emitted events")
    if prohibited_value_matches:
        raise AssertionError("audit serialization contains original or redacted message text")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "module9_governance_controls",
        "claim_scope": "synthetic_control_verification_not_production_validation",
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "model_inference_performed": False,
        "official_test_metrics_computed": False,
        "source_evidence": {
            "provenance": "synthetic_authored_non_customer_data",
            "pii_fixture_sha256": sha256_file(args.pii_fixture),
            "routing_fixture_sha256": sha256_file(args.routing_fixture),
            "calibration_report_sha256": calibration_report["report_sha256"],
            "model_adapter_sha256": model_sha256,
        },
        "privacy": {
            "policy_version": privacy_config.policy_version,
            "policy_sha256": privacy_config.config_sha256,
            "case_count": len(pii_cases),
            "expected_detection_cases": detection_outcomes["positive"],
            "expected_non_detection_cases": detection_outcomes["negative"],
            "detector_case_counts": dict(sorted(detector_case_counts.items())),
            "missing_registered_detectors": missing_detectors,
            "all_expected_outputs_matched": True,
        },
        "routing": {
            "policy_version": routing_config.policy_version,
            "policy_sha256": routing_config.config_sha256,
            "operating_mode": routing_config.operating_mode,
            "case_count": len(routing_cases),
            "action_counts": dict(sorted(routing_action_counts.items())),
            "reason_counts": dict(sorted(routing_reason_counts.items())),
            "all_expected_decisions_matched": True,
            "suggest_queue_count": routing_action_counts["suggest_queue"],
            "uncertainty_evidence_status": routing_config.uncertainty_status,
            "uncertainty_use": routing_config.uncertainty_use,
            "uncertainty_authorized_suggestion": (
                routing_config.uncertainty_may_authorize_suggestion
            ),
        },
        "audit": {
            "config_sha256": audit_config.config_sha256,
            "event_count": len(persisted_events),
            "validated_round_trip": True,
            "prohibited_value_matches": prohibited_value_matches,
            "directory_mode": directory_mode,
            "file_mode": file_mode,
            "message_hash_logged": False,
            "exact_input_length_logged": False,
        },
        "acceptance_gate": {
            "all_registered_detectors_exercised": not missing_detectors,
            "all_expected_redactions_matched": True,
            "all_expected_routes_matched": True,
            "no_suggestion_actions": routing_action_counts["suggest_queue"] == 0,
            "metadata_only_audit_round_trip": persisted_events == emitted_events,
            "no_original_or_redacted_values_in_audit": prohibited_value_matches == 0,
            "restrictive_sink_permissions": directory_mode == "0700" and file_mode == "0600",
            "module8_thresholds_remain_review_only": (
                routing_config.uncertainty_status == "failed_registered_gates"
                and routing_config.uncertainty_use == "review_signal_only"
                and routing_config.uncertainty_may_authorize_suggestion is False
            ),
        },
        "implementation_sha256": {
            "audit.py": sha256_file(Path("src/governed_banking/audit.py")),
            "policy.py": sha256_file(Path("src/governed_banking/policy.py")),
            "privacy.py": sha256_file(Path("src/governed_banking/privacy.py")),
            "run_governance_controls.py": sha256_file(Path(__file__)),
        },
        "limitations": [
            "Synthetic fixtures do not estimate production precision, recall, demographic parity, "
            "or attack resistance.",
            "The PII recognizers are bounded controls, not a replacement for enterprise DLP or "
            "secret scanning.",
            "Free-form names and contextual identifiers are outside the registered detector scope.",
            "Module 8 thresholds failed their registered gates and cannot authorize automated "
            "suggestions.",
            "The local JSONL sink demonstrates an event contract; production needs centralized "
            "access control, retention, integrity, and monitoring.",
        ],
    }
    report["acceptance_gate"]["all_passed"] = all(report["acceptance_gate"].values())
    report["report_sha256"] = stable_json_sha256(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "report_sha256": report["report_sha256"],
                "privacy_cases": len(pii_cases),
                "routing_cases": len(routing_cases),
                "audit_events": len(persisted_events),
                "suggest_queue_count": routing_action_counts["suggest_queue"],
                "all_passed": report["acceptance_gate"]["all_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
