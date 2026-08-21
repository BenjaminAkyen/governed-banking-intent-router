#!/usr/bin/env python3
"""Run registered real-MPS integration and latency checks for the Module 10 shadow API."""

from __future__ import annotations

import argparse
import json
import secrets
import stat
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from governed_banking.api import GovernedService, ServiceConfig, create_app
from governed_banking.audit import AuditConfig, AuditSink
from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.inference import LoRAPredictor
from governed_banking.policy import RoutingPolicyConfig
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.service_evaluation import (
    ServiceEvaluationConfig,
    validate_service_evaluation_report,
    write_service_evaluation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/service_evaluation.yaml"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/service/module10-api-evaluation.json")
    )
    return parser.parse_args()


def load_fixture(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected_keys = {"case_id", "message", "required_action", "required_reason"}
    if not rows or len(rows) > 100 or any(set(row) != expected_keys for row in rows):
        raise ValueError("service fixture has an invalid shape or size")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("service fixture case identifiers must be unique")
    return rows


def implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "api.py": Path("src/governed_banking/api.py"),
        "audit.py": Path("src/governed_banking/audit.py"),
        "inference.py": Path("src/governed_banking/inference.py"),
        "policy.py": Path("src/governed_banking/policy.py"),
        "privacy.py": Path("src/governed_banking/privacy.py"),
        "run_service_evaluation.py": script_path,
        "service_evaluation.py": Path("src/governed_banking/service_evaluation.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    evaluation_config = ServiceEvaluationConfig.from_yaml(args.config)
    service_config = ServiceConfig.from_yaml(evaluation_config.service_config_path)
    privacy_config = PrivacyConfig.from_yaml(service_config.privacy_config_path)
    routing_config = RoutingPolicyConfig.from_yaml(service_config.routing_config_path)
    audit_config = AuditConfig.from_yaml(service_config.audit_config_path)
    cases = load_fixture(evaluation_config.fixture_path)

    startup_started = time.perf_counter()
    predictor = LoRAPredictor(service_config.predictor)
    startup_seconds = time.perf_counter() - startup_started
    if predictor.runtime.selected != "mps":
        raise RuntimeError("registered Module 10 evaluation requires MPS")

    api_token = secrets.token_urlsafe(32)
    action_counts: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    latencies_ms: list[float] = []
    required_override_failures: list[str] = []
    prohibited_value_matches = 0
    response_status_counts: Counter[str] = Counter()
    boundary_checks: dict[str, bool] = {}

    with tempfile.TemporaryDirectory(prefix="module10-service-") as temporary:
        sink = AuditSink(Path(temporary), audit_config)
        service = GovernedService(
            config=service_config,
            predictor=predictor,
            privacy_config=privacy_config,
            routing_config=routing_config,
            audit_config=audit_config,
            audit_sink=sink,
        )
        client = TestClient(create_app(service, api_token=api_token))
        authorization = {"Authorization": f"Bearer {api_token}"}
        health = client.get("/health/live")
        boundary_checks = {
            "health_minimal": health.status_code == 200
            and set(health.json()) == {"status", "service_version", "operating_mode"},
            "docs_disabled": client.get("/docs").status_code == 404
            and client.get("/openapi.json").status_code == 404,
            "authentication_required": client.post(
                "/v1/route", json={"message": "synthetic unauthenticated request"}
            ).status_code
            == 401,
            "untrusted_host_rejected": client.post(
                "/v1/route",
                json={"message": "synthetic host check"},
                headers={**authorization, "Host": "attacker.example"},
            ).status_code
            == 400,
            "extra_fields_rejected": client.post(
                "/v1/route",
                json={"message": "synthetic schema check", "role": "admin"},
                headers=authorization,
            ).status_code
            == 422,
            "oversized_body_rejected": client.post(
                "/v1/route",
                content=json.dumps({"message": "x" * 9000}),
                headers={**authorization, "Content-Type": "application/json"},
            ).status_code
            == 413,
            "security_headers_present": all(
                health.headers.get(name) == value
                for name, value in {
                    "cache-control": "no-store",
                    "x-content-type-options": "nosniff",
                    "x-frame-options": "DENY",
                    "referrer-policy": "no-referrer",
                }.items()
            ),
        }
        warmup_case = cases[0]
        for _ in range(evaluation_config.warmup_requests):
            response = client.post(
                "/v1/route",
                json={"message": warmup_case["message"]},
                headers=authorization,
            )
            if response.status_code != 200:
                raise AssertionError("warm-up request failed")
        for _ in range(evaluation_config.measurement_repetitions):
            for case in cases:
                redacted = redact_pii(privacy_config, case["message"]).redacted_text
                started = time.perf_counter()
                response = client.post(
                    "/v1/route",
                    json={"message": case["message"]},
                    headers=authorization,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(latency_ms)
                response_status_counts[str(response.status_code)] += 1
                if response.status_code != 200:
                    continue
                body = response.json()
                action_counts[body["action"]] += 1
                processing_counts[body["processing_status"]] += 1
                reason_counts.update(body["reason_codes"])
                serialized = response.text
                if case["message"] in serialized or redacted in serialized:
                    prohibited_value_matches += 1
                if "uncertainty_score" in body or body["automated_action_authorized"] is not False:
                    raise AssertionError("API response exposed a prohibited field or authorization")
                if case["required_action"] and body["action"] != case["required_action"]:
                    required_override_failures.append(case["case_id"])
                if case["required_reason"] and case["required_reason"] not in body["reason_codes"]:
                    required_override_failures.append(case["case_id"])
        events = sink.read_validated()
        serialized_events = json.dumps(events, sort_keys=True)
        for case in cases:
            redacted = redact_pii(privacy_config, case["message"]).redacted_text
            if case["message"] in serialized_events or redacted in serialized_events:
                prohibited_value_matches += 1
        directory_mode = f"{stat.S_IMODE(sink.path.parent.stat().st_mode):04o}"
        file_mode = f"{stat.S_IMODE(sink.path.stat().st_mode):04o}"

    p50_ms = float(np.percentile(latencies_ms, 50, method="higher"))
    p95_ms = float(np.percentile(latencies_ms, 95, method="higher"))
    config_sha256 = sha256_file(args.config)
    implementation = implementation_hashes(Path(__file__))
    expected_event_count = evaluation_config.warmup_requests + len(latencies_ms)
    acceptance_gate = {
        "mps_selected": predictor.runtime.selected == "mps",
        "startup_within_registered_maximum": (
            startup_seconds <= evaluation_config.maximum_startup_seconds
        ),
        "p95_latency_within_registered_maximum": (
            p95_ms <= evaluation_config.maximum_p95_milliseconds
        ),
        "all_measured_http_responses_successful": response_status_counts
        == {"200": len(latencies_ms)},
        "zero_suggestion_actions": action_counts["suggest_queue"] == 0,
        "all_required_overrides_passed": not required_override_failures,
        "metadata_only_api_and_audit": prohibited_value_matches == 0
        and len(events) == expected_event_count,
        "restrictive_audit_permissions": directory_mode == "0700" and file_mode == "0600",
        "security_boundary_checks_passed": all(boundary_checks.values()),
        "module8_uncertainty_remains_experimental": (
            service_config.predictor.uncertainty_status == "experimental_review_only"
            and routing_config.uncertainty_may_authorize_suggestion is False
        ),
    }
    acceptance_gate["all_passed"] = all(acceptance_gate.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "module10_shadow_api_evaluation",
        "experiment_name": evaluation_config.experiment_name,
        "claim_scope": evaluation_config.claim_scope,
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "service_evaluation_config_sha256": config_sha256,
        "service_config_sha256": service_config.config_sha256,
        "fixture_sha256": evaluation_config.expected_fixture_sha256,
        "implementation_sha256": dict(sorted(implementation.items())),
        "data_boundary": {
            "fixture_provenance": evaluation_config.fixture_provenance,
            "model_inference_performed": True,
            "classification_metrics_computed": False,
            "official_test_access": False,
            "production_validation": False,
            "uncertainty_status": "experimental_review_only",
        },
        "runtime": predictor.runtime.to_dict(),
        "protocol": {
            "fixture_cases": len(cases),
            "warmup_requests": evaluation_config.warmup_requests,
            "measurement_repetitions": evaluation_config.measurement_repetitions,
            "measured_requests": len(latencies_ms),
            "latency_percentile_method": "higher",
        },
        "latency": {
            "startup_seconds": round(startup_seconds, 4),
            "mean_milliseconds": round(statistics.fmean(latencies_ms), 4),
            "p50_milliseconds": round(p50_ms, 4),
            "p95_milliseconds": round(p95_ms, 4),
            "maximum_milliseconds": round(max(latencies_ms), 4),
            "registered_maximum_p95_milliseconds": (evaluation_config.maximum_p95_milliseconds),
            "registered_maximum_startup_seconds": evaluation_config.maximum_startup_seconds,
        },
        "api": {
            "response_status_counts": dict(sorted(response_status_counts.items())),
            "processing_status_counts": dict(sorted(processing_counts.items())),
            "security_boundary_checks": dict(sorted(boundary_checks.items())),
            "prohibited_value_matches": prohibited_value_matches,
        },
        "routing": {
            "operating_mode": service_config.operating_mode,
            "action_counts": dict(sorted(action_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "suggest_queue_count": action_counts["suggest_queue"],
            "required_override_failures": sorted(set(required_override_failures)),
            "uncertainty_authorized_suggestion": False,
        },
        "audit": {
            "event_count": len(events),
            "expected_event_count": expected_event_count,
            "validated_round_trip": True,
            "directory_mode": directory_mode,
            "file_mode": file_mode,
            "message_hash_logged": False,
            "exact_input_length_logged": False,
        },
        "acceptance_gate": acceptance_gate,
        "limitations": [
            "Latency is an in-process sequential measurement on one Mac and is not a service-"
            "level objective.",
            "Synthetic requests do not estimate production accuracy, privacy performance, attack "
            "resistance, or traffic mix.",
            "No concurrency, sustained-load, network-hop, availability, rate-limit, or centralized-"
            "audit test was performed.",
            "Module 8 uncertainty thresholds failed their gates and remain experimental review "
            "observations.",
        ],
    }
    report["report_sha256"] = stable_json_sha256(report)
    validate_service_evaluation_report(
        report,
        config=evaluation_config,
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_service_evaluation_report(report, args.report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "runtime_device": predictor.runtime.selected,
                "startup_seconds": report["latency"]["startup_seconds"],
                "p50_milliseconds": report["latency"]["p50_milliseconds"],
                "p95_milliseconds": report["latency"]["p95_milliseconds"],
                "measured_requests": len(latencies_ms),
                "suggest_queue_count": action_counts["suggest_queue"],
                "all_passed": acceptance_gate["all_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
