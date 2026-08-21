#!/usr/bin/env python3
"""Record real-MPS, in-memory evidence for the Module 15 telemetry boundary."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from governed_banking.api import ServiceConfig
from governed_banking.audit import AuditConfig
from governed_banking.audit_store import LocalJsonlAuditStore
from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.deployment_service import DEPLOYMENT_SERVICE_VERSION
from governed_banking.observability import SERVICE_NAME, GovernedTelemetry, TelemetryIdentity
from governed_banking.observability_config import METRIC_NAMES, SPAN_NAMES, ObservabilityConfig
from governed_banking.observed_deployment_service import create_observed_deployment_app
from governed_banking.policy import RoutingPolicyConfig

EMAIL_CANARY = "module15.person@example.test"
SECRET_CANARY = "M15SyntheticAuthSecret987654"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/deployment/native-mps.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/observability/module15-native-mps-observability.json"),
    )
    return parser.parse_args()


def _in_memory_telemetry(
    profile: DeploymentProfile,
    config: ObservabilityConfig,
) -> tuple[GovernedTelemetry, InMemoryMetricReader, InMemorySpanExporter]:
    service = ServiceConfig.from_yaml(profile.legacy_service_config_path)
    policy = RoutingPolicyConfig.from_yaml(service.routing_config_path)
    identity = TelemetryIdentity(
        deployment_environment=profile.environment,
        service_version=DEPLOYMENT_SERVICE_VERSION,
        model_version=profile.model_release_id,
        policy_version=policy.policy_version,
    )
    resource = Resource(
        {
            "service.name": SERVICE_NAME,
            "service.version": identity.service_version,
            "deployment.environment": identity.deployment_environment,
            "model.version": identity.model_version,
            "policy.version": identity.policy_version,
        }
    )
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader], resource=resource)
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    def shutdown() -> None:
        meter_provider.force_flush()
        tracer_provider.force_flush()

    telemetry = GovernedTelemetry(
        config,
        identity,
        meter=metrics.get_meter("module15-mps-evidence", meter_provider=meter_provider),
        tracer=trace.get_tracer("module15-mps-evidence", tracer_provider=tracer_provider),
        shutdown=shutdown,
    )
    return telemetry, reader, exporter


def _metric_snapshot(reader: InMemoryMetricReader) -> tuple[set[str], list[dict[str, Any]]]:
    names: set[str] = set()
    attributes: list[dict[str, Any]] = []
    data = reader.get_metrics_data()
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                names.add(metric.name)
                attributes.extend(
                    dict(point.attributes) for point in metric.data.data_points
                )
    return names, attributes


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "observability.yaml": Path("configs/observability.yaml"),
        "observability.py": Path("src/governed_banking/observability.py"),
        "observability_config.py": Path("src/governed_banking/observability_config.py"),
        "observed_deployment_service.py": Path(
            "src/governed_banking/observed_deployment_service.py"
        ),
        "run_observability_smoke.py": Path(__file__),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    profile = DeploymentProfile.from_yaml(args.profile)
    if profile.expected_device != "mps" or profile.platform != "native_macos":
        raise ValueError("this evidence script requires the registered native MPS profile")
    config = ObservabilityConfig.from_yaml(
        Path("configs/observability.yaml"), deployment_profile=profile
    )
    telemetry, reader, exporter = _in_memory_telemetry(profile, config)
    environment = dict(os.environ)
    environment.pop("GOVERNED_BANKING_CONTAINER", None)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    token = secrets.token_urlsafe(32)
    environment[profile.authentication.secret_environment_variable] = token
    request_identifiers: list[str] = []
    action_counts = {"human_review": 0, "security_queue": 0}

    with tempfile.TemporaryDirectory(prefix="module15-audit-") as temporary:
        stores: list[LocalJsonlAuditStore] = []

        def audit_factory(_: Path, audit_config: AuditConfig) -> LocalJsonlAuditStore:
            store = LocalJsonlAuditStore(Path(temporary), audit_config)
            stores.append(store)
            return store

        app = create_observed_deployment_app(
            profile,
            environment=environment,
            audit_store_factory=audit_factory,
            telemetry=telemetry,
        )
        with TestClient(app) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            unauthorized = client.post(
                "/v1/route", json={"message": "synthetic request without credentials"}
            )
            missing = client.get("/not-a-real-route")
            for index in range(10):
                requests = (
                    f"Email {EMAIL_CANARY} about card delivery case {index}",
                    f"My password is {SECRET_CANARY}{index}; my account may be compromised",
                )
                for message in requests:
                    response = client.post(
                        "/v1/route",
                        json={"message": message},
                        headers={
                            "Authorization": f"Bearer {token}",
                            "X-Correlation-ID": str(uuid.uuid4()),
                        },
                    )
                    if response.status_code != 200:
                        raise AssertionError("synthetic observed route failed")
                    payload = response.json()
                    request_identifiers.append(payload["request_id"])
                    action_counts[payload["action"]] += 1
            loaded = app.state.deployment_runtime.loaded
            if loaded is None:
                raise AssertionError("observed deployment did not load")
            runtime = dict(loaded.runtime_metadata)

    metric_names, metric_attributes = _metric_snapshot(reader)
    spans = exporter.get_finished_spans()
    span_names = {span.name for span in spans}
    span_attributes = [dict(span.attributes or {}) for span in spans]
    resource_attributes = [dict(span.resource.attributes) for span in spans]
    serialized_telemetry = json.dumps(
        {
            "metric_attributes": metric_attributes,
            "span_attributes": span_attributes,
            "resource_attributes": resource_attributes,
        },
        sort_keys=True,
    )
    observed_keys = {
        key
        for attributes in metric_attributes + span_attributes + resource_attributes
        for key in attributes
    }
    redaction_categories = sorted(
        {
            attributes["privacy.redaction.category"]
            for attributes in metric_attributes
            if "privacy.redaction.category" in attributes
        }
    )
    checks = {
        "liveness_healthy": live.status_code == 200,
        "readiness_healthy": ready.status_code == 200,
        "authentication_error_observed": unauthorized.status_code == 401,
        "not_found_error_observed": missing.status_code == 404,
        "mps_selected_without_fallback": runtime.get("selected") == "mps",
        "all_registered_metrics_observed": metric_names == set(METRIC_NAMES),
        "only_registered_spans_observed": span_names == set(SPAN_NAMES),
        "human_review_and_security_actions_observed": all(action_counts.values()),
        "email_and_authentication_redactions_observed": {
            "email",
            "authentication_secret",
        }
        <= set(redaction_categories),
        "only_allowlisted_application_attributes": observed_keys
        <= set(config.allowed_attribute_keys),
        "customer_canaries_absent": EMAIL_CANARY not in serialized_telemetry
        and SECRET_CANARY not in serialized_telemetry,
        "request_identifiers_absent": all(
            identifier not in serialized_telemetry for identifier in request_identifiers
        ),
        "message_hashes_absent": "message_hash" not in serialized_telemetry,
        "rolling_distribution_minimum_reached": sum(action_counts.values())
        >= config.minimum_observations,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "module15_native_mps_observability_smoke",
        "claim_scope": "synthetic_in_memory_exporter_privacy_and_signal_smoke_not_production",
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_request_identifiers": False,
        "contains_message_hash": False,
        "deployment_profile_sha256": profile.config_sha256,
        "observability_config_sha256": config.config_sha256,
        "implementation_sha256": dict(sorted(_implementation_hashes().items())),
        "runtime": runtime,
        "observations": {
            "completed_governed_routes": sum(action_counts.values()),
            "routing_action_counts": dict(sorted(action_counts.items())),
            "metric_names": sorted(metric_names),
            "span_names": sorted(span_names),
            "attribute_keys": sorted(observed_keys),
            "redaction_categories": redaction_categories,
        },
        "checks": dict(sorted(checks.items())),
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "All requests are synthetic; no representative bank customer data was used.",
            "The in-memory exporter proves application emission, not Collector or "
            "backend operation.",
            "This is a single-process MPS smoke test, not a load or availability test.",
            "Routing-distribution change remains descriptive; no production alert "
            "threshold is approved.",
            "The service remains shadow-review-only because the registered Module 13 gates failed.",
        ],
    }
    report["report_sha256"] = stable_json_sha256(report)
    if not report["all_checks_passed"]:
        raise AssertionError("Module 15 observability smoke checks failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "selected_device": runtime.get("selected"),
                "completed_routes": sum(action_counts.values()),
                "metric_count": len(metric_names),
                "span_count": len(spans),
                "all_checks_passed": report["all_checks_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
