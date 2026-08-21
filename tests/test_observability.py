from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from governed_banking.audit import (
    AuditConfig,
    ModelAuditContext,
    build_audit_event,
)
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.observability import (
    SERVICE_NAME,
    GovernedTelemetry,
    ObservingAuditStore,
    RoutingDistributionMonitor,
    TelemetryAttributeGuard,
    TelemetryIdentity,
)
from governed_banking.observability_config import ObservabilityConfig
from governed_banking.otel_runtime import validate_otlp_endpoint
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.privacy import PrivacyConfig, redact_pii

CUSTOMER_CANARY = "customer-secret-alex@example.test-GB82WEST12345698765432"
REQUEST_CANARY = "2ecfc177-8ab9-4e05-a902-111111111111"
MESSAGE_HASH_CANARY = "f" * 64


class RecordingStore:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


def _config() -> ObservabilityConfig:
    return ObservabilityConfig.from_yaml(Path("configs/observability.yaml"))


def _identity() -> TelemetryIdentity:
    deployment = DeploymentProfile.from_yaml(Path("configs/deployment/native-mps.yaml"))
    return TelemetryIdentity(
        deployment_environment=deployment.environment,
        service_version="0.1.0",
        model_version=deployment.model_release_id,
        policy_version="module9-deterministic-risk-routing-v1",
    )


def _telemetry() -> tuple[
    GovernedTelemetry,
    InMemoryMetricReader,
    InMemorySpanExporter,
    MeterProvider,
    TracerProvider,
]:
    config = _config()
    identity = _identity()
    resource = Resource(
        {
            "service.name": SERVICE_NAME,
            "service.version": identity.service_version,
            "deployment.environment": identity.deployment_environment,
            "model.version": identity.model_version,
            "policy.version": identity.policy_version,
        }
    )
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader], resource=resource)
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    meter = metrics.get_meter("module15-test", meter_provider=meter_provider)
    tracer = trace.get_tracer("module15-test", tracer_provider=tracer_provider)
    telemetry = GovernedTelemetry(config, identity, meter=meter, tracer=tracer)
    return telemetry, metric_reader, span_exporter, meter_provider, tracer_provider


def _audit_event() -> dict[str, Any]:
    audit = AuditConfig.from_yaml(Path("configs/audit.yaml"))
    routing = RoutingPolicyConfig.from_yaml(Path("configs/routing_policy.yaml"))
    privacy = PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))
    redaction = redact_pii(privacy, CUSTOMER_CANARY)
    routing_input = RoutingInput(
        predicted_intent="card_arrival",
        model_seed=42,
        uncertainty_signal="max_probability",
        uncertainty_score=0.73,
        pii_type_counts=redaction.pii_type_counts,
        redaction_succeeded=True,
    )
    decision = route_request(routing, routing_input)
    return build_audit_event(
        audit,
        routing,
        ModelAuditContext("a" * 64),
        routing_input,
        redaction,
        decision,
        event_id=REQUEST_CANARY,
        occurred_at=datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
    )


def _metric_attributes(metric_reader: InMemoryMetricReader) -> list[dict[str, Any]]:
    data = metric_reader.get_metrics_data()
    attributes: list[dict[str, Any]] = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                attributes.extend(dict(point.attributes) for point in metric.data.data_points)
    return attributes


def test_guard_rejects_free_form_keys_and_unregistered_values() -> None:
    guard = TelemetryAttributeGuard(_config(), _identity())

    with pytest.raises(ValueError, match="non-allowlisted"):
        guard.validate({"message": CUSTOMER_CANARY})
    with pytest.raises(ValueError, match="unregistered value"):
        guard.validate({"route.action": CUSTOMER_CANARY})
    with pytest.raises(ValueError, match="integer from 100 to 599"):
        guard.validate({"http.response.status_code": True})


def test_manual_signals_emit_bounded_metadata_without_payload_or_identifiers() -> None:
    telemetry, reader, exporter, meter_provider, tracer_provider = _telemetry()
    store = RecordingStore()
    observing_store = ObservingAuditStore(store, telemetry)  # type: ignore[arg-type]

    with telemetry.request_span("route_v1", "POST") as span:
        observing_store.append(_audit_event())
        telemetry.finish_request(
            span,
            endpoint="route_v1",
            method="POST",
            status_code=200,
            duration_seconds=0.012,
            error_type=None,
        )
    with telemetry.model_load_span("mps") as span:
        telemetry.record_model_load(
            span,
            duration_seconds=0.5,
            device="mps",
            error_type=None,
        )
    telemetry.set_selected_device("mps")

    meter_provider.force_flush()
    tracer_provider.force_flush()
    metric_attributes = _metric_attributes(reader)
    spans = exporter.get_finished_spans()
    serialized = json.dumps(
        {
            "metric_attributes": metric_attributes,
            "span_attributes": [dict(span.attributes or {}) for span in spans],
            "resource_attributes": [dict(span.resource.attributes) for span in spans],
        },
        sort_keys=True,
    )

    assert {span.name for span in spans} == {
        "banking_router.http.request",
        "banking_router.model.load",
    }
    assert CUSTOMER_CANARY not in serialized
    assert REQUEST_CANARY not in serialized
    assert MESSAGE_HASH_CANARY not in serialized
    assert "request_id" not in serialized
    assert "correlation_id" not in serialized
    assert "message_hash" not in serialized
    allowed_keys = set(_config().allowed_attribute_keys)
    assert all(set(attributes) <= allowed_keys for attributes in metric_attributes)

    meter_provider.shutdown()
    tracer_provider.shutdown()


def test_distribution_shift_waits_for_minimum_and_uses_bounded_window() -> None:
    monitor = RoutingDistributionMonitor(_config())
    for _ in range(19):
        monitor.add("human_review")
    assert monitor.snapshot()["distribution_shift"] is None

    monitor.add("security_queue")
    snapshot = monitor.snapshot()
    assert snapshot["observations"] == 20
    assert snapshot["human_review_ratio"] == pytest.approx(0.95)
    assert snapshot["security_escalation_ratio"] == pytest.approx(0.05)
    assert snapshot["distribution_shift"] == pytest.approx(7 / 60)

    for _ in range(100):
        monitor.add("security_queue")
    assert monitor.snapshot()["observations"] == 100


@pytest.mark.parametrize(
    ("endpoint", "profile_path", "expected_insecure"),
    [
        ("http://localhost:4317", "configs/deployment/native-mps.yaml", True),
        ("https://collector.example.test:4317", "configs/deployment/linux-cpu.yaml", False),
        ("http://127.0.0.1:4317", "configs/deployment/linux-cpu.yaml", True),
    ],
)
def test_otlp_endpoint_transport_boundary(
    endpoint: str, profile_path: str, expected_insecure: bool
) -> None:
    deployment = DeploymentProfile.from_yaml(Path(profile_path))
    assert validate_otlp_endpoint(endpoint, deployment) is expected_insecure


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://collector.example.test:4317",
        "https://token@collector.example.test:4317",
        "https://collector.example.test:4317/v1/traces",
        "https://collector.example.test:4317?token=secret",
    ],
)
def test_unsafe_otlp_endpoint_fails_closed(endpoint: str) -> None:
    deployment = DeploymentProfile.from_yaml(Path("configs/deployment/linux-cpu.yaml"))
    with pytest.raises(ValueError, match="OTLP endpoint|credential-free|plaintext"):
        validate_otlp_endpoint(endpoint, deployment)
