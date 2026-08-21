"""Strict privacy and cardinality registration for Module 15 telemetry."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from governed_banking.data import sha256_file
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.privacy import REGISTERED_DETECTORS

OBSERVABILITY_SCHEMA_VERSION = 1
OBSERVABILITY_PROFILE_NAME = "module15-privacy-safe-otel-v1"
INSTRUMENTATION_NAME = "governed_banking.observability"
INSTRUMENTATION_VERSION = "module15-observability-v1"
METRIC_NAMES = (
    "banking_router.requests",
    "banking_router.errors",
    "banking_router.request.duration",
    "banking_router.model.load.duration",
    "banking_router.runtime.selected_device",
    "banking_router.routing.decisions",
    "banking_router.routing.human_reviews",
    "banking_router.routing.security_escalations",
    "banking_router.privacy.redactions",
    "banking_router.model.uncertainty",
    "banking_router.routing.human_review_ratio",
    "banking_router.routing.security_escalation_ratio",
    "banking_router.routing.distribution_shift",
    "banking_router.routing.window_observations",
)
SPAN_NAMES = (
    "banking_router.http.request",
    "banking_router.model.load",
)
ALLOWED_ATTRIBUTE_KEYS = (
    "deployment.device",
    "deployment.environment",
    "error.type",
    "http.request.method",
    "http.response.status_code",
    "http.route",
    "model.uncertainty.bucket",
    "model.version",
    "policy.version",
    "privacy.redaction.category",
    "route.action",
    "route.processing_status",
    "service.version",
    "telemetry.outcome",
)
PROHIBITED_ATTRIBUTE_KEYS = (
    "account",
    "authorization",
    "baggage",
    "content",
    "correlation_id",
    "customer",
    "email",
    "message",
    "message_hash",
    "name",
    "payload",
    "prompt",
    "query",
    "redacted_text",
    "request_id",
    "subject",
    "text",
    "token",
    "user",
)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Hash-bound OpenTelemetry configuration with no free-form attributes."""

    config_path: Path
    config_sha256: str
    project_root: Path
    profile_name: str
    instrumentation_name: str
    instrumentation_version: str
    endpoint_environment_variable: str
    metric_export_interval_milliseconds: int
    export_timeout_milliseconds: int
    metric_names: tuple[str, ...]
    span_names: tuple[str, ...]
    allowed_attribute_keys: tuple[str, ...]
    prohibited_attribute_keys: tuple[str, ...]
    reference_source_path: Path
    reference_source_sha256: str
    rolling_window_observations: int
    minimum_observations: int
    reference_distribution: Mapping[str, float]
    registered_deployment_profiles: Mapping[Path, str]

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        deployment_profile: DeploymentProfile | None = None,
        project_root: Path | None = None,
    ) -> ObservabilityConfig:
        resolved = path.resolve(strict=True)
        selected_root = (
            resolved.parent.parent
            if project_root is None
            else project_root.resolve(strict=True)
        )
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "profile_name",
            "instrumentation",
            "export",
            "signals",
            "attributes",
            "routing_distribution",
            "registered_deployment_profiles",
            "claims",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("observability configuration fields differ from registration")
        if raw.get("schema_version") != OBSERVABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported observability configuration schema")
        if raw.get("profile_name") != OBSERVABILITY_PROFILE_NAME:
            raise ValueError("unregistered observability profile")

        instrumentation = _exact_mapping(
            raw,
            "instrumentation",
            {
                "name",
                "version",
                "manual_instrumentation_only",
                "automatic_http_instrumentation",
                "inbound_trace_context",
                "baggage_propagation",
                "exception_stacktraces",
                "telemetry_logs",
            },
        )
        if instrumentation != {
            "name": INSTRUMENTATION_NAME,
            "version": INSTRUMENTATION_VERSION,
            "manual_instrumentation_only": True,
            "automatic_http_instrumentation": False,
            "inbound_trace_context": False,
            "baggage_propagation": False,
            "exception_stacktraces": False,
            "telemetry_logs": False,
        }:
            raise ValueError("observability instrumentation weakens the privacy registration")

        export = _exact_mapping(
            raw,
            "export",
            {
                "protocol",
                "endpoint_environment_variable",
                "endpoint_required",
                "exporter_headers_allowed",
                "metric_export_interval_milliseconds",
                "export_timeout_milliseconds",
            },
        )
        if (
            export.get("protocol") != "otlp_grpc"
            or export.get("endpoint_environment_variable") != "OTEL_EXPORTER_OTLP_ENDPOINT"
            or export.get("endpoint_required") is not True
            or export.get("exporter_headers_allowed") is not False
        ):
            raise ValueError("observability export boundary differs from registration")
        interval = _bounded_int(
            export.get("metric_export_interval_milliseconds"),
            "metric_export_interval_milliseconds",
            1000,
            300000,
        )
        timeout = _bounded_int(
            export.get("export_timeout_milliseconds"),
            "export_timeout_milliseconds",
            100,
            interval,
        )

        signals = _exact_mapping(raw, "signals", {"metric_names", "span_names"})
        if tuple(signals.get("metric_names", [])) != METRIC_NAMES:
            raise ValueError("metric names differ from the bounded registration")
        if tuple(signals.get("span_names", [])) != SPAN_NAMES:
            raise ValueError("span names differ from the bounded registration")
        attributes = _exact_mapping(raw, "attributes", {"allowed_keys", "prohibited_keys"})
        if tuple(attributes.get("allowed_keys", [])) != ALLOWED_ATTRIBUTE_KEYS:
            raise ValueError("telemetry attribute allowlist differs from registration")
        if tuple(attributes.get("prohibited_keys", [])) != PROHIBITED_ATTRIBUTE_KEYS:
            raise ValueError("telemetry prohibited-key list differs from registration")

        distribution = _exact_mapping(
            raw,
            "routing_distribution",
            {
                "dimension",
                "reference_source_path",
                "expected_reference_source_sha256",
                "reference_scope",
                "rolling_window_observations",
                "minimum_observations",
                "reference",
                "alerting_threshold_approved",
            },
        )
        if (
            distribution.get("dimension") != "route.action"
            or distribution.get("reference_scope")
            != "synthetic_module10_local_integration_not_production_baseline"
            or distribution.get("alerting_threshold_approved") is not False
        ):
            raise ValueError("routing-distribution claim boundary differs from registration")
        reference_path = _repository_path(
            selected_root,
            distribution.get("reference_source_path"),
            "reference_source_path",
        )
        reference_hash = _sha256(
            distribution.get("expected_reference_source_sha256"),
            "expected_reference_source_sha256",
        )
        _require_hash(reference_path, reference_hash, "routing reference source")
        reference = _reference_distribution(distribution.get("reference"))
        _validate_reference_source(reference_path, reference)
        window = _bounded_int(
            distribution.get("rolling_window_observations"),
            "rolling_window_observations",
            20,
            10000,
        )
        minimum = _bounded_int(
            distribution.get("minimum_observations"),
            "minimum_observations",
            2,
            window,
        )

        registrations = _deployment_registrations(
            selected_root, raw.get("registered_deployment_profiles")
        )
        if deployment_profile is not None:
            observed = registrations.get(deployment_profile.config_path)
            if observed != deployment_profile.config_sha256:
                raise ValueError("deployment profile is not registered for Module 15 telemetry")

        claims = _exact_mapping(
            raw,
            "claims",
            {
                "data_classification",
                "customer_text_collected",
                "redacted_text_collected",
                "identity_attributes_collected",
                "request_identifiers_collected",
                "message_hashes_collected",
                "automatic_header_capture",
                "production_alert_threshold_approved",
                "production_approved",
            },
        )
        if claims != {
            "data_classification": "aggregate_and_bounded_metadata_only",
            "customer_text_collected": False,
            "redacted_text_collected": False,
            "identity_attributes_collected": False,
            "request_identifiers_collected": False,
            "message_hashes_collected": False,
            "automatic_header_capture": False,
            "production_alert_threshold_approved": False,
            "production_approved": False,
        }:
            raise ValueError("observability claims overstate the current boundary")

        return cls(
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            project_root=selected_root,
            profile_name=OBSERVABILITY_PROFILE_NAME,
            instrumentation_name=INSTRUMENTATION_NAME,
            instrumentation_version=INSTRUMENTATION_VERSION,
            endpoint_environment_variable="OTEL_EXPORTER_OTLP_ENDPOINT",
            metric_export_interval_milliseconds=interval,
            export_timeout_milliseconds=timeout,
            metric_names=METRIC_NAMES,
            span_names=SPAN_NAMES,
            allowed_attribute_keys=ALLOWED_ATTRIBUTE_KEYS,
            prohibited_attribute_keys=PROHIBITED_ATTRIBUTE_KEYS,
            reference_source_path=reference_path,
            reference_source_sha256=reference_hash,
            rolling_window_observations=window,
            minimum_observations=minimum,
            reference_distribution=reference,
            registered_deployment_profiles=registrations,
        )


def _validate_reference_source(path: Path, reference: Mapping[str, float]) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    counts = report.get("routing", {}).get("action_counts")
    if not isinstance(counts, dict) or set(counts) != {"human_review", "security_queue"}:
        raise ValueError("routing reference report does not contain the registered actions")
    total = sum(counts.values())
    if not isinstance(total, int) or total <= 0:
        raise ValueError("routing reference counts are invalid")
    observed = {name: counts[name] / total for name in sorted(counts)}
    if any(not math.isclose(observed[key], reference[key], abs_tol=1e-12) for key in observed):
        raise ValueError("routing reference distribution differs from its source report")


def _reference_distribution(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != {"human_review", "security_queue"}:
        raise ValueError("routing reference must contain both shadow actions")
    result = {key: _unit_float(number, f"reference.{key}") for key, number in value.items()}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-12):
        raise ValueError("routing reference probabilities must sum to one")
    return dict(sorted(result.items()))


def _deployment_registrations(project_root: Path, value: Any) -> dict[Path, str]:
    if not isinstance(value, dict) or len(value) != 3:
        raise ValueError("exactly three deployment profiles must be registered")
    result: dict[Path, str] = {}
    for raw_path, raw_hash in value.items():
        path = _repository_path(project_root, raw_path, "deployment profile")
        digest = _sha256(raw_hash, "deployment profile hash")
        _require_hash(path, digest, "deployment profile")
        result[path] = digest
    return result


def _exact_mapping(
    parent: Mapping[str, Any], key: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"observability {key} fields differ from registration")
    return value


def _repository_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{name} must be a bounded repository path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be repository-relative")
    result = (project_root / relative).resolve(strict=True)
    if not result.is_relative_to(project_root):
        raise ValueError(f"{name} escapes the repository")
    return result


def _require_hash(path: Path, expected: str, name: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{name} hash differs from registration")


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _unit_float(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def registered_redaction_categories() -> tuple[str, ...]:
    """Expose the fixed category vocabulary without detector matches or values."""

    return REGISTERED_DETECTORS
