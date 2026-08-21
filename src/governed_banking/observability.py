"""Manual OpenTelemetry signals with an application-side privacy allowlist."""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from opentelemetry.metrics import Meter, Observation
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from governed_banking.audit_store import AuditStore
from governed_banking.observability_config import ObservabilityConfig
from governed_banking.privacy import REGISTERED_DETECTORS

SERVICE_NAME = "governed-banking-intent-router"
SAFE_ENDPOINTS = ("health_live", "health_ready", "route_v1", "other")
SAFE_HTTP_METHODS = ("GET", "POST", "OTHER")
SAFE_DEVICES = ("cpu", "mps", "cuda")
SAFE_OUTCOMES = ("success", "error")
SAFE_ACTIONS = ("human_review", "security_queue")
SAFE_PROCESSING_STATUSES = ("completed", "redaction_failed", "inference_failed")
SAFE_UNCERTAINTY_BUCKETS = (
    "missing_or_invalid",
    "lt_0_25",
    "lt_0_50",
    "lt_0_75",
    "lt_0_90",
    "gte_0_90",
)
SAFE_ERROR_TYPES = (
    "audit_unavailable",
    "authentication_failed",
    "backpressure",
    "inference_failed",
    "internal_error",
    "invalid_request",
    "model_load_failed",
    "not_found",
    "queue_timeout",
    "rate_limited",
    "redaction_failed",
    "request_timeout",
    "service_draining",
    "service_not_ready",
    "service_unavailable",
    "startup_timeout",
)


@dataclass(frozen=True)
class TelemetryIdentity:
    deployment_environment: str
    service_version: str
    model_version: str
    policy_version: str


class TelemetryAttributeGuard:
    """Reject every key and value not derived from a registered bounded vocabulary."""

    def __init__(self, config: ObservabilityConfig, identity: TelemetryIdentity) -> None:
        self._allowed = frozenset(config.allowed_attribute_keys)
        self._identity = identity

    def validate(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(attributes, Mapping) or not set(attributes) <= self._allowed:
            raise ValueError("telemetry attributes contain a non-allowlisted key")
        return {key: self._validate_value(key, value) for key, value in attributes.items()}

    def _validate_value(self, key: str, value: Any) -> Any:
        allowed_strings: dict[str, tuple[str, ...]] = {
            "deployment.device": SAFE_DEVICES,
            "deployment.environment": (self._identity.deployment_environment,),
            "error.type": SAFE_ERROR_TYPES,
            "http.request.method": SAFE_HTTP_METHODS,
            "http.route": SAFE_ENDPOINTS,
            "model.uncertainty.bucket": SAFE_UNCERTAINTY_BUCKETS,
            "model.version": (self._identity.model_version,),
            "policy.version": (self._identity.policy_version,),
            "privacy.redaction.category": REGISTERED_DETECTORS,
            "route.action": SAFE_ACTIONS,
            "route.processing_status": SAFE_PROCESSING_STATUSES,
            "service.name": (SERVICE_NAME,),
            "service.version": (self._identity.service_version,),
            "telemetry.outcome": SAFE_OUTCOMES,
        }
        if key == "http.response.status_code":
            if not isinstance(value, int) or isinstance(value, bool) or not 100 <= value <= 599:
                raise ValueError("telemetry HTTP status must be an integer from 100 to 599")
            return value
        if key not in allowed_strings or not isinstance(value, str):
            raise ValueError(f"telemetry attribute {key} has an invalid type")
        if value not in allowed_strings[key]:
            raise ValueError(f"telemetry attribute {key} has an unregistered value")
        return value


class RoutingDistributionMonitor:
    """Thread-safe rolling action ratios and total-variation shift from a synthetic reference."""

    def __init__(self, config: ObservabilityConfig) -> None:
        self._values: deque[str] = deque(maxlen=config.rolling_window_observations)
        self._minimum = config.minimum_observations
        self._reference = dict(config.reference_distribution)
        self._lock = threading.Lock()

    def add(self, action: str) -> None:
        if action not in SAFE_ACTIONS:
            raise ValueError("routing distribution received an unregistered action")
        with self._lock:
            self._values.append(action)

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            values = tuple(self._values)
        total = len(values)
        if total == 0:
            return {
                "observations": 0,
                "human_review_ratio": None,
                "security_escalation_ratio": None,
                "distribution_shift": None,
            }
        counts = Counter(values)
        human_ratio = counts["human_review"] / total
        security_ratio = counts["security_queue"] / total
        shift = None
        if total >= self._minimum:
            observed = {
                "human_review": human_ratio,
                "security_queue": security_ratio,
            }
            shift = 0.5 * sum(
                abs(observed[action] - self._reference[action]) for action in SAFE_ACTIONS
            )
        return {
            "observations": total,
            "human_review_ratio": human_ratio,
            "security_escalation_ratio": security_ratio,
            "distribution_shift": shift,
        }


class GovernedTelemetry:
    """OpenTelemetry instruments that accept only aggregate or bounded metadata."""

    def __init__(
        self,
        config: ObservabilityConfig,
        identity: TelemetryIdentity,
        *,
        meter: Meter,
        tracer: Tracer,
        shutdown: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self.identity = identity
        self.guard = TelemetryAttributeGuard(config, identity)
        self.distribution = RoutingDistributionMonitor(config)
        self._tracer = tracer
        self._shutdown = shutdown
        self._closed = False
        self._selected_device: str | None = None
        self._state_lock = threading.Lock()

        self._requests = meter.create_counter(
            "banking_router.requests",
            unit="{request}",
            description="Completed HTTP requests at the bounded service endpoint categories.",
        )
        self._errors = meter.create_counter(
            "banking_router.errors",
            unit="{error}",
            description="HTTP and governed-processing errors by bounded category.",
        )
        self._request_duration = meter.create_histogram(
            "banking_router.request.duration",
            unit="s",
            description="Completed HTTP request duration in seconds.",
        )
        self._model_load_duration = meter.create_histogram(
            "banking_router.model.load.duration",
            unit="s",
            description="Model and policy loading duration in seconds.",
        )
        self._decisions = meter.create_counter(
            "banking_router.routing.decisions",
            unit="{decision}",
            description="Governed routing decisions by bounded action.",
        )
        self._human_reviews = meter.create_counter(
            "banking_router.routing.human_reviews",
            unit="{decision}",
            description="Decisions sent to human review.",
        )
        self._security_escalations = meter.create_counter(
            "banking_router.routing.security_escalations",
            unit="{decision}",
            description="Decisions escalated to the security queue.",
        )
        self._redactions = meter.create_counter(
            "banking_router.privacy.redactions",
            unit="{finding}",
            description="Structured PII findings by registered category; values are never emitted.",
        )
        self._uncertainty = meter.create_histogram(
            "banking_router.model.uncertainty",
            unit="1",
            description="Experimental calibrated maximum-probability observations.",
        )
        meter.create_observable_gauge(
            "banking_router.runtime.selected_device",
            callbacks=[self._selected_device_observations],
            unit="1",
            description="One for the explicitly selected runtime device after readiness.",
        )
        meter.create_observable_gauge(
            "banking_router.routing.human_review_ratio",
            callbacks=[self._human_review_ratio_observations],
            unit="1",
            description="Human-review share in the bounded rolling action window.",
        )
        meter.create_observable_gauge(
            "banking_router.routing.security_escalation_ratio",
            callbacks=[self._security_ratio_observations],
            unit="1",
            description="Security-escalation share in the bounded rolling action window.",
        )
        meter.create_observable_gauge(
            "banking_router.routing.distribution_shift",
            callbacks=[self._distribution_shift_observations],
            unit="1",
            description="Total-variation shift from the synthetic Module 10 action reference.",
        )
        meter.create_observable_gauge(
            "banking_router.routing.window_observations",
            callbacks=[self._window_size_observations],
            unit="{decision}",
            description="Decision count currently held by the bounded rolling window.",
        )

    @contextmanager
    def request_span(self, endpoint: str, method: str) -> Iterator[Span]:
        attributes = self.guard.validate(
            {
                "http.route": endpoint,
                "http.request.method": method,
            }
        )
        with self._tracer.start_as_current_span(
            "banking_router.http.request", attributes=attributes
        ) as span:
            yield span

    @contextmanager
    def model_load_span(self, expected_device: str) -> Iterator[Span]:
        attributes = self.guard.validate({"deployment.device": expected_device})
        with self._tracer.start_as_current_span(
            "banking_router.model.load", attributes=attributes
        ) as span:
            yield span

    def finish_request(
        self,
        span: Span,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        duration_seconds: float,
        error_type: str | None,
    ) -> None:
        outcome = "error" if error_type is not None else "success"
        request_attributes = self.guard.validate(
            {
                "http.route": endpoint,
                "http.request.method": method,
                "http.response.status_code": status_code,
                "telemetry.outcome": outcome,
            }
        )
        self._requests.add(1, request_attributes)
        self._request_duration.record(
            _finite_nonnegative(duration_seconds, "duration_seconds"),
            self.guard.validate(
                {
                    "http.route": endpoint,
                    "telemetry.outcome": outcome,
                }
            ),
        )
        span.set_attributes(request_attributes)
        if error_type is not None:
            error_attributes = self.guard.validate(
                {"http.route": endpoint, "error.type": error_type}
            )
            self._errors.add(1, error_attributes)
            span.set_attribute("error.type", error_attributes["error.type"])
            span.set_status(Status(StatusCode.ERROR))

    def record_model_load(
        self,
        span: Span,
        *,
        duration_seconds: float,
        device: str,
        error_type: str | None,
    ) -> None:
        outcome = "error" if error_type is not None else "success"
        attributes: dict[str, Any] = {
            "deployment.device": device,
            "telemetry.outcome": outcome,
        }
        if error_type is not None:
            attributes["error.type"] = error_type
        checked = self.guard.validate(attributes)
        self._model_load_duration.record(
            _finite_nonnegative(duration_seconds, "duration_seconds"), checked
        )
        span.set_attributes(checked)
        if error_type is not None:
            self._errors.add(1, self.guard.validate({"error.type": error_type}))
            span.set_status(Status(StatusCode.ERROR))

    def set_selected_device(self, device: str) -> None:
        checked = self.guard.validate({"deployment.device": device})
        with self._state_lock:
            self._selected_device = checked["deployment.device"]

    def record_audit_event(self, event: Mapping[str, Any]) -> None:
        """Extract only fields already constrained by the Module 9 audit schema."""

        model = _mapping(event, "model")
        privacy = _mapping(event, "privacy")
        routing = _mapping(event, "routing")
        action = str(routing.get("action"))
        predicted_intent = str(model.get("predicted_intent"))
        if predicted_intent == "redaction_failed":
            processing_status = "redaction_failed"
        elif predicted_intent == "model_inference_failed":
            processing_status = "inference_failed"
        else:
            processing_status = "completed"

        decision_attributes = self.guard.validate({"route.action": action})
        self._decisions.add(1, decision_attributes)
        if action == "human_review":
            self._human_reviews.add(1)
        elif action == "security_queue":
            self._security_escalations.add(1)
        else:
            raise ValueError("audit event contains an unregistered routing action")
        self.distribution.add(action)

        counts = privacy.get("pii_type_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("audit event PII counts are invalid")
        for category, count in counts.items():
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError("audit event PII counts must be positive integers")
            self._redactions.add(
                count,
                self.guard.validate({"privacy.redaction.category": str(category)}),
            )

        uncertainty_score = model.get("uncertainty_score")
        if uncertainty_score is not None:
            score = _unit_float(uncertainty_score, "uncertainty_score")
            self._uncertainty.record(score)
        bucket = _uncertainty_bucket(uncertainty_score)
        span_attributes = self.guard.validate(
            {
                "route.action": action,
                "route.processing_status": processing_status,
                "model.uncertainty.bucket": bucket,
            }
        )
        current_span = trace.get_current_span()
        current_span.set_attributes(span_attributes)

        if processing_status != "completed":
            error_type = processing_status
            self._errors.add(1, self.guard.validate({"error.type": error_type}))
            current_span.set_attribute("error.type", error_type)
            current_span.set_status(Status(StatusCode.ERROR))

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        if self._shutdown is not None:
            self._shutdown()

    def _selected_device_observations(self, _: Any) -> list[Observation]:
        with self._state_lock:
            device = self._selected_device
        if device is None:
            return []
        return [Observation(1, self.guard.validate({"deployment.device": device}))]

    def _human_review_ratio_observations(self, _: Any) -> list[Observation]:
        value = self.distribution.snapshot()["human_review_ratio"]
        return [] if value is None else [Observation(float(value))]

    def _security_ratio_observations(self, _: Any) -> list[Observation]:
        value = self.distribution.snapshot()["security_escalation_ratio"]
        return [] if value is None else [Observation(float(value))]

    def _distribution_shift_observations(self, _: Any) -> list[Observation]:
        value = self.distribution.snapshot()["distribution_shift"]
        return [] if value is None else [Observation(float(value))]

    def _window_size_observations(self, _: Any) -> list[Observation]:
        value = int(self.distribution.snapshot()["observations"] or 0)
        return [Observation(value)]


class ObservingAuditStore:
    """Decorate an audit store; persist first, then emit only validated metadata fields."""

    def __init__(self, delegate: AuditStore, telemetry: GovernedTelemetry) -> None:
        self._delegate = delegate
        self._telemetry = telemetry

    def append(self, event: dict[str, Any]) -> None:
        self._delegate.append(event)
        self._telemetry.record_audit_event(event)

    def close(self) -> None:
        self._delegate.close()


def bounded_endpoint(path: str) -> str:
    return {
        "/health/live": "health_live",
        "/health/ready": "health_ready",
        "/v1/route": "route_v1",
    }.get(path, "other")


def bounded_method(method: str) -> str:
    normalized = method.upper()
    return normalized if normalized in {"GET", "POST"} else "OTHER"


def bounded_http_error(status_code: int) -> str | None:
    if status_code < 400:
        return None
    return {
        400: "invalid_request",
        401: "authentication_failed",
        404: "not_found",
        413: "invalid_request",
        422: "invalid_request",
        429: "rate_limited",
        500: "internal_error",
        503: "service_unavailable",
        504: "request_timeout",
    }.get(status_code, "internal_error")


def _uncertainty_bucket(value: Any) -> str:
    if value is None:
        return "missing_or_invalid"
    score = _unit_float(value, "uncertainty_score")
    if score < 0.25:
        return "lt_0_25"
    if score < 0.50:
        return "lt_0_50"
    if score < 0.75:
        return "lt_0_75"
    if score < 0.90:
        return "lt_0_90"
    return "gte_0_90"


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"audit event {key} must be a mapping")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _unit_float(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be no greater than one")
    return result
