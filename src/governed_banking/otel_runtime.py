"""OpenTelemetry SDK providers for explicit OTLP/gRPC export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from governed_banking.deployment_config import DeploymentProfile
from governed_banking.observability import (
    SERVICE_NAME,
    GovernedTelemetry,
    TelemetryAttributeGuard,
    TelemetryIdentity,
)
from governed_banking.observability_config import ObservabilityConfig


@dataclass
class _Providers:
    meter_provider: MeterProvider
    tracer_provider: TracerProvider
    closed: bool = False

    def shutdown(self) -> None:
        if self.closed:
            return
        self.meter_provider.force_flush()
        self.tracer_provider.force_flush()
        self.meter_provider.shutdown()
        self.tracer_provider.shutdown()
        self.closed = True


def build_otlp_telemetry(
    config: ObservabilityConfig,
    deployment: DeploymentProfile,
    *,
    service_version: str,
    policy_version: str,
    environment: Mapping[str, str],
) -> GovernedTelemetry:
    """Create isolated SDK providers; no globals or automatic HTTP instrumentation are used."""

    endpoint = environment.get(config.endpoint_environment_variable, "")
    insecure = validate_otlp_endpoint(endpoint, deployment)
    identity = TelemetryIdentity(
        deployment_environment=deployment.environment,
        service_version=service_version,
        model_version=deployment.model_release_id,
        policy_version=policy_version,
    )
    guard = TelemetryAttributeGuard(config, identity)
    resource_attributes = guard.validate(
        {
            "service.name": SERVICE_NAME,
            "service.version": identity.service_version,
            "deployment.environment": identity.deployment_environment,
            "model.version": identity.model_version,
            "policy.version": identity.policy_version,
        }
    )
    resource = Resource(resource_attributes)
    timeout = config.export_timeout_milliseconds / 1000.0
    metric_exporter = OTLPMetricExporter(
        endpoint=endpoint,
        insecure=insecure,
        headers=(),
        timeout=timeout,
    )
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=config.metric_export_interval_milliseconds,
        export_timeout_millis=config.export_timeout_milliseconds,
    )
    meter_provider = MeterProvider(
        metric_readers=[metric_reader],
        resource=resource,
        shutdown_on_exit=False,
    )
    tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=insecure,
                headers=(),
                timeout=timeout,
            )
        )
    )
    providers = _Providers(meter_provider=meter_provider, tracer_provider=tracer_provider)
    meter = metrics.get_meter(
        config.instrumentation_name,
        config.instrumentation_version,
        meter_provider=meter_provider,
    )
    tracer = trace.get_tracer(
        config.instrumentation_name,
        config.instrumentation_version,
        tracer_provider=tracer_provider,
    )
    return GovernedTelemetry(
        config,
        identity,
        meter=meter,
        tracer=tracer,
        shutdown=providers.shutdown,
    )


def validate_otlp_endpoint(endpoint: str, deployment: DeploymentProfile) -> bool:
    """Validate that export uses a credential-free origin and safe transport boundary."""

    invalid_endpoint = (
        not isinstance(endpoint, str)
        or not 1 <= len(endpoint) <= 512
        or endpoint.strip() != endpoint
    )
    if invalid_endpoint:
        raise ValueError("OTLP endpoint is required and must be a bounded URL")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("OTLP endpoint must be a credential-free HTTP(S) origin")
    insecure = parsed.scheme == "http"
    if insecure and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("plaintext OTLP is allowed only to a same-host Collector")
    if deployment.platform == "linux_container" and insecure and parsed.hostname != "127.0.0.1":
        raise ValueError("container plaintext OTLP must use a localhost Collector sidecar")
    return insecure
