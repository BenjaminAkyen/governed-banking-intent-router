"""Additive Module 15 observability boundary around the immutable Module 14 service."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from governed_banking.api import ServiceConfig
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.deployment_service import (
    DEPLOYMENT_SERVICE_VERSION,
    AuditStoreFactory,
    ServiceLoader,
    create_deployment_app,
)
from governed_banking.observability import (
    GovernedTelemetry,
    ObservabilityMiddleware,
    ObservingAuditStore,
)
from governed_banking.observability_config import ObservabilityConfig
from governed_banking.otel_runtime import build_otlp_telemetry
from governed_banking.policy import RoutingPolicyConfig


def create_observed_deployment_app(
    profile: DeploymentProfile,
    *,
    environment: Mapping[str, str] | None = None,
    service_loader: ServiceLoader | None = None,
    audit_store_factory: AuditStoreFactory | None = None,
    telemetry: GovernedTelemetry | None = None,
) -> FastAPI:
    """Instrument Module 14 without changing the implementation bound to its evidence."""

    selected_environment = dict(os.environ if environment is None else environment)
    selected_telemetry = telemetry or _build_registered_telemetry(
        profile, selected_environment
    )
    app = create_deployment_app(
        profile,
        environment=selected_environment,
        service_loader=service_loader,
        audit_store_factory=audit_store_factory,
    )
    base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def observed_lifespan(application: FastAPI):
        started = time.perf_counter()
        base_context = base_lifespan(application)
        entered = False
        try:
            with selected_telemetry.model_load_span(profile.expected_device) as span:
                try:
                    await base_context.__aenter__()
                    entered = True
                except Exception:
                    selected_telemetry.record_model_load(
                        span,
                        duration_seconds=time.perf_counter() - started,
                        device=profile.expected_device,
                        error_type="model_load_failed",
                    )
                    raise
                runtime = application.state.deployment_runtime
                if runtime.ready and runtime.loaded is not None:
                    observing_store = ObservingAuditStore(
                        runtime.loaded.audit_store, selected_telemetry
                    )
                    runtime.loaded.audit_store = observing_store  # type: ignore[assignment]
                    runtime.loaded.service.audit_sink = observing_store  # type: ignore[assignment]
                    selected_telemetry.set_selected_device(runtime.loaded.selected_device)
                    selected_telemetry.record_model_load(
                        span,
                        duration_seconds=time.perf_counter() - started,
                        device=runtime.loaded.selected_device,
                        error_type=None,
                    )
                else:
                    selected_telemetry.record_model_load(
                        span,
                        duration_seconds=time.perf_counter() - started,
                        device=profile.expected_device,
                        error_type=(
                            "startup_timeout"
                            if runtime.startup_failure == "TimeoutError"
                            else "model_load_failed"
                        ),
                    )
            yield
        finally:
            try:
                if entered:
                    await base_context.__aexit__(None, None, None)
            finally:
                selected_telemetry.close()

    app.router.lifespan_context = observed_lifespan
    app.add_middleware(ObservabilityMiddleware, telemetry=selected_telemetry)
    app.state.governed_telemetry = selected_telemetry
    return app


def create_observed_app_from_environment() -> FastAPI:
    """Uvicorn factory for the privacy-safe observed service profiles."""

    profile_path = Path(
        os.environ.get(
            "GOVERNED_BANKING_DEPLOYMENT_PROFILE",
            "configs/deployment/linux-cpu.yaml",
        )
    )
    environment = dict(os.environ)
    profile = DeploymentProfile.from_yaml(profile_path)
    return create_observed_deployment_app(profile, environment=environment)


def _build_registered_telemetry(
    profile: DeploymentProfile, environment: Mapping[str, str]
) -> GovernedTelemetry:
    service = ServiceConfig.from_yaml(profile.legacy_service_config_path)
    policy = RoutingPolicyConfig.from_yaml(service.routing_config_path)
    config = ObservabilityConfig.from_yaml(
        Path("configs/observability.yaml"), deployment_profile=profile
    )
    return build_otlp_telemetry(
        config,
        profile,
        service_version=DEPLOYMENT_SERVICE_VERSION,
        policy_version=policy.policy_version,
        environment=environment,
    )
