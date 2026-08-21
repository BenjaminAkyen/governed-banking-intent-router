from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from governed_banking.api import GovernedService, ServiceConfig
from governed_banking.audit import AuditConfig
from governed_banking.audit_store import require_audit_store
from governed_banking.deployment_config import (
    CapacityProfile,
    DeploymentProfile,
    LifecycleProfile,
)
from governed_banking.deployment_service import (
    CapacityController,
    CapacityRejected,
    FixedWindowRateLimiter,
    LoadedDeployment,
    create_deployment_app,
)
from governed_banking.inference import Prediction
from governed_banking.policy import RoutingPolicyConfig
from governed_banking.privacy import PrivacyConfig

pytestmark = pytest.mark.integration

DEV_TOKEN = "module14-development-token-000000000001"
GATEWAY_ASSERTION = "module14-gateway-assertion-0000000001"


@dataclass
class RecordingAuditStore:
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


@dataclass
class ControlledPredictor:
    intent: str = "card_arrival"
    delay_seconds: float = 0.0
    released: bool = False

    def predict(self, _: str) -> Prediction:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return Prediction(
            predicted_intent=self.intent,
            model_seed=42,
            uncertainty_signal="max_probability",
            uncertainty_score=0.99,
            model_artifact_sha256=(
                "b78b2dafce23a633c86b962cb672b8b91c8e07c5308debf124a73ffbcb21cca8"
            ),
        )

    def release_accelerator_cache(self) -> None:
        self.released = True


@dataclass
class LoaderHarness:
    selected_device: str
    predictor: ControlledPredictor = field(default_factory=ControlledPredictor)
    store: RecordingAuditStore = field(default_factory=RecordingAuditStore)
    loaded: LoadedDeployment | None = None

    def __call__(self, profile, _environment, _audit_factory) -> LoadedDeployment:
        legacy = ServiceConfig.from_yaml(profile.legacy_service_config_path)
        privacy = PrivacyConfig.from_yaml(legacy.privacy_config_path)
        routing = RoutingPolicyConfig.from_yaml(legacy.routing_config_path)
        audit = AuditConfig.from_yaml(legacy.audit_config_path)
        service = GovernedService(
            config=legacy,
            predictor=self.predictor,
            privacy_config=privacy,
            routing_config=routing,
            audit_config=audit,
            audit_sink=self.store,  # type: ignore[arg-type] - structural append contract
        )
        self.loaded = LoadedDeployment(
            service=service,
            audit_store=self.store,
            predictor=self.predictor,
            selected_device=self.selected_device,
            runtime_metadata={"selected": self.selected_device},
        )
        return self.loaded


def _native_profile() -> DeploymentProfile:
    return DeploymentProfile.from_yaml(Path("configs/deployment/native-mps.yaml"))


def _container_profile() -> DeploymentProfile:
    return DeploymentProfile.from_yaml(Path("configs/deployment/linux-cpu.yaml"))


def _native_environment() -> dict[str, str]:
    return {"GOVERNED_BANKING_DEV_API_TOKEN": DEV_TOKEN}


def _container_environment() -> dict[str, str]:
    return {
        "GOVERNED_BANKING_GATEWAY_ASSERTION": GATEWAY_ASSERTION,
        "GOVERNED_BANKING_RELEASE_ID": "module10-lora-seed42-research",
        "GOVERNED_BANKING_ROLLBACK_REFERENCE": (
            "registry.example.invalid/router@sha256:" + "a" * 64
        ),
    }


def test_liveness_readiness_versioned_route_and_graceful_close() -> None:
    profile = _native_profile()
    harness = LoaderHarness(selected_device="mps")
    app = create_deployment_app(
        profile,
        environment=_native_environment(),
        service_loader=harness,
    )
    correlation_id = str(uuid.uuid4())

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        response = client.post(
            "/v1/route",
            json={"message": "When will my card arrive?"},
            headers={
                "Authorization": f"Bearer {DEV_TOKEN}",
                "X-Correlation-ID": correlation_id,
            },
        )

        assert live.status_code == 200
        assert live.json()["status"] == "alive"
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        assert ready.json()["selected_device"] == "mps"
        assert response.status_code == 200
        assert response.headers["x-request-id"] == response.json()["request_id"]
        assert response.headers["x-correlation-id"] == correlation_id
        assert response.headers["x-model-release"] == profile.model_release_id
        assert len(harness.store.events) == 1

    assert harness.loaded is not None
    assert harness.loaded.closed is True
    assert harness.predictor.released is True
    assert harness.store.closed is True
    assert app.state.deployment_runtime.graceful_shutdown_completed is True


def test_failed_model_loading_is_live_but_not_ready() -> None:
    profile = _native_profile()

    def fail_loader(*_):
        raise RuntimeError("synthetic startup failure")

    app = create_deployment_app(
        profile,
        environment=_native_environment(),
        service_loader=fail_loader,
    )
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        route = client.post(
            "/v1/route",
            json={"message": "Where is my card?"},
            headers={"Authorization": f"Bearer {DEV_TOKEN}"},
        )

        assert live.status_code == 200
        assert ready.status_code == 503
        assert ready.json()["lifecycle_phase"] == "failed"
        assert route.status_code == 503
        assert route.json() == {"detail": "service_not_ready"}


def test_late_model_load_after_startup_timeout_is_closed() -> None:
    base = _native_profile()
    profile = replace(
        base,
        lifecycle=LifecycleProfile(
            startup_timeout_seconds=0.01,
            graceful_shutdown_seconds=0.1,
        ),
    )
    harness = LoaderHarness(selected_device="mps")

    def delayed_loader(*args):
        time.sleep(0.05)
        return harness(*args)

    app = create_deployment_app(
        profile,
        environment=_native_environment(),
        service_loader=delayed_loader,
    )
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["lifecycle_phase"] == "failed"
        time.sleep(0.08)

    assert harness.loaded is not None
    assert harness.loaded.closed is True


def test_development_authentication_rejects_legacy_and_incorrect_tokens() -> None:
    harness = LoaderHarness(selected_device="mps")
    app = create_deployment_app(
        _native_profile(),
        environment=_native_environment(),
        service_loader=harness,
    )

    with TestClient(app) as client:
        for authorization in (
            "",
            "Bearer incorrect-token-value-000000000000",
            "Bearer test-only-bearer-token-000000000001",
        ):
            response = client.post(
                "/v1/route",
                json={"message": "Where is my card?"},
                headers={"Authorization": authorization},
            )
            assert response.status_code == 401
    assert harness.store.events == []


def test_gateway_authentication_requires_origin_assertion_subject_and_issuer() -> None:
    profile = _container_profile()
    harness = LoaderHarness(selected_device="cpu")
    app = create_deployment_app(
        profile,
        environment=_container_environment(),
        service_loader=harness,
    )
    valid_headers = {
        "X-Governed-Gateway-Assertion": GATEWAY_ASSERTION,
        "X-Authenticated-Subject": "engineer@example.test",
        "X-Authenticated-Issuer": "https://identity.example.invalid/",
    }

    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/route",
                json={"message": "Where is my card?"},
                headers=valid_headers,
            ).status_code
            == 200
        )
        for missing in valid_headers:
            headers = dict(valid_headers)
            headers.pop(missing)
            assert (
                client.post(
                    "/v1/route",
                    json={"message": "Where is my card?"},
                    headers=headers,
                ).status_code
                == 401
            )


def test_invalid_correlation_identifier_is_rejected_without_inference() -> None:
    harness = LoaderHarness(selected_device="mps")
    app = create_deployment_app(
        _native_profile(),
        environment=_native_environment(),
        service_loader=harness,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/route",
            json={"message": "Where is my card?"},
            headers={
                "Authorization": f"Bearer {DEV_TOKEN}",
                "X-Correlation-ID": "customer-email@example.test",
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "invalid_correlation_id"}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert harness.store.events == []
    uuid.UUID(response.headers["x-request-id"])
    uuid.UUID(response.headers["x-correlation-id"])


def test_request_timeout_retains_capacity_until_background_inference_finishes() -> None:
    base = _native_profile()
    profile = replace(
        base,
        capacity=CapacityProfile(
            request_timeout_seconds=0.01,
            queue_timeout_seconds=0.01,
            maximum_concurrent_requests=1,
            maximum_queue_depth=0,
            rate_limit_requests=30,
            rate_limit_window_seconds=60,
        ),
    )
    harness = LoaderHarness(
        selected_device="mps",
        predictor=ControlledPredictor(delay_seconds=0.05),
    )
    app = create_deployment_app(
        profile,
        environment=_native_environment(),
        service_loader=harness,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/route",
            json={"message": "Where is my card?"},
            headers={"Authorization": f"Bearer {DEV_TOKEN}"},
        )
        immediate_response = client.post(
            "/v1/route",
            json={"message": "Where is my card?"},
            headers={"Authorization": f"Bearer {DEV_TOKEN}"},
        )
        time.sleep(0.08)

        assert response.status_code == 504
        assert immediate_response.status_code == 503
        assert immediate_response.json() == {"detail": "backpressure"}
        assert len(harness.store.events) == 1


def test_route_rate_limit_returns_retry_after() -> None:
    base = _native_profile()
    profile = replace(
        base,
        capacity=replace(base.capacity, rate_limit_requests=2),
    )
    app = create_deployment_app(
        profile,
        environment=_native_environment(),
        service_loader=LoaderHarness(selected_device="mps"),
    )

    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {DEV_TOKEN}"}
        assert (
            client.post(
                "/v1/route", json={"message": "Where is my card?"}, headers=headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/route", json={"message": "Where is my card?"}, headers=headers
            ).status_code
            == 200
        )
        limited = client.post("/v1/route", json={"message": "Where is my card?"}, headers=headers)

    assert limited.status_code == 429
    assert limited.json() == {"detail": "rate_limited"}
    assert int(limited.headers["retry-after"]) >= 1


def test_capacity_controller_backpressure_queue_timeout_and_drain() -> None:
    async def scenario() -> None:
        controller = CapacityController(
            maximum_concurrent_requests=1,
            maximum_queue_depth=0,
            queue_timeout_seconds=0.01,
        )
        await controller.acquire()
        with pytest.raises(CapacityRejected, match="backpressure"):
            await controller.acquire()
        drain = asyncio.create_task(controller.stop_accepting_and_drain(0.1))
        await asyncio.sleep(0)
        await controller.release()
        assert await drain is True

        queued_controller = CapacityController(
            maximum_concurrent_requests=1,
            maximum_queue_depth=1,
            queue_timeout_seconds=0.01,
        )
        await queued_controller.acquire()
        with pytest.raises(CapacityRejected, match="queue_timeout"):
            await queued_controller.acquire()
        await queued_controller.release()

    asyncio.run(scenario())


def test_fixed_window_rate_limit_returns_retry_after_and_resets() -> None:
    async def scenario() -> None:
        current = [100.0]
        limiter = FixedWindowRateLimiter(
            maximum_requests=2,
            window_seconds=10,
            clock=lambda: current[0],
        )

        assert await limiter.admit("principal") == (True, 0)
        assert await limiter.admit("principal") == (True, 0)
        admitted, retry_after = await limiter.admit("principal")
        assert admitted is False
        assert retry_after == 10
        current[0] += 10
        assert await limiter.admit("principal") == (True, 0)

    asyncio.run(scenario())


def test_audit_store_protocol_rejects_incomplete_plugins() -> None:
    class IncompleteStore:
        def append(self, _: dict[str, Any]) -> None:
            pass

    with pytest.raises(TypeError, match="append.*close"):
        require_audit_store(IncompleteStore())
