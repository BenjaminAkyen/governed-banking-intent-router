from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from governed_banking.api import GovernedService, ServiceConfig
from governed_banking.audit import AuditConfig
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.deployment_service import LoadedDeployment
from governed_banking.inference import Prediction
from governed_banking.observed_deployment_service import create_observed_deployment_app
from governed_banking.policy import RoutingPolicyConfig
from governed_banking.privacy import PrivacyConfig

pytestmark = pytest.mark.integration

DEV_TOKEN = "module15-development-token-000000000001"


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

    def predict(self, _: str) -> Prediction:
        return Prediction(
            predicted_intent=self.intent,
            model_seed=42,
            uncertainty_signal="max_probability",
            uncertainty_score=0.99,
            model_artifact_sha256="b" * 64,
        )

    def release_accelerator_cache(self) -> None:
        return None


@dataclass
class LoaderHarness:
    selected_device: str = "mps"
    store: RecordingAuditStore = field(default_factory=RecordingAuditStore)

    def __call__(self, profile, _environment, _audit_factory) -> LoadedDeployment:
        service_config = ServiceConfig.from_yaml(profile.legacy_service_config_path)
        service = GovernedService(
            config=service_config,
            predictor=ControlledPredictor(),
            privacy_config=PrivacyConfig.from_yaml(service_config.privacy_config_path),
            routing_config=RoutingPolicyConfig.from_yaml(service_config.routing_config_path),
            audit_config=AuditConfig.from_yaml(service_config.audit_config_path),
            audit_sink=self.store,  # type: ignore[arg-type] - structural append contract
        )
        return LoadedDeployment(
            service=service,
            audit_store=self.store,  # type: ignore[arg-type] - structural store contract
            predictor=service.predictor,
            selected_device=self.selected_device,
            runtime_metadata={"selected": self.selected_device},
        )


@dataclass
class RecordingTelemetry:
    requests: list[dict[str, Any]] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    selected_device: str | None = None
    closed: bool = False

    @contextmanager
    def request_span(self, endpoint: str, method: str):
        yield {"endpoint": endpoint, "method": method}

    @contextmanager
    def model_load_span(self, expected_device: str):
        yield {"expected_device": expected_device}

    def finish_request(self, _span: object, **values: Any) -> None:
        self.requests.append(values)

    def record_model_load(self, _span: object, **values: Any) -> None:
        self.loads.append(values)

    def set_selected_device(self, device: str) -> None:
        self.selected_device = device

    def record_audit_event(self, event: dict[str, Any]) -> None:
        self.audit_events.append(event)

    def close(self) -> None:
        self.closed = True


def test_observed_wrapper_covers_lifecycle_http_and_metadata_audit() -> None:
    profile = DeploymentProfile.from_yaml(Path("configs/deployment/native-mps.yaml"))
    harness = LoaderHarness()
    telemetry = RecordingTelemetry()
    app = create_observed_deployment_app(
        profile,
        environment={"GOVERNED_BANKING_DEV_API_TOKEN": DEV_TOKEN},
        service_loader=harness,
        telemetry=telemetry,  # type: ignore[arg-type] - lifecycle telemetry contract
    )
    correlation_id = str(uuid.uuid4())

    with TestClient(app) as client:
        response = client.post(
            "/v1/route",
            json={"message": "Email alex@example.test about my card"},
            headers={
                "Authorization": f"Bearer {DEV_TOKEN}",
                "X-Correlation-ID": correlation_id,
            },
        )
        missing = client.get("/not-a-real-route")

    assert response.status_code == 200
    assert missing.status_code == 404
    assert telemetry.selected_device == "mps"
    assert len(telemetry.loads) == 1
    assert telemetry.loads[0]["error_type"] is None
    assert len(telemetry.audit_events) == 1
    assert {request["endpoint"] for request in telemetry.requests} == {"route_v1", "other"}
    assert {request["error_type"] for request in telemetry.requests} == {None, "not_found"}
    serialized_requests = str(telemetry.requests)
    assert correlation_id not in serialized_requests
    assert response.json()["request_id"] not in serialized_requests
    assert "alex@example.test" not in serialized_requests
    assert telemetry.closed is True


def test_observed_wrapper_records_bounded_model_load_failure() -> None:
    profile = DeploymentProfile.from_yaml(Path("configs/deployment/native-mps.yaml"))
    telemetry = RecordingTelemetry()

    def fail_loader(*_):
        time.sleep(0.001)
        raise RuntimeError("failure text must not enter telemetry")

    app = create_observed_deployment_app(
        profile,
        environment={"GOVERNED_BANKING_DEV_API_TOKEN": DEV_TOKEN},
        service_loader=fail_loader,
        telemetry=telemetry,  # type: ignore[arg-type] - lifecycle telemetry contract
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert len(telemetry.loads) == 1
    assert telemetry.loads[0]["error_type"] == "model_load_failed"
    assert "failure text" not in str(telemetry.loads)
    assert telemetry.closed is True
