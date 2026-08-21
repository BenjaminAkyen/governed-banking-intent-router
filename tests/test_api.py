from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from governed_banking.api import (
    GovernedService,
    ServiceConfig,
    create_app,
    load_api_token,
)
from governed_banking.audit import AuditConfig
from governed_banking.inference import Prediction
from governed_banking.policy import RoutingPolicyConfig
from governed_banking.privacy import PrivacyConfig

pytestmark = pytest.mark.integration

TEST_TOKEN = "test-only-bearer-token-000000000001"
AUTHORIZATION = {"Authorization": f"Bearer {TEST_TOKEN}"}


@dataclass
class RecordingPredictor:
    intent: str = "card_arrival"
    score: float = 0.99
    should_fail: bool = False
    observed_inputs: list[str] = field(default_factory=list)

    def predict(self, redacted_text: str) -> Prediction:
        self.observed_inputs.append(redacted_text)
        if self.should_fail:
            raise RuntimeError("synthetic inference failure")
        return Prediction(
            predicted_intent=self.intent,
            model_seed=42,
            uncertainty_signal="max_probability",
            uncertainty_score=self.score,
            model_artifact_sha256="b78b2dafce23a633c86b962cb672b8b91c8e07c5308debf124a73ffbcb21cca8",
        )


class FailingAuditSink:
    def append(self, _: dict[str, Any]) -> None:
        raise OSError("synthetic audit outage")


class RecordingAuditSink:
    """Portable test double; production permission controls are tested separately."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            )
            handle.write("\n")

    def read_validated(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


def _service(
    tmp_path: Path,
    *,
    predictor: RecordingPredictor | None = None,
    audit_sink: Any | None = None,
) -> tuple[GovernedService, RecordingPredictor, Any]:
    config = ServiceConfig.from_yaml(Path("configs/service.yaml"))
    privacy_config = PrivacyConfig.from_yaml(config.privacy_config_path)
    routing_config = RoutingPolicyConfig.from_yaml(config.routing_config_path)
    audit_config = AuditConfig.from_yaml(config.audit_config_path)
    selected_predictor = predictor or RecordingPredictor()
    selected_sink = audit_sink or RecordingAuditSink(tmp_path / "audit" / "events.jsonl")
    return (
        GovernedService(
            config=config,
            predictor=selected_predictor,
            privacy_config=privacy_config,
            routing_config=routing_config,
            audit_config=audit_config,
            audit_sink=selected_sink,
        ),
        selected_predictor,
        selected_sink,
    )


def _client(
    tmp_path: Path,
    *,
    predictor: RecordingPredictor | None = None,
    audit_sink: Any | None = None,
    raise_server_exceptions: bool = True,
) -> tuple[TestClient, RecordingPredictor, Any]:
    service, selected_predictor, selected_sink = _service(
        tmp_path, predictor=predictor, audit_sink=audit_sink
    )
    return (
        TestClient(
            create_app(service, api_token=TEST_TOKEN),
            raise_server_exceptions=raise_server_exceptions,
        ),
        selected_predictor,
        selected_sink,
    )


def test_health_is_minimal_and_interactive_docs_are_disabled(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {
        "status": "ok",
        "service_version": "module10-shadow-api-v1",
        "operating_mode": "shadow_review_only",
    }
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer incorrect-token-value"}])
def test_route_requires_valid_bearer_authentication(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    client, predictor, sink = _client(tmp_path)

    response = client.post("/v1/route", json={"message": "Where is my card?"}, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert predictor.observed_inputs == []
    assert not sink.path.exists()


def test_original_pii_is_redacted_before_inference_and_absent_from_outputs(tmp_path: Path) -> None:
    client, predictor, sink = _client(tmp_path)
    original = "Email alex@example.test about my replacement card"

    response = client.post("/v1/route", json={"message": original}, headers=AUTHORIZATION)

    assert response.status_code == 200
    body = response.json()
    assert predictor.observed_inputs == ["Email [EMAIL] about my replacement card"]
    assert body["processing_status"] == "completed"
    assert body["action"] == "human_review"
    assert body["automated_action_authorized"] is False
    assert "uncertainty_score" not in body
    serialized_response = response.text
    serialized_audit = sink.path.read_text(encoding="utf-8")
    assert original not in serialized_response
    assert predictor.observed_inputs[0] not in serialized_response
    assert original not in serialized_audit
    assert predictor.observed_inputs[0] not in serialized_audit
    assert "message_hash" not in serialized_audit


def test_exposed_authentication_secret_routes_to_security(tmp_path: Path) -> None:
    client, predictor, sink = _client(tmp_path)
    secret = "TestSecret-0001"

    response = client.post(
        "/v1/route",
        json={"message": f"Password: {secret}"},
        headers=AUTHORIZATION,
    )

    assert response.status_code == 200
    assert predictor.observed_inputs == ["Password: [AUTHENTICATION_SECRET]"]
    assert response.json()["action"] == "security_queue"
    assert response.json()["queue"] == "security_operations"
    assert "EXPOSED_AUTHENTICATION_SECRET" in response.json()["reason_codes"]
    assert secret not in sink.path.read_text(encoding="utf-8")


def test_security_intent_overrides_high_confidence(tmp_path: Path) -> None:
    client, _, _ = _client(
        tmp_path, predictor=RecordingPredictor(intent="compromised_card", score=0.999)
    )

    response = client.post(
        "/v1/route", json={"message": "This card payment is not mine"}, headers=AUTHORIZATION
    )

    assert response.status_code == 200
    assert response.json()["action"] == "security_queue"
    assert "SECURITY_INTENT_OVERRIDE" in response.json()["reason_codes"]
    assert response.json()["automated_action_authorized"] is False


@pytest.mark.parametrize("score", [0.10, 0.99])
def test_experimental_threshold_never_authorizes_suggestion(tmp_path: Path, score: float) -> None:
    client, _, _ = _client(tmp_path, predictor=RecordingPredictor(score=score))

    response = client.post(
        "/v1/route", json={"message": "When will my card arrive?"}, headers=AUTHORIZATION
    )

    assert response.status_code == 200
    assert response.json()["action"] == "human_review"
    assert response.json()["uncertainty_policy"] == "experimental_review_only"
    assert response.json()["automated_action_authorized"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {"message": "Where is my card?", "role": "admin"},
    ],
)
def test_invalid_input_is_generic_and_does_not_echo_values(
    tmp_path: Path, payload: dict[str, str]
) -> None:
    client, predictor, sink = _client(tmp_path)

    response = client.post("/v1/route", json=payload, headers=AUTHORIZATION)

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_request"}
    assert all(value not in response.text for value in payload.values() if value)
    assert predictor.observed_inputs == []
    assert not sink.path.exists()


def test_oversized_body_is_rejected_before_prediction(tmp_path: Path) -> None:
    client, predictor, sink = _client(tmp_path)

    response = client.post(
        "/v1/route",
        content=json.dumps({"message": "x" * 9000}),
        headers={**AUTHORIZATION, "Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "request_too_large"}
    assert predictor.observed_inputs == []
    assert not sink.path.exists()


def test_untrusted_host_is_rejected(tmp_path: Path) -> None:
    client, predictor, _ = _client(tmp_path)

    response = client.post(
        "/v1/route",
        json={"message": "Where is my card?"},
        headers={**AUTHORIZATION, "Host": "attacker.example"},
    )

    assert response.status_code == 400
    assert predictor.observed_inputs == []


def test_security_headers_are_centralized(tmp_path: Path) -> None:
    client, _, _ = _client(tmp_path)

    response = client.get("/health/live")

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"


def test_redaction_failure_fails_closed_and_is_audited_without_text(tmp_path: Path) -> None:
    client, predictor, sink = _client(tmp_path)
    original = "contains\x00null"

    response = client.post("/v1/route", json={"message": original}, headers=AUTHORIZATION)

    assert response.status_code == 200
    assert response.json()["processing_status"] == "redaction_failed"
    assert response.json()["predicted_intent"] is None
    assert response.json()["action"] == "human_review"
    assert "REDACTION_FAILURE" in response.json()["reason_codes"]
    assert predictor.observed_inputs == []
    audit_event = sink.read_validated()[0]
    assert audit_event["privacy"]["redaction_succeeded"] is False
    assert original not in json.dumps(audit_event)


def test_inference_failure_fails_closed_and_is_audited(tmp_path: Path) -> None:
    predictor = RecordingPredictor(should_fail=True)
    client, _, sink = _client(tmp_path, predictor=predictor)

    response = client.post(
        "/v1/route", json={"message": "Where is my card?"}, headers=AUTHORIZATION
    )

    assert response.status_code == 200
    assert response.json()["processing_status"] == "inference_failed"
    assert response.json()["predicted_intent"] is None
    assert response.json()["action"] == "human_review"
    assert "UNSUPPORTED_INTENT" in response.json()["reason_codes"]
    assert sink.read_validated()[0]["model"]["predicted_intent"] == "model_inference_failed"


def test_audit_failure_rejects_the_route_response(tmp_path: Path) -> None:
    client, _, _ = _client(
        tmp_path,
        audit_sink=FailingAuditSink(),
        raise_server_exceptions=False,
    )

    response = client.post(
        "/v1/route", json={"message": "Where is my card?"}, headers=AUTHORIZATION
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "audit_unavailable"}


def test_service_configuration_is_hash_bound_and_token_is_environment_only() -> None:
    config = ServiceConfig.from_yaml(Path("configs/service.yaml"))
    if not config.predictor.checkpoint_directory.is_dir():
        pytest.skip("registered Module 10 adapter checkpoint is a local, unredistributed artifact")

    experiment, labels = config.predictor.validate_sources()

    assert experiment.encoder.repository == "FacebookAI/roberta-base"
    assert len(labels) == 77
    assert (
        load_api_token(config, environment={"GOVERNED_BANKING_API_TOKEN": TEST_TOKEN}) == TEST_TOKEN
    )
    assert TEST_TOKEN not in repr(config)
    with pytest.raises(ValueError, match="missing"):
        load_api_token(config, environment={})
