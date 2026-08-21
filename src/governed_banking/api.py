"""Secure-by-default FastAPI integration for the governed shadow router."""

from __future__ import annotations

import json
import math
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from governed_banking.audit import (
    AuditConfig,
    AuditSink,
    ModelAuditContext,
    PrivacyAuditContext,
    build_audit_event,
)
from governed_banking.data import sha256_file
from governed_banking.inference import (
    IntentPredictor,
    LoRAPredictor,
    LoRAPredictorConfig,
)
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.privacy import PrivacyConfig, redact_pii

SERVICE_CONFIG_SCHEMA_VERSION = 1
FAILURE_SENTINELS = {
    "redaction_failed": "redaction_failed",
    "inference_failed": "model_inference_failed",
}


@dataclass(frozen=True)
class ServiceConfig:
    service_version: str
    environment: str
    operating_mode: str
    config_sha256: str
    project_root: Path
    bind_host: str
    default_port: int
    allowed_hosts: tuple[str, ...]
    maximum_request_body_bytes: int
    token_environment_variable: str
    minimum_token_characters: int
    predictor: LoRAPredictorConfig
    privacy_config_path: Path
    routing_config_path: Path
    audit_config_path: Path
    cache_control: str

    @classmethod
    def from_yaml(cls, path: Path) -> ServiceConfig:
        resolved_path = path.resolve(strict=True)
        project_root = resolved_path.parent.parent
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        expected_top_level = {
            "schema_version",
            "service_version",
            "environment",
            "operating_mode",
            "network",
            "authentication",
            "model",
            "controls",
            "failure_policy",
            "response_policy",
        }
        if not isinstance(raw, dict) or set(raw) != expected_top_level:
            raise ValueError("service configuration fields differ from registration")
        if raw.get("schema_version") != SERVICE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported service configuration schema")
        if raw.get("environment") != "local_research":
            raise ValueError("Module 10 configuration is restricted to local research")
        if raw.get("operating_mode") != "shadow_review_only":
            raise ValueError("Module 10 must remain in shadow_review_only mode")
        network = _exact_mapping(
            raw,
            "network",
            {
                "bind_host",
                "default_port",
                "allowed_hosts",
                "docs_enabled",
                "cors_enabled",
                "maximum_request_body_bytes",
            },
        )
        if network.get("bind_host") != "127.0.0.1":
            raise ValueError("Module 10 must bind only to the IPv4 loopback address")
        if network.get("docs_enabled") is not False or network.get("cors_enabled") is not False:
            raise ValueError("Module 10 docs and CORS must remain disabled")
        allowed_hosts = tuple(network.get("allowed_hosts", []))
        if allowed_hosts != ("127.0.0.1", "localhost", "testserver"):
            raise ValueError("service allowed-host registration differs")
        authentication = _exact_mapping(
            raw,
            "authentication",
            {"scheme", "token_environment_variable", "minimum_token_characters"},
        )
        if authentication.get("scheme") != "bearer":
            raise ValueError("Module 10 requires Authorization bearer authentication")
        token_environment_variable = _bounded_string(
            authentication.get("token_environment_variable"),
            "authentication.token_environment_variable",
            1,
            128,
        )
        if token_environment_variable != "GOVERNED_BANKING_API_TOKEN":
            raise ValueError("unexpected API-token environment variable")
        model = _exact_mapping(
            raw,
            "model",
            {
                "seed",
                "device",
                "offline_only",
                "multiseed_config_path",
                "expected_multiseed_config_sha256",
                "manifest_path",
                "expected_manifest_file_sha256",
                "expected_manifest_sha256",
                "checkpoint_directory",
                "expected_checkpoint_files_sha256",
                "calibration_report_path",
                "expected_calibration_file_sha256",
                "expected_calibration_report_sha256",
                "temperature",
                "uncertainty_signal",
                "uncertainty_status",
                "model_cache_directory",
            },
        )
        expected_checkpoint_hashes = model.get("expected_checkpoint_files_sha256")
        if not isinstance(expected_checkpoint_hashes, dict) or set(expected_checkpoint_hashes) != {
            "README.md",
            "adapter_config.json",
            "adapter_model.safetensors",
        }:
            raise ValueError("checkpoint hash registration is invalid")
        controls = _exact_mapping(
            raw,
            "controls",
            {"privacy_config_path", "routing_config_path", "audit_config_path"},
        )
        failure_policy = _exact_mapping(
            raw,
            "failure_policy",
            {"redaction_failure", "inference_failure", "audit_failure"},
        )
        if failure_policy != {
            "redaction_failure": "human_review",
            "inference_failure": "human_review",
            "audit_failure": "reject_response",
        }:
            raise ValueError("service failure policy must remain fail closed")
        response_policy = _exact_mapping(
            raw,
            "response_policy",
            {
                "include_message_text",
                "include_redacted_text",
                "include_uncertainty_score",
                "cache_control",
            },
        )
        if (
            response_policy.get("include_message_text") is not False
            or response_policy.get("include_redacted_text") is not False
            or response_policy.get("include_uncertainty_score") is not False
            or response_policy.get("cache_control") != "no-store"
        ):
            raise ValueError("service response policy permits sensitive or experimental fields")
        predictor = LoRAPredictorConfig(
            seed=_bounded_int(model.get("seed"), "model.seed", 0, 2**31 - 1),
            device=_bounded_string(model.get("device"), "model.device", 1, 16),
            offline_only=_strict_bool(model.get("offline_only"), "model.offline_only"),
            multiseed_config_path=_project_path(
                project_root, model.get("multiseed_config_path"), "model.multiseed_config_path"
            ),
            expected_multiseed_config_sha256=_sha256(
                model.get("expected_multiseed_config_sha256"),
                "model.expected_multiseed_config_sha256",
            ),
            manifest_path=_project_path(
                project_root, model.get("manifest_path"), "model.manifest_path"
            ),
            expected_manifest_file_sha256=_sha256(
                model.get("expected_manifest_file_sha256"),
                "model.expected_manifest_file_sha256",
            ),
            expected_manifest_sha256=_sha256(
                model.get("expected_manifest_sha256"), "model.expected_manifest_sha256"
            ),
            checkpoint_directory=_project_path(
                project_root, model.get("checkpoint_directory"), "model.checkpoint_directory"
            ),
            expected_checkpoint_files_sha256={
                str(name): _sha256(value, f"checkpoint.{name}")
                for name, value in expected_checkpoint_hashes.items()
            },
            calibration_report_path=_project_path(
                project_root,
                model.get("calibration_report_path"),
                "model.calibration_report_path",
            ),
            expected_calibration_file_sha256=_sha256(
                model.get("expected_calibration_file_sha256"),
                "model.expected_calibration_file_sha256",
            ),
            expected_calibration_report_sha256=_sha256(
                model.get("expected_calibration_report_sha256"),
                "model.expected_calibration_report_sha256",
            ),
            temperature=_finite_float(model.get("temperature"), "model.temperature", 0.01, 100.0),
            uncertainty_signal=_bounded_string(
                model.get("uncertainty_signal"), "model.uncertainty_signal", 1, 64
            ),
            uncertainty_status=_bounded_string(
                model.get("uncertainty_status"), "model.uncertainty_status", 1, 64
            ),
            model_cache_directory=_project_path(
                project_root,
                model.get("model_cache_directory"),
                "model.model_cache_directory",
            ),
        )
        return cls(
            service_version=_bounded_string(raw.get("service_version"), "service_version", 1, 128),
            environment="local_research",
            operating_mode="shadow_review_only",
            config_sha256=sha256_file(resolved_path),
            project_root=project_root,
            bind_host="127.0.0.1",
            default_port=_bounded_int(
                network.get("default_port"), "network.default_port", 1024, 65535
            ),
            allowed_hosts=allowed_hosts,
            maximum_request_body_bytes=_bounded_int(
                network.get("maximum_request_body_bytes"),
                "network.maximum_request_body_bytes",
                1024,
                65536,
            ),
            token_environment_variable=token_environment_variable,
            minimum_token_characters=_bounded_int(
                authentication.get("minimum_token_characters"),
                "authentication.minimum_token_characters",
                32,
                256,
            ),
            predictor=predictor,
            privacy_config_path=_project_path(
                project_root, controls.get("privacy_config_path"), "controls.privacy_config_path"
            ),
            routing_config_path=_project_path(
                project_root, controls.get("routing_config_path"), "controls.routing_config_path"
            ),
            audit_config_path=_project_path(
                project_root, controls.get("audit_config_path"), "controls.audit_config_path"
            ),
            cache_control="no-store",
        )


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4096)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value


class RouteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    service_version: str
    operating_mode: Literal["shadow_review_only"]
    processing_status: Literal["completed", "redaction_failed", "inference_failed"]
    predicted_intent: str | None
    action: Literal["security_queue", "human_review"]
    queue: str
    reason_codes: list[str]
    pii_redaction_applied: bool
    uncertainty_observation: str
    uncertainty_policy: Literal["experimental_review_only"]
    automated_action_authorized: Literal[False]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "ready"]
    service_version: str
    operating_mode: Literal["shadow_review_only"]


class RequestBodyLimitMiddleware:
    """Buffer a small JSON body and reject declared or streamed over-limit requests."""

    def __init__(self, app: ASGIApp, *, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > self.maximum_bytes:
            await _send_problem(send, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "request_too_large")
            return
        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.maximum_bytes:
                await _send_problem(
                    send, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "request_too_large"
                )
                return
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay() -> Message:
            try:
                return next(iterator)
            except StopIteration:
                return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, *, cache_control: str) -> None:
        self.app = app
        self.cache_control = cache_control.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"cache-control", self.cache_control),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


@dataclass
class GovernedService:
    config: ServiceConfig
    predictor: IntentPredictor
    privacy_config: PrivacyConfig
    routing_config: RoutingPolicyConfig
    audit_config: AuditConfig
    audit_sink: AuditSink

    def route(self, message: str) -> RouteResponse:
        try:
            redaction = redact_pii(self.privacy_config, message)
        except (TypeError, ValueError, RuntimeError):
            privacy_context = PrivacyAuditContext.failed(
                self.privacy_config, input_character_count=len(message)
            )
            return self._fallback(
                status_name="redaction_failed",
                sentinel=FAILURE_SENTINELS["redaction_failed"],
                privacy=privacy_context,
            )
        try:
            prediction = self.predictor.predict(redaction.redacted_text)
        except Exception:
            privacy_context = PrivacyAuditContext.from_redaction(redaction)
            return self._fallback(
                status_name="inference_failed",
                sentinel=FAILURE_SENTINELS["inference_failed"],
                privacy=privacy_context,
            )
        routing_input = RoutingInput(
            predicted_intent=prediction.predicted_intent,
            model_seed=prediction.model_seed,
            uncertainty_signal=prediction.uncertainty_signal,
            uncertainty_score=prediction.uncertainty_score,
            pii_type_counts=redaction.pii_type_counts,
            redaction_succeeded=True,
        )
        decision = route_request(self.routing_config, routing_input)
        return self._persist_and_respond(
            status_name="completed",
            visible_intent=prediction.predicted_intent,
            model_artifact_sha256=prediction.model_artifact_sha256,
            routing_input=routing_input,
            privacy=PrivacyAuditContext.from_redaction(redaction),
            decision=decision,
        )

    def _fallback(
        self,
        *,
        status_name: Literal["redaction_failed", "inference_failed"],
        sentinel: str,
        privacy: PrivacyAuditContext,
    ) -> RouteResponse:
        routing_input = RoutingInput(
            predicted_intent=sentinel,
            model_seed=self.config.predictor.seed,
            uncertainty_signal=None,
            uncertainty_score=None,
            pii_type_counts=privacy.pii_type_counts,
            redaction_succeeded=privacy.redaction_succeeded,
        )
        decision = route_request(self.routing_config, routing_input)
        return self._persist_and_respond(
            status_name=status_name,
            visible_intent=None,
            model_artifact_sha256=self.config.predictor.expected_checkpoint_files_sha256[
                "adapter_model.safetensors"
            ],
            routing_input=routing_input,
            privacy=privacy,
            decision=decision,
        )

    def _persist_and_respond(
        self,
        *,
        status_name: Literal["completed", "redaction_failed", "inference_failed"],
        visible_intent: str | None,
        model_artifact_sha256: str,
        routing_input: RoutingInput,
        privacy: PrivacyAuditContext,
        decision: Any,
    ) -> RouteResponse:
        event = build_audit_event(
            self.audit_config,
            self.routing_config,
            ModelAuditContext(model_artifact_sha256),
            routing_input,
            privacy,
            decision,
        )
        try:
            self.audit_sink.append(event)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="audit_unavailable",
            ) from error
        return RouteResponse(
            request_id=event["event_id"],
            service_version=self.config.service_version,
            operating_mode="shadow_review_only",
            processing_status=status_name,
            predicted_intent=visible_intent,
            action=decision.action,
            queue=decision.queue,
            reason_codes=list(decision.reason_codes),
            pii_redaction_applied=privacy.redaction_applied,
            uncertainty_observation=decision.uncertainty_observation,
            uncertainty_policy="experimental_review_only",
            automated_action_authorized=False,
        )


def create_app(service: GovernedService, *, api_token: str) -> FastAPI:
    _validate_api_token(api_token, service.config.minimum_token_characters)
    app = FastAPI(
        title="Governed Banking Intent Router",
        version=service.config.service_version,
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(service.config.allowed_hosts),
    )
    app.add_middleware(SecurityHeadersMiddleware, cache_control=service.config.cache_control)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=service.config.maximum_request_body_bytes,
    )
    bearer = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> None:
        if (
            credentials is None
            or credentials.scheme.casefold() != "bearer"
            or not secrets.compare_digest(credentials.credentials, api_token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    @app.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service_version=service.config.service_version,
            operating_mode="shadow_review_only",
        )

    @app.get("/health/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        return HealthResponse(
            status="ready",
            service_version=service.config.service_version,
            operating_mode="shadow_review_only",
        )

    router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])

    @router.post("/route", response_model=RouteResponse)
    def governed_route(payload: RouteRequest) -> RouteResponse:
        return service.route(payload.message)

    app.include_router(router)
    return app


def build_runtime_app(
    config_path: Path = Path("configs/service.yaml"),
    *,
    environment: Mapping[str, str] | None = None,
) -> FastAPI:
    config = ServiceConfig.from_yaml(config_path)
    api_token = load_api_token(config, environment=environment)
    privacy_config = PrivacyConfig.from_yaml(config.privacy_config_path)
    routing_config = RoutingPolicyConfig.from_yaml(config.routing_config_path)
    audit_config = AuditConfig.from_yaml(config.audit_config_path)
    if privacy_config.maximum_input_characters > config.maximum_request_body_bytes:
        raise ValueError("request-body limit cannot be lower than the privacy character limit")
    threshold = routing_config.thresholds.get(config.predictor.seed)
    if threshold is None or threshold.signal != config.predictor.uncertainty_signal:
        raise ValueError("service predictor and experimental routing signal differ")
    predictor = LoRAPredictor(config.predictor)
    service = GovernedService(
        config=config,
        predictor=predictor,
        privacy_config=privacy_config,
        routing_config=routing_config,
        audit_config=audit_config,
        audit_sink=AuditSink(config.project_root, audit_config),
    )
    return create_app(service, api_token=api_token)


def load_api_token(
    config: ServiceConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    source = os.environ if environment is None else environment
    token = source.get(config.token_environment_variable, "")
    _validate_api_token(token, config.minimum_token_characters)
    return token


def _validate_api_token(token: str, minimum_characters: int) -> None:
    if (
        not isinstance(token, str)
        or len(token) < minimum_characters
        or len(token) > 512
        or token.strip() != token
    ):
        raise ValueError("API bearer token is missing or does not meet the registered length")


async def _send_problem(send: Send, status_code: int, detail: str) -> None:
    payload = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-content-type-options", b"nosniff"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def _content_length(scope: Scope) -> int | None:
    values = [value for key, value in scope.get("headers", []) if key.lower() == b"content-length"]
    if not values:
        return None
    if len(values) != 1:
        return 2**63 - 1
    try:
        result = int(values[0])
    except ValueError:
        return 2**63 - 1
    return result if result >= 0 else 2**63 - 1


def _exact_mapping(
    parent: Mapping[str, Any], key: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"service {key} fields differ from registration")
    return value


def _project_path(project_root: Path, value: Any, name: str) -> Path:
    relative = Path(_bounded_string(value, name, 1, 512))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe project-relative path")
    result = project_root / relative
    if not result.is_relative_to(project_root):
        raise ValueError(f"{name} escapes the project root")
    return result


def _bounded_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum or value.strip() != value:
        raise ValueError(f"{name} must be a bounded non-blank string")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _finite_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _sha256(value: Any, name: str) -> str:
    result = _bounded_string(value, name, 64, 64)
    if any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result
