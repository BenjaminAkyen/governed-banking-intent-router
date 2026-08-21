"""Lifecycle-managed Module 14 FastAPI service for native and container profiles."""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import secrets
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from governed_banking.api import (
    GovernedService,
    RequestBodyLimitMiddleware,
    RouteRequest,
    RouteResponse,
    SecurityHeadersMiddleware,
    ServiceConfig,
)
from governed_banking.audit import AuditConfig
from governed_banking.audit_store import AuditStore, LocalJsonlAuditStore, require_audit_store
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.policy import RoutingPolicyConfig
from governed_banking.portable_inference import PortableLoRAPredictor
from governed_banking.privacy import PrivacyConfig
from governed_banking.runtime_evidence import RuntimeProfile

DEPLOYMENT_SERVICE_VERSION = "module14-deployment-api-v1"
LifecyclePhase = Literal["starting", "loading", "ready", "draining", "failed", "stopped"]
AuditStoreFactory = Callable[[Path, AuditConfig], AuditStore]


class LiveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["alive"]
    service_version: Literal["module14-deployment-api-v1"]
    profile_name: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    service_version: Literal["module14-deployment-api-v1"]
    profile_name: str
    lifecycle_phase: LifecyclePhase
    model_release_id: str
    expected_device: str
    selected_device: str | None
    rollback_ready: bool


@dataclass(frozen=True)
class Principal:
    subject: str
    issuer: str

    @property
    def rate_limit_key(self) -> str:
        return f"{self.issuer}\x00{self.subject}"


@dataclass
class LoadedDeployment:
    """Fully initialized service plus bounded shutdown hooks."""

    service: GovernedService
    audit_store: AuditStore
    predictor: PortableLoRAPredictor | Any
    selected_device: str
    runtime_metadata: Mapping[str, Any]
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        try:
            release = getattr(self.predictor, "release_accelerator_cache", None)
            if callable(release):
                release()
        finally:
            self.audit_store.close()
            self.closed = True


ServiceLoader = Callable[
    [DeploymentProfile, Mapping[str, str], AuditStoreFactory | None], LoadedDeployment
]


class RequestContextMiddleware:
    """Create bounded UUID request/correlation identifiers for every HTTP response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        correlation_value = _header_value(scope, b"x-correlation-id")
        if correlation_value is None:
            correlation_id = str(uuid.uuid4())
        else:
            try:
                correlation_id = _canonical_uuid(correlation_value, "X-Correlation-ID")
            except ValueError:
                await _send_problem(
                    send,
                    status.HTTP_400_BAD_REQUEST,
                    "invalid_correlation_id",
                    request_id=str(uuid.uuid4()),
                    correlation_id=str(uuid.uuid4()),
                )
                return
        state = scope.setdefault("state", {})
        state["request_id"] = str(uuid.uuid4())
        state["correlation_id"] = correlation_id

        async def send_with_context(message: Message) -> None:
            if message["type"] == "http.response.start":
                excluded = {b"x-request-id", b"x-correlation-id"}
                headers = [
                    pair for pair in message.get("headers", []) if pair[0].lower() not in excluded
                ]
                headers.extend(
                    [
                        (b"x-request-id", state["request_id"].encode("ascii")),
                        (b"x-correlation-id", state["correlation_id"].encode("ascii")),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_context)


class RequestAuthenticator:
    """Separate local bearer authentication from trusted-gateway origin authentication."""

    def __init__(
        self,
        profile: DeploymentProfile,
        environment: Mapping[str, str],
    ) -> None:
        self._profile = profile
        authentication = profile.authentication
        secret = environment.get(authentication.secret_environment_variable, "")
        if (
            not isinstance(secret, str)
            or not authentication.minimum_secret_characters <= len(secret) <= 512
            or secret.strip() != secret
        ):
            raise ValueError("deployment authentication secret is missing or invalid")
        self._secret = secret

    def authenticate(self, request: Request) -> Principal:
        authentication = self._profile.authentication
        if authentication.mode == "development_bearer":
            authorization = request.headers.get("authorization", "")
            scheme, separator, credential = authorization.partition(" ")
            if (
                separator != " "
                or scheme.casefold() != "bearer"
                or not secrets.compare_digest(credential, self._secret)
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="unauthorized",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return Principal(subject="local-development", issuer="local-development")

        assert authentication.assertion_header is not None
        assert authentication.subject_header is not None
        assert authentication.issuer_header is not None
        assertion = request.headers.get(authentication.assertion_header, "")
        subject = request.headers.get(authentication.subject_header, "")
        issuer = request.headers.get(authentication.issuer_header, "")
        if (
            not secrets.compare_digest(assertion, self._secret)
            or issuer not in authentication.allowed_issuers
            or not _bounded_identity(subject)
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
        return Principal(subject=subject, issuer=issuer)


@dataclass
class _RateWindow:
    started_at: float
    count: int


class FixedWindowRateLimiter:
    """Per-authenticated-principal, per-process fixed-window admission control."""

    def __init__(
        self,
        *,
        maximum_requests: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._maximum_requests = maximum_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._windows: dict[str, _RateWindow] = {}
        self._lock = asyncio.Lock()

    async def admit(self, key: str) -> tuple[bool, int]:
        now = self._clock()
        async with self._lock:
            window = self._windows.get(key)
            if window is None or now - window.started_at >= self._window_seconds:
                self._windows[key] = _RateWindow(now, 1)
                self._discard_expired(now)
                return True, 0
            if window.count >= self._maximum_requests:
                remaining = max(1, math.ceil(self._window_seconds - (now - window.started_at)))
                return False, remaining
            window.count += 1
            return True, 0

    def _discard_expired(self, now: float) -> None:
        if len(self._windows) <= 10000:
            return
        self._windows = {
            key: value
            for key, value in self._windows.items()
            if now - value.started_at < self._window_seconds
        }


class CapacityRejected(RuntimeError):
    pass


class CapacityController:
    """Bound concurrent work, queue depth and queue wait before inference starts."""

    def __init__(
        self,
        *,
        maximum_concurrent_requests: int,
        maximum_queue_depth: int,
        queue_timeout_seconds: float,
    ) -> None:
        self._semaphore = asyncio.Semaphore(maximum_concurrent_requests)
        self._maximum_concurrent = maximum_concurrent_requests
        self._maximum_queue_depth = maximum_queue_depth
        self._queue_timeout = queue_timeout_seconds
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._accepting = True
        self._in_flight = 0
        self._queued = 0

    async def acquire(self) -> None:
        queued = False
        async with self._lock:
            if not self._accepting:
                raise CapacityRejected("service_draining")
            if self._in_flight >= self._maximum_concurrent:
                if self._queued >= self._maximum_queue_depth:
                    raise CapacityRejected("backpressure")
                self._queued += 1
                queued = True
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._queue_timeout)
        except TimeoutError as error:
            raise CapacityRejected("queue_timeout") from error
        finally:
            if queued:
                async with self._lock:
                    self._queued -= 1
        async with self._lock:
            if not self._accepting:
                self._semaphore.release()
                raise CapacityRejected("service_draining")
            self._in_flight += 1

    async def release(self) -> None:
        async with self._condition:
            if self._in_flight <= 0:
                raise RuntimeError("capacity release without an acquired slot")
            self._in_flight -= 1
            self._semaphore.release()
            self._condition.notify_all()

    async def stop_accepting_and_drain(self, timeout_seconds: float) -> bool:
        async with self._condition:
            self._accepting = False
            if self._in_flight == 0:
                return True
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._in_flight == 0),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                return False
            return True

    async def snapshot(self) -> dict[str, int | bool]:
        async with self._lock:
            return {
                "accepting": self._accepting,
                "in_flight": self._in_flight,
                "queued": self._queued,
            }


@dataclass
class DeploymentRuntime:
    profile: DeploymentProfile
    controller: CapacityController
    rollback_ready: bool
    phase: LifecyclePhase = "starting"
    loaded: LoadedDeployment | None = None
    startup_failure: str | None = None
    graceful_shutdown_completed: bool | None = None

    @property
    def ready(self) -> bool:
        return self.phase == "ready" and self.loaded is not None


def create_deployment_app(
    profile: DeploymentProfile,
    *,
    environment: Mapping[str, str] | None = None,
    service_loader: ServiceLoader | None = None,
    audit_store_factory: AuditStoreFactory | None = None,
) -> FastAPI:
    """Build an app that remains live while exposing honest model readiness."""

    selected_environment = dict(os.environ if environment is None else environment)
    rollback_ready = validate_release_environment(profile, selected_environment)
    authenticator = RequestAuthenticator(profile, selected_environment)
    loader = service_loader or load_deployment
    controller = CapacityController(
        maximum_concurrent_requests=profile.capacity.maximum_concurrent_requests,
        maximum_queue_depth=profile.capacity.maximum_queue_depth,
        queue_timeout_seconds=profile.capacity.queue_timeout_seconds,
    )
    rate_limiter = FixedWindowRateLimiter(
        maximum_requests=profile.capacity.rate_limit_requests,
        window_seconds=profile.capacity.rate_limit_window_seconds,
    )
    runtime = DeploymentRuntime(
        profile=profile,
        controller=controller,
        rollback_ready=rollback_ready,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.phase = "loading"
        try:
            runtime.loaded = await asyncio.wait_for(
                asyncio.to_thread(
                    loader,
                    profile,
                    selected_environment,
                    audit_store_factory,
                ),
                timeout=profile.lifecycle.startup_timeout_seconds,
            )
            require_audit_store(runtime.loaded.audit_store)
            if runtime.loaded.selected_device != profile.expected_device:
                raise RuntimeError("loaded service selected a different deployment device")
        except Exception as error:  # noqa: BLE001 - readiness records only a bounded category
            runtime.phase = "failed"
            runtime.startup_failure = type(error).__name__
        else:
            runtime.phase = "ready"
        yield
        runtime.phase = "draining"
        runtime.graceful_shutdown_completed = await controller.stop_accepting_and_drain(
            profile.lifecycle.graceful_shutdown_seconds
        )
        if runtime.loaded is not None:
            await asyncio.to_thread(runtime.loaded.close)
        runtime.phase = "stopped"

    app = FastAPI(
        title="Governed Banking Intent Router",
        version=DEPLOYMENT_SERVICE_VERSION,
        debug=False,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.deployment_runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(profile.allowed_hosts))
    app.add_middleware(SecurityHeadersMiddleware, cache_control="no-store")
    app.add_middleware(
        RequestBodyLimitMiddleware,
        maximum_bytes=ServiceConfig.from_yaml(
            profile.legacy_service_config_path
        ).maximum_request_body_bytes,
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    @app.get("/health/live", response_model=LiveResponse)
    async def live() -> LiveResponse:
        return LiveResponse(
            status="alive",
            service_version=DEPLOYMENT_SERVICE_VERSION,
            profile_name=profile.profile_name,
        )

    @app.get("/health/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse | JSONResponse:
        payload = ReadyResponse(
            status="ready" if runtime.ready else "not_ready",
            service_version=DEPLOYMENT_SERVICE_VERSION,
            profile_name=profile.profile_name,
            lifecycle_phase=runtime.phase,
            model_release_id=profile.model_release_id,
            expected_device=profile.expected_device,
            selected_device=(
                runtime.loaded.selected_device if runtime.loaded is not None else None
            ),
            rollback_ready=runtime.rollback_ready,
        )
        if runtime.ready:
            return payload
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))

    async def authenticate(request: Request) -> Principal:
        return authenticator.authenticate(request)

    router = APIRouter(prefix="/v1")

    @router.post("/route", response_model=RouteResponse)
    async def governed_route(
        payload: RouteRequest,
        request: Request,
        response: Response,
        principal: Principal = Depends(authenticate),  # noqa: B008
    ) -> RouteResponse:
        if not runtime.ready or runtime.loaded is None:
            raise HTTPException(status_code=503, detail="service_not_ready")
        admitted, retry_after = await rate_limiter.admit(principal.rate_limit_key)
        if not admitted:
            raise HTTPException(
                status_code=429,
                detail="rate_limited",
                headers={"Retry-After": str(retry_after)},
            )
        try:
            await controller.acquire()
        except CapacityRejected as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        release_on_return = True
        try:
            if not runtime.ready or runtime.loaded is None:
                raise HTTPException(status_code=503, detail="service_draining")
            inference_task = asyncio.create_task(
                asyncio.to_thread(runtime.loaded.service.route, payload.message)
            )
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(inference_task),
                    timeout=profile.capacity.request_timeout_seconds,
                )
            except TimeoutError as error:
                release_on_return = False
                inference_task.add_done_callback(
                    lambda task: _release_capacity_after_background_task(task, controller)
                )
                raise HTTPException(status_code=504, detail="request_timeout") from error
        finally:
            if release_on_return:
                await controller.release()
        request.state.request_id = result.request_id
        response.headers["X-Request-ID"] = result.request_id
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        response.headers["X-Model-Release"] = profile.model_release_id
        return result

    app.include_router(router)
    return app


def load_deployment(
    profile: DeploymentProfile,
    environment: Mapping[str, str],
    audit_store_factory: AuditStoreFactory | None,
) -> LoadedDeployment:
    """Load the registered service on the explicit real backend; never fall back."""

    validate_execution_platform(profile, environment)
    validate_release_environment(profile, environment)
    legacy = ServiceConfig.from_yaml(profile.legacy_service_config_path)
    runtime_profile = RuntimeProfile.from_yaml(profile.runtime_profile_path)
    privacy = PrivacyConfig.from_yaml(legacy.privacy_config_path)
    routing = RoutingPolicyConfig.from_yaml(legacy.routing_config_path)
    audit_config = AuditConfig.from_yaml(legacy.audit_config_path)
    predictor: PortableLoRAPredictor | None = None
    audit_store: AuditStore | None = None
    try:
        predictor = PortableLoRAPredictor(legacy.predictor, runtime_profile)
        if predictor.runtime.selected != profile.expected_device:
            raise RuntimeError("loaded model selected a different deployment device")
        factory = audit_store_factory or LocalJsonlAuditStore
        audit_store = require_audit_store(factory(profile.project_root, audit_config))
        service = GovernedService(
            config=legacy,
            predictor=predictor,
            privacy_config=privacy,
            routing_config=routing,
            audit_config=audit_config,
            audit_sink=audit_store,  # type: ignore[arg-type] - structural append contract
        )
        return LoadedDeployment(
            service=service,
            audit_store=audit_store,
            predictor=predictor,
            selected_device=predictor.runtime.selected,
            runtime_metadata=predictor.runtime.to_dict(),
        )
    except Exception:
        if predictor is not None:
            predictor.release_accelerator_cache()
        if audit_store is not None:
            audit_store.close()
        raise


def validate_execution_platform(
    profile: DeploymentProfile, environment: Mapping[str, str]
) -> None:
    observed = platform.system()
    declared_container = environment.get("GOVERNED_BANKING_CONTAINER", "") == "1"
    if profile.platform == "native_macos":
        if observed != "Darwin" or declared_container:
            raise RuntimeError("native MPS profile requires non-containerized macOS")
        if environment.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("native MPS profile prohibits PyTorch CPU fallback")
    elif observed != "Linux" or not declared_container:
        raise RuntimeError("Linux deployment profile requires an explicitly declared container")


def validate_release_environment(
    profile: DeploymentProfile, environment: Mapping[str, str]
) -> bool:
    current = environment.get(profile.current_release_environment_variable)
    if current is None and profile.environment == "native_development":
        current = profile.model_release_id
    if current != profile.model_release_id:
        raise ValueError("current release environment does not match the registered model release")
    rollback = environment.get(profile.rollback_reference_environment_variable, "")
    if profile.rollback_reference_required:
        if not 1 <= len(rollback) <= 512 or rollback.strip() != rollback:
            raise ValueError("container deployment requires a bounded rollback image reference")
        if rollback == current:
            raise ValueError("rollback reference must identify a different immutable revision")
        return True
    return bool(rollback and rollback != current)


def create_app_from_environment() -> FastAPI:
    """Uvicorn factory used by both container images."""

    profile_path = Path(
        os.environ.get(
            "GOVERNED_BANKING_DEPLOYMENT_PROFILE",
            "configs/deployment/linux-cpu.yaml",
        )
    )
    return create_deployment_app(DeploymentProfile.from_yaml(profile_path))


def _bounded_identity(value: str) -> bool:
    return (
        1 <= len(value) <= 256
        and value.strip() == value
        and all(character.isprintable() and character not in "\r\n\x00" for character in value)
    )


def _release_capacity_after_background_task(
    task: asyncio.Task[RouteResponse], controller: CapacityController
) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
    asyncio.create_task(controller.release())


def _header_value(scope: Scope, name: bytes) -> str | None:
    values = [value for key, value in scope.get("headers", []) if key.lower() == name]
    if not values:
        return None
    if len(values) != 1:
        return ""
    try:
        return values[0].decode("ascii")
    except UnicodeDecodeError:
        return ""


def _canonical_uuid(value: str, name: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise ValueError(f"{name} must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUID")
    return value


async def _send_problem(
    send: Send,
    status_code: int,
    detail: str,
    *,
    request_id: str,
    correlation_id: str,
) -> None:
    payload = json.dumps({"detail": detail}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-request-id", request_id.encode("ascii")),
                (b"x-correlation-id", correlation_id.encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})
