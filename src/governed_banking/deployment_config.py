"""Strict Module 14 deployment-profile registration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from governed_banking.api import ServiceConfig
from governed_banking.data import sha256_file
from governed_banking.runtime_evidence import RuntimeProfile

DEPLOYMENT_PROFILE_SCHEMA_VERSION = 1
DeploymentEnvironment = Literal["native_development", "container_staging"]
DeploymentPlatform = Literal["native_macos", "linux_container"]
AuthenticationMode = Literal["development_bearer", "trusted_gateway"]


@dataclass(frozen=True)
class AuthenticationProfile:
    mode: AuthenticationMode
    secret_environment_variable: str
    minimum_secret_characters: int
    subject_header: str | None
    issuer_header: str | None
    assertion_header: str | None
    allowed_issuers: tuple[str, ...]


@dataclass(frozen=True)
class CapacityProfile:
    request_timeout_seconds: float
    queue_timeout_seconds: float
    maximum_concurrent_requests: int
    maximum_queue_depth: int
    rate_limit_requests: int
    rate_limit_window_seconds: int


@dataclass(frozen=True)
class LifecycleProfile:
    startup_timeout_seconds: float
    graceful_shutdown_seconds: float


@dataclass(frozen=True)
class DeploymentProfile:
    """One immutable native or container service execution contract."""

    profile_name: str
    environment: DeploymentEnvironment
    platform: DeploymentPlatform
    config_path: Path
    config_sha256: str
    project_root: Path
    runtime_profile_path: Path
    runtime_profile_sha256: str
    expected_device: str
    container_required: bool
    legacy_service_config_path: Path
    legacy_service_config_sha256: str
    model_release_id: str
    model_artifact_sha256: str
    bind_host: str
    port: int
    allowed_hosts: tuple[str, ...]
    authentication: AuthenticationProfile
    capacity: CapacityProfile
    lifecycle: LifecycleProfile
    audit_backend: str
    audit_store_injection_supported: bool
    rollback_strategy: str
    current_release_environment_variable: str
    rollback_reference_environment_variable: str
    rollback_reference_required: bool

    @classmethod
    def from_yaml(cls, path: Path) -> DeploymentProfile:
        resolved = path.resolve(strict=True)
        project_root = resolved.parent.parent.parent
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "profile_name",
            "environment",
            "execution",
            "legacy_service",
            "model_release",
            "network",
            "authentication",
            "capacity",
            "lifecycle",
            "audit",
            "rollback",
            "claims",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("deployment-profile fields differ from registration")
        if raw.get("schema_version") != DEPLOYMENT_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported deployment-profile schema")
        profile_name = _bounded_string(raw.get("profile_name"), "profile_name", 1, 128)
        if profile_name not in {
            "module14-native-macos-mps-development",
            "module14-linux-cpu-container",
            "module14-linux-cuda-container",
        }:
            raise ValueError("unregistered Module 14 deployment profile")
        environment = raw.get("environment")
        if environment not in {"native_development", "container_staging"}:
            raise ValueError("deployment environment is invalid")

        execution = _exact_mapping(
            raw,
            "execution",
            {
                "platform",
                "runtime_profile_path",
                "expected_runtime_profile_sha256",
                "expected_device",
                "container_required",
                "mps_supported_in_linux_container",
            },
        )
        platform_name = execution.get("platform")
        if platform_name not in {"native_macos", "linux_container"}:
            raise ValueError("deployment platform is invalid")
        expected_device = execution.get("expected_device")
        if expected_device not in {"mps", "cpu", "cuda"}:
            raise ValueError("deployment expected device must be mps, cpu or cuda")
        if execution.get("mps_supported_in_linux_container") is not False:
            raise ValueError("standard Linux containers cannot register PyTorch MPS support")
        container_required = _strict_bool(
            execution.get("container_required"), "execution.container_required"
        )
        if platform_name == "native_macos":
            if expected_device != "mps" or container_required:
                raise ValueError("native macOS profile must use MPS without a container")
        elif expected_device not in {"cpu", "cuda"} or not container_required:
            raise ValueError("Linux container profiles must explicitly select CPU or CUDA")
        runtime_path = _repository_path(
            project_root, execution.get("runtime_profile_path"), "runtime_profile_path"
        )
        runtime_sha256 = _sha256(
            execution.get("expected_runtime_profile_sha256"),
            "expected_runtime_profile_sha256",
        )
        _require_hash(runtime_path, runtime_sha256, "runtime profile")
        runtime = RuntimeProfile.from_yaml(runtime_path)
        if runtime.device_preference != expected_device:
            raise ValueError("deployment and runtime profiles select different devices")
        if expected_device in {"mps", "cuda"} and not runtime.require_accelerator:
            raise ValueError("accelerated deployment profile must fail when unavailable")

        legacy = _exact_mapping(raw, "legacy_service", {"path", "expected_sha256"})
        legacy_path = _repository_path(project_root, legacy.get("path"), "legacy_service.path")
        legacy_sha256 = _sha256(
            legacy.get("expected_sha256"), "legacy_service.expected_sha256"
        )
        _require_hash(legacy_path, legacy_sha256, "legacy service configuration")
        legacy_config = ServiceConfig.from_yaml(legacy_path)
        if legacy_config.operating_mode != "shadow_review_only":
            raise ValueError("deployment source must retain shadow review mode")

        model_release = _exact_mapping(
            raw,
            "model_release",
            {"release_id", "expected_artifact_sha256", "champion_model"},
        )
        if model_release.get("release_id") != "module10-lora-seed42-research":
            raise ValueError("deployment release differs from the registered research model")
        model_sha256 = _sha256(
            model_release.get("expected_artifact_sha256"),
            "model_release.expected_artifact_sha256",
        )
        if (
            model_sha256
            != legacy_config.predictor.expected_checkpoint_files_sha256[
                "adapter_model.safetensors"
            ]
        ):
            raise ValueError("deployment release and legacy service bind different adapters")
        if model_release.get("champion_model") is not False:
            raise ValueError("the Module 10 LoRA research service is not the champion")

        network = _exact_mapping(raw, "network", {"bind_host", "port", "allowed_hosts"})
        bind_host = _bounded_string(network.get("bind_host"), "network.bind_host", 3, 64)
        if platform_name == "native_macos" and bind_host != "127.0.0.1":
            raise ValueError("native development must remain loopback only")
        if platform_name == "linux_container" and bind_host != "0.0.0.0":
            raise ValueError("container profiles must bind to the container interface")
        allowed_hosts = _string_tuple(network.get("allowed_hosts"), "network.allowed_hosts")
        if "testserver" not in allowed_hosts or "*" in allowed_hosts:
            raise ValueError("allowed hosts must be explicit and retain the test host")

        authentication = _authentication_profile(
            raw,
            expected_mode=(
                "development_bearer"
                if environment == "native_development"
                else "trusted_gateway"
            ),
        )
        capacity = _capacity_profile(raw)
        lifecycle = _lifecycle_profile(raw)
        audit = _exact_mapping(
            raw, "audit", {"backend", "store_injection_supported", "metadata_only_required"}
        )
        if audit.get("backend") != "local_jsonl":
            raise ValueError("the built-in Module 14 profile must use the registered JSONL store")
        if audit.get("store_injection_supported") is not True:
            raise ValueError("deployment profiles must support audit-store injection")
        if audit.get("metadata_only_required") is not True:
            raise ValueError("deployment audit stores must retain metadata-only events")

        rollback = _exact_mapping(
            raw,
            "rollback",
            {
                "strategy",
                "current_release_environment_variable",
                "rollback_reference_environment_variable",
                "rollback_reference_required",
            },
        )
        if rollback.get("strategy") != "immutable_process_or_container_revision":
            raise ValueError("hot model mutation is prohibited; rollback must replace the revision")
        rollback_required = _strict_bool(
            rollback.get("rollback_reference_required"),
            "rollback.rollback_reference_required",
        )
        if rollback_required is not (environment == "container_staging"):
            raise ValueError("container profiles must require a rollback image reference")

        claims = _exact_mapping(
            raw,
            "claims",
            {
                "operating_mode",
                "production_approved",
                "mps_in_standard_linux_container",
                "module13_gates_passed",
            },
        )
        if claims != {
            "operating_mode": "shadow_review_only",
            "production_approved": False,
            "mps_in_standard_linux_container": False,
            "module13_gates_passed": False,
        }:
            raise ValueError("deployment claims overstate the current evidence")

        return cls(
            profile_name=profile_name,
            environment=environment,
            platform=platform_name,
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            project_root=project_root,
            runtime_profile_path=runtime_path,
            runtime_profile_sha256=runtime_sha256,
            expected_device=expected_device,
            container_required=container_required,
            legacy_service_config_path=legacy_path,
            legacy_service_config_sha256=legacy_sha256,
            model_release_id="module10-lora-seed42-research",
            model_artifact_sha256=model_sha256,
            bind_host=bind_host,
            port=_bounded_int(network.get("port"), "network.port", 1024, 65535),
            allowed_hosts=allowed_hosts,
            authentication=authentication,
            capacity=capacity,
            lifecycle=lifecycle,
            audit_backend="local_jsonl",
            audit_store_injection_supported=True,
            rollback_strategy="immutable_process_or_container_revision",
            current_release_environment_variable=_environment_variable(
                rollback.get("current_release_environment_variable"),
                "rollback.current_release_environment_variable",
            ),
            rollback_reference_environment_variable=_environment_variable(
                rollback.get("rollback_reference_environment_variable"),
                "rollback.rollback_reference_environment_variable",
            ),
            rollback_reference_required=rollback_required,
        )


def _authentication_profile(
    raw: Mapping[str, Any], *, expected_mode: AuthenticationMode
) -> AuthenticationProfile:
    value = _exact_mapping(
        raw,
        "authentication",
        {
            "mode",
            "secret_environment_variable",
            "minimum_secret_characters",
            "subject_header",
            "issuer_header",
            "assertion_header",
            "allowed_issuers",
        },
    )
    if value.get("mode") != expected_mode:
        raise ValueError("authentication mode differs from the deployment environment")
    secret_variable = _environment_variable(
        value.get("secret_environment_variable"),
        "authentication.secret_environment_variable",
    )
    minimum = _bounded_int(
        value.get("minimum_secret_characters"),
        "authentication.minimum_secret_characters",
        32,
        256,
    )
    if expected_mode == "development_bearer":
        if secret_variable != "GOVERNED_BANKING_DEV_API_TOKEN":
            raise ValueError("development bearer must use its dedicated environment variable")
        if any(
            value.get(key) is not None
            for key in ("subject_header", "issuer_header", "assertion_header")
        ) or value.get("allowed_issuers") != []:
            raise ValueError("development bearer cannot register trusted-gateway fields")
        return AuthenticationProfile(
            mode="development_bearer",
            secret_environment_variable=secret_variable,
            minimum_secret_characters=minimum,
            subject_header=None,
            issuer_header=None,
            assertion_header=None,
            allowed_issuers=(),
        )
    if secret_variable != "GOVERNED_BANKING_GATEWAY_ASSERTION":
        raise ValueError("container origin authentication must use the gateway assertion secret")
    subject_header = _header_name(value.get("subject_header"), "subject_header")
    issuer_header = _header_name(value.get("issuer_header"), "issuer_header")
    assertion_header = _header_name(value.get("assertion_header"), "assertion_header")
    allowed_issuers = _string_tuple(value.get("allowed_issuers"), "allowed_issuers")
    if not allowed_issuers:
        raise ValueError("trusted-gateway authentication requires an issuer allowlist")
    return AuthenticationProfile(
        mode="trusted_gateway",
        secret_environment_variable=secret_variable,
        minimum_secret_characters=minimum,
        subject_header=subject_header,
        issuer_header=issuer_header,
        assertion_header=assertion_header,
        allowed_issuers=allowed_issuers,
    )


def _capacity_profile(raw: Mapping[str, Any]) -> CapacityProfile:
    value = _exact_mapping(
        raw,
        "capacity",
        {
            "request_timeout_seconds",
            "queue_timeout_seconds",
            "maximum_concurrent_requests",
            "maximum_queue_depth",
            "rate_limit_requests",
            "rate_limit_window_seconds",
        },
    )
    request_timeout = _bounded_float(
        value.get("request_timeout_seconds"), "request_timeout_seconds", 0.1, 120.0
    )
    queue_timeout = _bounded_float(
        value.get("queue_timeout_seconds"), "queue_timeout_seconds", 0.01, request_timeout
    )
    return CapacityProfile(
        request_timeout_seconds=request_timeout,
        queue_timeout_seconds=queue_timeout,
        maximum_concurrent_requests=_bounded_int(
            value.get("maximum_concurrent_requests"),
            "maximum_concurrent_requests",
            1,
            128,
        ),
        maximum_queue_depth=_bounded_int(
            value.get("maximum_queue_depth"), "maximum_queue_depth", 0, 4096
        ),
        rate_limit_requests=_bounded_int(
            value.get("rate_limit_requests"), "rate_limit_requests", 1, 100000
        ),
        rate_limit_window_seconds=_bounded_int(
            value.get("rate_limit_window_seconds"),
            "rate_limit_window_seconds",
            1,
            3600,
        ),
    )


def _lifecycle_profile(raw: Mapping[str, Any]) -> LifecycleProfile:
    value = _exact_mapping(
        raw, "lifecycle", {"startup_timeout_seconds", "graceful_shutdown_seconds"}
    )
    return LifecycleProfile(
        startup_timeout_seconds=_bounded_float(
            value.get("startup_timeout_seconds"), "startup_timeout_seconds", 1.0, 900.0
        ),
        graceful_shutdown_seconds=_bounded_float(
            value.get("graceful_shutdown_seconds"),
            "graceful_shutdown_seconds",
            1.0,
            300.0,
        ),
    )


def _exact_mapping(
    parent: Mapping[str, Any], key: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"deployment {key} fields differ from registration")
    return value


def _repository_path(root: Path, value: Any, name: str) -> Path:
    relative = Path(_bounded_string(value, name, 1, 512))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
    result = root / relative
    if not result.is_relative_to(root):
        raise ValueError(f"{name} escapes the repository root")
    return result


def _require_hash(path: Path, expected: str, name: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{name} hash differs from deployment registration")


def _environment_variable(value: Any, name: str) -> str:
    result = _bounded_string(value, name, 1, 128)
    if not result.replace("_", "").isalnum() or result.upper() != result:
        raise ValueError(f"{name} must be an uppercase environment-variable name")
    return result


def _header_name(value: Any, name: str) -> str:
    result = _bounded_string(value, name, 3, 128)
    if not result.lower().startswith("x-") or any(
        not (character.isalnum() or character == "-") for character in result
    ):
        raise ValueError(f"{name} must be a private HTTP header name")
    return result.lower()


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(_bounded_string(item, name, 1, 256) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


def _bounded_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or value.strip() != value
    ):
        raise ValueError(f"{name} must be a bounded non-blank string")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
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
