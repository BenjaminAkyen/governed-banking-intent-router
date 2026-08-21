"""Registered configuration and artifact checks for Module 10 service evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from governed_banking.baseline import assert_text_free_artifact, write_json_artifact
from governed_banking.data import sha256_file, stable_json_sha256

SERVICE_EVALUATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ServiceEvaluationConfig:
    experiment_name: str
    claim_scope: str
    project_root: Path
    service_config_path: Path
    fixture_path: Path
    expected_fixture_sha256: str
    warmup_requests: int
    measurement_repetitions: int
    maximum_p95_milliseconds: float
    maximum_startup_seconds: float
    fixture_provenance: str

    @classmethod
    def from_yaml(cls, path: Path) -> ServiceEvaluationConfig:
        resolved_path = path.resolve(strict=True)
        root = resolved_path.parent.parent
        raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        expected_top_level = {
            "schema_version",
            "experiment_name",
            "claim_scope",
            "service_config_path",
            "fixture_path",
            "expected_fixture_sha256",
            "warmup_requests",
            "measurement_repetitions",
            "latency",
            "acceptance_gate",
            "boundary",
        }
        if not isinstance(raw, dict) or set(raw) != expected_top_level:
            raise ValueError("service-evaluation fields differ from registration")
        if raw.get("schema_version") != SERVICE_EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported service-evaluation schema")
        if raw.get("claim_scope") != "synthetic_local_integration_not_production_validation":
            raise ValueError("service evaluation must retain its synthetic non-production scope")
        latency = _exact_mapping(
            raw, "latency", {"statistic", "maximum_milliseconds", "maximum_startup_seconds"}
        )
        if latency.get("statistic") != "p95":
            raise ValueError("service evaluation must use the registered p95 statistic")
        gates = _exact_mapping(
            raw,
            "acceptance_gate",
            {
                "require_mps",
                "require_all_http_responses_successful",
                "require_zero_suggestion_actions",
                "require_all_required_overrides",
                "require_metadata_only_audit",
                "require_restrictive_audit_permissions",
                "require_security_boundary_checks",
            },
        )
        if any(value is not True for value in gates.values()):
            raise ValueError("all registered service-evaluation controls must be required")
        boundary = _exact_mapping(
            raw,
            "boundary",
            {
                "fixture_provenance",
                "model_inference_performed",
                "official_test_access",
                "production_validation",
                "uncertainty_status",
            },
        )
        if boundary != {
            "fixture_provenance": "synthetic_authored_non_customer_data",
            "model_inference_performed": True,
            "official_test_access": False,
            "production_validation": False,
            "uncertainty_status": "experimental_review_only",
        }:
            raise ValueError("service-evaluation boundary differs from registration")
        fixture_path = _project_path(root, raw.get("fixture_path"), "fixture_path")
        expected_fixture_sha256 = _sha256(
            raw.get("expected_fixture_sha256"), "expected_fixture_sha256"
        )
        if sha256_file(fixture_path) != expected_fixture_sha256:
            raise ValueError("service fixture hash differs from registration")
        return cls(
            experiment_name=_bounded_string(raw.get("experiment_name"), "experiment_name"),
            claim_scope="synthetic_local_integration_not_production_validation",
            project_root=root,
            service_config_path=_project_path(
                root, raw.get("service_config_path"), "service_config_path"
            ),
            fixture_path=fixture_path,
            expected_fixture_sha256=expected_fixture_sha256,
            warmup_requests=_bounded_int(raw.get("warmup_requests"), "warmup_requests", 1, 100),
            measurement_repetitions=_bounded_int(
                raw.get("measurement_repetitions"), "measurement_repetitions", 1, 100
            ),
            maximum_p95_milliseconds=_finite_float(
                latency.get("maximum_milliseconds"), "latency.maximum_milliseconds", 1, 60000
            ),
            maximum_startup_seconds=_finite_float(
                latency.get("maximum_startup_seconds"),
                "latency.maximum_startup_seconds",
                1,
                600,
            ),
            fixture_provenance="synthetic_authored_non_customer_data",
        )


def validate_service_evaluation_report(
    report: Mapping[str, Any],
    *,
    config: ServiceEvaluationConfig,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if report.get("artifact_type") != "module10_shadow_api_evaluation":
        raise ValueError("unexpected service-evaluation artifact type")
    body = dict(report)
    expected_hash = body.pop("report_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("service-evaluation report content hash check failed")
    expected = {
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "service_evaluation_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "fixture_sha256": config.expected_fixture_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"service-evaluation report has an invalid {key}")
    boundary = report.get("data_boundary", {})
    if boundary != {
        "fixture_provenance": config.fixture_provenance,
        "model_inference_performed": True,
        "classification_metrics_computed": False,
        "official_test_access": False,
        "production_validation": False,
        "uncertainty_status": "experimental_review_only",
    }:
        raise ValueError("service-evaluation report overstates its data boundary")
    if report.get("acceptance_gate", {}).get("all_passed") is not True:
        raise ValueError("service-evaluation acceptance gate did not pass")
    if report.get("routing", {}).get("suggest_queue_count") != 0:
        raise ValueError("service-evaluation report contains suggestion actions")
    assert_text_free_artifact(report)


def write_service_evaluation_report(report: Mapping[str, Any], destination: Path) -> None:
    write_json_artifact(report, destination)


def _exact_mapping(
    parent: Mapping[str, Any], key: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"service-evaluation {key} fields differ from registration")
    return value


def _project_path(root: Path, value: Any, name: str) -> Path:
    relative = Path(_bounded_string(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe project-relative path")
    result = root / relative
    if not result.is_relative_to(root):
        raise ValueError(f"{name} escapes the project root")
    return result


def _bounded_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512 or value.strip() != value:
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


def _sha256(value: Any, name: str) -> str:
    result = _bounded_string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result
