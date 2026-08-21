"""Real-hardware prediction-parity evidence for Module 11."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from governed_banking.api import ServiceConfig
from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.portable_inference import PortableLoRAPredictor
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.runtime_evidence import RuntimeProfile

PARITY_CONFIG_SCHEMA_VERSION = 1
BACKEND_REPORT_SCHEMA_VERSION = 1
COMPARISON_REPORT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "cross_device_numerical_and_decision_parity_not_model_quality"
REGISTERED_BACKENDS = ("mps", "cuda")
CONFIG_KEYS = {
    "schema_version",
    "experiment_name",
    "claim_scope",
    "legacy_service_config_path",
    "expected_legacy_service_config_sha256",
    "fixture_path",
    "expected_fixture_sha256",
    "runtime_profiles",
    "reference_backend",
    "candidate_backend",
    "probability_absolute_tolerance",
    "require_identical_predicted_intents",
    "require_identical_routing_actions",
    "output_policy",
    "data_boundary",
}


@dataclass(frozen=True)
class RegisteredRuntimeProfile:
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class PredictionParityConfig:
    experiment_name: str
    config_path: Path
    config_sha256: str
    project_root: Path
    legacy_service_config_path: Path
    expected_legacy_service_config_sha256: str
    fixture_path: Path
    expected_fixture_sha256: str
    runtime_profiles: dict[str, RegisteredRuntimeProfile]
    reference_backend: str
    candidate_backend: str
    probability_absolute_tolerance: float

    @classmethod
    def from_yaml(cls, path: Path) -> PredictionParityConfig:
        resolved = path.resolve(strict=True)
        project_root = resolved.parent.parent
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
            raise ValueError("prediction-parity fields differ from the registered schema")
        if raw.get("schema_version") != PARITY_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-parity schema")
        if raw.get("experiment_name") != "module11-mps-cuda-prediction-parity":
            raise ValueError("prediction-parity experiment name differs from registration")
        if raw.get("claim_scope") != CLAIM_SCOPE:
            raise ValueError("prediction-parity claim scope differs from registration")
        if raw.get("reference_backend") != "mps" or raw.get("candidate_backend") != "cuda":
            raise ValueError("Module 11 parity must compare MPS with CUDA")
        tolerance = raw.get("probability_absolute_tolerance")
        if (
            not isinstance(tolerance, int | float)
            or isinstance(tolerance, bool)
            or not math.isfinite(float(tolerance))
            or not 0.0 < float(tolerance) <= 0.01
        ):
            raise ValueError("probability tolerance must be finite and in (0, 0.01]")
        if raw.get("require_identical_predicted_intents") is not True:
            raise ValueError("top-1 prediction parity cannot be disabled")
        if raw.get("require_identical_routing_actions") is not True:
            raise ValueError("routing-action parity cannot be disabled")
        if raw.get("output_policy") != {
            "include_input_text": False,
            "include_redacted_text": False,
            "include_probability_vector": True,
        }:
            raise ValueError("parity output policy could expose text or omit required evidence")
        if raw.get("data_boundary") != {
            "official_test_access": False,
            "customer_data_access": False,
            "fixture_is_synthetic": True,
        }:
            raise ValueError("parity data boundary differs from registration")

        service_path = _repository_path(
            project_root, raw.get("legacy_service_config_path"), "legacy service config"
        )
        fixture_path = _repository_path(project_root, raw.get("fixture_path"), "fixture")
        service_sha = _sha256(
            raw.get("expected_legacy_service_config_sha256"), "legacy service config SHA-256"
        )
        fixture_sha = _sha256(raw.get("expected_fixture_sha256"), "fixture SHA-256")
        _require_file_hash(service_path, service_sha, "legacy service config")
        _require_file_hash(fixture_path, fixture_sha, "fixture")

        profile_values = raw.get("runtime_profiles")
        if not isinstance(profile_values, dict) or tuple(profile_values) != REGISTERED_BACKENDS:
            raise ValueError("runtime profiles must register MPS followed by CUDA")
        runtime_profiles: dict[str, RegisteredRuntimeProfile] = {}
        for backend in REGISTERED_BACKENDS:
            value = profile_values.get(backend)
            if not isinstance(value, dict) or set(value) != {"path", "expected_sha256"}:
                raise ValueError(f"{backend} runtime-profile registration is invalid")
            profile_path = _repository_path(
                project_root, value.get("path"), f"{backend} runtime profile"
            )
            profile_sha = _sha256(
                value.get("expected_sha256"), f"{backend} runtime profile SHA-256"
            )
            _require_file_hash(profile_path, profile_sha, f"{backend} runtime profile")
            runtime_profile = RuntimeProfile.from_yaml(profile_path)
            if runtime_profile.device_preference != backend:
                raise ValueError(f"{backend} runtime profile selects a different backend")
            runtime_profiles[backend] = RegisteredRuntimeProfile(profile_path, profile_sha)

        return cls(
            experiment_name="module11-mps-cuda-prediction-parity",
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            project_root=project_root,
            legacy_service_config_path=service_path,
            expected_legacy_service_config_sha256=service_sha,
            fixture_path=fixture_path,
            expected_fixture_sha256=fixture_sha,
            runtime_profiles=runtime_profiles,
            reference_backend="mps",
            candidate_backend="cuda",
            probability_absolute_tolerance=float(tolerance),
        )


def run_backend_evidence(
    config: PredictionParityConfig,
    *,
    backend: str,
    report_path: Path,
    implementation_paths: dict[str, Path],
) -> dict[str, Any]:
    """Run registered synthetic cases on one real backend and write metadata-only evidence."""

    if backend not in REGISTERED_BACKENDS:
        raise ValueError("backend must be mps or cuda")
    if not implementation_paths:
        raise ValueError("implementation_paths must not be empty")
    registered_profile = config.runtime_profiles[backend]
    _require_file_hash(
        registered_profile.path,
        registered_profile.expected_sha256,
        f"{backend} runtime profile",
    )
    runtime_profile = RuntimeProfile.from_yaml(registered_profile.path)
    if runtime_profile.device_preference != backend:
        raise ValueError("runtime profile does not match requested parity backend")

    service_config = ServiceConfig.from_yaml(config.legacy_service_config_path)
    if service_config.config_sha256 != config.expected_legacy_service_config_sha256:
        raise ValueError("legacy service configuration differs from parity registration")
    privacy_config = PrivacyConfig.from_yaml(service_config.privacy_config_path)
    routing_config = RoutingPolicyConfig.from_yaml(service_config.routing_config_path)
    cases = _load_fixture(config.fixture_path, config.expected_fixture_sha256)
    predictor = PortableLoRAPredictor(service_config.predictor, runtime_profile)

    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            redaction = redact_pii(privacy_config, case["message"])
            prediction = predictor.predict(redaction.redacted_text)
            routing_input = RoutingInput(
                predicted_intent=prediction.predicted_intent,
                model_seed=prediction.model_seed,
                uncertainty_signal=prediction.uncertainty_signal,
                uncertainty_score=prediction.uncertainty_score,
                pii_type_counts=redaction.pii_type_counts,
                redaction_succeeded=True,
            )
            decision = route_request(routing_config, routing_input)
            results.append(
                {
                    "case_id": case["case_id"],
                    "predicted_index": prediction.predicted_index,
                    "predicted_intent": prediction.predicted_intent,
                    "probabilities": list(prediction.probabilities),
                    "uncertainty_signal": prediction.uncertainty_signal,
                    "uncertainty_score": prediction.uncertainty_score,
                    "routing_action": decision.action,
                    "routing_queue": decision.queue,
                    "reason_codes": list(decision.reason_codes),
                    "pii_type_counts": dict(redaction.pii_type_counts),
                }
            )
    finally:
        predictor.release_accelerator_cache()

    report: dict[str, Any] = {
        "schema_version": BACKEND_REPORT_SCHEMA_VERSION,
        "artifact_type": "module11_real_hardware_prediction_report",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": config.experiment_name,
        "claim_scope": CLAIM_SCOPE,
        "backend": backend,
        "runtime_profile": runtime_profile.to_dict(),
        "runtime": predictor.runtime.to_dict(),
        "config_sha256": config.config_sha256,
        "source_evidence": {
            "legacy_service_config_sha256": service_config.config_sha256,
            "fixture_sha256": config.expected_fixture_sha256,
            "privacy_config_sha256": privacy_config.config_sha256,
            "routing_config_sha256": routing_config.config_sha256,
            "checkpoint_files_sha256": dict(
                sorted(service_config.predictor.expected_checkpoint_files_sha256.items())
            ),
            "calibration_report_sha256": (
                service_config.predictor.expected_calibration_report_sha256
            ),
        },
        "label_names": list(predictor.label_names),
        "case_count": len(results),
        "cases": results,
        "data_boundary": {
            "fixture_is_synthetic": True,
            "input_text_persisted": False,
            "redacted_text_persisted": False,
            "official_test_access": False,
            "customer_data_access": False,
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(implementation_paths.items())
        },
    }
    report["report_sha256"] = stable_json_sha256(report)
    validate_backend_report(report, config=config, expected_backend=backend)
    _atomic_write_json(report_path, report)
    return report


def validate_backend_report(
    report: dict[str, Any],
    *,
    config: PredictionParityConfig,
    expected_backend: str,
) -> None:
    """Validate integrity, hardware identity, privacy boundary and numeric shape."""

    body = dict(report)
    expected_hash = body.pop("report_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("backend report content hash check failed")
    if report.get("schema_version") != BACKEND_REPORT_SCHEMA_VERSION:
        raise ValueError("backend report schema differs from registration")
    if report.get("artifact_type") != "module11_real_hardware_prediction_report":
        raise ValueError("backend report artifact type is invalid")
    if report.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("backend report claim scope is invalid")
    if expected_backend not in REGISTERED_BACKENDS or report.get("backend") != expected_backend:
        raise ValueError("backend report identifies a different accelerator")
    if report.get("config_sha256") != config.config_sha256:
        raise ValueError("backend report uses a different parity registration")
    runtime = report.get("runtime", {})
    if (
        runtime.get("requested") != expected_backend
        or runtime.get("selected") != expected_backend
        or runtime.get("real_hardware_observed") is not True
    ):
        raise ValueError("backend report does not prove the explicitly requested real device")
    boundary = report.get("data_boundary")
    if boundary != {
        "fixture_is_synthetic": True,
        "input_text_persisted": False,
        "redacted_text_persisted": False,
        "official_test_access": False,
        "customer_data_access": False,
    }:
        raise ValueError("backend report crossed the registered privacy or data boundary")
    if _contains_prohibited_text_keys(report):
        raise ValueError("backend report contains a prohibited message-text field")

    labels = report.get("label_names")
    cases = report.get("cases")
    if not isinstance(labels, list) or len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("backend report label taxonomy is invalid")
    if not isinstance(cases, list) or report.get("case_count") != len(cases) or not cases:
        raise ValueError("backend report case collection is invalid")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("backend report case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_ids:
            raise ValueError("backend report case IDs must be unique non-blank strings")
        case_ids.add(case_id)
        probabilities = case.get("probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(labels):
            raise ValueError("backend report probability-vector shape is invalid")
        values = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("backend report contains an invalid probability")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("backend report probability vector does not sum to one")
        predicted_index = case.get("predicted_index")
        if (
            not isinstance(predicted_index, int)
            or isinstance(predicted_index, bool)
            or not 0 <= predicted_index < len(labels)
            or case.get("predicted_intent") != labels[predicted_index]
            or predicted_index != max(range(len(values)), key=values.__getitem__)
        ):
            raise ValueError("backend report prediction is inconsistent with its probabilities")


def compare_backend_reports(
    config: PredictionParityConfig,
    *,
    reference_report_path: Path,
    candidate_report_path: Path,
    comparison_path: Path,
) -> dict[str, Any]:
    """Compare independently produced real MPS and CUDA reports against registered gates."""

    reference = json.loads(reference_report_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    validate_backend_report(reference, config=config, expected_backend=config.reference_backend)
    validate_backend_report(candidate, config=config, expected_backend=config.candidate_backend)
    if reference.get("source_evidence") != candidate.get("source_evidence"):
        raise ValueError("MPS and CUDA reports do not use identical source evidence")
    if reference.get("label_names") != candidate.get("label_names"):
        raise ValueError("MPS and CUDA reports do not use the same label order")

    reference_cases = {case["case_id"]: case for case in reference["cases"]}
    candidate_cases = {case["case_id"]: case for case in candidate["cases"]}
    if tuple(reference_cases) != tuple(candidate_cases):
        raise ValueError("MPS and CUDA reports do not contain the same ordered cases")

    details: list[dict[str, Any]] = []
    global_max_delta = 0.0
    prediction_mismatches = 0
    routing_mismatches = 0
    for case_id, reference_case in reference_cases.items():
        candidate_case = candidate_cases[case_id]
        deltas = [
            abs(float(left) - float(right))
            for left, right in zip(
                reference_case["probabilities"],
                candidate_case["probabilities"],
                strict=True,
            )
        ]
        maximum_delta = max(deltas)
        global_max_delta = max(global_max_delta, maximum_delta)
        same_prediction = (
            reference_case["predicted_intent"] == candidate_case["predicted_intent"]
        )
        same_routing = (
            reference_case["routing_action"] == candidate_case["routing_action"]
            and reference_case["routing_queue"] == candidate_case["routing_queue"]
        )
        prediction_mismatches += int(not same_prediction)
        routing_mismatches += int(not same_routing)
        details.append(
            {
                "case_id": case_id,
                "maximum_absolute_probability_delta": maximum_delta,
                "predicted_intent_match": same_prediction,
                "routing_action_and_queue_match": same_routing,
            }
        )

    tolerance_passed = global_max_delta <= config.probability_absolute_tolerance
    prediction_gate_passed = prediction_mismatches == 0
    routing_gate_passed = routing_mismatches == 0
    all_gates_passed = tolerance_passed and prediction_gate_passed and routing_gate_passed
    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
        "artifact_type": "module11_mps_cuda_prediction_parity_comparison",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_name": config.experiment_name,
        "claim_scope": CLAIM_SCOPE,
        "config_sha256": config.config_sha256,
        "reference": {
            "backend": config.reference_backend,
            "report_sha256": reference["report_sha256"],
        },
        "candidate": {
            "backend": config.candidate_backend,
            "report_sha256": candidate["report_sha256"],
        },
        "registered_gates": {
            "probability_absolute_tolerance": config.probability_absolute_tolerance,
            "require_identical_predicted_intents": True,
            "require_identical_routing_actions": True,
        },
        "results": {
            "case_count": len(details),
            "maximum_absolute_probability_delta": global_max_delta,
            "prediction_mismatch_count": prediction_mismatches,
            "routing_mismatch_count": routing_mismatches,
            "probability_tolerance_passed": tolerance_passed,
            "predicted_intent_gate_passed": prediction_gate_passed,
            "routing_action_gate_passed": routing_gate_passed,
            "all_gates_passed": all_gates_passed,
            "cases": details,
        },
        "data_boundary": {
            "fixture_is_synthetic": True,
            "official_test_access": False,
            "customer_data_access": False,
        },
    }
    comparison["report_sha256"] = stable_json_sha256(comparison)
    _atomic_write_json(comparison_path, comparison)
    return comparison


def _load_fixture(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    _require_file_hash(path, expected_sha256, "prediction-parity fixture")
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "case_id",
            "message",
            "required_action",
            "required_reason",
        }:
            raise ValueError(f"fixture line {line_number} has an invalid schema")
        if not isinstance(value.get("case_id"), str) or not value["case_id"]:
            raise ValueError(f"fixture line {line_number} has an invalid case ID")
        if not isinstance(value.get("message"), str) or not value["message"].strip():
            raise ValueError(f"fixture line {line_number} has an invalid message")
        cases.append(value)
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("fixture must contain unique, non-empty cases")
    return cases


def _contains_prohibited_text_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in {"message", "input_text", "redacted_text", "text"}:
                return True
            if _contains_prohibited_text_keys(child):
                return True
    elif isinstance(value, list):
        return any(_contains_prohibited_text_keys(child) for child in value)
    return False


def _repository_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} path must be a non-blank repository-relative string")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{name} path must remain inside the repository")
    resolved = (project_root / candidate).resolve(strict=True)
    if not resolved.is_relative_to(project_root):
        raise ValueError(f"{name} path escapes the repository")
    return resolved


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_file_hash(path: Path, expected: str, name: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{name} file hash differs from registration")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)
