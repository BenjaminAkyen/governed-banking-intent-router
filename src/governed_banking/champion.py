"""Fail-closed champion–challenger registry and promotion gates for Module 12."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from governed_banking.data import sha256_file, stable_json_sha256

FRAMEWORK_SCHEMA_VERSION = 1
REGISTRY_REPORT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "governance_registry_not_new_model_evaluation"
EXPECTED_MODEL_IDS = (
    "tfidf-word-char-c4",
    "frozen-roberta-mean-c1024",
    "lora-roberta-r8-original",
    "lora-roberta-r8-revised",
    "full-roberta-base",
)
EXPECTED_SEEDS = (17, 42, 73)
EvidenceScope = Literal[
    "historical_official_test_already_observed",
    "post_test_development_validation",
    "post_test_development_calibration",
    "post_test_synthetic_possible_ood",
]


@dataclass(frozen=True)
class EvidenceRegistration:
    evidence_id: str
    path: Path
    expected_file_sha256: str
    content_hash_field: str
    expected_content_sha256: str
    scope: EvidenceScope
    promotion_eligible: bool


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    role: str
    lifecycle_status: str
    architecture: str
    evidence: tuple[EvidenceRegistration, ...]


@dataclass(frozen=True)
class ChampionChallengerConfig:
    framework_version: str
    claim_scope: str
    decision_owner: str
    current_champion_id: str
    models: tuple[ModelRegistration, ...]
    service_alignment: dict[str, Any]
    data_boundary: dict[str, Any]
    promotion_gates: dict[str, Any]
    current_decision: dict[str, str]
    config_path: Path
    config_sha256: str
    project_root: Path

    @classmethod
    def from_yaml(cls, path: Path) -> ChampionChallengerConfig:
        resolved = path.resolve(strict=True)
        project_root = resolved.parent.parent
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "framework_version",
            "claim_scope",
            "decision_owner",
            "current_champion_id",
            "models",
            "service_alignment",
            "data_boundary",
            "promotion_gates",
            "current_decision",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("champion–challenger fields differ from the registered schema")
        if raw.get("schema_version") != FRAMEWORK_SCHEMA_VERSION:
            raise ValueError("unsupported champion–challenger schema")
        if raw.get("framework_version") != "module12-champion-challenger-v1":
            raise ValueError("framework version differs from registration")
        if raw.get("claim_scope") != CLAIM_SCOPE:
            raise ValueError("champion–challenger claim scope differs from registration")
        if raw.get("decision_owner") != "human_model_risk_review":
            raise ValueError("promotion decisions must remain human owned")
        if raw.get("current_champion_id") != "tfidf-word-char-c4":
            raise ValueError("TF-IDF must remain champion until a valid promotion decision")

        models = _parse_models(raw.get("models"), project_root)
        _validate_service_alignment(raw.get("service_alignment"))
        _validate_data_boundary(raw.get("data_boundary"))
        _validate_promotion_gates(raw.get("promotion_gates"))
        if raw.get("current_decision") != {
            "action": "retain_champion",
            "reason": "external_evaluation_lock_missing",
        }:
            raise ValueError("current decision must retain the champion while the lock is missing")

        return cls(
            framework_version="module12-champion-challenger-v1",
            claim_scope=CLAIM_SCOPE,
            decision_owner="human_model_risk_review",
            current_champion_id="tfidf-word-char-c4",
            models=models,
            service_alignment=dict(raw["service_alignment"]),
            data_boundary=dict(raw["data_boundary"]),
            promotion_gates=dict(raw["promotion_gates"]),
            current_decision=dict(raw["current_decision"]),
            config_path=resolved,
            config_sha256=sha256_file(resolved),
            project_root=project_root,
        )


@dataclass(frozen=True)
class SeedComparison:
    champion_macro_f1: float
    challenger_macro_f1: float
    champion_ece: float
    challenger_ece: float
    champion_selective_risk: float
    challenger_selective_risk: float
    champion_known_coverage: float
    challenger_known_coverage: float
    challenger_possible_ood_recall: float
    champion_security_intent_f1: float
    challenger_security_intent_f1: float


@dataclass(frozen=True)
class PairedIntervals:
    macro_f1_delta: tuple[float, float]
    ece_delta: tuple[float, float]
    selective_risk_delta: tuple[float, float]


def build_registry_report(
    config: ChampionChallengerConfig,
    *,
    report_path: Path,
    implementation_paths: dict[str, Path],
) -> dict[str, Any]:
    """Validate registered evidence and write a self-hashing, text-free registry snapshot."""

    if not implementation_paths:
        raise ValueError("implementation_paths must not be empty")
    model_rows = [_model_report(model) for model in config.models]
    historical_rows = [
        row
        for row in model_rows
        if row.get("historical_official_test_macro_f1") is not None
    ]
    historical_ranking = sorted(
        (
            {
                "model_id": row["model_id"],
                "macro_f1": row["historical_official_test_macro_f1"],
                "promotion_eligible": False,
            }
            for row in historical_rows
        ),
        key=lambda value: (-value["macro_f1"], value["model_id"]),
    )
    if historical_ranking[0]["model_id"] != config.current_champion_id:
        raise ValueError("registered champion is not the best historical comparison")

    report: dict[str, Any] = {
        "schema_version": REGISTRY_REPORT_SCHEMA_VERSION,
        "artifact_type": "module12_champion_challenger_registry",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "framework_version": config.framework_version,
        "claim_scope": config.claim_scope,
        "config_sha256": config.config_sha256,
        "current_champion_id": config.current_champion_id,
        "historical_ranking_context_only": historical_ranking,
        "models": model_rows,
        "service_alignment": config.service_alignment,
        "promotion_readiness": {
            "external_evaluation_lock_status": config.data_boundary[
                "external_evaluation_lock_status"
            ],
            "eligible_challenger_evaluations": 0,
            "automatic_promotion_permitted": False,
            "human_approval_required": True,
        },
        "current_decision": {
            "action": "retain_champion",
            "reason_codes": [
                "EXTERNAL_EVALUATION_LOCK_MISSING",
                "NO_PROMOTION_ELIGIBLE_CHALLENGER_EVIDENCE",
                "SHADOW_SERVICE_MODEL_DIFFERS_FROM_CHAMPION",
                "REVISED_LORA_UNCERTAINTY_GATES_FAILED",
            ],
            "production_deployment_approved": False,
        },
        "data_boundary": {
            "source_reports_only": True,
            "message_text_loaded": False,
            "official_banking77_test_loaded": False,
            "external_evaluation_loaded": False,
            "new_model_metrics_computed": False,
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(implementation_paths.items())
        },
    }
    report["report_sha256"] = stable_json_sha256(report)
    validate_registry_report(report, config=config)
    _atomic_write_json(report_path, report)
    return report


def validate_registry_report(
    report: dict[str, Any], *, config: ChampionChallengerConfig
) -> None:
    body = dict(report)
    expected_hash = body.pop("report_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("champion registry content hash check failed")
    if report.get("schema_version") != REGISTRY_REPORT_SCHEMA_VERSION:
        raise ValueError("champion registry schema differs from registration")
    if report.get("artifact_type") != "module12_champion_challenger_registry":
        raise ValueError("champion registry artifact type is invalid")
    if report.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("champion registry claim scope is invalid")
    if report.get("config_sha256") != config.config_sha256:
        raise ValueError("champion registry uses a different configuration")
    if report.get("current_champion_id") != config.current_champion_id:
        raise ValueError("champion registry changed the current champion")
    if report.get("current_decision") != {
        "action": "retain_champion",
        "reason_codes": [
            "EXTERNAL_EVALUATION_LOCK_MISSING",
            "NO_PROMOTION_ELIGIBLE_CHALLENGER_EVIDENCE",
            "SHADOW_SERVICE_MODEL_DIFFERS_FROM_CHAMPION",
            "REVISED_LORA_UNCERTAINTY_GATES_FAILED",
        ],
        "production_deployment_approved": False,
    }:
        raise ValueError("registry decision is not the registered fail-closed decision")
    if report.get("data_boundary") != {
        "source_reports_only": True,
        "message_text_loaded": False,
        "official_banking77_test_loaded": False,
        "external_evaluation_loaded": False,
        "new_model_metrics_computed": False,
    }:
        raise ValueError("champion registry crossed its registered data boundary")
    models = report.get("models")
    if not isinstance(models, list) or tuple(row.get("model_id") for row in models) != (
        EXPECTED_MODEL_IDS
    ):
        raise ValueError("champion registry model order differs from registration")
    if any(
        evidence.get("promotion_eligible") is not False
        for model in models
        for evidence in model.get("evidence", [])
    ):
        raise ValueError("historical or development evidence was made promotion eligible")


def evaluate_registered_gates(
    config: ChampionChallengerConfig,
    *,
    comparisons_by_seed: dict[int, SeedComparison],
    intervals: PairedIntervals,
    privacy_tests_passed: bool,
    routing_tests_passed: bool,
    audit_tests_passed: bool,
    matched_coverage: bool,
) -> dict[str, Any]:
    """Evaluate frozen numeric gates; this never mutates or automatically promotes a model."""

    if tuple(sorted(comparisons_by_seed)) != EXPECTED_SEEDS:
        raise ValueError("promotion evidence must contain exactly seeds 17, 42 and 73")
    _validate_intervals(intervals)
    comparisons = [comparisons_by_seed[seed] for seed in EXPECTED_SEEDS]
    for comparison in comparisons:
        _validate_seed_comparison(comparison)

    macro_deltas = [
        row.challenger_macro_f1 - row.champion_macro_f1 for row in comparisons
    ]
    ece_deltas = [row.challenger_ece - row.champion_ece for row in comparisons]
    risk_deltas = [
        row.challenger_selective_risk - row.champion_selective_risk for row in comparisons
    ]
    security_deltas = [
        row.challenger_security_intent_f1 - row.champion_security_intent_f1
        for row in comparisons
    ]
    gates = config.promotion_gates
    superiority = gates["superiority"]
    noninferiority = gates["noninferiority"]
    calibration = gates["calibration_route"]
    selective = gates["selective_risk_route"]
    safety = gates["safety_vetoes"]

    superiority_passed = all(delta > 0.0 for delta in macro_deltas) and (
        intervals.macro_f1_delta[0]
        > superiority["mean_macro_f1_delta_ci_lower_strictly_above"]
    )
    noninferiority_passed = (
        intervals.macro_f1_delta[0]
        >= noninferiority["mean_macro_f1_delta_ci_lower_at_least"]
        and min(macro_deltas) >= noninferiority["minimum_per_seed_macro_f1_delta"]
    )
    calibration_passed = (
        -_mean(ece_deltas) >= calibration["minimum_mean_ece_reduction"]
        and all(delta <= 0.0 for delta in ece_deltas)
        and intervals.ece_delta[1]
        < calibration["mean_ece_delta_ci_upper_strictly_below"]
    )
    selective_risk_passed = (
        -_mean(risk_deltas) >= selective["minimum_mean_selective_risk_reduction"]
        and matched_coverage is True
        and all(
            row.challenger_known_coverage
            >= selective["minimum_known_coverage_each_seed"]
            for row in comparisons
        )
        and all(
            row.challenger_possible_ood_recall
            >= selective["minimum_possible_ood_recall_each_seed"]
            for row in comparisons
        )
        and intervals.selective_risk_delta[1]
        < selective["mean_selective_risk_delta_ci_upper_strictly_below"]
    )
    operational_route_passed = noninferiority_passed and (
        calibration_passed or selective_risk_passed
    )
    controls_passed = (
        privacy_tests_passed is True
        and routing_tests_passed is True
        and audit_tests_passed is True
    )
    security_veto_passed = min(security_deltas) >= safety[
        "minimum_security_intent_f1_delta"
    ]
    eligible = (
        (superiority_passed or operational_route_passed)
        and controls_passed
        and security_veto_passed
    )
    return {
        "classification_superiority_passed": superiority_passed,
        "classification_noninferiority_passed": noninferiority_passed,
        "calibration_improvement_passed": calibration_passed,
        "selective_risk_improvement_passed": selective_risk_passed,
        "operational_improvement_route_passed": operational_route_passed,
        "privacy_routing_audit_controls_passed": controls_passed,
        "security_intent_veto_passed": security_veto_passed,
        "eligible_for_human_approval": eligible,
        "automatic_promotion_permitted": False,
        "decision": "await_human_approval" if eligible else "retain_champion",
        "observed_deltas": {
            "macro_f1_by_seed": dict(zip(EXPECTED_SEEDS, macro_deltas, strict=True)),
            "mean_macro_f1": _mean(macro_deltas),
            "mean_ece": _mean(ece_deltas),
            "mean_selective_risk": _mean(risk_deltas),
            "minimum_security_intent_f1": min(security_deltas),
        },
    }


def _parse_models(value: Any, project_root: Path) -> tuple[ModelRegistration, ...]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_MODEL_IDS):
        raise ValueError("models must contain the five registered entries")
    expected = {
        "tfidf-word-char-c4": (
            "champion",
            "active_historical_champion",
            "tfidf_word_char_logistic_regression",
            1,
        ),
        "frozen-roberta-mean-c1024": (
            "challenger",
            "historical_challenger",
            "frozen_roberta_mean_pooling_logistic_regression",
            1,
        ),
        "lora-roberta-r8-original": (
            "challenger",
            "retired_historical_challenger",
            "roberta_base_lora_rank8",
            1,
        ),
        "lora-roberta-r8-revised": (
            "challenger",
            "active_development_challenger",
            "roberta_base_lora_rank8_revised_stopping",
            3,
        ),
        "full-roberta-base": (
            "challenger",
            "planned_cuda_challenger",
            "roberta_base_full_finetuning",
            0,
        ),
    }
    rows: list[ModelRegistration] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "model_id",
            "role",
            "lifecycle_status",
            "architecture",
            "evidence",
        }:
            raise ValueError("model registration fields are invalid")
        model_id = item.get("model_id")
        if model_id != EXPECTED_MODEL_IDS[index]:
            raise ValueError("model registration order differs from policy")
        role, status, architecture, evidence_count = expected[model_id]
        if (
            item.get("role") != role
            or item.get("lifecycle_status") != status
            or item.get("architecture") != architecture
        ):
            raise ValueError(f"model registration differs for {model_id}")
        evidence_values = item.get("evidence")
        if not isinstance(evidence_values, list) or len(evidence_values) != evidence_count:
            raise ValueError(f"evidence count differs for {model_id}")
        evidence = tuple(
            _parse_evidence(entry, project_root, model_id) for entry in evidence_values
        )
        rows.append(ModelRegistration(model_id, role, status, architecture, evidence))
    return tuple(rows)


def _parse_evidence(value: Any, project_root: Path, model_id: str) -> EvidenceRegistration:
    if not isinstance(value, dict) or set(value) != {
        "evidence_id",
        "path",
        "expected_file_sha256",
        "content_hash_field",
        "expected_content_sha256",
        "scope",
        "promotion_eligible",
    }:
        raise ValueError(f"evidence registration fields are invalid for {model_id}")
    if value.get("promotion_eligible") is not False:
        raise ValueError("existing historical and development evidence cannot promote a model")
    scope = value.get("scope")
    if scope not in {
        "historical_official_test_already_observed",
        "post_test_development_validation",
        "post_test_development_calibration",
        "post_test_synthetic_possible_ood",
    }:
        raise ValueError("evidence scope is invalid")
    path = _repository_path(project_root, value.get("path"), "evidence")
    file_sha = _sha256(value.get("expected_file_sha256"), "evidence file SHA-256")
    content_sha = _sha256(
        value.get("expected_content_sha256"), "evidence content SHA-256"
    )
    content_field = value.get("content_hash_field")
    if content_field not in {"evaluation_sha256", "aggregate_sha256"}:
        raise ValueError("evidence content hash field is invalid")
    if sha256_file(path) != file_sha:
        raise ValueError(f"registered evidence file hash differs: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_embedded_hash(payload, content_field, content_sha)
    _validate_evidence_boundary(payload, scope)
    evidence_id = value.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("evidence ID must be a non-blank string")
    return EvidenceRegistration(
        evidence_id=evidence_id,
        path=path,
        expected_file_sha256=file_sha,
        content_hash_field=content_field,
        expected_content_sha256=content_sha,
        scope=scope,
        promotion_eligible=False,
    )


def _model_report(model: ModelRegistration) -> dict[str, Any]:
    evidence_rows = []
    extracted: dict[str, Any] = {}
    for evidence in model.evidence:
        payload = json.loads(evidence.path.read_text(encoding="utf-8"))
        _validate_embedded_hash(
            payload, evidence.content_hash_field, evidence.expected_content_sha256
        )
        evidence_rows.append(
            {
                "evidence_id": evidence.evidence_id,
                "path": _portable_path(evidence.path),
                "file_sha256": evidence.expected_file_sha256,
                "content_sha256": evidence.expected_content_sha256,
                "scope": evidence.scope,
                "promotion_eligible": False,
            }
        )
        if evidence.scope == "historical_official_test_already_observed":
            extracted["historical_official_test_macro_f1"] = float(
                payload["test_result"]["metrics"]["macro_f1"]
            )
        elif evidence.evidence_id == "multiseed_validation":
            extracted["development_validation_macro_f1"] = payload["validation_metrics"][
                "macro_f1"
            ]
        elif evidence.evidence_id == "calibration_assessment":
            extracted["development_calibrated_ece"] = payload["assessment_metrics"][
                "expected_calibration_error"
            ]["calibrated"]
            extracted["calibration_point_gates_passed"] = payload["acceptance_gate"][
                "all_seeds_passed"
            ]
        elif evidence.evidence_id == "uncertainty_assessment":
            extracted["development_selective_risk"] = payload["assessment_metrics"][
                "selective_risk"
            ]
            extracted["development_possible_ood_recall"] = payload["assessment_metrics"][
                "possible_ood_recall"
            ]
            extracted["uncertainty_gates_passed"] = payload["acceptance_gate"][
                "all_seeds_passed"
            ]
    return {
        "model_id": model.model_id,
        "role": model.role,
        "lifecycle_status": model.lifecycle_status,
        "architecture": model.architecture,
        "evidence": evidence_rows,
        "promotion_ready": False,
        "external_locked_evaluation_present": False,
        **extracted,
    }


def _validate_embedded_hash(
    payload: dict[str, Any], hash_field: str, expected_content_sha256: str
) -> None:
    body = dict(payload)
    actual = body.pop(hash_field, None)
    if actual != expected_content_sha256 or stable_json_sha256(body) != actual:
        raise ValueError("registered evidence content hash check failed")


def _validate_evidence_boundary(payload: dict[str, Any], scope: str) -> None:
    boundary = payload.get("data_boundary", {})
    if scope == "historical_official_test_already_observed":
        if boundary.get("evaluation_split") != "test":
            raise ValueError("historical test evidence does not identify the test split")
    elif boundary.get("test_split_loaded") is not False:
        raise ValueError("post-test development evidence unexpectedly loaded the test split")
    if scope in {
        "post_test_development_calibration",
        "post_test_synthetic_possible_ood",
    } and boundary.get("independent_model_evaluation") is not False:
        raise ValueError("development evidence is incorrectly presented as independent")
    if scope == "post_test_synthetic_possible_ood" and (
        payload.get("acceptance_gate", {}).get("all_seeds_passed") is not False
    ):
        raise ValueError("registered uncertainty evidence must preserve its failed gates")


def _validate_service_alignment(value: Any) -> None:
    expected = {
        "service_module": 10,
        "operating_mode": "shadow_review_only",
        "currently_served_model_id": "lora-roberta-r8-revised",
        "champion_aligned": False,
        "production_deployment_approved": False,
    }
    if value != expected:
        raise ValueError("service/champion alignment differs from registration")


def _validate_data_boundary(value: Any) -> None:
    expected = {
        "official_banking77_test_previously_observed": True,
        "official_banking77_test_access_in_module12": False,
        "banking77_test_eligible_for_promotion": False,
        "development_validation_eligible_for_promotion": False,
        "synthetic_possible_ood_eligible_for_promotion": False,
        "required_promotion_dataset": "new_locked_external_evaluation",
        "external_evaluation_lock_status": "missing",
    }
    if value != expected:
        raise ValueError("champion–challenger data boundary differs from registration")


def _validate_promotion_gates(value: Any) -> None:
    expected = {
        "required_seeds": [17, 42, 73],
        "confidence_level": 0.95,
        "paired_bootstrap_resamples": 5000,
        "bootstrap_seed": 120017,
        "superiority": {
            "require_positive_macro_f1_delta_each_seed": True,
            "mean_macro_f1_delta_ci_lower_strictly_above": 0.0,
        },
        "noninferiority": {
            "mean_macro_f1_delta_ci_lower_at_least": -0.005,
            "minimum_per_seed_macro_f1_delta": -0.01,
        },
        "calibration_route": {
            "minimum_mean_ece_reduction": 0.01,
            "require_non_worse_ece_each_seed": True,
            "mean_ece_delta_ci_upper_strictly_below": 0.0,
        },
        "selective_risk_route": {
            "minimum_mean_selective_risk_reduction": 0.02,
            "matched_coverage_required": True,
            "minimum_known_coverage_each_seed": 0.7,
            "minimum_possible_ood_recall_each_seed": 0.9,
            "mean_selective_risk_delta_ci_upper_strictly_below": 0.0,
        },
        "safety_vetoes": {
            "minimum_security_intent_f1_delta": -0.02,
            "privacy_tests_must_pass": True,
            "routing_tests_must_pass": True,
            "audit_tests_must_pass": True,
            "complete_seed_evidence_required": True,
        },
        "approval": {
            "automatic_promotion_permitted": False,
            "human_approval_required": True,
        },
    }
    if value != expected:
        raise ValueError("promotion gates differ from registration")


def _validate_seed_comparison(value: SeedComparison) -> None:
    for name, number in vars(value).items():
        if not isinstance(number, int | float) or isinstance(number, bool):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(number)) or not 0.0 <= float(number) <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")


def _validate_intervals(value: PairedIntervals) -> None:
    for name, interval in vars(value).items():
        if len(interval) != 2:
            raise ValueError(f"{name} must contain two bounds")
        lower, upper = interval
        if not all(math.isfinite(number) for number in interval) or lower > upper:
            raise ValueError(f"{name} interval is invalid")
        if lower < -1.0 or upper > 1.0:
            raise ValueError(f"{name} interval must remain within [-1, 1]")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


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


def _portable_path(path: Path) -> str:
    for candidate in path.parents:
        if (candidate / "pyproject.toml").is_file():
            return path.relative_to(candidate).as_posix()
    raise ValueError("evidence path is outside the repository")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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
