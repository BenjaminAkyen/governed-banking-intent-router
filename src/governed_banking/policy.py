"""Fail-closed deterministic routing policy for advisory banking support decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from governed_banking.data import sha256_file

RoutingAction = Literal["security_queue", "human_review", "suggest_queue"]

POLICY_SCHEMA_VERSION = 1
REGISTERED_SEEDS = (17, 42, 73)
REGISTERED_ACTIONS = {
    "security": "security_queue",
    "review": "human_review",
    "suggestion": "suggest_queue",
}
REGISTERED_PRECEDENCE = (
    "exposed_authentication_secret",
    "security_intent",
    "redaction_failure",
    "unsupported_intent",
    "sensitive_pii",
    "experimental_uncertainty",
    "shadow_default",
)
REASON_ORDER = (
    "EXPOSED_AUTHENTICATION_SECRET",
    "SECURITY_INTENT_OVERRIDE",
    "REDACTION_FAILURE",
    "UNSUPPORTED_INTENT",
    "SENSITIVE_PII_REVIEW",
    "UNCERTAINTY_METADATA_INVALID",
    "EXPERIMENTAL_UNCERTAINTY_BELOW_THRESHOLD",
    "EXPERIMENTAL_UNCERTAINTY_NOT_AUTHORIZED",
    "SHADOW_MODE_REQUIRES_REVIEW",
)


@dataclass(frozen=True)
class ExperimentalThreshold:
    signal: str
    threshold: float


@dataclass(frozen=True)
class RoutingPolicyConfig:
    policy_version: str
    operating_mode: str
    config_sha256: str
    actions: Mapping[str, str]
    uncertainty_aggregate_path: Path
    uncertainty_file_sha256: str
    uncertainty_aggregate_sha256: str
    uncertainty_status: str
    uncertainty_use: str
    uncertainty_may_authorize_suggestion: bool
    thresholds: Mapping[int, ExperimentalThreshold]
    precedence: tuple[str, ...]
    queues: Mapping[str, str]
    security_intents: frozenset[str]
    review_intents: frozenset[str]
    security_pii_types: frozenset[str]
    sensitive_pii_types: frozenset[str]
    allowed_intents: frozenset[str]

    @classmethod
    def from_yaml(cls, path: Path) -> RoutingPolicyConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported routing-policy schema")
        actions = _mapping(raw, "actions")
        if actions != REGISTERED_ACTIONS:
            raise ValueError("routing actions differ from the registered advisory boundary")
        if raw.get("operating_mode") != "shadow_review_only":
            raise ValueError("Module 9 must remain in shadow_review_only mode")
        uncertainty = _mapping(raw, "uncertainty_evidence")
        if uncertainty.get("status") != "failed_registered_gates":
            raise ValueError("Module 8 failed-gate status must remain explicit")
        if uncertainty.get("use") != "review_signal_only":
            raise ValueError("Module 8 uncertainty may be used only as a review signal")
        if uncertainty.get("may_authorize_suggest_queue") is not False:
            raise ValueError("failed Module 8 thresholds cannot authorize suggestions")
        aggregate_path = Path(
            _non_blank(uncertainty.get("aggregate_path"), "uncertainty_evidence.aggregate_path")
        )
        expected_file_sha256 = _sha256(
            uncertainty.get("expected_file_sha256"),
            "uncertainty_evidence.expected_file_sha256",
        )
        if sha256_file(aggregate_path) != expected_file_sha256:
            raise ValueError("Module 8 aggregate file hash differs from policy registration")
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        expected_aggregate_sha256 = _sha256(
            uncertainty.get("expected_aggregate_sha256"),
            "uncertainty_evidence.expected_aggregate_sha256",
        )
        if aggregate.get("aggregate_sha256") != expected_aggregate_sha256:
            raise ValueError("Module 8 aggregate content hash differs from policy registration")
        if aggregate.get("acceptance_gate", {}).get("all_seeds_passed") is not False:
            raise ValueError("Module 9 expects Module 8 acceptance gates to have failed")
        if aggregate.get("data_boundary", {}).get("test_split_loaded") is not False:
            raise ValueError("Module 8 evidence unexpectedly reports official-test access")
        by_seed = _mapping(uncertainty, "by_seed")
        if tuple(sorted(int(seed) for seed in by_seed)) != REGISTERED_SEEDS:
            raise ValueError("routing policy must bind all registered Module 8 seeds")
        thresholds = {
            int(seed): ExperimentalThreshold(
                signal=_non_blank(value.get("signal"), f"by_seed.{seed}.signal"),
                threshold=_unit_float(value.get("threshold"), f"by_seed.{seed}.threshold"),
            )
            for seed, value in by_seed.items()
            if isinstance(value, dict)
        }
        expected_signals = {
            int(seed): value for seed, value in aggregate["selected_signals"].items()
        }
        expected_thresholds = {
            int(seed): float(value) for seed, value in aggregate["selected_thresholds"].items()
        }
        if {seed: value.signal for seed, value in thresholds.items()} != expected_signals:
            raise ValueError("policy signals differ from locked Module 8 selections")
        if {seed: value.threshold for seed, value in thresholds.items()} != expected_thresholds:
            raise ValueError("policy thresholds differ from locked Module 8 selections")
        precedence = tuple(raw.get("precedence", []))
        if precedence != REGISTERED_PRECEDENCE:
            raise ValueError("routing precedence differs from registration")
        queues = _mapping(raw, "queues")
        if set(queues) != {"security", "review"} or any(
            not isinstance(value, str) or not value for value in queues.values()
        ):
            raise ValueError("routing queues must contain non-empty security and review values")
        allowed_intents = _string_set(raw.get("allowed_intents"), "allowed_intents")
        if len(allowed_intents) != 77:
            raise ValueError("routing policy must register all 77 BANKING77 intents")
        security_intents = _string_set(raw.get("security_intents"), "security_intents")
        review_intents = _string_set(raw.get("review_intents"), "review_intents")
        if not security_intents <= allowed_intents or not review_intents <= allowed_intents:
            raise ValueError("risk-tier intents must be part of the registered taxonomy")
        if security_intents & review_intents:
            raise ValueError("security and review intent tiers cannot overlap")
        security_pii_types = _string_set(raw.get("security_pii_types"), "security_pii_types")
        sensitive_pii_types = _string_set(raw.get("sensitive_pii_types"), "sensitive_pii_types")
        if not security_pii_types <= sensitive_pii_types:
            raise ValueError("security PII types must also be sensitive")
        return cls(
            policy_version=_non_blank(raw.get("policy_version"), "policy_version"),
            operating_mode="shadow_review_only",
            config_sha256=sha256_file(path),
            actions=dict(actions),
            uncertainty_aggregate_path=aggregate_path,
            uncertainty_file_sha256=expected_file_sha256,
            uncertainty_aggregate_sha256=expected_aggregate_sha256,
            uncertainty_status="failed_registered_gates",
            uncertainty_use="review_signal_only",
            uncertainty_may_authorize_suggestion=False,
            thresholds=thresholds,
            precedence=precedence,
            queues=dict(queues),
            security_intents=frozenset(security_intents),
            review_intents=frozenset(review_intents),
            security_pii_types=frozenset(security_pii_types),
            sensitive_pii_types=frozenset(sensitive_pii_types),
            allowed_intents=frozenset(allowed_intents),
        )


@dataclass(frozen=True)
class RoutingInput:
    predicted_intent: str
    model_seed: int
    uncertainty_signal: str | None
    uncertainty_score: float | None
    pii_type_counts: Mapping[str, int]
    redaction_succeeded: bool


@dataclass(frozen=True)
class RoutingDecision:
    action: RoutingAction
    queue: str
    reason_codes: tuple[str, ...]
    policy_version: str
    policy_sha256: str
    uncertainty_observation: str
    uncertainty_authorized_suggestion: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "queue": self.queue,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "uncertainty_observation": self.uncertainty_observation,
            "uncertainty_authorized_suggestion": self.uncertainty_authorized_suggestion,
        }


def route_request(config: RoutingPolicyConfig, value: RoutingInput) -> RoutingDecision:
    """Apply registered precedence without allowing experimental scores to lower risk."""

    counts = _validated_pii_counts(value.pii_type_counts)
    reasons: list[str] = []
    security_pii_present = any(counts.get(name, 0) > 0 for name in config.security_pii_types)
    if security_pii_present:
        reasons.append("EXPOSED_AUTHENTICATION_SECRET")
    if value.predicted_intent in config.security_intents:
        reasons.append("SECURITY_INTENT_OVERRIDE")
    if reasons:
        return _decision(
            config,
            action="security_queue",
            queue=config.queues["security"],
            reasons=reasons,
            uncertainty_observation=_uncertainty_observation(config, value),
        )
    if not value.redaction_succeeded:
        reasons.append("REDACTION_FAILURE")
    if value.predicted_intent not in config.allowed_intents:
        reasons.append("UNSUPPORTED_INTENT")
    if any(counts.get(name, 0) > 0 for name in config.sensitive_pii_types):
        reasons.append("SENSITIVE_PII_REVIEW")
    observation = _uncertainty_observation(config, value)
    if observation == "invalid_or_missing":
        reasons.append("UNCERTAINTY_METADATA_INVALID")
    elif observation == "below_experimental_threshold":
        reasons.append("EXPERIMENTAL_UNCERTAINTY_BELOW_THRESHOLD")
    else:
        reasons.append("EXPERIMENTAL_UNCERTAINTY_NOT_AUTHORIZED")
    reasons.append("SHADOW_MODE_REQUIRES_REVIEW")
    return _decision(
        config,
        action="human_review",
        queue=config.queues["review"],
        reasons=reasons,
        uncertainty_observation=observation,
    )


def _uncertainty_observation(config: RoutingPolicyConfig, value: RoutingInput) -> str:
    threshold = config.thresholds.get(value.model_seed)
    if (
        threshold is None
        or value.uncertainty_signal != threshold.signal
        or not isinstance(value.uncertainty_score, int | float)
        or isinstance(value.uncertainty_score, bool)
        or not math.isfinite(float(value.uncertainty_score))
        or not 0.0 <= float(value.uncertainty_score) <= 1.0
    ):
        return "invalid_or_missing"
    if float(value.uncertainty_score) < threshold.threshold:
        return "below_experimental_threshold"
    return "at_or_above_experimental_threshold"


def _decision(
    config: RoutingPolicyConfig,
    *,
    action: RoutingAction,
    queue: str,
    reasons: list[str],
    uncertainty_observation: str,
) -> RoutingDecision:
    if action == "suggest_queue":
        raise AssertionError("Module 9 shadow policy cannot authorize suggest_queue")
    ordered = tuple(reason for reason in REASON_ORDER if reason in set(reasons))
    return RoutingDecision(
        action=action,
        queue=queue,
        reason_codes=ordered,
        policy_version=config.policy_version,
        policy_sha256=config.config_sha256,
        uncertainty_observation=uncertainty_observation,
        uncertainty_authorized_suggestion=False,
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _non_blank(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} cannot be blank")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _non_blank(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _unit_float(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _string_set(value: Any, name: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = {_non_blank(item, name) for item in value}
    if len(result) != len(value):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _validated_pii_counts(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("pii_type_counts must be a mapping")
    counts = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(count, int)
            or isinstance(count, bool)
        ):
            raise ValueError("PII counts must use non-empty string keys and integer values")
        if count < 0 or count > 100:
            raise ValueError("PII counts must be between zero and 100")
        counts[key] = count
    return counts
