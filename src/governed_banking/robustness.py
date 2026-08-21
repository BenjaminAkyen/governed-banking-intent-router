"""Versioned synthetic robustness-pack validation and leakage checks."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from governed_banking.data import NORMALIZATION_VERSION, normalize_text, sha256_file

ROBUSTNESS_CONFIG_SCHEMA_VERSION = 1
ROBUSTNESS_CASE_SCHEMA_VERSION = 1
ROBUSTNESS_MANIFEST_SCHEMA_VERSION = 1
REGISTERED_FAMILIES = (
    "typographical_error",
    "speech_transcription_error",
    "paraphrase",
    "multi_intent",
    "short_ambiguous",
    "code_switching",
    "pii_bearing",
    "prompt_like_manipulation",
    "non_banking_adversarial",
    "high_risk_security",
)
REGISTERED_ACTIONS = ("human_review", "security_queue")
REGISTERED_RISK_SEVERITIES = ("low", "medium", "high", "critical")
REGISTERED_REASON_CODES = frozenset(
    {
        "AMBIGUOUS_REQUEST",
        "CODE_SWITCHING_REVIEW",
        "MULTI_INTENT_REQUIRES_DISAMBIGUATION",
        "NON_BANKING_OR_OUT_OF_SCOPE",
        "PII_REQUIRES_CONTROLLED_REVIEW",
        "PROMPT_LIKE_MANIPULATION",
        "ROBUSTNESS_CHALLENGE",
        "SECURITY_INCIDENT",
        "SHADOW_MODE_REQUIRES_REVIEW",
    }
)
REGISTERED_PROVENANCE = {
    "origin": "project_authored_synthetic",
    "creation_method": "human_directed_ai_assisted_scenario_authoring",
    "contains_customer_data": False,
    "derived_from_production_data": False,
    "derived_from_banking77_text": False,
}
REGISTERED_LICENCE = {
    "spdx_id": "MIT",
    "copyright": "Copyright (c) 2026 Benjamin Akyen",
    "reference": "../../../LICENSE",
}
CASE_KEYS = {
    "case_id",
    "schema_version",
    "text",
    "primary_family",
    "robustness_tags",
    "semantic_group",
    "acceptable_intents",
    "out_of_scope",
    "expected_routing_action",
    "risk_severity",
    "escalation_required",
    "escalation_reason_codes",
    "escalation_rationale",
    "expected_pii_types",
    "provenance",
    "licence",
}


@dataclass(frozen=True)
class RobustnessCase:
    """One fully annotated, project-authored synthetic evaluation case."""

    case_id: str
    text: str
    primary_family: str
    robustness_tags: tuple[str, ...]
    semantic_group: str
    acceptable_intents: tuple[str, ...]
    out_of_scope: bool
    expected_routing_action: str
    risk_severity: str
    escalation_reason_codes: tuple[str, ...]
    escalation_rationale: str
    expected_pii_types: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessEvaluationConfig:
    """Hash-bound Module 13 evaluation registration."""

    config_path: Path
    project_root: Path
    pack_version: str
    pack_path: Path
    expected_pack_sha256: str
    manifest_path: Path
    taxonomy_manifest_path: Path
    dataset_config_path: Path
    raw_directory: Path
    model_role: str
    service_config_path: Path
    runtime_profile_path: Path
    report_path: Path
    minimum_in_scope_acceptable_intent_rate: float
    minimum_expected_security_routing_recall: float
    minimum_overall_routing_action_match_rate: float
    minimum_cases_per_family: int
    near_duplicate_ngram_size: int
    near_duplicate_jaccard_threshold: float
    minimum_near_duplicate_characters: int

    @classmethod
    def from_yaml(cls, path: Path) -> RobustnessEvaluationConfig:
        resolved = path.resolve(strict=True)
        project_root = resolved.parent.parent
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "experiment_name",
            "claim_scope",
            "pack",
            "taxonomy",
            "leakage_checks",
            "evaluation",
            "acceptance_gate",
            "assessment_gate",
            "output_policy",
            "data_boundary",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("robustness-evaluation fields differ from registration")
        if raw.get("schema_version") != ROBUSTNESS_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported robustness-evaluation schema")
        if raw.get("experiment_name") != "module13-synthetic-robustness-v1":
            raise ValueError("unexpected robustness experiment name")
        if raw.get("claim_scope") != "synthetic_stress_test_not_production_validation":
            raise ValueError("robustness claim scope must remain synthetic and non-production")

        pack = _exact_mapping(
            raw,
            "pack",
            {"version", "path", "expected_sha256", "manifest_path", "licence_spdx"},
        )
        if pack.get("version") != "module13-synthetic-v1":
            raise ValueError("unexpected robustness-pack version")
        if pack.get("licence_spdx") != "MIT":
            raise ValueError("robustness-pack licence differs from the project registration")
        pack_path = _repository_path(project_root, pack.get("path"), "pack.path")
        expected_pack_sha256 = _sha256(pack.get("expected_sha256"), "pack.expected_sha256")
        if sha256_file(pack_path) != expected_pack_sha256:
            raise ValueError("robustness-pack hash differs from registration")

        taxonomy = _exact_mapping(
            raw,
            "taxonomy",
            {"manifest_path", "expected_label_count"},
        )
        if taxonomy.get("expected_label_count") != 77:
            raise ValueError("robustness evaluation requires the 77-label taxonomy")

        leakage = _exact_mapping(
            raw,
            "leakage_checks",
            {
                "dataset_config_path",
                "raw_directory",
                "normalization",
                "near_duplicate_ngram_size",
                "near_duplicate_jaccard_threshold",
                "minimum_near_duplicate_characters",
            },
        )
        if leakage.get("normalization") != NORMALIZATION_VERSION:
            raise ValueError("leakage normalization differs from the dataset contract")
        ngram_size = _bounded_int(
            leakage.get("near_duplicate_ngram_size"),
            "near_duplicate_ngram_size",
            3,
            8,
        )
        threshold = _bounded_float(
            leakage.get("near_duplicate_jaccard_threshold"),
            "near_duplicate_jaccard_threshold",
            0.75,
            1.0,
        )
        minimum_characters = _bounded_int(
            leakage.get("minimum_near_duplicate_characters"),
            "minimum_near_duplicate_characters",
            8,
            100,
        )

        evaluation = _exact_mapping(
            raw,
            "evaluation",
            {"service_config_path", "runtime_profile_path", "report_path", "model_role"},
        )
        if evaluation.get("model_role") != "module10_lora_research_service_not_champion":
            raise ValueError("Module 13 model role must preserve the champion mismatch")
        gates = _exact_mapping(
            raw,
            "acceptance_gate",
            {
                "minimum_cases_per_family",
                "require_all_families",
                "require_zero_exact_duplicates",
                "require_zero_unregistered_near_duplicates",
                "require_zero_banking77_exact_matches",
                "require_zero_banking77_near_duplicates",
                "require_all_pii_expectations",
                "require_zero_suggestion_actions",
            },
        )
        boolean_gates = {
            key: value
            for key, value in gates.items()
            if key != "minimum_cases_per_family"
        }
        if any(value is not True for value in boolean_gates.values()):
            raise ValueError("all registered robustness gates must remain enabled")
        assessment_gates = _exact_mapping(
            raw,
            "assessment_gate",
            {
                "minimum_in_scope_acceptable_intent_rate",
                "minimum_expected_security_routing_recall",
                "minimum_overall_routing_action_match_rate",
                "require_all_pii_expectations",
                "require_zero_suggestion_actions",
                "require_text_free_report",
            },
        )
        for key in (
            "require_all_pii_expectations",
            "require_zero_suggestion_actions",
            "require_text_free_report",
        ):
            if assessment_gates.get(key) is not True:
                raise ValueError(f"assessment gate {key} must remain enabled")
        _bounded_float(
            assessment_gates.get("minimum_in_scope_acceptable_intent_rate"),
            "minimum_in_scope_acceptable_intent_rate",
            0.8,
            1.0,
        )
        _bounded_float(
            assessment_gates.get("minimum_expected_security_routing_recall"),
            "minimum_expected_security_routing_recall",
            0.95,
            1.0,
        )
        _bounded_float(
            assessment_gates.get("minimum_overall_routing_action_match_rate"),
            "minimum_overall_routing_action_match_rate",
            0.9,
            1.0,
        )

        if raw.get("output_policy") != {
            "include_input_text": False,
            "include_redacted_text": False,
            "include_message_hash": False,
            "include_case_level_prediction_metadata": True,
        }:
            raise ValueError("robustness output policy differs from registration")
        if raw.get("data_boundary") != {
            "fixture_is_synthetic": True,
            "customer_data_access": False,
            "production_data_access": False,
            "official_test_used_for_model_scoring": False,
            "banking77_text_used_only_for_leakage_detection": True,
            "production_validation": False,
        }:
            raise ValueError("robustness data boundary differs from registration")

        return cls(
            config_path=resolved,
            project_root=project_root,
            pack_version="module13-synthetic-v1",
            pack_path=pack_path,
            expected_pack_sha256=expected_pack_sha256,
            manifest_path=_repository_path(
                project_root, pack.get("manifest_path"), "pack.manifest_path"
            ),
            taxonomy_manifest_path=_repository_path(
                project_root, taxonomy.get("manifest_path"), "taxonomy.manifest_path"
            ),
            dataset_config_path=_repository_path(
                project_root,
                leakage.get("dataset_config_path"),
                "leakage_checks.dataset_config_path",
            ),
            raw_directory=_repository_path(
                project_root, leakage.get("raw_directory"), "leakage_checks.raw_directory"
            ),
            model_role="module10_lora_research_service_not_champion",
            service_config_path=_repository_path(
                project_root,
                evaluation.get("service_config_path"),
                "evaluation.service_config_path",
            ),
            runtime_profile_path=_repository_path(
                project_root,
                evaluation.get("runtime_profile_path"),
                "evaluation.runtime_profile_path",
            ),
            report_path=_repository_path(
                project_root, evaluation.get("report_path"), "evaluation.report_path"
            ),
            minimum_in_scope_acceptable_intent_rate=_bounded_float(
                assessment_gates.get("minimum_in_scope_acceptable_intent_rate"),
                "minimum_in_scope_acceptable_intent_rate",
                0.8,
                1.0,
            ),
            minimum_expected_security_routing_recall=_bounded_float(
                assessment_gates.get("minimum_expected_security_routing_recall"),
                "minimum_expected_security_routing_recall",
                0.95,
                1.0,
            ),
            minimum_overall_routing_action_match_rate=_bounded_float(
                assessment_gates.get("minimum_overall_routing_action_match_rate"),
                "minimum_overall_routing_action_match_rate",
                0.9,
                1.0,
            ),
            minimum_cases_per_family=_bounded_int(
                gates.get("minimum_cases_per_family"), "minimum_cases_per_family", 1, 100
            ),
            near_duplicate_ngram_size=ngram_size,
            near_duplicate_jaccard_threshold=threshold,
            minimum_near_duplicate_characters=minimum_characters,
        )


@dataclass(frozen=True)
class LeakageFinding:
    case_id: str
    source_collection: str
    source_reference: str
    similarity: float
    match_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_collection": self.source_collection,
            "source_reference": self.source_reference,
            "similarity": round(self.similarity, 6),
            "match_type": self.match_type,
        }


def load_robustness_cases(
    path: Path,
    *,
    allowed_intents: Iterable[str],
    allowed_pii_types: Iterable[str],
) -> tuple[RobustnessCase, ...]:
    """Load the JSONL pack and enforce the complete per-case evidence contract."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("robustness pack must be non-empty JSONL without blank lines")
    records = [json.loads(line) for line in lines]
    allowed_intent_set = frozenset(allowed_intents)
    allowed_pii_set = frozenset(allowed_pii_types)
    cases = tuple(
        _parse_case(
            record,
            allowed_intents=allowed_intent_set,
            allowed_pii_types=allowed_pii_set,
        )
        for record in records
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("robustness case IDs must be unique")
    if case_ids != sorted(case_ids):
        raise ValueError("robustness cases must be ordered by case_id")
    return cases


def find_leakage(
    cases: Sequence[RobustnessCase],
    source_collections: Mapping[str, Sequence[tuple[str, str]]],
    *,
    ngram_size: int,
    jaccard_threshold: float,
    minimum_characters: int,
) -> tuple[LeakageFinding, ...]:
    """Find normalized exact and high-overlap character-ngram matches without persisting text."""

    findings: list[LeakageFinding] = []
    for collection_name, rows in source_collections.items():
        prepared = [
            (reference, normalize_text(text), None)
            for reference, text in rows
        ]
        prepared = [
            (
                reference,
                normalized,
                _character_ngrams(normalized, ngram_size)
                if len(normalized) >= minimum_characters
                else frozenset(),
            )
            for reference, normalized, _ in prepared
        ]
        exact_index: dict[str, list[str]] = defaultdict(list)
        for reference, normalized, _ in prepared:
            exact_index[normalized].append(reference)

        for case in cases:
            normalized_case = normalize_text(case.text)
            for reference in exact_index.get(normalized_case, []):
                findings.append(
                    LeakageFinding(case.case_id, collection_name, reference, 1.0, "exact")
                )
            if len(normalized_case) < minimum_characters:
                continue
            case_ngrams = _character_ngrams(normalized_case, ngram_size)
            for reference, normalized_source, source_ngrams in prepared:
                if normalized_case == normalized_source or not source_ngrams:
                    continue
                upper_bound = min(len(case_ngrams), len(source_ngrams)) / max(
                    len(case_ngrams), len(source_ngrams)
                )
                if upper_bound < jaccard_threshold:
                    continue
                similarity = _jaccard(case_ngrams, source_ngrams)
                if similarity >= jaccard_threshold:
                    findings.append(
                        LeakageFinding(
                            case.case_id,
                            collection_name,
                            reference,
                            similarity,
                            "near_duplicate",
                        )
                    )
    return tuple(
        sorted(
            findings,
            key=lambda value: (
                value.case_id,
                value.source_collection,
                value.source_reference,
                value.match_type,
            ),
        )
    )


def find_internal_leakage(
    cases: Sequence[RobustnessCase],
    *,
    ngram_size: int,
    jaccard_threshold: float,
    minimum_characters: int,
) -> tuple[LeakageFinding, ...]:
    """Compare every case pair once and return text-free duplicate evidence."""

    findings: list[LeakageFinding] = []
    for index, case in enumerate(cases):
        normalized_case = normalize_text(case.text)
        case_ngrams = (
            _character_ngrams(normalized_case, ngram_size)
            if len(normalized_case) >= minimum_characters
            else frozenset()
        )
        for other in cases[index + 1 :]:
            normalized_other = normalize_text(other.text)
            if normalized_case == normalized_other:
                findings.append(
                    LeakageFinding(case.case_id, "robustness_pack", other.case_id, 1.0, "exact")
                )
                continue
            if (
                len(normalized_case) < minimum_characters
                or len(normalized_other) < minimum_characters
            ):
                continue
            other_ngrams = _character_ngrams(normalized_other, ngram_size)
            upper_bound = min(len(case_ngrams), len(other_ngrams)) / max(
                len(case_ngrams), len(other_ngrams)
            )
            if upper_bound < jaccard_threshold:
                continue
            similarity = _jaccard(case_ngrams, other_ngrams)
            if similarity >= jaccard_threshold:
                findings.append(
                    LeakageFinding(
                        case.case_id,
                        "robustness_pack",
                        other.case_id,
                        similarity,
                        "near_duplicate",
                    )
                )
    return tuple(findings)


def summarize_cases(cases: Sequence[RobustnessCase]) -> dict[str, Any]:
    """Return text-free aggregate coverage metadata for a pack manifest."""

    family_counts = Counter(case.primary_family for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case.robustness_tags)
    severity_counts = Counter(case.risk_severity for case in cases)
    action_counts = Counter(case.expected_routing_action for case in cases)
    pii_counts = Counter(pii_type for case in cases for pii_type in case.expected_pii_types)
    return {
        "case_count": len(cases),
        "primary_family_counts": dict(sorted(family_counts.items())),
        "robustness_tag_counts": dict(sorted(tag_counts.items())),
        "risk_severity_counts": dict(sorted(severity_counts.items())),
        "expected_routing_action_counts": dict(sorted(action_counts.items())),
        "expected_pii_type_case_counts": dict(sorted(pii_counts.items())),
        "out_of_scope_case_count": sum(case.out_of_scope for case in cases),
        "ambiguous_label_case_count": sum(len(case.acceptable_intents) > 1 for case in cases),
        "all_cases_require_escalation": True,
    }


def _parse_case(
    record: Any,
    *,
    allowed_intents: frozenset[str],
    allowed_pii_types: frozenset[str],
) -> RobustnessCase:
    if not isinstance(record, dict) or set(record) != CASE_KEYS:
        raise ValueError("robustness case fields differ from the registered schema")
    if record.get("schema_version") != ROBUSTNESS_CASE_SCHEMA_VERSION:
        raise ValueError("unsupported robustness case schema")
    case_id = _bounded_string(record.get("case_id"), "case_id", 6, 64)
    if re.fullmatch(r"m13-[a-z0-9-]+-[0-9]{3}", case_id) is None:
        raise ValueError(f"invalid robustness case ID: {case_id}")
    text = _bounded_string(record.get("text"), f"{case_id}.text", 1, 1000)
    if "\x00" in text:
        raise ValueError(f"{case_id} contains a null byte")
    primary_family = _registered_string(
        record.get("primary_family"), REGISTERED_FAMILIES, f"{case_id}.primary_family"
    )
    tags = _string_tuple(record.get("robustness_tags"), f"{case_id}.robustness_tags")
    if primary_family not in tags or any(tag not in REGISTERED_FAMILIES for tag in tags):
        raise ValueError(f"{case_id} tags must be registered and include the primary family")
    semantic_group = _bounded_string(
        record.get("semantic_group"), f"{case_id}.semantic_group", 3, 96
    )
    if re.fullmatch(r"[a-z0-9_]+", semantic_group) is None:
        raise ValueError(f"{case_id} semantic_group must be snake_case")
    acceptable_intents = _string_tuple(
        record.get("acceptable_intents"),
        f"{case_id}.acceptable_intents",
        allow_empty=True,
    )
    if any(intent not in allowed_intents for intent in acceptable_intents):
        raise ValueError(f"{case_id} contains an unregistered acceptable intent")
    out_of_scope = record.get("out_of_scope")
    if not isinstance(out_of_scope, bool):
        raise ValueError(f"{case_id}.out_of_scope must be boolean")
    if out_of_scope != (not acceptable_intents):
        raise ValueError(f"{case_id} out_of_scope must match an empty acceptable-intent set")
    expected_action = _registered_string(
        record.get("expected_routing_action"),
        REGISTERED_ACTIONS,
        f"{case_id}.expected_routing_action",
    )
    severity = _registered_string(
        record.get("risk_severity"),
        REGISTERED_RISK_SEVERITIES,
        f"{case_id}.risk_severity",
    )
    if record.get("escalation_required") is not True:
        raise ValueError(f"{case_id} must require escalation in shadow mode")
    reason_codes = _string_tuple(
        record.get("escalation_reason_codes"), f"{case_id}.escalation_reason_codes"
    )
    if (
        "SHADOW_MODE_REQUIRES_REVIEW" not in reason_codes
        or any(code not in REGISTERED_REASON_CODES for code in reason_codes)
    ):
        raise ValueError(f"{case_id} has invalid escalation reason codes")
    if expected_action == "security_queue" and "SECURITY_INCIDENT" not in reason_codes:
        raise ValueError(f"{case_id} security routing requires SECURITY_INCIDENT")
    rationale = _bounded_string(
        record.get("escalation_rationale"), f"{case_id}.escalation_rationale", 15, 500
    )
    expected_pii_types = _string_tuple(
        record.get("expected_pii_types"), f"{case_id}.expected_pii_types", allow_empty=True
    )
    if any(pii_type not in allowed_pii_types for pii_type in expected_pii_types):
        raise ValueError(f"{case_id} contains an unregistered expected PII type")
    if record.get("provenance") != REGISTERED_PROVENANCE:
        raise ValueError(f"{case_id} provenance differs from the synthetic registration")
    if record.get("licence") != REGISTERED_LICENCE:
        raise ValueError(f"{case_id} licence differs from the pack registration")
    return RobustnessCase(
        case_id=case_id,
        text=text,
        primary_family=primary_family,
        robustness_tags=tags,
        semantic_group=semantic_group,
        acceptable_intents=acceptable_intents,
        out_of_scope=out_of_scope,
        expected_routing_action=expected_action,
        risk_severity=severity,
        escalation_reason_codes=reason_codes,
        escalation_rationale=rationale,
        expected_pii_types=expected_pii_types,
    )


def _character_ngrams(value: str, size: int) -> frozenset[str]:
    padded = f" {' '.join(value.split())} "
    return frozenset(padded[index : index + size] for index in range(len(padded) - size + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _exact_mapping(
    parent: Mapping[str, Any], key: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"robustness-evaluation {key} fields differ from registration")
    return value


def _repository_path(root: Path, value: Any, name: str) -> Path:
    relative = Path(_bounded_string(value, name, 1, 512))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be a safe repository-relative path")
    result = root / relative
    if not result.is_relative_to(root):
        raise ValueError(f"{name} escapes the repository root")
    return result


def _bounded_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or value.strip() != value
    ):
        raise ValueError(f"{name} must be a bounded non-blank string")
    return value


def _registered_string(value: Any, allowed: Iterable[str], name: str) -> str:
    result = _bounded_string(value, name, 1, 128)
    if result not in allowed:
        raise ValueError(f"{name} is not registered")
    return result


def _string_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a list of strings")
    result = tuple(_bounded_string(item, name, 1, 128) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


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


def _sha256(value: Any, name: str) -> str:
    result = _bounded_string(value, name, 64, 64)
    if any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result
