"""Leakage-aware selective prediction and possible-OOD evaluation."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PeftModel
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from governed_banking.baseline import assert_record_text_absent, assert_text_free_artifact
from governed_banking.calibration import probabilities_from_logits, write_calibration_artifact
from governed_banking.data import BankingRecord, normalize_text, sha256_file, stable_json_sha256
from governed_banking.device import select_device
from governed_banking.frozen_baseline import ModelSnapshot
from governed_banking.lora_baseline import (
    build_base_sequence_classifier,
    hash_checkpoint_files,
)

UNCERTAINTY_ARTIFACT_SCHEMA_VERSION = 1
REGISTERED_SEEDS = (17, 42, 73)
REGISTERED_SIGNALS = (
    "max_probability",
    "top_two_margin",
    "inverse_normalized_entropy",
)
PERMITTED_MODEL_SOURCES = ("validation", "synthetic_fixture")
PROHIBITED_MODEL_SPLITS = ("train", "test")


@dataclass(frozen=True)
class PossibleOODRecord:
    fixture_id: str
    domain: str
    scenario_group: str
    text: str


@dataclass(frozen=True)
class KnownPartitionConfig:
    development_role: str
    assessment_role: str
    development_fraction: float
    seed_offset: int


@dataclass(frozen=True)
class OODConfig:
    fixture_path: Path
    provenance: str
    development_role: str
    assessment_role: str
    development_fraction: float
    random_seed: int
    registry_path: Path


@dataclass(frozen=True)
class SelectionConfig:
    target_ood_recall: float
    target_selective_risk: float
    minimum_known_coverage: float


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int
    confidence_level: float
    seed_offset: int


@dataclass(frozen=True)
class UncertaintyConfig:
    experiment_name: str
    claim_scope: str
    seeds: tuple[int, ...]
    calibration_config_path: Path
    calibration_registry_path: Path
    calibration_report_pattern: str
    manifest_pattern: str
    checkpoint_pattern: str
    known_source_role: str
    known_source_role_history: str
    known_partition: KnownPartitionConfig
    possible_ood: OODConfig
    signals: tuple[str, ...]
    selection: SelectionConfig
    bootstrap: BootstrapConfig
    official_test_access_history: str
    official_test_evaluation: bool
    independent_model_evaluation: bool

    @classmethod
    def from_yaml(cls, path: Path) -> UncertaintyConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("unsupported uncertainty configuration schema")
        source = _mapping(raw, "source")
        known = _mapping(raw, "known_partition")
        ood = _mapping(raw, "possible_ood")
        signals = _mapping(raw, "signals")
        selection = _mapping(raw, "selection")
        metrics = _mapping(raw, "metrics")
        bootstrap = _mapping(raw, "bootstrap")
        boundary = _mapping(raw, "boundary")
        if tuple(raw.get("seeds", [])) != REGISTERED_SEEDS:
            raise ValueError(f"Module 8 seeds must be {REGISTERED_SEEDS}")
        claim_scope = _non_blank(raw.get("claim_scope"), "claim_scope")
        if claim_scope != "post_calibration_post_test_exploratory":
            raise ValueError("Module 8 must disclose calibration and test history")
        if source.get("known_source_role") != "calibration_assessment":
            raise ValueError("known source must be Module 7 calibration_assessment")
        expected_history = (
            "used_for_module6_checkpoint_selection_and_module7_calibration_assessment"
        )
        if source.get("known_source_role_history") != expected_history:
            raise ValueError("known source role history is incomplete")
        if known.get("development_role") != "threshold_known_development":
            raise ValueError("unexpected known development role")
        if known.get("assessment_role") != "selective_known_assessment":
            raise ValueError("unexpected known assessment role")
        if known.get("group_by_normalized_text") is not True:
            raise ValueError("known partitions must preserve normalized-text groups")
        if ood.get("development_role") != "threshold_ood_development":
            raise ValueError("unexpected possible-OOD development role")
        if ood.get("assessment_role") != "possible_ood_assessment":
            raise ValueError("unexpected possible-OOD assessment role")
        if ood.get("group_field") != "scenario_group" or ood.get("stratum_field") != "domain":
            raise ValueError("possible-OOD fixture must split scenario groups by domain")
        if ood.get("provenance") != "synthetic_authored_non_customer_data":
            raise ValueError("possible-OOD provenance must remain explicit")
        candidates = tuple(signals.get("candidates", []))
        if candidates != REGISTERED_SIGNALS or signals.get("direction") != (
            "higher_is_more_in_distribution"
        ):
            raise ValueError("uncertainty signals or direction differ from registration")
        if selection.get("objective") != "maximize_known_coverage":
            raise ValueError("selection objective must maximize known coverage")
        if tuple(selection.get("tie_breakers", [])) != (
            "maximize_possible_ood_recall",
            "minimize_selective_risk",
            "signal_registration_order",
        ):
            raise ValueError("selection tie breakers differ from registration")
        required_metrics = {
            "include_risk_coverage_curve": True,
            "include_aurc": True,
            "include_auroc": True,
            "include_average_precision": True,
            "include_error_capture": True,
        }
        if any(metrics.get(key) is not value for key, value in required_metrics.items()):
            raise ValueError("all registered Module 8 metrics must remain enabled")
        permitted = tuple(boundary.get("permitted_model_sources", []))
        prohibited = tuple(boundary.get("prohibited_model_splits", []))
        test_evaluation = _strict_bool(
            boundary.get("official_test_evaluation_in_this_module"),
            "official_test_evaluation_in_this_module",
        )
        independent = _strict_bool(
            boundary.get("independent_model_evaluation"), "independent_model_evaluation"
        )
        if permitted != PERMITTED_MODEL_SOURCES or prohibited != PROHIBITED_MODEL_SPLITS:
            raise ValueError("Module 8 model-access boundary changed")
        if boundary.get("official_test_access_history") != "observed_in_module_5":
            raise ValueError("official test access history is incomplete")
        if test_evaluation or independent:
            raise ValueError("Module 8 cannot claim test or independent evaluation")
        return cls(
            experiment_name=_non_blank(raw.get("experiment_name"), "experiment_name"),
            claim_scope=claim_scope,
            seeds=REGISTERED_SEEDS,
            calibration_config_path=Path(
                _non_blank(source.get("calibration_config"), "source.calibration_config")
            ),
            calibration_registry_path=Path(
                _non_blank(source.get("calibration_registry"), "source.calibration_registry")
            ),
            calibration_report_pattern=_seed_pattern(
                source.get("calibration_report_pattern"), "source.calibration_report_pattern"
            ),
            manifest_pattern=_seed_pattern(
                source.get("manifest_pattern"), "source.manifest_pattern"
            ),
            checkpoint_pattern=_seed_pattern(
                source.get("checkpoint_pattern"), "source.checkpoint_pattern"
            ),
            known_source_role="calibration_assessment",
            known_source_role_history=expected_history,
            known_partition=KnownPartitionConfig(
                development_role="threshold_known_development",
                assessment_role="selective_known_assessment",
                development_fraction=_bounded_float(
                    known.get("development_fraction"),
                    "known_partition.development_fraction",
                    0.1,
                    0.9,
                ),
                seed_offset=_bounded_int(
                    known.get("seed_offset"), "known_partition.seed_offset", 1, 1_000_000
                ),
            ),
            possible_ood=OODConfig(
                fixture_path=Path(_non_blank(ood.get("fixture_path"), "possible_ood.fixture_path")),
                provenance="synthetic_authored_non_customer_data",
                development_role="threshold_ood_development",
                assessment_role="possible_ood_assessment",
                development_fraction=_bounded_float(
                    ood.get("development_fraction"),
                    "possible_ood.development_fraction",
                    0.1,
                    0.9,
                ),
                random_seed=_bounded_int(
                    ood.get("random_seed"), "possible_ood.random_seed", 1, 1_000_000
                ),
                registry_path=Path(
                    _non_blank(ood.get("registry_path"), "possible_ood.registry_path")
                ),
            ),
            signals=candidates,
            selection=SelectionConfig(
                target_ood_recall=_bounded_float(
                    selection.get("target_possible_ood_recall"),
                    "selection.target_possible_ood_recall",
                    0.5,
                    1.0,
                ),
                target_selective_risk=_bounded_float(
                    selection.get("target_selective_risk"),
                    "selection.target_selective_risk",
                    0.0,
                    0.5,
                ),
                minimum_known_coverage=_bounded_float(
                    selection.get("minimum_known_coverage"),
                    "selection.minimum_known_coverage",
                    0.1,
                    1.0,
                ),
            ),
            bootstrap=BootstrapConfig(
                resamples=_bounded_int(
                    bootstrap.get("resamples"), "bootstrap.resamples", 100, 100_000
                ),
                confidence_level=_bounded_float(
                    bootstrap.get("confidence_level"),
                    "bootstrap.confidence_level",
                    0.5,
                    0.999,
                ),
                seed_offset=_bounded_int(
                    bootstrap.get("seed_offset"), "bootstrap.seed_offset", 1, 1_000_000
                ),
            ),
            official_test_access_history="observed_in_module_5",
            official_test_evaluation=False,
            independent_model_evaluation=False,
        )

    def calibration_report_path(self, seed: int) -> Path:
        return Path(self.calibration_report_pattern.format(seed=_registered_seed(seed, self.seeds)))

    def manifest_path(self, seed: int) -> Path:
        return Path(self.manifest_pattern.format(seed=_registered_seed(seed, self.seeds)))

    def checkpoint_path(self, seed: int) -> Path:
        return Path(self.checkpoint_pattern.format(seed=_registered_seed(seed, self.seeds)))


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


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _seed_pattern(value: Any, name: str) -> str:
    result = _non_blank(value, name)
    if result.count("{seed}") != 1:
        raise ValueError(f"{name} must contain exactly one {{seed}}")
    return result


def _registered_seed(seed: int, registered: Sequence[int]) -> int:
    if seed not in registered:
        raise ValueError(f"unregistered seed: {seed}")
    return seed


def assert_uncertainty_source_permitted(source_name: str) -> None:
    if source_name in PROHIBITED_MODEL_SPLITS or source_name not in PERMITTED_MODEL_SOURCES:
        raise ValueError(f"Module 8 prohibits model access to source: {source_name}")


def load_possible_ood_fixture(path: Path) -> tuple[PossibleOODRecord, ...]:
    records: list[PossibleOODRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {"id", "domain", "scenario_group", "text"}:
            raise ValueError(f"invalid possible-OOD fixture schema on line {line_number}")
        records.append(
            PossibleOODRecord(
                fixture_id=_non_blank(value["id"], f"line {line_number} id"),
                domain=_non_blank(value["domain"], f"line {line_number} domain"),
                scenario_group=_non_blank(
                    value["scenario_group"], f"line {line_number} scenario_group"
                ),
                text=_non_blank(value["text"], f"line {line_number} text"),
            )
        )
    if len(records) < 40:
        raise ValueError("possible-OOD fixture is too small for this exploratory protocol")
    ids = [record.fixture_id for record in records]
    normalized = [normalize_text(record.text) for record in records]
    if len(ids) != len(set(ids)) or len(normalized) != len(set(normalized)):
        raise ValueError("possible-OOD fixture ids and normalized texts must be unique")
    groups: dict[str, list[PossibleOODRecord]] = defaultdict(list)
    for record in records:
        groups[record.scenario_group].append(record)
    for group, members in groups.items():
        if len({member.domain for member in members}) != 1:
            raise ValueError(f"possible-OOD scenario group crosses domains: {group}")
    domain_group_counts = Counter(members[0].domain for members in groups.values())
    if len(domain_group_counts) < 6 or min(domain_group_counts.values()) < 2:
        raise ValueError("possible-OOD fixture needs multiple domains and groups per domain")
    return tuple(records)


def partition_known_records(
    records: Sequence[BankingRecord],
    *,
    development_fraction: float,
    random_seed: int,
) -> tuple[tuple[BankingRecord, ...], tuple[BankingRecord, ...]]:
    groups: dict[str, list[BankingRecord]] = defaultdict(list)
    for record in records:
        groups[normalize_text(record.text)].append(record)
    if not groups:
        raise ValueError("known calibration-assessment source cannot be empty")
    if any(len({row.category for row in group}) != 1 for group in groups.values()):
        raise ValueError("known source contains conflicting normalized-text groups")
    keys = sorted(groups, key=lambda key: min(row.source_index for row in groups[key]))
    labels = [groups[key][0].category for key in keys]
    development_keys, assessment_keys = train_test_split(
        keys,
        train_size=development_fraction,
        random_state=random_seed,
        stratify=labels,
    )
    development_set = set(development_keys)
    assessment_set = set(assessment_keys)
    development = tuple(
        record for record in records if normalize_text(record.text) in development_set
    )
    assessment = tuple(
        record for record in records if normalize_text(record.text) in assessment_set
    )
    _assert_partition(
        development_set, assessment_set, len(development), len(assessment), len(records)
    )
    if {row.category for row in development} != {row.category for row in assessment}:
        raise ValueError("known development and assessment label coverage differs")
    return development, assessment


def partition_possible_ood_records(
    records: Sequence[PossibleOODRecord],
    *,
    development_fraction: float,
    random_seed: int,
) -> tuple[tuple[PossibleOODRecord, ...], tuple[PossibleOODRecord, ...]]:
    groups: dict[str, list[PossibleOODRecord]] = defaultdict(list)
    for record in records:
        groups[record.scenario_group].append(record)
    keys = sorted(groups)
    domains = [groups[key][0].domain for key in keys]
    development_keys, assessment_keys = train_test_split(
        keys,
        train_size=development_fraction,
        random_state=random_seed,
        stratify=domains,
    )
    development_set = set(development_keys)
    assessment_set = set(assessment_keys)
    development = tuple(row for row in records if row.scenario_group in development_set)
    assessment = tuple(row for row in records if row.scenario_group in assessment_set)
    _assert_partition(
        development_set, assessment_set, len(development), len(assessment), len(records)
    )
    if {row.domain for row in development} != {row.domain for row in assessment}:
        raise ValueError("possible-OOD development and assessment domain coverage differs")
    return development, assessment


def _assert_partition(
    development_groups: set[str],
    assessment_groups: set[str],
    development_count: int,
    assessment_count: int,
    source_count: int,
) -> None:
    if development_groups & assessment_groups:
        raise AssertionError("development and assessment groups overlap")
    if development_count + assessment_count != source_count:
        raise AssertionError("partition lost source rows")


def build_uncertainty_registry(
    config: UncertaintyConfig,
    known_sources: Mapping[int, Sequence[BankingRecord]],
    possible_ood_records: Sequence[PossibleOODRecord],
    *,
    source_calibration_report_sha256: Mapping[int, str],
    source_manifest_sha256: Mapping[int, str],
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    known_entries = []
    for seed in config.seeds:
        source = known_sources[seed]
        development, assessment = partition_known_records(
            source,
            development_fraction=config.known_partition.development_fraction,
            random_seed=config.known_partition.seed_offset + seed,
        )
        known_entries.append(
            {
                "seed": seed,
                "source_calibration_report_sha256": source_calibration_report_sha256[seed],
                "source_manifest_sha256": source_manifest_sha256[seed],
                "source_indices_sha256": stable_json_sha256(
                    [record.source_index for record in source]
                ),
                config.known_partition.development_role: _known_partition_manifest(development),
                config.known_partition.assessment_role: _known_partition_manifest(assessment),
                "normalized_text_group_overlap": 0,
                "source_index_overlap": 0,
            }
        )
    ood_development, ood_assessment = partition_possible_ood_records(
        possible_ood_records,
        development_fraction=config.possible_ood.development_fraction,
        random_seed=config.possible_ood.random_seed,
    )
    artifact: dict[str, Any] = {
        "schema_version": UNCERTAINTY_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "uncertainty_role_registry",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "uncertainty_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "known_source_role": config.known_source_role,
        "known_source_role_history": config.known_source_role_history,
        "known_entries": known_entries,
        "possible_ood": {
            "fixture_sha256": sha256_file(config.possible_ood.fixture_path),
            "provenance": config.possible_ood.provenance,
            "source_count": len(possible_ood_records),
            config.possible_ood.development_role: _ood_partition_manifest(ood_development),
            config.possible_ood.assessment_role: _ood_partition_manifest(ood_assessment),
            "scenario_group_overlap": 0,
            "fixture_id_overlap": 0,
        },
        "model_access_boundary": {
            "permitted_model_sources": list(PERMITTED_MODEL_SOURCES),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
        },
        "interpretation_notice": (
            "Threshold development and assessment roles are disjoint, but known rows were already "
            "used for checkpoint selection and Module 7 calibration assessment. Possible-OOD rows "
            "are synthetic and do not estimate production prevalence or diversity."
        ),
    }
    artifact["registry_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    for records in known_sources.values():
        assert_record_text_absent(artifact, records)
    _assert_ood_text_absent(artifact, possible_ood_records)
    return artifact


def _known_partition_manifest(records: Sequence[BankingRecord]) -> dict[str, Any]:
    indices = [record.source_index for record in records]
    return {
        "count": len(records),
        "source_indices": indices,
        "source_indices_sha256": stable_json_sha256(indices),
        "label_distribution": dict(sorted(Counter(row.category for row in records).items())),
    }


def _ood_partition_manifest(records: Sequence[PossibleOODRecord]) -> dict[str, Any]:
    ids = [record.fixture_id for record in records]
    groups = sorted({record.scenario_group for record in records})
    return {
        "count": len(records),
        "fixture_ids": ids,
        "fixture_ids_sha256": stable_json_sha256(ids),
        "scenario_groups": groups,
        "scenario_groups_sha256": stable_json_sha256(groups),
        "domain_distribution": dict(sorted(Counter(row.domain for row in records).items())),
    }


def _assert_ood_text_absent(value: Any, records: Sequence[PossibleOODRecord]) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    for record in records:
        if record.text in serialized or normalize_text(record.text) in normalize_text(serialized):
            raise ValueError("artifact contains possible-OOD fixture text")


def validate_uncertainty_registry(
    artifact: Mapping[str, Any],
    *,
    config: UncertaintyConfig,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "uncertainty_role_registry":
        raise ValueError("expected uncertainty role registry")
    _validate_content_hash(artifact, "registry_sha256")
    expected = {
        "claim_scope": config.claim_scope,
        "seeds": list(config.seeds),
        "uncertainty_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "known_source_role_history": config.known_source_role_history,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"uncertainty registry has invalid {key}")
    entries = artifact.get("known_entries", [])
    if [entry.get("seed") for entry in entries] != list(config.seeds):
        raise ValueError("uncertainty registry seed entries are invalid")
    for entry in entries:
        development = entry[config.known_partition.development_role]
        assessment = entry[config.known_partition.assessment_role]
        development_indices = development.get("source_indices", [])
        assessment_indices = assessment.get("source_indices", [])
        if len(development_indices) != development.get("count") or len(
            assessment_indices
        ) != assessment.get("count"):
            raise ValueError("known partition count mismatch")
        if len(development_indices) != len(set(development_indices)) or len(
            assessment_indices
        ) != len(set(assessment_indices)):
            raise ValueError("known partition contains duplicate indices")
        if set(development_indices) & set(assessment_indices):
            raise ValueError("known uncertainty roles overlap")
        if development.get("source_indices_sha256") != stable_json_sha256(development_indices):
            raise ValueError("known development index hash mismatch")
        if assessment.get("source_indices_sha256") != stable_json_sha256(assessment_indices):
            raise ValueError("known assessment index hash mismatch")
    ood = artifact.get("possible_ood", {})
    development = ood[config.possible_ood.development_role]
    assessment = ood[config.possible_ood.assessment_role]
    development_ids = development.get("fixture_ids", [])
    assessment_ids = assessment.get("fixture_ids", [])
    if len(development_ids) != development.get("count") or len(assessment_ids) != assessment.get(
        "count"
    ):
        raise ValueError("possible-OOD partition count mismatch")
    if len(development_ids) != len(set(development_ids)) or len(assessment_ids) != len(
        set(assessment_ids)
    ):
        raise ValueError("possible-OOD partition contains duplicate fixture ids")
    if set(development_ids) & set(assessment_ids):
        raise ValueError("possible-OOD fixture roles overlap")
    development_groups = development.get("scenario_groups", [])
    assessment_groups = assessment.get("scenario_groups", [])
    if set(development_groups) & set(assessment_groups):
        raise ValueError("possible-OOD scenario groups overlap")
    if development.get("fixture_ids_sha256") != stable_json_sha256(development_ids):
        raise ValueError("possible-OOD development id hash mismatch")
    if assessment.get("fixture_ids_sha256") != stable_json_sha256(assessment_ids):
        raise ValueError("possible-OOD assessment id hash mismatch")
    if development.get("scenario_groups_sha256") != stable_json_sha256(development_groups):
        raise ValueError("possible-OOD development group hash mismatch")
    if assessment.get("scenario_groups_sha256") != stable_json_sha256(assessment_groups):
        raise ValueError("possible-OOD assessment group hash mismatch")
    if len(development_ids) + len(assessment_ids) != ood.get("source_count"):
        raise ValueError("possible-OOD partitions do not cover the registered fixture")
    if ood.get("fixture_sha256") != sha256_file(config.possible_ood.fixture_path):
        raise ValueError("possible-OOD fixture hash mismatch")
    boundary = artifact.get("model_access_boundary", {})
    if boundary.get("permitted_model_sources") != list(PERMITTED_MODEL_SOURCES):
        raise ValueError("uncertainty registry permits unexpected sources")
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("uncertainty registry reports official-test access")
    assert_text_free_artifact(artifact)


def select_known_partition(
    records: Sequence[BankingRecord], partition: Mapping[str, Any]
) -> tuple[BankingRecord, ...]:
    indices = partition.get("source_indices", [])
    lookup = {record.source_index: record for record in records}
    if any(index not in lookup for index in indices):
        raise ValueError("known uncertainty index is absent from source")
    selected = tuple(lookup[index] for index in indices)
    if len(selected) != partition.get("count"):
        raise ValueError("known uncertainty partition count mismatch")
    return selected


def select_ood_partition(
    records: Sequence[PossibleOODRecord], partition: Mapping[str, Any]
) -> tuple[PossibleOODRecord, ...]:
    ids = partition.get("fixture_ids", [])
    lookup = {record.fixture_id: record for record in records}
    if any(fixture_id not in lookup for fixture_id in ids):
        raise ValueError("possible-OOD fixture id is absent from source")
    selected = tuple(lookup[fixture_id] for fixture_id in ids)
    if len(selected) != partition.get("count"):
        raise ValueError("possible-OOD partition count mismatch")
    return selected


class _TextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: Sequence[str], tokenizer: Any, *, max_length: int) -> None:
        self._encodings = tokenizer(
            list(texts),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self._encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self._encodings.items()
        }


def extract_uncertainty_logits(
    known_records: Sequence[BankingRecord],
    ood_records: Sequence[PossibleOODRecord],
    *,
    snapshot: ModelSnapshot,
    checkpoint_directory: Path,
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    attention_implementation: str,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint_hashes = hash_checkpoint_files(checkpoint_directory)
    base_model = build_base_sequence_classifier(
        snapshot,
        label_names=label_names,
        attention_implementation=attention_implementation,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_directory, is_trainable=False)
    device, runtime = select_device(device_name)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True)
    texts = [record.text for record in known_records] + [record.text for record in ood_records]
    dataset = _TextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: tokenizer.pad(rows, padding=True, return_tensors="pt"),
    )
    parts: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            parts.append(model(**inputs).logits.detach().cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    logits = torch.cat(parts).numpy().astype(np.float64)
    known_count = len(known_records)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    known_labels = np.asarray(
        [label_to_id[record.category] for record in known_records], dtype=np.int64
    )
    known_logits = logits[:known_count]
    ood_logits = logits[known_count:]
    metadata = {
        "known_row_count": known_count,
        "possible_ood_row_count": len(ood_records),
        "known_logits_sha256": stable_json_sha256(known_logits.round(10).tolist()),
        "possible_ood_logits_sha256": stable_json_sha256(ood_logits.round(10).tolist()),
        "known_labels_sha256": stable_json_sha256(known_labels.tolist()),
        "checkpoint_files_sha256": checkpoint_hashes,
        "extraction_seconds": _metric_float(seconds),
        "runtime_device": runtime.to_dict(),
    }
    return known_logits, known_labels, ood_logits, metadata


def uncertainty_signals(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have at least two classes")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError("probabilities must be finite and sum to one")
    ordered = np.sort(probabilities, axis=1)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)), axis=1
    )
    normalized_entropy = entropy / math.log(probabilities.shape[1])
    return {
        "max_probability": ordered[:, -1],
        "top_two_margin": ordered[:, -1] - ordered[:, -2],
        "inverse_normalized_entropy": 1.0 - normalized_entropy,
    }


def selection_metrics(
    known_scores: np.ndarray,
    known_correct: np.ndarray,
    ood_scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    if not len(known_scores) or not len(ood_scores) or len(known_scores) != len(known_correct):
        raise ValueError("selection metrics require non-empty aligned arrays")
    accepted = known_scores >= threshold
    rejected_ood = ood_scores < threshold
    accepted_count = int(accepted.sum())
    accepted_errors = int(np.sum(accepted & ~known_correct))
    total_errors = int((~known_correct).sum())
    rejected_errors = int(np.sum(~accepted & ~known_correct))
    return {
        "known_count": len(known_scores),
        "known_accepted": accepted_count,
        "known_coverage": _metric_float(accepted_count / len(known_scores)),
        "selective_risk": _metric_float(
            accepted_errors / accepted_count if accepted_count else 0.0
        ),
        "accepted_errors": accepted_errors,
        "false_automation_rate": _metric_float(accepted_errors / len(known_scores)),
        "known_errors": total_errors,
        "error_capture_rate": _metric_float(
            rejected_errors / total_errors if total_errors else 1.0
        ),
        "possible_ood_count": len(ood_scores),
        "possible_ood_rejected": int(rejected_ood.sum()),
        "possible_ood_recall": _metric_float(rejected_ood.mean()),
        "possible_ood_false_acceptance_rate": _metric_float((~rejected_ood).mean()),
    }


def select_signal_and_threshold(
    known_signals: Mapping[str, np.ndarray],
    known_correct: np.ndarray,
    ood_signals: Mapping[str, np.ndarray],
    *,
    config: UncertaintyConfig,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for signal_order, signal in enumerate(config.signals):
        known_scores = np.asarray(known_signals[signal], dtype=np.float64)
        ood_scores = np.asarray(ood_signals[signal], dtype=np.float64)
        thresholds = np.unique(np.concatenate((known_scores, ood_scores)))
        for threshold in thresholds:
            metrics = selection_metrics(
                known_scores, known_correct, ood_scores, threshold=float(threshold)
            )
            feasible = (
                metrics["known_coverage"] >= config.selection.minimum_known_coverage
                and metrics["selective_risk"] <= config.selection.target_selective_risk
                and metrics["possible_ood_recall"] >= config.selection.target_ood_recall
            )
            coverage_shortfall = (
                max(0.0, config.selection.minimum_known_coverage - metrics["known_coverage"])
                / config.selection.minimum_known_coverage
            )
            risk_excess = max(
                0.0, metrics["selective_risk"] - config.selection.target_selective_risk
            ) / (1.0 - config.selection.target_selective_risk)
            recall_shortfall = (
                max(0.0, config.selection.target_ood_recall - metrics["possible_ood_recall"])
                / config.selection.target_ood_recall
            )
            evaluated.append(
                {
                    "signal": signal,
                    "signal_registration_order": signal_order,
                    "threshold": float(threshold),
                    "feasible": feasible,
                    "constraint_violation": coverage_shortfall + risk_excess + recall_shortfall,
                    "metrics": metrics,
                }
            )
    feasible = [candidate for candidate in evaluated if candidate["feasible"]]
    if feasible:
        winner = min(
            feasible,
            key=lambda row: (
                -row["metrics"]["known_coverage"],
                -row["metrics"]["possible_ood_recall"],
                row["metrics"]["selective_risk"],
                row["signal_registration_order"],
                row["threshold"],
            ),
        )
    else:
        winner = min(
            evaluated,
            key=lambda row: (
                row["constraint_violation"],
                -row["metrics"]["known_coverage"],
                -row["metrics"]["possible_ood_recall"],
                row["metrics"]["selective_risk"],
                row["signal_registration_order"],
                row["threshold"],
            ),
        )
    by_signal = {}
    for signal in config.signals:
        rows = [row for row in evaluated if row["signal"] == signal]
        feasible_rows = [row for row in rows if row["feasible"]]
        representative = min(
            feasible_rows or rows,
            key=lambda row: (
                0.0 if row["feasible"] else row["constraint_violation"],
                -row["metrics"]["known_coverage"],
                -row["metrics"]["possible_ood_recall"],
                row["metrics"]["selective_risk"],
                row["threshold"],
            ),
        )
        by_signal[signal] = {
            "thresholds_evaluated": len(rows),
            "has_feasible_threshold": bool(feasible_rows),
            "representative_threshold": _metric_float(representative["threshold"]),
            "representative_metrics": representative["metrics"],
        }
    return {
        "development_constraints_feasible": bool(feasible),
        "selected_signal": winner["signal"],
        "selected_threshold": _metric_float(winner["threshold"]),
        "selected_development_metrics": winner["metrics"],
        "selected_constraint_violation": _metric_float(winner["constraint_violation"]),
        "candidate_summary": by_signal,
    }


def detection_metrics(known_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate(
        (np.zeros(len(known_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64))
    )
    uncertainty = -np.concatenate((known_scores, ood_scores))
    return {
        "possible_ood_auroc": _metric_float(roc_auc_score(labels, uncertainty)),
        "possible_ood_average_precision": _metric_float(
            average_precision_score(labels, uncertainty)
        ),
    }


def risk_coverage_evidence(scores: np.ndarray, correct: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, kind="stable")
    ordered_errors = (~correct[order]).astype(np.float64)
    cumulative_risk = np.cumsum(ordered_errors) / np.arange(1, len(scores) + 1)
    aurc = float(cumulative_risk.mean())
    points = []
    for target in np.linspace(0.1, 1.0, 10):
        count = max(1, int(math.ceil(target * len(scores))))
        points.append(
            {
                "coverage": _metric_float(count / len(scores)),
                "risk": _metric_float(cumulative_risk[count - 1]),
                "accepted": count,
            }
        )
    return {"aurc": _metric_float(aurc), "curve": points}


def per_domain_ood_recall(
    records: Sequence[PossibleOODRecord], scores: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    domains: dict[str, list[bool]] = defaultdict(list)
    for record, score in zip(records, scores, strict=True):
        domains[record.domain].append(bool(score < threshold))
    return {
        domain: {"count": len(values), "recall": _metric_float(statistics.fmean(values))}
        for domain, values in sorted(domains.items())
    }


def bootstrap_assessment(
    known_scores: np.ndarray,
    known_correct: np.ndarray,
    ood_scores: np.ndarray,
    *,
    threshold: float,
    config: BootstrapConfig,
    random_seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(random_seed)
    names = (
        "known_coverage",
        "selective_risk",
        "false_automation_rate",
        "error_capture_rate",
        "possible_ood_recall",
        "possible_ood_false_acceptance_rate",
        "possible_ood_auroc",
        "possible_ood_average_precision",
    )
    values = {name: np.empty(config.resamples, dtype=np.float64) for name in names}
    for iteration in range(config.resamples):
        known_indices = generator.integers(0, len(known_scores), size=len(known_scores))
        ood_indices = generator.integers(0, len(ood_scores), size=len(ood_scores))
        metrics = selection_metrics(
            known_scores[known_indices],
            known_correct[known_indices],
            ood_scores[ood_indices],
            threshold=threshold,
        )
        metrics.update(detection_metrics(known_scores[known_indices], ood_scores[ood_indices]))
        for name in names:
            values[name][iteration] = float(metrics[name])
    alpha = 1.0 - config.confidence_level
    return {
        "resamples": config.resamples,
        "confidence_level": config.confidence_level,
        "random_seed": random_seed,
        "intervals": {
            name: {
                "lower": _metric_float(np.quantile(samples, alpha / 2.0)),
                "median": _metric_float(np.quantile(samples, 0.5)),
                "upper": _metric_float(np.quantile(samples, 1.0 - alpha / 2.0)),
            }
            for name, samples in values.items()
        },
    }


def build_seed_uncertainty_report(
    config: UncertaintyConfig,
    *,
    seed: int,
    temperature: float,
    known_development_logits: np.ndarray,
    known_development_labels: np.ndarray,
    known_assessment_logits: np.ndarray,
    known_assessment_labels: np.ndarray,
    ood_development_logits: np.ndarray,
    ood_assessment_logits: np.ndarray,
    ood_assessment_records: Sequence[PossibleOODRecord],
    partition_entry: Mapping[str, Any],
    ood_partition: Mapping[str, Any],
    extraction_metadata: Mapping[str, Any],
    source_calibration_report_sha256: str,
    source_manifest_sha256: str,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    known_development_probabilities = probabilities_from_logits(
        known_development_logits, temperature
    )
    known_assessment_probabilities = probabilities_from_logits(known_assessment_logits, temperature)
    ood_development_probabilities = probabilities_from_logits(ood_development_logits, temperature)
    ood_assessment_probabilities = probabilities_from_logits(ood_assessment_logits, temperature)
    known_development_correct = (
        known_development_probabilities.argmax(axis=1) == known_development_labels
    )
    known_assessment_correct = (
        known_assessment_probabilities.argmax(axis=1) == known_assessment_labels
    )
    known_development_signals = uncertainty_signals(known_development_probabilities)
    known_assessment_signals = uncertainty_signals(known_assessment_probabilities)
    ood_development_signals = uncertainty_signals(ood_development_probabilities)
    ood_assessment_signals = uncertainty_signals(ood_assessment_probabilities)
    selection = select_signal_and_threshold(
        known_development_signals,
        known_development_correct,
        ood_development_signals,
        config=config,
    )
    signal = selection["selected_signal"]
    threshold = selection["selected_threshold"]
    known_scores = known_assessment_signals[signal]
    ood_scores = ood_assessment_signals[signal]
    assessment = selection_metrics(
        known_scores, known_assessment_correct, ood_scores, threshold=threshold
    )
    assessment.update(detection_metrics(known_scores, ood_scores))
    risk_coverage = risk_coverage_evidence(known_scores, known_assessment_correct)
    assessment["aurc"] = risk_coverage["aurc"]
    gates = {
        "known_coverage_at_or_above_minimum": (
            assessment["known_coverage"] >= config.selection.minimum_known_coverage
        ),
        "selective_risk_at_or_below_target": (
            assessment["selective_risk"] <= config.selection.target_selective_risk
        ),
        "possible_ood_recall_at_or_above_target": (
            assessment["possible_ood_recall"] >= config.selection.target_ood_recall
        ),
    }
    report: dict[str, Any] = {
        "schema_version": UNCERTAINTY_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "selective_prediction_possible_ood_assessment",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seed": seed,
        "uncertainty_config_sha256": config_sha256,
        "partition_registry_sha256": registry_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "source": {
            "manifest_sha256": source_manifest_sha256,
            "calibration_report_sha256": source_calibration_report_sha256,
            "temperature": temperature,
            "known_role_history": config.known_source_role_history,
            "possible_ood_provenance": config.possible_ood.provenance,
        },
        "data_boundary": {
            "loaded_model_sources": list(PERMITTED_MODEL_SOURCES),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "development_and_assessment_roles_disjoint": True,
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
        },
        "partitions": {
            config.known_partition.development_role: {
                "count": partition_entry[config.known_partition.development_role]["count"],
                "source_indices_sha256": partition_entry[config.known_partition.development_role][
                    "source_indices_sha256"
                ],
            },
            config.known_partition.assessment_role: {
                "count": partition_entry[config.known_partition.assessment_role]["count"],
                "source_indices_sha256": partition_entry[config.known_partition.assessment_role][
                    "source_indices_sha256"
                ],
            },
            config.possible_ood.development_role: {
                "count": ood_partition[config.possible_ood.development_role]["count"],
                "fixture_ids_sha256": ood_partition[config.possible_ood.development_role][
                    "fixture_ids_sha256"
                ],
            },
            config.possible_ood.assessment_role: {
                "count": ood_partition[config.possible_ood.assessment_role]["count"],
                "fixture_ids_sha256": ood_partition[config.possible_ood.assessment_role][
                    "fixture_ids_sha256"
                ],
            },
        },
        "extraction": dict(extraction_metadata),
        "development_selection": selection,
        "assessment": {
            "selected_signal": signal,
            "locked_threshold": threshold,
            "metrics": assessment,
            "risk_coverage": risk_coverage,
            "possible_ood_recall_by_domain": per_domain_ood_recall(
                ood_assessment_records, ood_scores, threshold=threshold
            ),
        },
        "bootstrap_assessment": bootstrap_assessment(
            known_scores,
            known_assessment_correct,
            ood_scores,
            threshold=threshold,
            config=config.bootstrap,
            random_seed=config.bootstrap.seed_offset + seed,
        ),
        "acceptance_gate": {
            **gates,
            "all_passed": all(gates.values()),
            "registered_minimum_known_coverage": config.selection.minimum_known_coverage,
            "registered_maximum_selective_risk": config.selection.target_selective_risk,
            "registered_minimum_possible_ood_recall": config.selection.target_ood_recall,
        },
        "interpretation_notice": (
            "The locked threshold was selected only on development roles. Assessment roles are "
            "disjoint, but known data have prior selection and calibration-assessment history, and "
            "possible-OOD data are synthetic. This is not production OOD validation."
        ),
        "software": software_versions(),
    }
    report["report_sha256"] = stable_json_sha256(report)
    assert_text_free_artifact(report)
    return report


def aggregate_uncertainty_reports(
    config: UncertaintyConfig,
    reports: Sequence[Mapping[str, Any]],
    *,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if [report.get("seed") for report in reports] != list(config.seeds):
        raise ValueError("uncertainty reports must follow registered seed order")
    metric_names = (
        "known_coverage",
        "selective_risk",
        "false_automation_rate",
        "error_capture_rate",
        "possible_ood_recall",
        "possible_ood_false_acceptance_rate",
        "possible_ood_auroc",
        "possible_ood_average_precision",
        "aurc",
    )
    aggregate_metrics = {}
    for name in metric_names:
        values = [float(report["assessment"]["metrics"][name]) for report in reports]
        aggregate_metrics[name] = _descriptive_statistics(values, reports)
    artifact: dict[str, Any] = {
        "schema_version": UNCERTAINTY_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "multiseed_selective_prediction_possible_ood_aggregate",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "uncertainty_config_sha256": config_sha256,
        "partition_registry_sha256": registry_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "report_hashes": {str(report["seed"]): report["report_sha256"] for report in reports},
        "selected_signals": {
            str(report["seed"]): report["assessment"]["selected_signal"] for report in reports
        },
        "selected_thresholds": {
            str(report["seed"]): report["assessment"]["locked_threshold"] for report in reports
        },
        "assessment_metrics": aggregate_metrics,
        "acceptance_gate": {
            "all_seeds_passed": all(
                bool(report["acceptance_gate"]["all_passed"]) for report in reports
            ),
            "by_seed": {str(report["seed"]): report["acceptance_gate"] for report in reports},
        },
        "data_boundary": {
            "development_and_assessment_roles_disjoint": True,
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
            "known_source_role_history": config.known_source_role_history,
            "possible_ood_provenance": config.possible_ood.provenance,
        },
        "interpretation_notice": (
            "Aggregate assessment is post-selection and post-test exploratory. Synthetic possible-"
            "OOD results are a controlled challenge test, not evidence of real-world OOD coverage."
        ),
    }
    artifact["aggregate_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    return artifact


def _descriptive_statistics(
    values: Sequence[float], reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    half_width = float(student_t.ppf(0.975, df=len(values) - 1)) * std / math.sqrt(len(values))
    return {
        "values_by_seed": {
            str(report["seed"]): _metric_float(value)
            for report, value in zip(reports, values, strict=True)
        },
        "mean": _metric_float(mean),
        "sample_standard_deviation": _metric_float(std),
        "confidence_interval_95": [
            _metric_float(mean - half_width),
            _metric_float(mean + half_width),
        ],
    }


def validate_seed_uncertainty_report(
    artifact: Mapping[str, Any],
    *,
    config: UncertaintyConfig,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "selective_prediction_possible_ood_assessment":
        raise ValueError("expected selective-prediction assessment")
    _validate_content_hash(artifact, "report_sha256")
    expected = {
        "claim_scope": config.claim_scope,
        "partition_registry_sha256": registry_sha256,
        "uncertainty_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"uncertainty report has invalid {key}")
    if artifact.get("seed") not in config.seeds:
        raise ValueError("uncertainty report uses unregistered seed")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("uncertainty report accessed official test")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("uncertainty report computed official-test metrics")
    if boundary.get("development_and_assessment_roles_disjoint") is not True:
        raise ValueError("uncertainty development and assessment roles overlap")
    if "assessment" not in artifact or "development_selection" not in artifact:
        raise ValueError("uncertainty report is missing role-specific evidence")
    selection = artifact["development_selection"]
    assessment = artifact["assessment"]
    if selection.get("selected_signal") not in config.signals:
        raise ValueError("uncertainty report selected an unregistered signal")
    if assessment.get("selected_signal") != selection.get("selected_signal"):
        raise ValueError("assessment signal differs from development selection")
    if assessment.get("locked_threshold") != selection.get("selected_threshold"):
        raise ValueError("assessment threshold differs from development selection")
    assert_text_free_artifact(artifact)


def validate_uncertainty_aggregate(
    artifact: Mapping[str, Any],
    *,
    config: UncertaintyConfig,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "multiseed_selective_prediction_possible_ood_aggregate":
        raise ValueError("expected multi-seed uncertainty aggregate")
    _validate_content_hash(artifact, "aggregate_sha256")
    expected = {
        "claim_scope": config.claim_scope,
        "seeds": list(config.seeds),
        "partition_registry_sha256": registry_sha256,
        "uncertainty_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"uncertainty aggregate has invalid {key}")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("uncertainty aggregate reports official-test access")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("uncertainty aggregate reports official-test metrics")
    assert_text_free_artifact(artifact)


def _validate_content_hash(artifact: Mapping[str, Any], hash_field: str) -> None:
    body = dict(artifact)
    expected_hash = body.pop(hash_field, None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError(f"{hash_field} content hash check failed")


def _metric_float(value: float) -> float:
    return round(float(value), 10)


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
    }


def write_uncertainty_artifact(value: Mapping[str, Any], destination: Path) -> None:
    write_calibration_artifact(value, destination)
