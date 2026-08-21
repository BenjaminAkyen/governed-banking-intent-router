"""Validation-partitioned temperature scaling and calibration assessment."""

from __future__ import annotations

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
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import t as student_t
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from governed_banking.baseline import (
    assert_record_text_absent,
    assert_text_free_artifact,
    write_json_artifact,
)
from governed_banking.data import BankingRecord, normalize_text, stable_json_sha256
from governed_banking.device import select_device
from governed_banking.frozen_baseline import ModelSnapshot
from governed_banking.lora_baseline import (
    TokenizedBankingDataset,
    build_base_sequence_classifier,
    hash_checkpoint_files,
    token_length_audit,
)

CALIBRATION_CONFIG_SCHEMA_VERSION = 1
CALIBRATION_ARTIFACT_SCHEMA_VERSION = 1
REGISTERED_SEEDS = (17, 42, 73)
PERMITTED_MODEL_SPLITS = ("validation",)
PROHIBITED_MODEL_SPLITS = ("train", "test")


@dataclass(frozen=True)
class PartitionConfig:
    development_role: str
    assessment_role: str
    development_fraction: float
    seed_offset: int
    registry_path: Path


@dataclass(frozen=True)
class TemperatureConfig:
    minimum: float
    maximum: float
    absolute_tolerance: float
    maximum_iterations: int


@dataclass(frozen=True)
class BootstrapConfig:
    resamples: int
    confidence_level: float
    seed_offset: int


@dataclass(frozen=True)
class CalibrationConfig:
    experiment_name: str
    claim_scope: str
    seeds: tuple[int, ...]
    multiseed_config_path: Path
    manifest_pattern: str
    validation_report_pattern: str
    checkpoint_pattern: str
    source_split: str
    source_role_history: str
    partition: PartitionConfig
    temperature: TemperatureConfig
    ece_bins: int
    bootstrap: BootstrapConfig
    calibrated_ece_maximum: float
    permitted_model_splits: tuple[str, ...]
    prohibited_model_splits: tuple[str, ...]
    official_test_access_history: str
    official_test_evaluation: bool
    independent_model_evaluation: bool

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("unsupported calibration configuration schema")
        source = _mapping(raw, "source")
        partition = _mapping(raw, "partition")
        temperature = _mapping(raw, "temperature_scaling")
        metrics = _mapping(raw, "metrics")
        bootstrap = _mapping(raw, "bootstrap")
        gate = _mapping(raw, "acceptance_gate")
        boundary = _mapping(raw, "boundary")
        if tuple(raw.get("seeds", [])) != REGISTERED_SEEDS:
            raise ValueError(f"Module 7 seeds must be {REGISTERED_SEEDS}")
        claim_scope = _non_blank(raw.get("claim_scope"), "claim_scope")
        if claim_scope != "post_selection_post_test_exploratory":
            raise ValueError("Module 7 claim scope must disclose selection and test history")
        source_split = _non_blank(source.get("split"), "source.split")
        if source_split != "validation":
            raise ValueError("Module 7 must partition only Module 6 validation rows")
        role_history = _non_blank(source.get("role_history"), "source.role_history")
        if role_history != "used_for_module6_checkpoint_selection":
            raise ValueError("Module 7 must disclose prior checkpoint-selection use")
        development_role = _non_blank(
            partition.get("development_role"), "partition.development_role"
        )
        assessment_role = _non_blank(partition.get("assessment_role"), "partition.assessment_role")
        if (development_role, assessment_role) != (
            "temperature_fit",
            "calibration_assessment",
        ):
            raise ValueError("unexpected calibration partition roles")
        if (
            _strict_bool(
                partition.get("group_by_normalized_text"), "partition.group_by_normalized_text"
            )
            is not True
        ):
            raise ValueError("calibration partitioning must keep normalized-text groups together")
        if temperature.get("parameterization") != "log_temperature":
            raise ValueError("temperature must use a positive log-temperature parameterization")
        if temperature.get("optimizer") != "scipy_bounded":
            raise ValueError("Module 7 requires the bounded scalar optimizer")
        required_metric_policy = {
            "ece_binning": "fixed_width",
            "include_nll": True,
            "include_multiclass_brier": True,
            "include_mce": True,
            "include_confidence_gap": True,
        }
        for key, expected in required_metric_policy.items():
            if metrics.get(key) != expected:
                raise ValueError(f"metrics.{key} must be {expected!r}")
        if gate.get("calibrated_ece_must_be_lower") is not True:
            raise ValueError("calibration gate must require lower assessment ECE")
        permitted = tuple(str(value) for value in boundary.get("permitted_model_splits", []))
        prohibited = tuple(str(value) for value in boundary.get("prohibited_model_splits", []))
        if permitted != PERMITTED_MODEL_SPLITS or prohibited != PROHIBITED_MODEL_SPLITS:
            raise ValueError("Module 7 may access only the validation split")
        test_history = _non_blank(
            boundary.get("official_test_access_history"), "official_test_access_history"
        )
        test_evaluation = _strict_bool(
            boundary.get("official_test_evaluation_in_this_module"),
            "official_test_evaluation_in_this_module",
        )
        independent = _strict_bool(
            boundary.get("independent_model_evaluation"), "independent_model_evaluation"
        )
        if test_history != "observed_in_module_5" or test_evaluation or independent:
            raise ValueError("Module 7 boundary overstates independence or permits test use")
        manifest_pattern = _pattern(source.get("manifest_pattern"), "manifest_pattern")
        report_pattern = _pattern(
            source.get("validation_report_pattern"), "validation_report_pattern"
        )
        checkpoint_pattern = _pattern(source.get("checkpoint_pattern"), "checkpoint_pattern")
        minimum_temperature = _bounded_float(
            temperature.get("minimum_temperature"),
            "temperature.minimum_temperature",
            0.001,
            1.0,
        )
        maximum_temperature = _bounded_float(
            temperature.get("maximum_temperature"),
            "temperature.maximum_temperature",
            1.0,
            100.0,
        )
        if minimum_temperature >= maximum_temperature:
            raise ValueError("temperature bounds are invalid")
        confidence_level = _bounded_float(
            bootstrap.get("confidence_level"), "bootstrap.confidence_level", 0.5, 0.999
        )
        return cls(
            experiment_name=_non_blank(raw.get("experiment_name"), "experiment_name"),
            claim_scope=claim_scope,
            seeds=REGISTERED_SEEDS,
            multiseed_config_path=Path(
                _non_blank(source.get("multiseed_config"), "source.multiseed_config")
            ),
            manifest_pattern=manifest_pattern,
            validation_report_pattern=report_pattern,
            checkpoint_pattern=checkpoint_pattern,
            source_split=source_split,
            source_role_history=role_history,
            partition=PartitionConfig(
                development_role=development_role,
                assessment_role=assessment_role,
                development_fraction=_bounded_float(
                    partition.get("development_fraction"),
                    "partition.development_fraction",
                    0.1,
                    0.9,
                ),
                seed_offset=_bounded_int(
                    partition.get("seed_offset"), "partition.seed_offset", 1, 1_000_000
                ),
                registry_path=Path(
                    _non_blank(partition.get("registry_path"), "partition.registry_path")
                ),
            ),
            temperature=TemperatureConfig(
                minimum=minimum_temperature,
                maximum=maximum_temperature,
                absolute_tolerance=_bounded_float(
                    temperature.get("absolute_tolerance"),
                    "temperature.absolute_tolerance",
                    1e-12,
                    0.01,
                ),
                maximum_iterations=_bounded_int(
                    temperature.get("maximum_iterations"),
                    "temperature.maximum_iterations",
                    10,
                    10_000,
                ),
            ),
            ece_bins=_bounded_int(metrics.get("ece_bins"), "metrics.ece_bins", 2, 100),
            bootstrap=BootstrapConfig(
                resamples=_bounded_int(
                    bootstrap.get("resamples"), "bootstrap.resamples", 100, 100_000
                ),
                confidence_level=confidence_level,
                seed_offset=_bounded_int(
                    bootstrap.get("seed_offset"), "bootstrap.seed_offset", 1, 1_000_000
                ),
            ),
            calibrated_ece_maximum=_bounded_float(
                gate.get("calibrated_ece_maximum"),
                "acceptance_gate.calibrated_ece_maximum",
                0.0,
                1.0,
            ),
            permitted_model_splits=permitted,
            prohibited_model_splits=prohibited,
            official_test_access_history=test_history,
            official_test_evaluation=test_evaluation,
            independent_model_evaluation=independent,
        )

    def manifest_path(self, seed: int) -> Path:
        return Path(self.manifest_pattern.format(seed=_registered_seed(seed, self.seeds)))

    def validation_report_path(self, seed: int) -> Path:
        return Path(self.validation_report_pattern.format(seed=_registered_seed(seed, self.seeds)))

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


def _pattern(value: Any, name: str) -> str:
    result = _non_blank(value, name)
    if result.count("{seed}") != 1:
        raise ValueError(f"{name} must contain exactly one {{seed}}")
    return result


def _registered_seed(seed: int, registered: Sequence[int]) -> int:
    if seed not in registered:
        raise ValueError(f"unregistered seed: {seed}")
    return seed


def assert_calibration_split_permitted(split_name: str) -> None:
    if split_name in PROHIBITED_MODEL_SPLITS or split_name not in PERMITTED_MODEL_SPLITS:
        raise ValueError(f"Module 7 prohibits model access to split: {split_name}")


def partition_calibration_records(
    records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    development_fraction: float,
    random_seed: int,
) -> tuple[tuple[BankingRecord, ...], tuple[BankingRecord, ...]]:
    """Split normalized-text groups so fitting and assessment rows never overlap."""

    if not records:
        raise ValueError("validation records cannot be empty")
    groups: dict[str, list[BankingRecord]] = defaultdict(list)
    for record in records:
        if record.source_split != "official_train":
            raise ValueError("calibration records must come from official_train validation")
        groups[normalize_text(record.text)].append(record)
    if any(len({record.category for record in group}) != 1 for group in groups.values()):
        raise ValueError("calibration source contains a conflicting normalized-text group")
    keys = sorted(groups, key=lambda key: min(row.source_index for row in groups[key]))
    group_labels = [groups[key][0].category for key in keys]
    development_keys, assessment_keys = train_test_split(
        keys,
        train_size=development_fraction,
        random_state=random_seed,
        stratify=group_labels,
    )
    development_set = set(development_keys)
    assessment_set = set(assessment_keys)
    development = tuple(
        record for record in records if normalize_text(record.text) in development_set
    )
    assessment = tuple(
        record for record in records if normalize_text(record.text) in assessment_set
    )
    if development_set & assessment_set:
        raise AssertionError("calibration normalized-text groups overlap")
    if {record.source_index for record in development} & {
        record.source_index for record in assessment
    }:
        raise AssertionError("calibration source indices overlap")
    expected_labels = set(label_names)
    for role, partition in (
        ("temperature_fit", development),
        ("calibration_assessment", assessment),
    ):
        if {record.category for record in partition} != expected_labels:
            raise ValueError(f"{role} partition is missing registered labels")
    if len(development) + len(assessment) != len(records):
        raise AssertionError("calibration partition lost source rows")
    return development, assessment


def build_calibration_registry(
    config: CalibrationConfig,
    source_records: Mapping[int, Sequence[BankingRecord]],
    *,
    label_names_by_seed: Mapping[int, Sequence[str]],
    source_manifest_sha256: Mapping[int, str],
    source_run_sha256: Mapping[int, str],
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for seed in config.seeds:
        records = source_records[seed]
        labels = label_names_by_seed[seed]
        development, assessment = partition_calibration_records(
            records,
            label_names=labels,
            development_fraction=config.partition.development_fraction,
            random_seed=config.partition.seed_offset + seed,
        )
        entries.append(
            {
                "seed": seed,
                "source_manifest_sha256": source_manifest_sha256[seed],
                "source_run_sha256": source_run_sha256[seed],
                "source_validation_indices_sha256": stable_json_sha256(
                    [record.source_index for record in records]
                ),
                "temperature_fit": _partition_manifest(development, labels),
                "calibration_assessment": _partition_manifest(assessment, labels),
                "normalized_text_group_overlap": 0,
                "source_index_overlap": 0,
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "calibration_partition_registry",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "calibration_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "source_role": "validation",
        "source_role_history": config.source_role_history,
        "partition_policy": {
            "development_role": config.partition.development_role,
            "assessment_role": config.partition.assessment_role,
            "development_fraction": config.partition.development_fraction,
            "seed_offset": config.partition.seed_offset,
            "group_by_normalized_text": True,
        },
        "model_access_boundary": {
            "permitted_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_splits": list(PROHIBITED_MODEL_SPLITS),
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
        },
        "entries": entries,
        "interpretation_notice": (
            "Temperature fitting and calibration assessment use disjoint rows, but both came from "
            "the validation pool previously used for Module 6 checkpoint selection."
        ),
    }
    artifact["registry_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    for records in source_records.values():
        assert_record_text_absent(artifact, records)
    return artifact


def _partition_manifest(
    records: Sequence[BankingRecord], label_names: Sequence[str]
) -> dict[str, Any]:
    indices = [record.source_index for record in records]
    counts = Counter(record.category for record in records)
    return {
        "count": len(records),
        "source_indices": indices,
        "source_indices_sha256": stable_json_sha256(indices),
        "label_distribution": {label: counts[label] for label in label_names},
    }


def validate_calibration_registry(
    artifact: Mapping[str, Any],
    *,
    config: CalibrationConfig,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "calibration_partition_registry":
        raise ValueError("expected a calibration partition registry")
    body = dict(artifact)
    expected_hash = body.pop("registry_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("calibration registry content hash check failed")
    expected = {
        "claim_scope": config.claim_scope,
        "seeds": list(config.seeds),
        "calibration_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "source_role_history": config.source_role_history,
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"calibration registry has an invalid {key}")
    boundary = artifact.get("model_access_boundary", {})
    if boundary.get("permitted_splits") != list(PERMITTED_MODEL_SPLITS):
        raise ValueError("calibration registry permits an invalid split")
    if boundary.get("prohibited_splits") != list(PROHIBITED_MODEL_SPLITS):
        raise ValueError("calibration registry does not prohibit train and test")
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("calibration registry reports test access")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("calibration registry reports official-test metrics")
    entries = artifact.get("entries", [])
    if [entry.get("seed") for entry in entries] != list(config.seeds):
        raise ValueError("calibration registry seed entries are invalid")
    for entry in entries:
        fit = entry.get("temperature_fit", {})
        assessment = entry.get("calibration_assessment", {})
        fit_indices = fit.get("source_indices", [])
        assessment_indices = assessment.get("source_indices", [])
        if set(fit_indices) & set(assessment_indices):
            raise ValueError("calibration registry partitions overlap")
        if fit.get("source_indices_sha256") != stable_json_sha256(fit_indices):
            raise ValueError("temperature-fit index hash check failed")
        if assessment.get("source_indices_sha256") != stable_json_sha256(assessment_indices):
            raise ValueError("calibration-assessment index hash check failed")
        if entry.get("normalized_text_group_overlap") != 0:
            raise ValueError("calibration registry reports group overlap")
    assert_text_free_artifact(artifact)


def select_partition_records(
    records: Sequence[BankingRecord], partition: Mapping[str, Any]
) -> tuple[BankingRecord, ...]:
    indices = partition.get("source_indices")
    if not isinstance(indices, list) or len(indices) != len(set(indices)):
        raise ValueError("calibration partition indices must be a unique list")
    if partition.get("source_indices_sha256") != stable_json_sha256(indices):
        raise ValueError("calibration partition index hash check failed")
    lookup = {record.source_index: record for record in records}
    if any(index not in lookup for index in indices):
        raise ValueError("calibration partition index is absent from source validation")
    selected = tuple(lookup[index] for index in indices)
    if len(selected) != partition.get("count"):
        raise ValueError("calibration partition count mismatch")
    return selected


def extract_logits(
    records: Sequence[BankingRecord],
    *,
    snapshot: ModelSnapshot,
    checkpoint_directory: Path,
    label_names: Sequence[str],
    max_length: int,
    batch_size: int,
    attention_implementation: str,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    label_to_id = {label: index for index, label in enumerate(label_names)}
    dataset = TokenizedBankingDataset(
        records, tokenizer, label_to_id=label_to_id, max_length=max_length
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            parts.append(model(**inputs).logits.detach().cpu())
    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    logits = torch.cat(parts).numpy().astype(np.float64)
    labels = np.asarray([label_to_id[record.category] for record in records], dtype=np.int64)
    metadata = {
        "row_count": len(records),
        "logit_shape": list(logits.shape),
        "logits_sha256": stable_json_sha256(logits.round(10).tolist()),
        "labels_sha256": stable_json_sha256(labels.tolist()),
        "checkpoint_files_sha256": checkpoint_hashes,
        "extraction_seconds": _metric_float(seconds),
        "token_length_audit": token_length_audit(records, tokenizer, max_length=max_length),
        "runtime_device": runtime.to_dict(),
    }
    return logits, labels, metadata


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, config: TemperatureConfig
) -> dict[str, Any]:
    _validate_logits_and_labels(logits, labels)
    lower = math.log(config.minimum)
    upper = math.log(config.maximum)

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        scaled = logits / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    raw_nll = objective(0.0)
    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": config.absolute_tolerance, "maxiter": config.maximum_iterations},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimisation failed: {result.message}")
    temperature = math.exp(float(result.x))
    at_boundary = math.isclose(
        temperature, config.minimum, rel_tol=0.0, abs_tol=1e-5
    ) or math.isclose(temperature, config.maximum, rel_tol=0.0, abs_tol=1e-5)
    return {
        "temperature": _metric_float(temperature),
        "log_temperature": _metric_float(float(result.x)),
        "raw_nll": _metric_float(raw_nll),
        "calibrated_nll": _metric_float(float(result.fun)),
        "nll_change": _metric_float(float(result.fun) - raw_nll),
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(result.nit),
        "optimizer_function_evaluations": int(result.nfev),
        "temperature_at_registered_boundary": at_boundary,
    }


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return softmax(logits / temperature, axis=1)


def calibration_metrics(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int
) -> dict[str, Any]:
    if probabilities.ndim != 2 or len(labels) != probabilities.shape[0]:
        raise ValueError("labels and probability rows are misaligned")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correctness = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.searchsorted(edges[1:-1], confidences, side="right")
    reliability: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        accuracy = float(correctness[mask].mean()) if count else 0.0
        mean_confidence = float(confidences[mask].mean()) if count else 0.0
        gap = abs(accuracy - mean_confidence) if count else 0.0
        ece += count / len(labels) * gap
        mce = max(mce, gap)
        reliability.append(
            {
                "bin": index + 1,
                "lower": _metric_float(edges[index]),
                "upper": _metric_float(edges[index + 1]),
                "count": count,
                "accuracy": _metric_float(accuracy),
                "mean_confidence": _metric_float(mean_confidence),
                "absolute_gap": _metric_float(gap),
            }
        )
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    accuracy = float(accuracy_score(labels, predictions))
    mean_confidence = float(confidences.mean())
    return {
        "count": len(labels),
        "accuracy": _metric_float(accuracy),
        "macro_f1": _metric_float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(probabilities.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        "negative_log_likelihood": _metric_float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
        ),
        "expected_calibration_error": _metric_float(ece),
        "maximum_calibration_error": _metric_float(mce),
        "multiclass_brier_score": _metric_float(
            float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        ),
        "mean_confidence": _metric_float(mean_confidence),
        "signed_confidence_gap": _metric_float(mean_confidence - accuracy),
        "reliability_bins": reliability,
    }


def bootstrap_calibration_deltas(
    labels: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    *,
    bins: int,
    config: BootstrapConfig,
    random_seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(random_seed)
    names = (
        "negative_log_likelihood",
        "expected_calibration_error",
        "maximum_calibration_error",
        "multiclass_brier_score",
        "signed_confidence_gap",
    )
    values = {name: np.empty(config.resamples, dtype=np.float64) for name in names}
    row_count = len(labels)
    for iteration in range(config.resamples):
        indices = generator.integers(0, row_count, size=row_count)
        raw = calibration_metrics(labels[indices], raw_probabilities[indices], bins=bins)
        calibrated = calibration_metrics(
            labels[indices], calibrated_probabilities[indices], bins=bins
        )
        for name in names:
            values[name][iteration] = calibrated[name] - raw[name]
    alpha = 1.0 - config.confidence_level
    return {
        "resamples": config.resamples,
        "confidence_level": config.confidence_level,
        "random_seed": random_seed,
        "delta_definition": "calibrated_minus_raw",
        "intervals": {
            name: {
                "lower": _metric_float(np.quantile(samples, alpha / 2.0)),
                "median": _metric_float(np.quantile(samples, 0.5)),
                "upper": _metric_float(np.quantile(samples, 1.0 - alpha / 2.0)),
            }
            for name, samples in values.items()
        },
    }


def build_seed_calibration_report(
    config: CalibrationConfig,
    *,
    seed: int,
    development_logits: np.ndarray,
    development_labels: np.ndarray,
    assessment_logits: np.ndarray,
    assessment_labels: np.ndarray,
    partition_entry: Mapping[str, Any],
    extraction_metadata: Mapping[str, Any],
    source_run_sha256: str,
    source_manifest_sha256: str,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    fitted = fit_temperature(development_logits, development_labels, config.temperature)
    temperature = fitted["temperature"]
    assessment_raw = probabilities_from_logits(assessment_logits)
    assessment_calibrated = probabilities_from_logits(assessment_logits, temperature)
    assessment_metrics_raw = calibration_metrics(
        assessment_labels, assessment_raw, bins=config.ece_bins
    )
    assessment_metrics_calibrated = calibration_metrics(
        assessment_labels, assessment_calibrated, bins=config.ece_bins
    )
    raw_predictions = assessment_raw.argmax(axis=1)
    calibrated_predictions = assessment_calibrated.argmax(axis=1)
    changed = int(np.sum(raw_predictions != calibrated_predictions))
    if changed:
        raise AssertionError("positive scalar temperature changed predicted classes")
    delta_names = (
        "negative_log_likelihood",
        "expected_calibration_error",
        "maximum_calibration_error",
        "multiclass_brier_score",
        "mean_confidence",
        "signed_confidence_gap",
    )
    deltas = {
        name: _metric_float(assessment_metrics_calibrated[name] - assessment_metrics_raw[name])
        for name in delta_names
    }
    ece_raw = assessment_metrics_raw["expected_calibration_error"]
    ece_calibrated = assessment_metrics_calibrated["expected_calibration_error"]
    gates = {
        "calibrated_ece_lower_than_raw": ece_calibrated < ece_raw,
        "calibrated_ece_at_or_below_maximum": (ece_calibrated <= config.calibrated_ece_maximum),
        "predicted_classes_unchanged": changed == 0,
    }
    report: dict[str, Any] = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "temperature_scaling_assessment",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seed": seed,
        "calibration_config_sha256": config_sha256,
        "partition_registry_sha256": registry_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "source": {
            "manifest_sha256": source_manifest_sha256,
            "module6_run_sha256": source_run_sha256,
            "role": "validation",
            "role_history": config.source_role_history,
        },
        "data_boundary": {
            "loaded_model_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "temperature_fit_role": config.partition.development_role,
            "assessment_role": config.partition.assessment_role,
            "fit_and_assessment_rows_disjoint": True,
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
        },
        "partitions": {
            "temperature_fit": {
                "count": partition_entry["temperature_fit"]["count"],
                "source_indices_sha256": partition_entry["temperature_fit"][
                    "source_indices_sha256"
                ],
            },
            "calibration_assessment": {
                "count": partition_entry["calibration_assessment"]["count"],
                "source_indices_sha256": partition_entry["calibration_assessment"][
                    "source_indices_sha256"
                ],
            },
        },
        "extraction": dict(extraction_metadata),
        "temperature_fit": fitted,
        "assessment_metrics": {
            "raw": assessment_metrics_raw,
            "calibrated": assessment_metrics_calibrated,
            "calibrated_minus_raw": deltas,
        },
        "prediction_invariance": {
            "changed_class_predictions": changed,
            "unchanged_class_predictions": len(assessment_labels) - changed,
        },
        "paired_bootstrap": bootstrap_calibration_deltas(
            assessment_labels,
            assessment_raw,
            assessment_calibrated,
            bins=config.ece_bins,
            config=config.bootstrap,
            random_seed=config.bootstrap.seed_offset + seed,
        ),
        "acceptance_gate": {
            **gates,
            "all_passed": all(gates.values()),
            "registered_ece_maximum": config.calibrated_ece_maximum,
        },
        "interpretation_notice": (
            "Assessment rows were not used to fit temperature, but were previously part of Module "
            "6 checkpoint selection; this is not independent model evaluation."
        ),
        "software": software_versions(),
    }
    report["report_sha256"] = stable_json_sha256(report)
    assert_text_free_artifact(report)
    return report


def aggregate_calibration_reports(
    config: CalibrationConfig,
    reports: Sequence[Mapping[str, Any]],
    *,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if [report.get("seed") for report in reports] != list(config.seeds):
        raise ValueError("calibration reports must follow registered seed order")
    names = (
        "negative_log_likelihood",
        "expected_calibration_error",
        "maximum_calibration_error",
        "multiclass_brier_score",
        "mean_confidence",
        "signed_confidence_gap",
    )
    metrics: dict[str, Any] = {}
    for name in names:
        raw_values = [float(report["assessment_metrics"]["raw"][name]) for report in reports]
        calibrated_values = [
            float(report["assessment_metrics"]["calibrated"][name]) for report in reports
        ]
        delta_values = [cal - raw for raw, cal in zip(raw_values, calibrated_values, strict=True)]
        metrics[name] = {
            "raw": _descriptive_statistics(raw_values, reports),
            "calibrated": _descriptive_statistics(calibrated_values, reports),
            "calibrated_minus_raw": _descriptive_statistics(delta_values, reports),
        }
    temperatures = [float(report["temperature_fit"]["temperature"]) for report in reports]
    artifact: dict[str, Any] = {
        "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "multiseed_temperature_scaling_aggregate",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "calibration_config_sha256": config_sha256,
        "partition_registry_sha256": registry_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "data_boundary": {
            "loaded_model_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "fit_and_assessment_rows_disjoint": True,
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "independent_model_evaluation": False,
            "source_role_history": config.source_role_history,
        },
        "report_hashes": {str(report["seed"]): report["report_sha256"] for report in reports},
        "temperatures": _descriptive_statistics(temperatures, reports),
        "assessment_metrics": metrics,
        "acceptance_gate": {
            "all_seeds_passed": all(
                bool(report["acceptance_gate"]["all_passed"]) for report in reports
            ),
            "by_seed": {str(report["seed"]): report["acceptance_gate"] for report in reports},
        },
        "interpretation_notice": (
            "The aggregate measures calibration on disjoint fit/assessment rows after checkpoint "
            "selection. It is post-selection, post-test exploratory evidence and not an "
            "independent "
            "generalisation estimate."
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


def validate_seed_calibration_report(
    artifact: Mapping[str, Any],
    *,
    config: CalibrationConfig,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "temperature_scaling_assessment":
        raise ValueError("expected a temperature-scaling assessment")
    body = dict(artifact)
    expected_hash = body.pop("report_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("calibration report content hash check failed")
    expected = {
        "claim_scope": config.claim_scope,
        "partition_registry_sha256": registry_sha256,
        "calibration_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"calibration report has an invalid {key}")
    if artifact.get("seed") not in config.seeds:
        raise ValueError("calibration report uses an unregistered seed")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("loaded_model_splits") != list(PERMITTED_MODEL_SPLITS):
        raise ValueError("calibration report loaded an invalid split")
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("calibration report accessed official test")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("calibration report computed official-test metrics")
    if boundary.get("fit_and_assessment_rows_disjoint") is not True:
        raise ValueError("calibration fit and assessment roles are not disjoint")
    if artifact.get("prediction_invariance", {}).get("changed_class_predictions") != 0:
        raise ValueError("temperature scaling changed predicted classes")
    assert_text_free_artifact(artifact)


def validate_calibration_aggregate(
    artifact: Mapping[str, Any],
    *,
    config: CalibrationConfig,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "multiseed_temperature_scaling_aggregate":
        raise ValueError("expected a multi-seed calibration aggregate")
    body = dict(artifact)
    expected_hash = body.pop("aggregate_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("calibration aggregate content hash check failed")
    expected = {
        "claim_scope": config.claim_scope,
        "seeds": list(config.seeds),
        "partition_registry_sha256": registry_sha256,
        "calibration_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"calibration aggregate has an invalid {key}")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("calibration aggregate reports test access")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("calibration aggregate contains official-test metrics")
    if boundary.get("fit_and_assessment_rows_disjoint") is not True:
        raise ValueError("calibration aggregate roles are not disjoint")
    assert_text_free_artifact(artifact)


def _validate_logits_and_labels(logits: np.ndarray, labels: np.ndarray) -> None:
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("logits and labels have incompatible shapes")
    if not len(labels) or not np.all(np.isfinite(logits)):
        raise ValueError("logits must be non-empty and finite")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("labels are outside the logit class range")


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


def write_calibration_artifact(value: Mapping[str, Any], destination: Path) -> None:
    write_json_artifact(value, destination)
