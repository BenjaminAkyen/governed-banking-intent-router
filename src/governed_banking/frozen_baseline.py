"""Frozen RoBERTa embedding extraction and leakage-safe linear evaluation."""

from __future__ import annotations

import json
import math
import platform
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from governed_banking.baseline import (
    assert_record_text_absent,
    assert_text_free_artifact,
    classification_metrics,
    write_json_artifact,
    write_prediction_jsonl,
)
from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256
from governed_banking.device import seed_everything, select_device

FROZEN_CONFIG_SCHEMA_VERSION = 1
FROZEN_ARTIFACT_SCHEMA_VERSION = 1
POOLING_NAMES = ("cls", "mean")


@dataclass(frozen=True)
class EncoderConfig:
    repository: str
    revision: str
    license_name: str
    max_length: int
    batch_size: int
    device: str
    normalize_embeddings: bool
    snapshot_files: tuple[str, ...]


@dataclass(frozen=True)
class FrozenCandidateConfig:
    name: str
    pooling: str
    c_value: float
    solver: str
    max_iter: int
    tolerance: float


@dataclass(frozen=True)
class FrozenBaselineConfig:
    experiment_name: str
    random_seed: int
    thread_limit: int
    encoder: EncoderConfig
    selection_metric: str
    tie_breakers: tuple[str, ...]
    amendment_round: int
    amendment_reason: str
    test_accessed_before_amendment: bool
    candidates: tuple[FrozenCandidateConfig, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> FrozenBaselineConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("frozen baseline configuration must be a mapping")
        if raw.get("schema_version") != FROZEN_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported frozen baseline configuration schema")

        experiment_name = str(raw.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValueError("experiment_name cannot be blank")
        model = _mapping(raw, "model")
        selection = _mapping(raw, "selection")
        if selection.get("metric") != "macro_f1":
            raise ValueError("selection.metric must be macro_f1")
        tie_breakers = tuple(str(value) for value in selection.get("tie_breakers", []))
        expected_tie_breakers = ("accuracy", "negative_log_loss", "candidate_name")
        if tie_breakers != expected_tie_breakers:
            raise ValueError(f"selection.tie_breakers must be {expected_tie_breakers}")
        amendment_round = _bounded_integer(
            selection.get("amendment_round"), "selection.amendment_round", 1, 100
        )
        amendment_reason = str(selection.get("amendment_reason", "")).strip()
        if not amendment_reason:
            raise ValueError("selection.amendment_reason cannot be blank")
        test_accessed_before_amendment = _boolean(
            selection.get("test_accessed_before_amendment"),
            "selection.test_accessed_before_amendment",
        )
        if test_accessed_before_amendment:
            raise ValueError("the candidate search cannot be amended after test access")

        repository = str(model.get("repository", "")).strip()
        revision = str(model.get("revision", "")).strip()
        if repository != "FacebookAI/roberta-base":
            raise ValueError("Module 4 is registered for FacebookAI/roberta-base")
        invalid_revision = len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        )
        if invalid_revision:
            raise ValueError("model.revision must be a full lowercase Git commit")
        snapshot_files_value = model.get("snapshot_files")
        if not isinstance(snapshot_files_value, list) or not snapshot_files_value:
            raise ValueError("model.snapshot_files must be a non-empty list")
        snapshot_files = tuple(str(value) for value in snapshot_files_value)
        if len(snapshot_files) != len(set(snapshot_files)):
            raise ValueError("model.snapshot_files cannot contain duplicates")
        if any(Path(filename).name != filename for filename in snapshot_files):
            raise ValueError("model snapshot filenames cannot contain directory traversal")

        device = str(model.get("device", ""))
        if device not in {"mps", "cpu"}:
            raise ValueError("model.device must be mps or cpu")
        encoder = EncoderConfig(
            repository=repository,
            revision=revision,
            license_name=str(model.get("license", "")).strip(),
            max_length=_bounded_integer(model.get("max_length"), "model.max_length", 8, 512),
            batch_size=_bounded_integer(model.get("batch_size"), "model.batch_size", 1, 512),
            device=device,
            normalize_embeddings=_boolean(
                model.get("normalize_embeddings"), "model.normalize_embeddings"
            ),
            snapshot_files=snapshot_files,
        )

        candidate_values = raw.get("candidates")
        if not isinstance(candidate_values, list) or not candidate_values:
            raise ValueError("candidates must be a non-empty list")
        candidates = tuple(_parse_candidate(value) for value in candidate_values)
        candidate_names = [candidate.name for candidate in candidates]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("candidate names must be unique")
        if not {candidate.pooling for candidate in candidates}.issubset(set(POOLING_NAMES)):
            raise ValueError("candidate pooling must be cls or mean")

        return cls(
            experiment_name=experiment_name,
            random_seed=_bounded_integer(
                raw.get("random_seed"), "random_seed", 0, 2**32 - 1
            ),
            thread_limit=_bounded_integer(raw.get("thread_limit"), "thread_limit", 1, 128),
            encoder=encoder,
            selection_metric="macro_f1",
            tie_breakers=tie_breakers,
            amendment_round=amendment_round,
            amendment_reason=amendment_reason,
            test_accessed_before_amendment=test_accessed_before_amendment,
            candidates=candidates,
        )


@dataclass(frozen=True)
class ModelSnapshot:
    path: Path
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class EmbeddingBundle:
    split_name: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _parse_candidate(value: Any) -> FrozenCandidateConfig:
    if not isinstance(value, dict):
        raise ValueError("each candidate must be a mapping")
    name = str(value.get("name", "")).strip()
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("candidate name must contain only letters, numbers and underscores")
    pooling = str(value.get("pooling", ""))
    logistic = _mapping(value, "logistic_regression")
    solver = str(logistic.get("solver", ""))
    if solver != "lbfgs":
        raise ValueError("the frozen baseline currently requires the lbfgs solver")
    c_value = float(logistic.get("C", 0.0))
    tolerance = float(logistic.get("tolerance", 0.0))
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("logistic_regression.C must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("logistic_regression.tolerance must be finite and positive")
    return FrozenCandidateConfig(
        name=name,
        pooling=pooling,
        c_value=c_value,
        solver=solver,
        max_iter=_bounded_integer(
            logistic.get("max_iter"), f"{name}.logistic_regression.max_iter", 1, 100_000
        ),
        tolerance=tolerance,
    )


def resolve_model_snapshot(
    encoder: EncoderConfig,
    *,
    cache_directory: Path | None = None,
    offline: bool = False,
) -> ModelSnapshot:
    """Resolve only registered files from a full Hugging Face model revision."""

    snapshot_path = Path(
        snapshot_download(
            repo_id=encoder.repository,
            revision=encoder.revision,
            allow_patterns=list(encoder.snapshot_files),
            cache_dir=None if cache_directory is None else str(cache_directory),
            local_files_only=offline,
        )
    )
    missing = [
        filename
        for filename in encoder.snapshot_files
        if not (snapshot_path / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"model snapshot is missing registered files: {missing}")
    hashes = {
        filename: sha256_file(snapshot_path / filename) for filename in encoder.snapshot_files
    }
    return ModelSnapshot(path=snapshot_path, file_sha256=hashes)


def freeze_encoder(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze every encoder parameter and return total/trainable counts."""

    model.requires_grad_(False)
    model.eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != 0:
        raise AssertionError("frozen encoder still has trainable parameters")
    return total, trainable


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    *,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    """Create CLS and content-token mean embeddings from the final hidden layer."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape batch x tokens x hidden")
    expected_shape = hidden_states.shape[:2]
    if attention_mask.shape != expected_shape or special_tokens_mask.shape != expected_shape:
        raise ValueError("attention and special-token masks must match hidden-state tokens")

    cls_embeddings = hidden_states[:, 0, :]
    content_mask = attention_mask.bool() & ~special_tokens_mask.bool()
    empty_rows = content_mask.sum(dim=1) == 0
    if empty_rows.any():
        content_mask = content_mask.clone()
        content_mask[empty_rows] = attention_mask[empty_rows].bool()
    weights = content_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    mean_embeddings = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    pooled = {"cls": cls_embeddings, "mean": mean_embeddings}
    if normalize:
        pooled = {
            name: torch.nn.functional.normalize(value, p=2, dim=1)
            for name, value in pooled.items()
        }
    return pooled


def extract_or_load_embeddings(
    records: Sequence[BankingRecord],
    *,
    split_name: str,
    source_indices_sha256: str,
    config: FrozenBaselineConfig,
    snapshot: ModelSnapshot,
    cache_directory: Path,
    implementation_sha256: Mapping[str, str],
    force: bool = False,
) -> EmbeddingBundle:
    """Return verified cached embeddings or extract them with the frozen encoder."""

    if not records:
        raise ValueError("embedding records cannot be empty")
    cache_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_directory / f"{split_name}-metadata.json"
    expected_contract = {
        "split_name": split_name,
        "row_count": len(records),
        "source_indices_sha256": source_indices_sha256,
        "encoder_repository": config.encoder.repository,
        "encoder_revision": config.encoder.revision,
        "encoder_files_sha256": dict(sorted(snapshot.file_sha256.items())),
        "max_length": config.encoder.max_length,
        "requested_device": config.encoder.device,
        "normalize_embeddings": config.encoder.normalize_embeddings,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    if metadata_path.exists() and not force:
        return load_embedding_cache(
            cache_directory,
            split_name=split_name,
            expected_contract=expected_contract,
        )

    seed_everything(config.random_seed)
    device, device_metadata = select_device(config.encoder.device)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot.path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        snapshot.path,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation="eager",
        add_pooling_layer=False,
    )
    total_parameters, trainable_parameters = freeze_encoder(model)
    model.to(device)

    token_lengths: list[int] = []
    pooled_batches: dict[str, list[np.ndarray]] = {name: [] for name in POOLING_NAMES}
    started = time.perf_counter()
    for start in range(0, len(records), config.encoder.batch_size):
        batch_records = records[start : start + config.encoder.batch_size]
        texts = [record.text for record in batch_records]
        length_encoding = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        token_lengths.extend(int(value) for value in length_encoding["length"])
        encoding = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=config.encoder.max_length,
            padding=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        model_inputs = {
            "input_ids": encoding["input_ids"].to(device),
            "attention_mask": encoding["attention_mask"].to(device),
        }
        with torch.inference_mode():
            outputs = model(**model_inputs)
            pooled = pool_hidden_states(
                outputs.last_hidden_state,
                model_inputs["attention_mask"],
                encoding["special_tokens_mask"].to(device),
                normalize=config.encoder.normalize_embeddings,
            )
        for pooling_name, values in pooled.items():
            pooled_batches[pooling_name].append(values.float().cpu().numpy())

    extraction_seconds = time.perf_counter() - started
    arrays = {
        name: np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        for name, batches in pooled_batches.items()
    }
    if any(array.shape[0] != len(records) for array in arrays.values()):
        raise AssertionError("embedding row count changed during extraction")

    array_files: dict[str, dict[str, Any]] = {}
    for pooling_name, array in arrays.items():
        array_path = cache_directory / f"{split_name}-{pooling_name}.npy"
        _write_numpy(array, array_path)
        array_files[pooling_name] = {
            "filename": array_path.name,
            "sha256": sha256_file(array_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    metadata: dict[str, Any] = {
        "schema_version": FROZEN_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "frozen_embedding_cache",
        "contains_message_text": False,
        **expected_contract,
        "device": device_metadata.to_dict(),
        "total_encoder_parameters": total_parameters,
        "trainable_encoder_parameters": trainable_parameters,
        "pooling": {
            "cls": "final hidden state of the first token",
            "mean": "mean of final-layer non-special, non-padding token states",
        },
        "token_length": {
            "maximum_before_truncation": max(token_lengths),
            "p95_before_truncation": _metric_float(np.percentile(token_lengths, 95)),
            "truncated_rows": sum(length > config.encoder.max_length for length in token_lengths),
        },
        "extraction_seconds": _metric_float(extraction_seconds),
        "rows_per_second": _metric_float(len(records) / extraction_seconds),
        "arrays": array_files,
    }
    metadata["cache_metadata_sha256"] = stable_json_sha256(metadata)
    assert_text_free_artifact(metadata)
    assert_record_text_absent(metadata, records)
    write_json_artifact(metadata, metadata_path)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return EmbeddingBundle(split_name=split_name, arrays=arrays, metadata=metadata)


def load_embedding_cache(
    cache_directory: Path,
    *,
    split_name: str,
    expected_contract: Mapping[str, Any],
) -> EmbeddingBundle:
    metadata_path = cache_directory / f"{split_name}-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_type") != "frozen_embedding_cache":
        raise ValueError("expected a frozen-embedding cache artifact")
    expected_hash = metadata.get("cache_metadata_sha256")
    body = dict(metadata)
    body.pop("cache_metadata_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("embedding cache metadata hash check failed")
    for key, value in expected_contract.items():
        if metadata.get(key) != value:
            raise ValueError(f"embedding cache contract mismatch: {key}")

    arrays: dict[str, np.ndarray] = {}
    for pooling_name in POOLING_NAMES:
        details = metadata.get("arrays", {}).get(pooling_name)
        if not isinstance(details, dict):
            raise ValueError(f"embedding cache is missing {pooling_name} details")
        path = cache_directory / str(details.get("filename"))
        if sha256_file(path) != details.get("sha256"):
            raise ValueError(f"{pooling_name} embedding file hash check failed")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != details.get("shape") or str(array.dtype) != details.get("dtype"):
            raise ValueError(f"{pooling_name} embedding shape or dtype check failed")
        arrays[pooling_name] = array
    assert_text_free_artifact(metadata)
    return EmbeddingBundle(split_name=split_name, arrays=arrays, metadata=metadata)


def _write_numpy(array: np.ndarray, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_frozen_baseline(
    config: FrozenBaselineConfig,
    train_records: Sequence[BankingRecord],
    validation_records: Sequence[BankingRecord],
    train_embeddings: EmbeddingBundle,
    validation_embeddings: EmbeddingBundle,
    *,
    label_names: Sequence[str],
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
) -> dict[str, Any]:
    _validate_records_and_embeddings(train_records, train_embeddings, "official_train", "train")
    _validate_records_and_embeddings(
        validation_records, validation_embeddings, "official_train", "validation"
    )
    train_indices = {record.source_index for record in train_records}
    validation_indices = {record.source_index for record in validation_records}
    if train_indices & validation_indices:
        raise ValueError("training and validation source indices overlap")

    results = [
        _fit_and_score_classifier(
            candidate,
            config,
            train_embeddings.arrays[candidate.pooling],
            [record.category for record in train_records],
            validation_embeddings.arrays[candidate.pooling],
            [record.category for record in validation_records],
            label_names=label_names,
        )[1]
        for candidate in config.candidates
    ]
    ranked = sorted(
        results,
        key=lambda result: (
            -result["metrics"]["macro_f1"],
            -result["metrics"]["accuracy"],
            result["metrics"]["log_loss"],
            result["candidate_name"],
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result["validation_rank"] = rank

    artifact: dict[str, Any] = {
        "schema_version": FROZEN_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "frozen_validation_selection",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "frozen_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "encoder": {
            "repository": config.encoder.repository,
            "revision": config.encoder.revision,
            "license": config.encoder.license_name,
            "files_sha256": dict(sorted(model_files_sha256.items())),
            "parameters_frozen": True,
            "max_length": config.encoder.max_length,
        },
        "embedding_cache": {
            "train_metadata_sha256": train_embeddings.metadata["cache_metadata_sha256"],
            "validation_metadata_sha256": validation_embeddings.metadata[
                "cache_metadata_sha256"
            ],
        },
        "random_seed": config.random_seed,
        "thread_limit": config.thread_limit,
        "data_boundary": {
            "fit_split": "train",
            "selection_split": "validation",
            "test_split_loaded": False,
            "test_embeddings_created": False,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
        },
        "selection_policy": {
            "metric": config.selection_metric,
            "tie_breakers": list(config.tie_breakers),
            "amendment_round": config.amendment_round,
            "amendment_reason": config.amendment_reason,
            "test_accessed_before_amendment": config.test_accessed_before_amendment,
        },
        "selected_candidate": ranked[0]["candidate_name"],
        "candidate_results": ranked,
        "software": frozen_software_versions(),
    }
    artifact["selection_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_record_text_absent(artifact, (*train_records, *validation_records))
    return artifact


def evaluate_locked_frozen_baseline(
    config: FrozenBaselineConfig,
    selection: Mapping[str, Any],
    train_records: Sequence[BankingRecord],
    test_records: Sequence[BankingRecord],
    train_embeddings: EmbeddingBundle,
    test_embeddings: EmbeddingBundle,
    *,
    label_names: Sequence[str],
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
) -> tuple[LogisticRegression, dict[str, Any], list[dict[str, Any]]]:
    validate_frozen_selection_artifact(
        selection,
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=model_files_sha256,
    )
    selection_cache = selection.get("embedding_cache", {})
    if (
        train_embeddings.metadata.get("cache_metadata_sha256")
        != selection_cache.get("train_metadata_sha256")
    ):
        raise ValueError("training embeddings differ from the selection lock")
    _validate_records_and_embeddings(train_records, train_embeddings, "official_train", "train")
    _validate_records_and_embeddings(test_records, test_embeddings, "official_test", "test")
    candidates = {candidate.name: candidate for candidate in config.candidates}
    selected_name = selection["selected_candidate"]
    if selected_name not in candidates:
        raise ValueError("selected candidate is absent from the frozen baseline configuration")
    candidate = candidates[selected_name]
    classifier, result, predictions = _fit_and_score_classifier(
        candidate,
        config,
        train_embeddings.arrays[candidate.pooling],
        [record.category for record in train_records],
        test_embeddings.arrays[candidate.pooling],
        [record.category for record in test_records],
        label_names=label_names,
    )
    prediction_rows = [
        {
            "source_index": record.source_index,
            "true_label": record.category,
            "predicted_label": predicted,
            "confidence_uncalibrated": _metric_float(float(max(probabilities))),
        }
        for record, predicted, probabilities in zip(
            test_records, predictions["labels"], predictions["probabilities"], strict=True
        )
    ]
    artifact: dict[str, Any] = {
        "schema_version": FROZEN_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "frozen_locked_test_evaluation",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "frozen_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "selection_sha256": selection["selection_sha256"],
        "selected_candidate": selected_name,
        "encoder": selection["encoder"],
        "embedding_cache": {
            "train_metadata_sha256": train_embeddings.metadata["cache_metadata_sha256"],
            "test_metadata_sha256": test_embeddings.metadata["cache_metadata_sha256"],
        },
        "random_seed": config.random_seed,
        "thread_limit": config.thread_limit,
        "data_boundary": {
            "fit_split": "train",
            "selection_was_locked_before_test": True,
            "test_embeddings_created_after_selection_lock": True,
            "evaluation_split": "test",
            "train_rows": len(train_records),
            "test_rows": len(test_records),
        },
        "test_result": result,
        "test_predictions_sha256": stable_json_sha256(prediction_rows),
        "confidence_notice": "Probabilities are uncalibrated until the calibration module.",
        "software": frozen_software_versions(),
    }
    artifact["evaluation_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_text_free_artifact(prediction_rows)
    assert_record_text_absent(artifact, (*train_records, *test_records))
    assert_record_text_absent(prediction_rows, test_records)
    return classifier, artifact, prediction_rows


def _fit_and_score_classifier(
    candidate: FrozenCandidateConfig,
    config: FrozenBaselineConfig,
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    evaluation_embeddings: np.ndarray,
    evaluation_labels: Sequence[str],
    *,
    label_names: Sequence[str],
) -> tuple[LogisticRegression, dict[str, Any], dict[str, Any]]:
    if train_embeddings.ndim != 2 or evaluation_embeddings.ndim != 2:
        raise ValueError("classifier embeddings must be two-dimensional")
    if train_embeddings.shape[1] != evaluation_embeddings.shape[1]:
        raise ValueError("training and evaluation embedding dimensions differ")
    classifier = LogisticRegression(
        C=candidate.c_value,
        solver=candidate.solver,
        max_iter=candidate.max_iter,
        tol=candidate.tolerance,
        random_state=config.random_seed,
    )
    with (
        threadpool_limits(limits=config.thread_limit),
        warnings.catch_warnings(record=True) as seen,
    ):
        warnings.simplefilter("always", ConvergenceWarning)
        fit_started = time.perf_counter()
        classifier.fit(train_embeddings, train_labels)
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        predicted_labels = classifier.predict(evaluation_embeddings)
        probabilities = classifier.predict_proba(evaluation_embeddings)
        prediction_seconds = time.perf_counter() - prediction_started

    convergence_warnings = [
        str(item.message) for item in seen if issubclass(item.category, ConvergenceWarning)
    ]
    classes = [str(value) for value in classifier.classes_]
    result = {
        "candidate_name": candidate.name,
        "candidate": asdict(candidate),
        "embedding_dimension": int(train_embeddings.shape[1]),
        "classifier_parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
        "fit_seconds": _metric_float(fit_seconds),
        "prediction_seconds": _metric_float(prediction_seconds),
        "converged": not convergence_warnings,
        "convergence_warnings": convergence_warnings,
        "iterations": [int(value) for value in np.atleast_1d(classifier.n_iter_)],
        "metrics": classification_metrics(
            evaluation_labels,
            predicted_labels,
            probabilities,
            classes=classes,
            label_names=label_names,
        ),
    }
    return classifier, result, {"labels": predicted_labels, "probabilities": probabilities}


def _validate_records_and_embeddings(
    records: Sequence[BankingRecord],
    embeddings: EmbeddingBundle,
    expected_source: str,
    expected_split: str,
) -> None:
    if not records:
        raise ValueError(f"{expected_split} records cannot be empty")
    if any(record.source_split != expected_source for record in records):
        raise ValueError(f"{expected_split} records must come only from {expected_source}")
    if embeddings.split_name != expected_split:
        raise ValueError(f"expected {expected_split} embeddings")
    if embeddings.metadata.get("trainable_encoder_parameters") != 0:
        raise ValueError("embedding cache was not produced by a fully frozen encoder")
    for pooling_name in POOLING_NAMES:
        array = embeddings.arrays.get(pooling_name)
        if not isinstance(array, np.ndarray) or array.shape[0] != len(records):
            raise ValueError(f"{expected_split} {pooling_name} embedding count mismatch")


def validate_frozen_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "frozen_validation_selection":
        raise ValueError("expected a frozen-validation-selection artifact")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("selection must prove that the test split was not loaded")
    if boundary.get("test_embeddings_created") is not False:
        raise ValueError("selection must prove that test embeddings were not created")
    expected_hash = artifact.get("selection_sha256")
    body = dict(artifact)
    body.pop("selection_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("frozen selection artifact content hash check failed")
    if artifact.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("frozen selection uses a different dataset manifest")
    if artifact.get("frozen_config_sha256") != config_sha256:
        raise ValueError("frozen selection uses a different configuration")
    if artifact.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise ValueError("frozen selection uses a different implementation")
    encoder = artifact.get("encoder", {})
    if encoder.get("files_sha256") != dict(sorted(model_files_sha256.items())):
        raise ValueError("frozen selection uses different encoder files")
    if encoder.get("parameters_frozen") is not True:
        raise ValueError("encoder parameters must be frozen")
    assert_text_free_artifact(artifact)


def validate_frozen_evaluation_artifact(
    artifact: Mapping[str, Any],
    *,
    selection_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "frozen_locked_test_evaluation":
        raise ValueError("expected a frozen locked-test evaluation artifact")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("selection_was_locked_before_test") is not True:
        raise ValueError("frozen evaluation must follow a locked selection")
    if boundary.get("test_embeddings_created_after_selection_lock") is not True:
        raise ValueError("test embeddings must follow the selection lock")
    expected_hash = artifact.get("evaluation_sha256")
    body = dict(artifact)
    body.pop("evaluation_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("frozen evaluation artifact content hash check failed")
    if artifact.get("selection_sha256") != selection_sha256:
        raise ValueError("frozen evaluation uses a different selection lock")
    if artifact.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("frozen evaluation uses a different dataset manifest")
    if artifact.get("frozen_config_sha256") != config_sha256:
        raise ValueError("frozen evaluation uses a different configuration")
    if artifact.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise ValueError("frozen evaluation uses a different implementation")
    encoder = artifact.get("encoder", {})
    if encoder.get("files_sha256") != dict(sorted(model_files_sha256.items())):
        raise ValueError("frozen evaluation uses different encoder files")
    prediction_artifact = artifact.get("prediction_artifact", {})
    if prediction_artifact.get("sha256") != artifact.get("test_predictions_sha256"):
        raise ValueError("frozen prediction artifact hash does not match evaluation")
    assert_text_free_artifact(artifact)


def frozen_software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "huggingface_hub": version("huggingface-hub"),
        "joblib": version("joblib"),
    }


def write_classifier(classifier: LogisticRegression, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(classifier, temporary)
    temporary.replace(destination)
    return sha256_file(destination)


def write_frozen_predictions(
    rows: Sequence[Mapping[str, Any]], destination: Path
) -> str:
    return write_prediction_jsonl(rows, destination)


def write_frozen_report(value: Mapping[str, Any], destination: Path) -> None:
    write_json_artifact(value, destination)


def _metric_float(value: float) -> float:
    return round(float(value), 10)
