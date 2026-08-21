"""Leakage-safe TF-IDF logistic-regression baseline experiments."""

from __future__ import annotations

import json
import math
import platform
import tempfile
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from threadpoolctl import threadpool_limits

from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256

BASELINE_CONFIG_SCHEMA_VERSION = 1
BASELINE_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureConfig:
    """A bounded TF-IDF feature-space definition."""

    ngram_range: tuple[int, int]
    min_df: int
    max_features: int


@dataclass(frozen=True)
class CandidateConfig:
    """One predeclared model candidate eligible for validation selection."""

    name: str
    word: FeatureConfig
    char: FeatureConfig | None
    c_value: float
    solver: str
    max_iter: int
    tolerance: float


@dataclass(frozen=True)
class BaselineConfig:
    """Validated experiment policy loaded before model fitting."""

    experiment_name: str
    random_seed: int
    thread_limit: int
    selection_metric: str
    tie_breakers: tuple[str, ...]
    candidates: tuple[CandidateConfig, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> BaselineConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("baseline configuration must be a mapping")
        if raw.get("schema_version") != BASELINE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported baseline configuration schema")

        selection = _mapping(raw, "selection")
        if selection.get("metric") != "macro_f1":
            raise ValueError("selection.metric must be macro_f1")
        tie_breakers = tuple(str(item) for item in selection.get("tie_breakers", []))
        expected_tie_breakers = ("accuracy", "negative_log_loss", "candidate_name")
        if tie_breakers != expected_tie_breakers:
            raise ValueError(f"selection.tie_breakers must be {expected_tie_breakers}")

        candidate_values = raw.get("candidates")
        if not isinstance(candidate_values, list) or not candidate_values:
            raise ValueError("candidates must be a non-empty list")
        candidates = tuple(_parse_candidate(value) for value in candidate_values)
        candidate_names = [candidate.name for candidate in candidates]
        if len(candidate_names) != len(set(candidate_names)):
            raise ValueError("candidate names must be unique")

        random_seed = _positive_integer(raw.get("random_seed"), "random_seed", allow_zero=True)
        thread_limit = _positive_integer(raw.get("thread_limit"), "thread_limit")
        experiment_name = str(raw.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValueError("experiment_name cannot be blank")
        return cls(
            experiment_name=experiment_name,
            random_seed=random_seed,
            thread_limit=thread_limit,
            selection_metric="macro_f1",
            tie_breakers=tie_breakers,
            candidates=candidates,
        )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    lower_bound = 0 if allow_zero else 1
    if value < lower_bound:
        raise ValueError(f"{name} must be at least {lower_bound}")
    return value


def _parse_feature(value: Any, name: str) -> FeatureConfig:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    ngram_range = value.get("ngram_range")
    if (
        not isinstance(ngram_range, list)
        or len(ngram_range) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in ngram_range)
    ):
        raise ValueError(f"{name}.ngram_range must contain two integers")
    ngram_min, ngram_max = ngram_range
    if not 1 <= ngram_min <= ngram_max <= 6:
        raise ValueError(f"{name}.ngram_range must satisfy 1 <= min <= max <= 6")
    return FeatureConfig(
        ngram_range=(ngram_min, ngram_max),
        min_df=_positive_integer(value.get("min_df"), f"{name}.min_df"),
        max_features=_positive_integer(value.get("max_features"), f"{name}.max_features"),
    )


def _parse_candidate(value: Any) -> CandidateConfig:
    if not isinstance(value, dict):
        raise ValueError("each candidate must be a mapping")
    name = str(value.get("name", "")).strip()
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("candidate name must contain only letters, numbers and underscores")
    logistic = _mapping(value, "logistic_regression")
    solver = str(logistic.get("solver", ""))
    if solver != "saga":
        raise ValueError("the reproducible baseline currently requires the saga solver")
    c_value = float(logistic.get("C", 0.0))
    tolerance = float(logistic.get("tolerance", 0.0))
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("logistic_regression.C must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("logistic_regression.tolerance must be finite and positive")
    char_value = value.get("char")
    return CandidateConfig(
        name=name,
        word=_parse_feature(value.get("word"), f"{name}.word"),
        char=None if char_value is None else _parse_feature(char_value, f"{name}.char"),
        c_value=c_value,
        solver=solver,
        max_iter=_positive_integer(
            logistic.get("max_iter"), f"{name}.logistic_regression.max_iter"
        ),
        tolerance=tolerance,
    )


def build_pipeline(candidate: CandidateConfig, *, random_seed: int) -> Pipeline:
    """Create a pipeline whose vectorisers can only fit inside the supplied training call."""

    transformers: list[tuple[str, TfidfVectorizer]] = [
        ("word", _vectorizer(candidate.word, analyzer="word"))
    ]
    if candidate.char is not None:
        transformers.append(("char", _vectorizer(candidate.char, analyzer="char_wb")))

    return Pipeline(
        steps=[
            ("features", FeatureUnion(transformer_list=transformers)),
            (
                "classifier",
                LogisticRegression(
                    C=candidate.c_value,
                    solver=candidate.solver,
                    max_iter=candidate.max_iter,
                    tol=candidate.tolerance,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _vectorizer(config: FeatureConfig, *, analyzer: str) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        norm="l2",
        dtype=np.float64,
    )


def select_baseline(
    config: BaselineConfig,
    train_records: Sequence[BankingRecord],
    validation_records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Fit predeclared candidates on train and select only with validation metrics."""

    _validate_modelling_records(train_records, "official_train", "train")
    _validate_modelling_records(validation_records, "official_train", "validation")
    train_indices = {record.source_index for record in train_records}
    validation_indices = {record.source_index for record in validation_records}
    if train_indices & validation_indices:
        raise ValueError("training and validation source indices overlap")

    results: list[dict[str, Any]] = []
    for candidate in config.candidates:
        _, result, _ = _fit_and_score(
            candidate,
            config,
            train_records,
            validation_records,
            label_names=label_names,
        )
        results.append(result)

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
        "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "validation_selection",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "baseline_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "random_seed": config.random_seed,
        "thread_limit": config.thread_limit,
        "data_boundary": {
            "fit_split": "train",
            "selection_split": "validation",
            "test_split_loaded": False,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
        },
        "selection_policy": {
            "metric": config.selection_metric,
            "tie_breakers": list(config.tie_breakers),
        },
        "selected_candidate": ranked[0]["candidate_name"],
        "candidate_results": ranked,
        "software": software_versions(),
    }
    artifact["selection_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_record_text_absent(artifact, (*train_records, *validation_records))
    return artifact


def evaluate_locked_baseline(
    config: BaselineConfig,
    selection_artifact: Mapping[str, Any],
    train_records: Sequence[BankingRecord],
    test_records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Evaluate test once, after verifying the validation-selection lock."""

    validate_selection_artifact(
        selection_artifact,
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    _validate_modelling_records(train_records, "official_train", "train")
    _validate_modelling_records(test_records, "official_test", "test")
    selected_name = selection_artifact["selected_candidate"]
    candidates = {candidate.name: candidate for candidate in config.candidates}
    if selected_name not in candidates:
        raise ValueError("selected candidate is absent from the current baseline configuration")

    model, fit_result, predictions = _fit_and_score(
        candidates[selected_name],
        config,
        train_records,
        test_records,
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
    prediction_sha256 = stable_json_sha256(prediction_rows)

    artifact: dict[str, Any] = {
        "schema_version": BASELINE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "locked_test_evaluation",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "baseline_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "selection_sha256": selection_artifact["selection_sha256"],
        "selected_candidate": selected_name,
        "random_seed": config.random_seed,
        "thread_limit": config.thread_limit,
        "data_boundary": {
            "fit_split": "train",
            "selection_was_locked_before_test": True,
            "evaluation_split": "test",
            "train_rows": len(train_records),
            "test_rows": len(test_records),
        },
        "test_result": fit_result,
        "test_predictions_sha256": prediction_sha256,
        "confidence_notice": "Probabilities are uncalibrated until the calibration module.",
        "software": software_versions(),
    }
    artifact["evaluation_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_text_free_artifact(prediction_rows)
    assert_record_text_absent(artifact, (*train_records, *test_records))
    assert_record_text_absent(prediction_rows, test_records)
    return model, artifact, prediction_rows


def _fit_and_score(
    candidate: CandidateConfig,
    config: BaselineConfig,
    train_records: Sequence[BankingRecord],
    evaluation_records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    model = build_pipeline(candidate, random_seed=config.random_seed)
    train_texts = [record.text for record in train_records]
    train_labels = [record.category for record in train_records]
    evaluation_texts = [record.text for record in evaluation_records]
    evaluation_labels = [record.category for record in evaluation_records]

    with (
        threadpool_limits(limits=config.thread_limit),
        warnings.catch_warnings(record=True) as seen,
    ):
        warnings.simplefilter("always", ConvergenceWarning)
        fit_started = time.perf_counter()
        model.fit(train_texts, train_labels)
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        predicted_labels = model.predict(evaluation_texts)
        probabilities = model.predict_proba(evaluation_texts)
        prediction_seconds = time.perf_counter() - prediction_started

    convergence_warnings = [
        str(item.message) for item in seen if issubclass(item.category, ConvergenceWarning)
    ]
    classifier = model.named_steps["classifier"]
    classes = [str(value) for value in classifier.classes_]
    result = {
        "candidate_name": candidate.name,
        "candidate": _candidate_dict(candidate),
        "feature_count": int(len(model.named_steps["features"].get_feature_names_out())),
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
    return model, result, {"labels": predicted_labels, "probabilities": probabilities}


def classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    probabilities: np.ndarray,
    *,
    classes: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, Any]:
    """Return stable aggregate, per-intent and confusion metrics."""

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_intent = {
        label: {
            "precision": _metric_float(report[label]["precision"]),
            "recall": _metric_float(report[label]["recall"]),
            "f1": _metric_float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in label_names
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=label_names)
    confusions = sorted(
        (
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": int(matrix[true_index, predicted_index]),
            }
            for true_index, true_label in enumerate(label_names)
            for predicted_index, predicted_label in enumerate(label_names)
            if true_index != predicted_index and matrix[true_index, predicted_index] > 0
        ),
        key=lambda item: (-item["count"], item["true_label"], item["predicted_label"]),
    )[:20]
    confidences = probabilities.max(axis=1)
    return {
        "count": len(true_labels),
        "accuracy": _metric_float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": _metric_float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": _metric_float(
            f1_score(true_labels, predicted_labels, labels=label_names, average="macro")
        ),
        "weighted_f1": _metric_float(
            f1_score(true_labels, predicted_labels, labels=label_names, average="weighted")
        ),
        "log_loss": _metric_float(log_loss(true_labels, probabilities, labels=classes)),
        "top_3_accuracy": _metric_float(
            top_k_accuracy_score(true_labels, probabilities, k=3, labels=classes)
        ),
        "mean_max_confidence_uncalibrated": _metric_float(float(confidences.mean())),
        "per_intent": per_intent,
        "top_confusions": confusions,
    }


def _validate_modelling_records(
    records: Sequence[BankingRecord], expected_source: str, split_name: str
) -> None:
    if not records:
        raise ValueError(f"{split_name} records cannot be empty")
    if any(record.source_split != expected_source for record in records):
        raise ValueError(f"{split_name} records must come only from {expected_source}")


def _candidate_dict(candidate: CandidateConfig) -> dict[str, Any]:
    value = asdict(candidate)
    value["c_value"] = _metric_float(candidate.c_value)
    value["tolerance"] = _metric_float(candidate.tolerance)
    return value


def _metric_float(value: float) -> float:
    return round(float(value), 10)


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "joblib": version("joblib"),
    }


def validate_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "validation_selection":
        raise ValueError("expected a validation-selection artifact")
    if artifact.get("data_boundary", {}).get("test_split_loaded") is not False:
        raise ValueError("selection artifact must prove that the test split was not loaded")
    expected_hash = artifact.get("selection_sha256")
    body = dict(artifact)
    body.pop("selection_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("selection artifact content hash check failed")
    if artifact.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("selection artifact uses a different dataset manifest")
    if artifact.get("baseline_config_sha256") != config_sha256:
        raise ValueError("selection artifact uses a different baseline configuration")
    if artifact.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise ValueError("selection artifact uses a different implementation")
    assert_text_free_artifact(artifact)


def validate_evaluation_artifact(
    artifact: Mapping[str, Any],
    *,
    selection_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "locked_test_evaluation":
        raise ValueError("expected a locked-test-evaluation artifact")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("selection_was_locked_before_test") is not True:
        raise ValueError("evaluation must follow a locked selection")
    expected_hash = artifact.get("evaluation_sha256")
    body = dict(artifact)
    body.pop("evaluation_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("evaluation artifact content hash check failed")
    if artifact.get("selection_sha256") != selection_sha256:
        raise ValueError("evaluation artifact uses a different selection lock")
    if artifact.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("evaluation artifact uses a different dataset manifest")
    if artifact.get("baseline_config_sha256") != config_sha256:
        raise ValueError("evaluation artifact uses a different baseline configuration")
    if artifact.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise ValueError("evaluation artifact uses a different implementation")
    prediction_artifact = artifact.get("prediction_artifact", {})
    if prediction_artifact.get("sha256") != artifact.get("test_predictions_sha256"):
        raise ValueError("prediction artifact hash does not match the test evaluation")
    assert_text_free_artifact(artifact)


def assert_text_free_artifact(value: Any) -> None:
    """Reject fields that could accidentally persist raw message content."""

    forbidden_keys = {"text", "message", "raw_text", "redacted_text"}
    if isinstance(value, Mapping):
        keys = {str(key).casefold() for key in value}
        if keys & forbidden_keys:
            raise ValueError("experiment artifacts cannot contain message-text fields")
        for child in value.values():
            assert_text_free_artifact(child)
    elif isinstance(value, list | tuple):
        for child in value:
            assert_text_free_artifact(child)


def assert_record_text_absent(value: Any, records: Iterable[BankingRecord]) -> None:
    """Detect accidental exact-message persistence even under an unexpected field name."""

    persisted_strings = set(_string_values(value))
    if any(record.text in persisted_strings for record in records):
        raise ValueError("experiment artifact contains a source message value")


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _string_values(child)


def write_json_artifact(value: Mapping[str, Any], destination: Path) -> None:
    assert_text_free_artifact(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)


def write_prediction_jsonl(rows: Iterable[Mapping[str, Any]], destination: Path) -> str:
    row_list = list(rows)
    assert_text_free_artifact(row_list)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in row_list:
                temporary.write(json.dumps(row, sort_keys=True) + "\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return stable_json_sha256(row_list)


def write_model(model: Pipeline, destination: Path) -> str:
    """Persist a local-only model; joblib files must never be loaded from untrusted sources."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(destination)
    return sha256_file(destination)
