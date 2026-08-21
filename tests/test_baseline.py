from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.baseline import (
    BaselineConfig,
    CandidateConfig,
    FeatureConfig,
    assert_record_text_absent,
    assert_text_free_artifact,
    build_pipeline,
    evaluate_locked_baseline,
    select_baseline,
    validate_evaluation_artifact,
    validate_selection_artifact,
)
from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256

LABELS = ("alpha", "beta", "gamma", "delta")


def _config() -> BaselineConfig:
    return BaselineConfig(
        experiment_name="synthetic-baseline-test",
        random_seed=42,
        thread_limit=1,
        selection_metric="macro_f1",
        tie_breakers=("accuracy", "negative_log_loss", "candidate_name"),
        candidates=(
            CandidateConfig(
                name="word_test",
                word=FeatureConfig(ngram_range=(1, 2), min_df=1, max_features=500),
                char=None,
                c_value=2.0,
                solver="saga",
                max_iter=500,
                tolerance=0.0001,
            ),
        ),
    )


def _records(source: str, examples_per_label: int, *, marker: str) -> tuple[BankingRecord, ...]:
    rows = [
        BankingRecord(
            source_split=source,
            source_index=index,
            text=f"{label} banking request {marker} example {example}",
            category=label,
        )
        for index, (label, example) in enumerate(
            (label, example)
            for label in LABELS
            for example in range(examples_per_label)
        )
    ]
    return tuple(rows)


def _hashes() -> dict[str, str]:
    return {"baseline.py": "a" * 64, "data.py": "b" * 64}


def test_registered_baseline_configuration_is_valid() -> None:
    config = BaselineConfig.from_yaml(Path("configs/baseline_tfidf.yaml"))

    assert config.selection_metric == "macro_f1"
    assert [candidate.name for candidate in config.candidates] == [
        "word_12_c2",
        "word_char_c2",
        "word_char_c4",
    ]
    assert config.thread_limit == 1


def test_committed_baseline_evidence_matches_current_implementation() -> None:
    manifest = json.loads(
        Path("data/manifests/banking77-seed-42.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        Path("reports/baseline/tfidf-logreg-selection.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        Path("reports/baseline/tfidf-logreg-test.json").read_text(encoding="utf-8")
    )
    implementation_sha256 = {
        "baseline.py": sha256_file(Path("src/governed_banking/baseline.py")),
        "data.py": sha256_file(Path("src/governed_banking/data.py")),
        "run_tfidf_baseline.py": sha256_file(Path("scripts/run_tfidf_baseline.py")),
    }
    config_sha256 = sha256_file(Path("configs/baseline_tfidf.yaml"))

    validate_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    validate_evaluation_artifact(
        evaluation,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    prediction_rows = [
        json.loads(line)
        for line in Path(
            "reports/baseline/tfidf-logreg-test-predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert stable_json_sha256(prediction_rows) == evaluation["test_predictions_sha256"]
    assert evaluation["test_result"]["metrics"]["macro_f1"] == 0.9053010357


def test_vectorizer_vocabulary_is_fitted_on_training_text_only() -> None:
    config = _config()
    train = _records("official_train", 4, marker="training")
    validation = list(_records("official_train", 1, marker="validationonlytoken"))
    model = build_pipeline(config.candidates[0], random_seed=config.random_seed)

    model.fit([record.text for record in train], [record.category for record in train])
    feature_names = set(model.named_steps["features"].get_feature_names_out())

    assert not any("validationonlytoken" in feature for feature in feature_names)
    assert model.predict([record.text for record in validation]).shape == (len(validation),)


def test_selection_and_locked_test_evaluation_are_text_free() -> None:
    config = _config()
    train = _records("official_train", 8, marker="training")
    validation = tuple(
        BankingRecord(
            source_split="official_train",
            source_index=100 + record.source_index,
            text=record.text,
            category=record.category,
        )
        for record in _records("official_train", 2, marker="validation")
    )
    selection = select_baseline(
        config,
        train,
        validation,
        label_names=LABELS,
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=_hashes(),
    )

    validate_selection_artifact(
        selection,
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=_hashes(),
    )
    assert selection["data_boundary"]["test_split_loaded"] is False
    assert selection["selected_candidate"] == "word_test"

    test = _records("official_test", 2, marker="test")
    _, evaluation, predictions = evaluate_locked_baseline(
        config,
        selection,
        train,
        test,
        label_names=LABELS,
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=_hashes(),
    )

    assert evaluation["data_boundary"]["selection_was_locked_before_test"] is True
    assert evaluation["test_result"]["metrics"]["count"] == len(test)
    assert len(predictions) == len(test)
    assert_text_free_artifact(evaluation)
    assert_text_free_artifact(predictions)
    evaluation_body = dict(evaluation)
    evaluation_body.pop("evaluation_sha256")
    evaluation_body["prediction_artifact"] = {
        "sha256": evaluation["test_predictions_sha256"]
    }
    evaluation_body["evaluation_sha256"] = stable_json_sha256(evaluation_body)
    validate_evaluation_artifact(
        evaluation_body,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=_hashes(),
    )


def test_tampered_selection_artifact_cannot_unlock_test_evaluation() -> None:
    config = _config()
    train = _records("official_train", 4, marker="training")
    validation = tuple(
        BankingRecord(
            source_split="official_train",
            source_index=100 + record.source_index,
            text=record.text,
            category=record.category,
        )
        for record in _records("official_train", 1, marker="validation")
    )
    selection = select_baseline(
        config,
        train,
        validation,
        label_names=LABELS,
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=_hashes(),
    )
    selection["selected_candidate"] = "changed_after_selection"

    with pytest.raises(ValueError, match="content hash"):
        evaluate_locked_baseline(
            config,
            selection,
            train,
            _records("official_test", 1, marker="test"),
            label_names=LABELS,
            dataset_manifest_sha256="c" * 64,
            config_sha256="d" * 64,
            implementation_sha256=_hashes(),
        )


def test_artifact_minimisation_rejects_message_fields() -> None:
    with pytest.raises(ValueError, match="message-text"):
        assert_text_free_artifact({"metrics": {}, "text": "do not persist me"})

    records = _records("official_train", 1, marker="private-source-value")
    with pytest.raises(ValueError, match="source message"):
        assert_record_text_absent({"unexpected_field": records[0].text}, records)
