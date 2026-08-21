from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256
from governed_banking.frozen_baseline import (
    EmbeddingBundle,
    EncoderConfig,
    FrozenBaselineConfig,
    FrozenCandidateConfig,
    evaluate_locked_frozen_baseline,
    freeze_encoder,
    load_embedding_cache,
    pool_hidden_states,
    select_frozen_baseline,
    validate_frozen_evaluation_artifact,
    validate_frozen_selection_artifact,
)

LABELS = ("alpha", "beta", "gamma", "delta")


def _config() -> FrozenBaselineConfig:
    return FrozenBaselineConfig(
        experiment_name="synthetic-frozen-test",
        random_seed=42,
        thread_limit=1,
        encoder=EncoderConfig(
            repository="FacebookAI/roberta-base",
            revision="a" * 40,
            license_name="MIT",
            max_length=32,
            batch_size=4,
            device="cpu",
            normalize_embeddings=True,
            snapshot_files=("config.json", "model.safetensors"),
        ),
        selection_metric="macro_f1",
        tie_breakers=("accuracy", "negative_log_loss", "candidate_name"),
        amendment_round=1,
        amendment_reason="initial synthetic test registration",
        test_accessed_before_amendment=False,
        candidates=(
            FrozenCandidateConfig(
                name="mean_test",
                pooling="mean",
                c_value=2.0,
                solver="lbfgs",
                max_iter=500,
                tolerance=0.00001,
            ),
        ),
    )


def _records(
    source: str, examples_per_label: int, *, index_offset: int = 0
) -> tuple[BankingRecord, ...]:
    return tuple(
        BankingRecord(
            source_split=source,
            source_index=index_offset + index,
            text=f"{label} frozen example {example}",
            category=label,
        )
        for index, (label, example) in enumerate(
            (label, example)
            for label in LABELS
            for example in range(examples_per_label)
        )
    )


def _bundle(split_name: str, records: tuple[BankingRecord, ...]) -> EmbeddingBundle:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    values = np.zeros((len(records), len(LABELS)), dtype=np.float32)
    for row_index, record in enumerate(records):
        values[row_index, label_to_index[record.category]] = 1.0
        values[row_index] += (row_index % 3) * 0.001
    return EmbeddingBundle(
        split_name=split_name,
        arrays={"cls": values.copy(), "mean": values.copy()},
        metadata={
            "cache_metadata_sha256": f"{split_name}-cache-hash",
            "trainable_encoder_parameters": 0,
        },
    )


def _implementation_hashes() -> dict[str, str]:
    return {"frozen_baseline.py": "b" * 64, "data.py": "c" * 64}


def _model_hashes() -> dict[str, str]:
    return {"config.json": "d" * 64, "model.safetensors": "e" * 64}


def test_registered_frozen_configuration_is_valid() -> None:
    config = FrozenBaselineConfig.from_yaml(Path("configs/frozen_roberta.yaml"))

    assert config.encoder.revision == "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
    assert config.encoder.device == "mps"
    assert config.encoder.max_length == 96
    assert [candidate.name for candidate in config.candidates] == [
        "cls_c1",
        "mean_c1",
        "mean_c4",
        "mean_c16",
        "mean_c64",
        "mean_c256",
        "mean_c1024",
        "mean_c4096",
    ]


def test_committed_frozen_evidence_matches_current_implementation() -> None:
    manifest = json.loads(
        Path("data/manifests/banking77-seed-42.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        Path("reports/frozen-roberta/selection.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        Path("reports/frozen-roberta/test.json").read_text(encoding="utf-8")
    )
    implementation_sha256 = {
        "baseline.py": sha256_file(Path("src/governed_banking/baseline.py")),
        "data.py": sha256_file(Path("src/governed_banking/data.py")),
        "device.py": sha256_file(Path("src/governed_banking/device.py")),
        "frozen_baseline.py": sha256_file(Path("src/governed_banking/frozen_baseline.py")),
        "run_frozen_roberta_baseline.py": sha256_file(
            Path("scripts/run_frozen_roberta_baseline.py")
        ),
    }
    model_hashes = selection["encoder"]["files_sha256"]
    config_sha256 = sha256_file(Path("configs/frozen_roberta.yaml"))

    validate_frozen_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=model_hashes,
    )
    validate_frozen_evaluation_artifact(
        evaluation,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=model_hashes,
    )
    prediction_rows = [
        json.loads(line)
        for line in Path("reports/frozen-roberta/test-predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert stable_json_sha256(prediction_rows) == evaluation["test_predictions_sha256"]
    assert evaluation["test_result"]["metrics"]["macro_f1"] == 0.8964173425

    extraction = json.loads(
        Path("reports/frozen-roberta/embedding-extraction.json").read_text(encoding="utf-8")
    )
    extraction_body = dict(extraction)
    extraction_hash = extraction_body.pop("evidence_sha256")
    assert stable_json_sha256(extraction_body) == extraction_hash
    assert extraction["implementation_sha256"] == {
        "summarize_frozen_embedding_evidence.py": sha256_file(
            Path("scripts/summarize_frozen_embedding_evidence.py")
        )
    }
    assert all(
        details["device"]["selected"] == "mps"
        for details in extraction["splits"].values()
    )
    assert all(
        details["trainable_encoder_parameters"] == 0
        for details in extraction["splits"].values()
    )


def test_pooling_excludes_special_and_padding_tokens() -> None:
    hidden = torch.tensor(
        [[[10.0, 0.0], [2.0, 2.0], [4.0, 2.0], [99.0, 99.0], [0.0, 50.0]]]
    )
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])

    pooled = pool_hidden_states(hidden, attention, special, normalize=False)

    assert torch.equal(pooled["cls"], torch.tensor([[10.0, 0.0]]))
    assert torch.equal(pooled["mean"], torch.tensor([[3.0, 2.0]]))
    normalized = pool_hidden_states(hidden, attention, special, normalize=True)
    assert torch.allclose(torch.linalg.vector_norm(normalized["mean"], dim=1), torch.ones(1))


def test_freeze_encoder_removes_all_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))

    total, trainable = freeze_encoder(model)

    assert total == 23
    assert trainable == 0
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_embedding_cache_detects_array_tampering(tmp_path: Path) -> None:
    expected_contract = {
        "split_name": "train",
        "row_count": 2,
        "source_indices_sha256": "a" * 64,
    }
    arrays = {
        "cls": np.arange(8, dtype=np.float32).reshape(2, 4),
        "mean": np.arange(8, dtype=np.float32).reshape(2, 4) / 2,
    }
    details = {}
    for name, array in arrays.items():
        path = tmp_path / f"train-{name}.npy"
        np.save(path, array, allow_pickle=False)
        details[name] = {
            "filename": path.name,
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    metadata = {
        "schema_version": 1,
        "artifact_type": "frozen_embedding_cache",
        "contains_message_text": False,
        **expected_contract,
        "trainable_encoder_parameters": 0,
        "arrays": details,
    }
    metadata["cache_metadata_sha256"] = stable_json_sha256(metadata)
    (tmp_path / "train-metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    loaded = load_embedding_cache(
        tmp_path,
        split_name="train",
        expected_contract=expected_contract,
    )
    assert np.array_equal(loaded.arrays["mean"], arrays["mean"])

    with (tmp_path / "train-mean.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        load_embedding_cache(
            tmp_path,
            split_name="train",
            expected_contract=expected_contract,
        )


def test_frozen_selection_and_evaluation_enforce_the_lock() -> None:
    config = _config()
    train = _records("official_train", 8)
    validation = _records("official_train", 2, index_offset=100)
    train_embeddings = _bundle("train", train)
    validation_embeddings = _bundle("validation", validation)
    selection = select_frozen_baseline(
        config,
        train,
        validation,
        train_embeddings,
        validation_embeddings,
        label_names=LABELS,
        dataset_manifest_sha256="f" * 64,
        config_sha256="1" * 64,
        implementation_sha256=_implementation_hashes(),
        model_files_sha256=_model_hashes(),
    )
    validate_frozen_selection_artifact(
        selection,
        dataset_manifest_sha256="f" * 64,
        config_sha256="1" * 64,
        implementation_sha256=_implementation_hashes(),
        model_files_sha256=_model_hashes(),
    )
    assert selection["data_boundary"]["test_embeddings_created"] is False

    test = _records("official_test", 2)
    test_embeddings = _bundle("test", test)
    _, evaluation, predictions = evaluate_locked_frozen_baseline(
        config,
        selection,
        train,
        test,
        train_embeddings,
        test_embeddings,
        label_names=LABELS,
        dataset_manifest_sha256="f" * 64,
        config_sha256="1" * 64,
        implementation_sha256=_implementation_hashes(),
        model_files_sha256=_model_hashes(),
    )
    assert evaluation["test_result"]["metrics"]["accuracy"] == 1.0
    assert len(predictions) == len(test)

    body = dict(evaluation)
    body.pop("evaluation_sha256")
    body["prediction_artifact"] = {"sha256": evaluation["test_predictions_sha256"]}
    body["evaluation_sha256"] = stable_json_sha256(body)
    validate_frozen_evaluation_artifact(
        body,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256="f" * 64,
        config_sha256="1" * 64,
        implementation_sha256=_implementation_hashes(),
        model_files_sha256=_model_hashes(),
    )

    selection["selected_candidate"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        evaluate_locked_frozen_baseline(
            config,
            selection,
            train,
            test,
            train_embeddings,
            test_embeddings,
            label_names=LABELS,
            dataset_manifest_sha256="f" * 64,
            config_sha256="1" * 64,
            implementation_sha256=_implementation_hashes(),
            model_files_sha256=_model_hashes(),
        )
