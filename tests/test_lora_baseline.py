from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256
from governed_banking.lora_baseline import (
    LoraExperimentConfig,
    TokenizedBankingDataset,
    assert_trainable_parameter_policy,
    evaluate_model,
    hash_checkpoint_files,
    validate_lora_evaluation_artifact,
    validate_lora_selection_artifact,
)


class FakeTokenizer:
    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        if kwargs.get("return_length"):
            return {"length": [len(text.split()) + 2 for text in texts]}
        maximum = int(kwargs["max_length"])
        input_ids = torch.zeros((len(texts), maximum), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, text in enumerate(texts):
            length = min(len(text.split()) + 2, maximum)
            input_ids[index, :length] = torch.arange(1, length + 1)
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class IdentityLogitModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(logits=input_ids.float())


def test_registered_lora_configuration_is_valid() -> None:
    config = LoraExperimentConfig.from_yaml(Path("configs/lora_roberta.yaml"))

    assert config.encoder.revision == "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
    assert config.encoder.device == "mps"
    assert config.encoder.max_length == 96
    assert config.gradient_accumulation_steps == 2
    assert config.training.epochs == 3
    assert [candidate.name for candidate in config.candidates] == [
        "lora_r8_lr2e4",
        "lora_r16_lr1e4",
    ]
    assert all(candidate.target_modules == ("query", "value") for candidate in config.candidates)
    assert all(candidate.modules_to_save == ("classifier",) for candidate in config.candidates)


def test_tokenized_dataset_keeps_labels_aligned_without_storing_records() -> None:
    records = (
        BankingRecord("official_train", 10, "cash withdrawal failed", "cash_withdrawal"),
        BankingRecord("official_train", 11, "card payment reversed", "card_payment"),
    )
    dataset = TokenizedBankingDataset(
        records,
        FakeTokenizer(),
        label_to_id={"cash_withdrawal": 0, "card_payment": 1},
        max_length=8,
    )

    assert len(dataset) == 2
    assert dataset[0]["labels"].item() == 0
    assert dataset[1]["labels"].item() == 1
    assert dataset[0]["attention_mask"].sum().item() == 5
    assert not hasattr(dataset, "records")


def test_trainable_parameter_policy_rejects_base_encoder_updates() -> None:
    valid = (
        "base_model.encoder.layer.0.query.lora_A.default.weight",
        "base_model.encoder.layer.0.query.lora_B.default.weight",
        "base_model.classifier.modules_to_save.default.out_proj.weight",
    )
    assert_trainable_parameter_policy(valid)

    with pytest.raises(ValueError, match="unexpected trainable"):
        assert_trainable_parameter_policy((*valid, "base_model.encoder.layer.0.dense.weight"))


def test_neural_probability_order_is_adapted_for_sklearn_metrics() -> None:
    label_names = ("zeta", "alpha", "mu", "beta")
    records = tuple(
        BankingRecord("official_train", index, f"example {index}", label)
        for index, label in enumerate(label_names)
    )
    loader = torch.utils.data.DataLoader(
        [
            {
                "input_ids": torch.nn.functional.one_hot(
                    torch.tensor(index), num_classes=len(label_names)
                ).float()
                * 10,
                "attention_mask": torch.ones(len(label_names)),
            }
            for index in range(len(label_names))
        ],
        batch_size=2,
    )

    metrics, predictions, probabilities = evaluate_model(
        IdentityLogitModel(),
        loader,
        records,
        label_names=label_names,
        device=torch.device("cpu"),
    )

    assert metrics["accuracy"] == 1.0
    assert list(predictions) == list(label_names)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2, 3]


def test_checkpoint_hashing_requires_safetensors_and_detects_changes(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "adapter_model.safetensors"
    weights.write_bytes(b"safe-adapter")
    hashes = hash_checkpoint_files(tmp_path)

    assert set(hashes) == {"adapter_config.json", "adapter_model.safetensors"}
    weights.write_bytes(b"changed-adapter")
    assert hash_checkpoint_files(tmp_path) != hashes

    (tmp_path / "unsafe.bin").write_bytes(b"pickle")
    with pytest.raises(ValueError, match="pickle-based"):
        hash_checkpoint_files(tmp_path)


def test_selection_and_evaluation_locks_detect_tampering(tmp_path: Path) -> None:
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"safe-adapter")
    checkpoint_hashes = hash_checkpoint_files(tmp_path)
    implementation = {"lora_baseline.py": "a" * 64}
    model_hashes = {"model.safetensors": "b" * 64}
    selection = {
        "schema_version": 1,
        "artifact_type": "lora_validation_selection",
        "contains_message_text": False,
        "dataset_manifest_sha256": "c" * 64,
        "lora_config_sha256": "d" * 64,
        "implementation_sha256": implementation,
        "encoder": {"files_sha256": model_hashes},
        "data_boundary": {"test_split_loaded": False},
        "selected_candidate": "registered_candidate",
        "selected_checkpoint_files_sha256": checkpoint_hashes,
        "candidate_results": [
            {
                "candidate_name": "registered_candidate",
                "checkpoint": {"path": str(tmp_path), "files_sha256": checkpoint_hashes},
            }
        ],
    }
    selection["selection_sha256"] = stable_json_sha256(selection)
    validate_lora_selection_artifact(
        selection,
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=implementation,
        model_files_sha256=model_hashes,
        verify_checkpoint=True,
    )

    tampered = json.loads(json.dumps(selection))
    tampered["selected_candidate"] = "post_test_change"
    with pytest.raises(ValueError, match="content hash"):
        validate_lora_selection_artifact(
            tampered,
            dataset_manifest_sha256="c" * 64,
            config_sha256="d" * 64,
            implementation_sha256=implementation,
            model_files_sha256=model_hashes,
            verify_checkpoint=False,
        )

    evaluation = {
        "schema_version": 1,
        "artifact_type": "lora_locked_test_evaluation",
        "contains_message_text": False,
        "selection_sha256": selection["selection_sha256"],
        "dataset_manifest_sha256": "c" * 64,
        "lora_config_sha256": "d" * 64,
        "implementation_sha256": implementation,
        "encoder_files_sha256": model_hashes,
        "data_boundary": {"selection_was_locked_before_test": True},
        "test_predictions_sha256": "e" * 64,
        "prediction_artifact": {"sha256": "e" * 64},
    }
    evaluation["evaluation_sha256"] = stable_json_sha256(evaluation)
    validate_lora_evaluation_artifact(
        evaluation,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        implementation_sha256=implementation,
        model_files_sha256=model_hashes,
    )


def test_committed_lora_evidence_matches_current_implementation_when_present() -> None:
    selection_path = Path("reports/lora-roberta/selection.json")
    evaluation_path = Path("reports/lora-roberta/test.json")
    predictions_path = Path("reports/lora-roberta/test-predictions.jsonl")
    if not selection_path.exists():
        pytest.skip("Module 5 evidence has not been generated yet")
    manifest = json.loads(Path("data/manifests/banking77-seed-42.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    implementation_paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "run_lora_roberta.py": Path("scripts/run_lora_roberta.py"),
    }
    implementation = {name: sha256_file(path) for name, path in implementation_paths.items()}
    model_hashes = selection["encoder"]["files_sha256"]
    validate_lora_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=sha256_file(Path("configs/lora_roberta.yaml")),
        implementation_sha256=implementation,
        model_files_sha256=model_hashes,
        verify_checkpoint=False,
    )
    validate_lora_evaluation_artifact(
        evaluation,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=sha256_file(Path("configs/lora_roberta.yaml")),
        implementation_sha256=implementation,
        model_files_sha256=model_hashes,
    )
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    assert stable_json_sha256(rows) == evaluation["test_predictions_sha256"]
