"""Leakage-safe LoRA adaptation of RoBERTa for BANKING77 intent routing."""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from governed_banking.baseline import (
    assert_record_text_absent,
    assert_text_free_artifact,
    classification_metrics,
    write_json_artifact,
    write_prediction_jsonl,
)
from governed_banking.data import BankingRecord, sha256_file, stable_json_sha256
from governed_banking.device import seed_everything, select_device
from governed_banking.frozen_baseline import EncoderConfig, ModelSnapshot

LORA_CONFIG_SCHEMA_VERSION = 1
LORA_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoraCandidateConfig:
    name: str
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    target_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    warmup_ratio: float
    weight_decay: float
    max_gradient_norm: float


@dataclass(frozen=True)
class LoraExperimentConfig:
    experiment_name: str
    random_seed: int
    thread_limit: int
    encoder: EncoderConfig
    train_batch_size: int
    evaluation_batch_size: int
    gradient_accumulation_steps: int
    attention_implementation: str
    training: TrainingConfig
    selection_metric: str
    tie_breakers: tuple[str, ...]
    candidates: tuple[LoraCandidateConfig, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> LoraExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != LORA_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported LoRA configuration schema")
        model = _mapping(raw, "model")
        training = _mapping(raw, "training")
        selection = _mapping(raw, "selection")
        snapshot_files_value = model.get("snapshot_files")
        if not isinstance(snapshot_files_value, list) or not snapshot_files_value:
            raise ValueError("model.snapshot_files must be a non-empty list")
        snapshot_files = tuple(str(value) for value in snapshot_files_value)
        if len(snapshot_files) != len(set(snapshot_files)):
            raise ValueError("model.snapshot_files cannot contain duplicates")
        if any(Path(filename).name != filename for filename in snapshot_files):
            raise ValueError("model snapshot filenames cannot contain directory traversal")
        repository = str(model.get("repository", "")).strip()
        revision = str(model.get("revision", "")).strip()
        if repository != "FacebookAI/roberta-base":
            raise ValueError("Module 5 is registered for FacebookAI/roberta-base")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("model.revision must be a full lowercase Git commit")
        device = str(model.get("device", ""))
        if device not in {"mps", "cpu"}:
            raise ValueError("model.device must be mps or cpu")
        attention_implementation = str(model.get("attention_implementation", ""))
        if attention_implementation != "eager":
            raise ValueError("Module 5 requires the registered eager attention implementation")

        selection_metric = str(selection.get("metric", ""))
        tie_breakers = tuple(str(value) for value in selection.get("tie_breakers", []))
        expected_tie_breakers = ("accuracy", "negative_log_loss", "candidate_name")
        if selection_metric != "macro_f1" or tie_breakers != expected_tie_breakers:
            raise ValueError("LoRA selection policy differs from the registered policy")

        candidate_values = raw.get("candidates")
        if not isinstance(candidate_values, list) or len(candidate_values) < 2:
            raise ValueError("at least two predeclared LoRA candidates are required")
        candidates = tuple(_parse_candidate(value) for value in candidate_values)
        names = [candidate.name for candidate in candidates]
        if len(names) != len(set(names)):
            raise ValueError("candidate names must be unique")

        encoder = EncoderConfig(
            repository=repository,
            revision=revision,
            license_name=str(model.get("license", "")).strip(),
            max_length=_bounded_int(model.get("max_length"), "model.max_length", 8, 512),
            batch_size=_bounded_int(
                model.get("evaluation_batch_size"), "model.evaluation_batch_size", 1, 512
            ),
            device=device,
            normalize_embeddings=False,
            snapshot_files=snapshot_files,
        )
        return cls(
            experiment_name=_non_blank(raw.get("experiment_name"), "experiment_name"),
            random_seed=_bounded_int(raw.get("random_seed"), "random_seed", 0, 2**32 - 1),
            thread_limit=_bounded_int(raw.get("thread_limit"), "thread_limit", 1, 128),
            encoder=encoder,
            train_batch_size=_bounded_int(
                model.get("train_batch_size"), "model.train_batch_size", 1, 512
            ),
            evaluation_batch_size=encoder.batch_size,
            gradient_accumulation_steps=_bounded_int(
                model.get("gradient_accumulation_steps"),
                "model.gradient_accumulation_steps",
                1,
                128,
            ),
            attention_implementation=attention_implementation,
            training=TrainingConfig(
                epochs=_bounded_int(training.get("epochs"), "training.epochs", 1, 20),
                warmup_ratio=_bounded_float(
                    training.get("warmup_ratio"), "training.warmup_ratio", 0.0, 0.5
                ),
                weight_decay=_bounded_float(
                    training.get("weight_decay"), "training.weight_decay", 0.0, 1.0
                ),
                max_gradient_norm=_bounded_float(
                    training.get("max_gradient_norm"),
                    "training.max_gradient_norm",
                    0.01,
                    100.0,
                ),
            ),
            selection_metric=selection_metric,
            tie_breakers=tie_breakers,
            candidates=candidates,
        )


class TokenizedBankingDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory tensors; message text is never written to an experiment artifact."""

    def __init__(
        self,
        records: Sequence[BankingRecord],
        tokenizer: Any,
        *,
        label_to_id: Mapping[str, int],
        max_length: int,
    ) -> None:
        if not records:
            raise ValueError("records cannot be empty")
        unknown = sorted({record.category for record in records} - set(label_to_id))
        if unknown:
            raise ValueError(f"records contain unknown labels: {unknown}")
        encoded = tokenizer(
            [record.text for record in records],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.features = {
            key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}
        }
        self.labels = torch.tensor(
            [label_to_id[record.category] for record in records], dtype=torch.long
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{name: values[index] for name, values in self.features.items()},
            "labels": self.labels[index],
        }


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


def _parse_candidate(value: Any) -> LoraCandidateConfig:
    if not isinstance(value, dict):
        raise ValueError("each candidate must be a mapping")
    name = _non_blank(value.get("name"), "candidate.name")
    if not name.replace("_", "").isalnum():
        raise ValueError("candidate name must contain only letters, numbers and underscores")
    target_modules = tuple(str(item) for item in value.get("target_modules", []))
    modules_to_save = tuple(str(item) for item in value.get("modules_to_save", []))
    if target_modules != ("query", "value"):
        raise ValueError("Module 5 candidates must target query and value projections")
    if modules_to_save != ("classifier",):
        raise ValueError("the sequence-classification head must be trained and saved")
    return LoraCandidateConfig(
        name=name,
        rank=_bounded_int(value.get("rank"), f"{name}.rank", 1, 256),
        alpha=_bounded_int(value.get("alpha"), f"{name}.alpha", 1, 1024),
        dropout=_bounded_float(value.get("dropout"), f"{name}.dropout", 0.0, 0.9),
        learning_rate=_bounded_float(
            value.get("learning_rate"), f"{name}.learning_rate", 1e-7, 0.1
        ),
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def token_length_audit(
    records: Sequence[BankingRecord], tokenizer: Any, *, max_length: int
) -> dict[str, int]:
    lengths = tokenizer(
        [record.text for record in records],
        add_special_tokens=True,
        truncation=False,
        return_length=True,
    )["length"]
    return {
        "rows": len(records),
        "maximum_tokens": int(max(lengths)),
        "rows_exceeding_max_length": int(sum(length > max_length for length in lengths)),
    }


def build_lora_model(
    snapshot: ModelSnapshot,
    candidate: LoraCandidateConfig,
    *,
    label_names: Sequence[str],
    attention_implementation: str,
) -> tuple[PeftModel, dict[str, int | float]]:
    base_model = build_base_sequence_classifier(
        snapshot,
        label_names=label_names,
        attention_implementation=attention_implementation,
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=candidate.rank,
        lora_alpha=candidate.alpha,
        lora_dropout=candidate.dropout,
        target_modules=list(candidate.target_modules),
        bias="none",
        modules_to_save=list(candidate.modules_to_save),
        init_lora_weights=True,
    )
    model = get_peft_model(base_model, peft_config)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert_trainable_parameter_policy(
        [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    )
    return model, {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": round(100.0 * trainable / total, 10),
    }


def build_base_sequence_classifier(
    snapshot: ModelSnapshot,
    *,
    label_names: Sequence[str],
    attention_implementation: str,
) -> torch.nn.Module:
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        snapshot.path,
        local_files_only=True,
        use_safetensors=True,
        num_labels=len(label_names),
        label2id=label_to_id,
        id2label=id_to_label,
        trust_remote_code=False,
        attn_implementation=attention_implementation,
    )


def assert_trainable_parameter_policy(names: Sequence[str]) -> None:
    if not names:
        raise ValueError("LoRA model has no trainable parameters")
    invalid = [
        name
        for name in names
        if not ("lora_A" in name or "lora_B" in name or "modules_to_save" in name)
    ]
    if invalid:
        raise ValueError(f"unexpected trainable parameters: {invalid[:5]}")
    if not any("lora_A" in name for name in names) or not any("lora_B" in name for name in names):
        raise ValueError("both LoRA matrix families must be trainable")
    if not any("classifier" in name and "modules_to_save" in name for name in names):
        raise ValueError("classification head must be trainable through modules_to_save")


def train_and_select_lora(
    config: LoraExperimentConfig,
    train_records: Sequence[BankingRecord],
    validation_records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    snapshot: ModelSnapshot,
    checkpoint_root: Path,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    _validate_split(train_records, "official_train", "train")
    _validate_split(validation_records, "official_train", "validation")
    if {record.source_index for record in train_records} & {
        record.source_index for record in validation_records
    }:
        raise ValueError("training and validation source indices overlap")
    device, runtime = select_device(config.encoder.device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    train_data = TokenizedBankingDataset(
        train_records, tokenizer, label_to_id=label_to_id, max_length=config.encoder.max_length
    )
    validation_data = TokenizedBankingDataset(
        validation_records,
        tokenizer,
        label_to_id=label_to_id,
        max_length=config.encoder.max_length,
    )
    token_audit = {
        "train": token_length_audit(train_records, tokenizer, max_length=config.encoder.max_length),
        "validation": token_length_audit(
            validation_records, tokenizer, max_length=config.encoder.max_length
        ),
    }
    candidate_results: list[dict[str, Any]] = []
    torch.set_num_threads(config.thread_limit)
    for candidate in config.candidates:
        seed_everything(config.random_seed)
        model, parameter_counts = build_lora_model(
            snapshot,
            candidate,
            label_names=label_names,
            attention_implementation=config.attention_implementation,
        )
        model.to(device)
        generator = torch.Generator().manual_seed(config.random_seed)
        train_loader = DataLoader(
            train_data,
            batch_size=config.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_data,
            batch_size=config.evaluation_batch_size,
            shuffle=False,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=candidate.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_updates = updates_per_epoch * config.training.epochs
        warmup_updates = int(total_updates * config.training.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates
        )
        epoch_results: list[dict[str, Any]] = []
        best_sort_key: tuple[float, float, float, int] | None = None
        best_metrics: dict[str, Any] | None = None
        best_epoch = 0
        checkpoint_directory = checkpoint_root / candidate.name
        training_started = time.perf_counter()
        for epoch in range(1, config.training.epochs + 1):
            train_loss = _train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device=device,
                accumulation_steps=config.gradient_accumulation_steps,
                max_gradient_norm=config.training.max_gradient_norm,
            )
            metrics, _, _ = evaluate_model(
                model, validation_loader, validation_records, label_names=label_names, device=device
            )
            sort_key = (
                -metrics["macro_f1"],
                -metrics["accuracy"],
                metrics["log_loss"],
                epoch,
            )
            if best_sort_key is None or sort_key < best_sort_key:
                best_sort_key = sort_key
                best_metrics = metrics
                best_epoch = epoch
                model.save_pretrained(checkpoint_directory, safe_serialization=True)
            epoch_results.append(
                {
                    "epoch": epoch,
                    "mean_training_loss": _metric_float(train_loss),
                    "validation": _aggregate_metrics(metrics),
                }
            )
        _synchronize(device)
        training_seconds = time.perf_counter() - training_started
        if best_metrics is None:
            raise AssertionError("candidate did not produce validation metrics")
        checkpoint_files = hash_checkpoint_files(checkpoint_directory)
        candidate_results.append(
            {
                "candidate_name": candidate.name,
                "candidate": _candidate_dict(candidate),
                "parameters": parameter_counts,
                "optimizer": "AdamW",
                "scheduler": "linear_with_warmup",
                "total_optimizer_updates": total_updates,
                "warmup_updates": warmup_updates,
                "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics,
                "epochs": epoch_results,
                "training_seconds": _metric_float(training_seconds),
                "checkpoint": {
                    "path": str(checkpoint_directory),
                    "files_sha256": checkpoint_files,
                    "total_bytes": sum(
                        (checkpoint_directory / name).stat().st_size for name in checkpoint_files
                    ),
                },
            }
        )
        del model, optimizer, scheduler
        if device.type == "mps":
            torch.mps.empty_cache()

    ranked = sorted(
        candidate_results,
        key=lambda result: (
            -result["best_validation_metrics"]["macro_f1"],
            -result["best_validation_metrics"]["accuracy"],
            result["best_validation_metrics"]["log_loss"],
            result["candidate_name"],
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result["validation_rank"] = rank
    artifact: dict[str, Any] = {
        "schema_version": LORA_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "lora_validation_selection",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lora_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "encoder": {
            "repository": config.encoder.repository,
            "revision": config.encoder.revision,
            "license": config.encoder.license_name,
            "files_sha256": dict(sorted(snapshot.file_sha256.items())),
            "maximum_length": config.encoder.max_length,
            "attention_implementation": config.attention_implementation,
        },
        "random_seed": config.random_seed,
        "data_boundary": {
            "fit_split": "train",
            "selection_split": "validation",
            "test_split_loaded": False,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
        },
        "token_length_audit": token_audit,
        "selection_policy": {
            "metric": config.selection_metric,
            "tie_breakers": list(config.tie_breakers),
        },
        "selected_candidate": ranked[0]["candidate_name"],
        "selected_checkpoint_files_sha256": ranked[0]["checkpoint"]["files_sha256"],
        "candidate_results": ranked,
        "runtime_device": runtime.to_dict(),
        "software": software_versions(),
        "reproducibility_notice": (
            "Seeded training is best-effort on MPS; exact cross-version bitwise reproducibility "
            "is not claimed."
        ),
    }
    artifact["selection_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_record_text_absent(artifact, (*train_records, *validation_records))
    return artifact


def _train_epoch(
    model: PeftModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    device: torch.device,
    accumulation_steps: int,
    max_gradient_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        moved = {key: value.to(device) for key, value in batch.items()}
        loss = model(**moved).loss
        accumulated_loss += float(loss.detach().cpu())
        (loss / accumulation_steps).backward()
        if batch_index % accumulation_steps == 0 or batch_index == len(loader):
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_gradient_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    _synchronize(device)
    return accumulated_loss / len(loader)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            model_inputs = {
                key: value.to(device) for key, value in batch.items() if key != "labels"
            }
            logits_parts.append(model(**model_inputs).logits.detach().cpu())
    _synchronize(device)
    logits = torch.cat(logits_parts).numpy()
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = np.asarray([label_names[index] for index in predicted_ids])
    metric_classes = tuple(sorted(label_names))
    probability_indices = [label_names.index(label) for label in metric_classes]
    metrics = classification_metrics(
        [record.category for record in records],
        predicted_labels,
        probabilities[:, probability_indices],
        classes=metric_classes,
        label_names=label_names,
    )
    return metrics, predicted_labels, probabilities


def evaluate_locked_lora(
    config: LoraExperimentConfig,
    selection: Mapping[str, Any],
    test_records: Sequence[BankingRecord],
    *,
    label_names: Sequence[str],
    snapshot: ModelSnapshot,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_lora_selection_artifact(
        selection,
        dataset_manifest_sha256=dataset_manifest_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
        verify_checkpoint=True,
    )
    _validate_split(test_records, "official_test", "test")
    selected_name = str(selection["selected_candidate"])
    candidate_lookup = {candidate.name: candidate for candidate in config.candidates}
    if selected_name not in candidate_lookup:
        raise ValueError("selected candidate is absent from current LoRA configuration")
    selected_result = next(
        result
        for result in selection["candidate_results"]
        if result["candidate_name"] == selected_name
    )
    checkpoint_directory = Path(selected_result["checkpoint"]["path"])
    base_model = build_base_sequence_classifier(
        snapshot,
        label_names=label_names,
        attention_implementation=config.attention_implementation,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_directory, is_trainable=False)
    device, runtime = select_device(config.encoder.device)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    test_data = TokenizedBankingDataset(
        test_records, tokenizer, label_to_id=label_to_id, max_length=config.encoder.max_length
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.evaluation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    started = time.perf_counter()
    metrics, predicted_labels, probabilities = evaluate_model(
        model, test_loader, test_records, label_names=label_names, device=device
    )
    inference_seconds = time.perf_counter() - started
    predictions = [
        {
            "source_index": record.source_index,
            "true_label": record.category,
            "predicted_label": str(predicted),
            "confidence_uncalibrated": _metric_float(float(max(probs))),
        }
        for record, predicted, probs in zip(
            test_records, predicted_labels, probabilities, strict=True
        )
    ]
    prediction_sha256 = stable_json_sha256(predictions)
    artifact: dict[str, Any] = {
        "schema_version": LORA_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "lora_locked_test_evaluation",
        "experiment_name": config.experiment_name,
        "contains_message_text": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lora_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "selection_sha256": selection["selection_sha256"],
        "selected_candidate": selected_name,
        "selected_checkpoint_files_sha256": selection["selected_checkpoint_files_sha256"],
        "encoder_files_sha256": dict(sorted(snapshot.file_sha256.items())),
        "data_boundary": {
            "selection_was_locked_before_test": True,
            "evaluation_split": "test",
            "test_rows": len(test_records),
        },
        "test_token_length_audit": token_length_audit(
            test_records, tokenizer, max_length=config.encoder.max_length
        ),
        "test_result": {
            "metrics": metrics,
            "inference_seconds": _metric_float(inference_seconds),
        },
        "test_predictions_sha256": prediction_sha256,
        "confidence_notice": "Probabilities are uncalibrated until Module 7.",
        "runtime_device": runtime.to_dict(),
        "software": software_versions(),
    }
    artifact["evaluation_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_text_free_artifact(predictions)
    assert_record_text_absent(artifact, test_records)
    assert_record_text_absent(predictions, test_records)
    return artifact, predictions


def hash_checkpoint_files(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"adapter checkpoint directory does not exist: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError("adapter checkpoint contains no files")
    relative_names = [str(path.relative_to(directory)) for path in files]
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(relative_names):
        raise ValueError("adapter checkpoint is missing safe PEFT files")
    if any(name.endswith((".bin", ".pt", ".pth")) for name in relative_names):
        raise ValueError("adapter checkpoint contains a disallowed pickle-based weight file")
    return {name: sha256_file(directory / name) for name in relative_names}


def validate_lora_selection_artifact(
    artifact: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
    verify_checkpoint: bool,
) -> None:
    if artifact.get("artifact_type") != "lora_validation_selection":
        raise ValueError("expected a LoRA validation-selection artifact")
    if artifact.get("data_boundary", {}).get("test_split_loaded") is not False:
        raise ValueError("selection artifact must prove the test split was not loaded")
    body = dict(artifact)
    expected_hash = body.pop("selection_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("LoRA selection content hash check failed")
    if artifact.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("LoRA selection uses a different dataset manifest")
    if artifact.get("lora_config_sha256") != config_sha256:
        raise ValueError("LoRA selection uses a different configuration")
    if artifact.get("implementation_sha256") != dict(sorted(implementation_sha256.items())):
        raise ValueError("LoRA selection uses a different implementation")
    if artifact.get("encoder", {}).get("files_sha256") != dict(sorted(model_files_sha256.items())):
        raise ValueError("LoRA selection uses different encoder files")
    selected_name = artifact.get("selected_candidate")
    selected_results = [
        result
        for result in artifact.get("candidate_results", [])
        if result.get("candidate_name") == selected_name
    ]
    if len(selected_results) != 1:
        raise ValueError("selected LoRA candidate is missing or duplicated")
    registered_hashes = selected_results[0].get("checkpoint", {}).get("files_sha256")
    if registered_hashes != artifact.get("selected_checkpoint_files_sha256"):
        raise ValueError("selected adapter hash mapping is inconsistent")
    if verify_checkpoint:
        checkpoint_path = Path(selected_results[0]["checkpoint"]["path"])
        if hash_checkpoint_files(checkpoint_path) != registered_hashes:
            raise ValueError("selected adapter checkpoint hash verification failed")
    assert_text_free_artifact(artifact)


def validate_lora_evaluation_artifact(
    artifact: Mapping[str, Any],
    *,
    selection_sha256: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "lora_locked_test_evaluation":
        raise ValueError("expected a LoRA locked-test artifact")
    if artifact.get("data_boundary", {}).get("selection_was_locked_before_test") is not True:
        raise ValueError("LoRA test evaluation must follow locked selection")
    body = dict(artifact)
    expected_hash = body.pop("evaluation_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("LoRA evaluation content hash check failed")
    expected_values = {
        "selection_sha256": selection_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lora_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "encoder_files_sha256": dict(sorted(model_files_sha256.items())),
    }
    for key, value in expected_values.items():
        if artifact.get(key) != value:
            raise ValueError(f"LoRA evaluation has an invalid {key}")
    prediction_artifact = artifact.get("prediction_artifact", {})
    if prediction_artifact.get("sha256") != artifact.get("test_predictions_sha256"):
        raise ValueError("LoRA prediction artifact hash does not match evaluation")
    assert_text_free_artifact(artifact)


def _validate_split(
    records: Sequence[BankingRecord], expected_source: str, split_name: str
) -> None:
    if not records:
        raise ValueError(f"{split_name} records cannot be empty")
    if any(record.source_split != expected_source for record in records):
        raise ValueError(f"{split_name} records must come only from {expected_source}")


def _candidate_dict(candidate: LoraCandidateConfig) -> dict[str, Any]:
    value = asdict(candidate)
    value["dropout"] = _metric_float(candidate.dropout)
    value["learning_rate"] = _metric_float(candidate.learning_rate)
    return value


def _aggregate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "top_3_accuracy",
        "mean_max_confidence_uncalibrated",
    )
    return {name: metrics[name] for name in names}


def _metric_float(value: float) -> float:
    return round(float(value), 10)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "scikit_learn": version("scikit-learn"),
    }


def write_lora_report(value: Mapping[str, Any], destination: Path) -> None:
    write_json_artifact(value, destination)


def write_lora_predictions(rows: Sequence[Mapping[str, Any]], destination: Path) -> str:
    return write_prediction_jsonl(rows, destination)
