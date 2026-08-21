"""Post-test exploratory, validation-only multi-seed LoRA experiments."""

from __future__ import annotations

import math
import platform
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import yaml
from scipy.stats import t as student_t
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from governed_banking.baseline import (
    assert_record_text_absent,
    assert_text_free_artifact,
    write_json_artifact,
)
from governed_banking.data import (
    BankingRecord,
    DatasetConfig,
    build_manifest,
    prepare_splits,
    read_banking_csv,
    read_categories,
    sha256_file,
    stable_json_sha256,
    validate_manifest,
    write_manifest,
)
from governed_banking.device import seed_everything, select_device
from governed_banking.frozen_baseline import EncoderConfig, ModelSnapshot
from governed_banking.lora_baseline import (
    LoraCandidateConfig,
    TokenizedBankingDataset,
    build_lora_model,
    evaluate_model,
    hash_checkpoint_files,
    token_length_audit,
)

MULTISEED_CONFIG_SCHEMA_VERSION = 1
MULTISEED_ARTIFACT_SCHEMA_VERSION = 1
REGISTERED_SEEDS = (17, 42, 73)
PERMITTED_MODEL_SPLITS = ("train", "validation")
PROHIBITED_MODEL_SPLITS = ("test",)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    minimum_epochs: int
    maximum_epochs: int
    patience: int
    min_delta: float
    monitored_metric: str


@dataclass(frozen=True)
class MultiSeedTrainingConfig:
    early_stopping: EarlyStoppingConfig
    warmup_ratio: float
    weight_decay: float
    max_gradient_norm: float


@dataclass(frozen=True)
class MultiSeedExperimentConfig:
    experiment_name: str
    claim_scope: str
    seeds: tuple[int, ...]
    thread_limit: int
    manifest_path_pattern: str
    registry_path: Path
    encoder: EncoderConfig
    train_batch_size: int
    evaluation_batch_size: int
    gradient_accumulation_steps: int
    attention_implementation: str
    training: MultiSeedTrainingConfig
    candidate: LoraCandidateConfig
    permitted_model_splits: tuple[str, ...]
    prohibited_model_splits: tuple[str, ...]
    official_test_access_history: str
    official_test_evaluation: bool
    confirmatory_claims_permitted: bool

    @classmethod
    def from_yaml(cls, path: Path) -> MultiSeedExperimentConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("unsupported multi-seed configuration schema")
        model = _mapping(raw, "model")
        training = _mapping(raw, "training")
        manifests = _mapping(raw, "manifests")
        boundary = _mapping(raw, "boundary")
        seeds_value = raw.get("seeds")
        if not isinstance(seeds_value, list) or tuple(seeds_value) != REGISTERED_SEEDS:
            raise ValueError(f"Module 6 seeds must be {REGISTERED_SEEDS}")
        claim_scope = _non_blank(raw.get("claim_scope"), "claim_scope")
        if claim_scope != "post_test_exploratory":
            raise ValueError("Module 6 must remain explicitly post-test exploratory")
        permitted = tuple(str(value) for value in boundary.get("permitted_model_splits", []))
        prohibited = tuple(str(value) for value in boundary.get("prohibited_model_splits", []))
        if permitted != PERMITTED_MODEL_SPLITS or prohibited != PROHIBITED_MODEL_SPLITS:
            raise ValueError("Module 6 model access must be train/validation only")
        test_history = _non_blank(
            boundary.get("official_test_access_history"), "official_test_access_history"
        )
        if test_history != "observed_in_module_5":
            raise ValueError("Module 6 must disclose prior official-test access")
        test_evaluation = _strict_bool(
            boundary.get("official_test_evaluation_in_this_module"),
            "official_test_evaluation_in_this_module",
        )
        confirmatory = _strict_bool(
            boundary.get("confirmatory_claims_permitted"),
            "confirmatory_claims_permitted",
        )
        if test_evaluation or confirmatory:
            raise ValueError("Module 6 cannot evaluate official test or make confirmatory claims")

        snapshot_files_value = model.get("snapshot_files")
        if not isinstance(snapshot_files_value, list) or not snapshot_files_value:
            raise ValueError("model.snapshot_files must be a non-empty list")
        snapshot_files = tuple(str(value) for value in snapshot_files_value)
        if len(snapshot_files) != len(set(snapshot_files)):
            raise ValueError("model.snapshot_files cannot contain duplicates")
        if any(Path(filename).name != filename for filename in snapshot_files):
            raise ValueError("model snapshot filenames cannot contain directory traversal")
        repository = _non_blank(model.get("repository"), "model.repository")
        revision = _non_blank(model.get("revision"), "model.revision")
        if repository != "FacebookAI/roberta-base":
            raise ValueError("Module 6 is registered for FacebookAI/roberta-base")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("model.revision must be a full lowercase Git commit")
        device = _non_blank(model.get("device"), "model.device")
        if device not in {"mps", "cpu"}:
            raise ValueError("model.device must be mps or cpu")
        attention = _non_blank(
            model.get("attention_implementation"), "model.attention_implementation"
        )
        if attention != "eager":
            raise ValueError("Module 6 requires eager attention")

        minimum_epochs = _bounded_int(
            training.get("minimum_epochs"), "training.minimum_epochs", 1, 30
        )
        maximum_epochs = _bounded_int(
            training.get("maximum_epochs"), "training.maximum_epochs", 1, 30
        )
        if minimum_epochs >= maximum_epochs:
            raise ValueError("minimum_epochs must be lower than maximum_epochs")
        monitored = _non_blank(training.get("monitored_metric"), "training.monitored_metric")
        if monitored != "macro_f1":
            raise ValueError("early stopping must monitor validation macro_f1")
        path_pattern = _non_blank(manifests.get("path_pattern"), "manifests.path_pattern")
        if path_pattern.count("{seed}") != 1:
            raise ValueError("manifest path pattern must contain exactly one {seed}")

        return cls(
            experiment_name=_non_blank(raw.get("experiment_name"), "experiment_name"),
            claim_scope=claim_scope,
            seeds=REGISTERED_SEEDS,
            thread_limit=_bounded_int(raw.get("thread_limit"), "thread_limit", 1, 128),
            manifest_path_pattern=path_pattern,
            registry_path=Path(_non_blank(manifests.get("registry_path"), "registry_path")),
            encoder=EncoderConfig(
                repository=repository,
                revision=revision,
                license_name=_non_blank(model.get("license"), "model.license"),
                max_length=_bounded_int(model.get("max_length"), "model.max_length", 8, 512),
                batch_size=_bounded_int(
                    model.get("evaluation_batch_size"),
                    "model.evaluation_batch_size",
                    1,
                    512,
                ),
                device=device,
                normalize_embeddings=False,
                snapshot_files=snapshot_files,
            ),
            train_batch_size=_bounded_int(
                model.get("train_batch_size"), "model.train_batch_size", 1, 512
            ),
            evaluation_batch_size=_bounded_int(
                model.get("evaluation_batch_size"), "model.evaluation_batch_size", 1, 512
            ),
            gradient_accumulation_steps=_bounded_int(
                model.get("gradient_accumulation_steps"),
                "model.gradient_accumulation_steps",
                1,
                128,
            ),
            attention_implementation=attention,
            training=MultiSeedTrainingConfig(
                early_stopping=EarlyStoppingConfig(
                    minimum_epochs=minimum_epochs,
                    maximum_epochs=maximum_epochs,
                    patience=_bounded_int(
                        training.get("early_stopping_patience"),
                        "training.early_stopping_patience",
                        1,
                        10,
                    ),
                    min_delta=_bounded_float(
                        training.get("early_stopping_min_delta"),
                        "training.early_stopping_min_delta",
                        0.0,
                        0.1,
                    ),
                    monitored_metric=monitored,
                ),
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
            candidate=_parse_candidate(_mapping(raw, "candidate")),
            permitted_model_splits=permitted,
            prohibited_model_splits=prohibited,
            official_test_access_history=test_history,
            official_test_evaluation=test_evaluation,
            confirmatory_claims_permitted=confirmatory,
        )

    def manifest_path(self, seed: int) -> Path:
        if seed not in self.seeds:
            raise ValueError(f"unregistered seed: {seed}")
        return Path(self.manifest_path_pattern.format(seed=seed))


@dataclass
class EarlyStoppingTracker:
    config: EarlyStoppingConfig
    significant_best: float = -math.inf
    stale_epochs: int = 0

    def observe(self, epoch: int, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("early-stopping score must be finite")
        if score > self.significant_best + self.config.min_delta:
            self.significant_best = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return epoch >= self.config.minimum_epochs and self.stale_epochs >= self.config.patience


def assert_model_split_permitted(split_name: str) -> None:
    """Fail closed before any Module 6 modelling data access."""

    if split_name in PROHIBITED_MODEL_SPLITS or split_name not in PERMITTED_MODEL_SPLITS:
        raise ValueError(f"Module 6 prohibits model access to split: {split_name}")


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


def _parse_candidate(value: Mapping[str, Any]) -> LoraCandidateConfig:
    name = _non_blank(value.get("name"), "candidate.name")
    if name != "lora_r8_lr2e4_revised_stopping":
        raise ValueError("Module 6 must reuse the registered rank-8 architecture")
    targets = tuple(str(item) for item in value.get("target_modules", []))
    saved = tuple(str(item) for item in value.get("modules_to_save", []))
    if targets != ("query", "value") or saved != ("classifier",):
        raise ValueError("Module 6 LoRA targets or saved modules changed")
    candidate = LoraCandidateConfig(
        name=name,
        rank=_bounded_int(value.get("rank"), "candidate.rank", 1, 256),
        alpha=_bounded_int(value.get("alpha"), "candidate.alpha", 1, 1024),
        dropout=_bounded_float(value.get("dropout"), "candidate.dropout", 0.0, 0.9),
        learning_rate=_bounded_float(
            value.get("learning_rate"), "candidate.learning_rate", 1e-7, 0.1
        ),
        target_modules=targets,
        modules_to_save=saved,
    )
    expected = (8, 16, 0.1, 0.0002)
    observed = (candidate.rank, candidate.alpha, candidate.dropout, candidate.learning_rate)
    if observed != expected:
        raise ValueError("Module 6 candidate must reuse Module 5's validation winner")
    return candidate


def build_multiseed_manifests(
    config: MultiSeedExperimentConfig,
    *,
    dataset_config_path: Path,
    raw_directory: Path,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Generate exact seed-specific development splits and a text-free registry."""

    dataset_config = DatasetConfig.from_yaml(dataset_config_path)
    source_paths = {
        name: raw_directory / filename for name, filename in dataset_config.files.items()
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"offline source is missing: {path}")
        if sha256_file(path) != dataset_config.expected_sha256[name]:
            raise ValueError(f"source hash mismatch for {name}")
    labels = read_categories(
        source_paths["categories"], expected_count=dataset_config.expected_label_count
    )
    official_train = read_banking_csv(
        source_paths["official_train"],
        source_split="official_train",
        expected_rows=dataset_config.expected_train_rows,
    )
    # Integrity preparation reads the official test source only to quarantine exact duplicates.
    # The modelling runner below has no path that can load or score this split.
    official_test = read_banking_csv(
        source_paths["official_test"],
        source_split="official_test",
        expected_rows=dataset_config.expected_test_rows,
    )
    entries: list[dict[str, Any]] = []
    for seed in config.seeds:
        seeded_config = replace(dataset_config, seed=seed)
        splits = prepare_splits(
            official_train,
            official_test,
            labels,
            seed=seed,
            validation_fraction=dataset_config.validation_fraction,
        )
        manifest = build_manifest(seeded_config, source_paths, splits)
        validate_manifest(manifest)
        path = config.manifest_path(seed)
        write_manifest(manifest, path)
        entries.append(
            {
                "seed": seed,
                "path": str(path),
                "manifest_sha256": manifest["manifest_sha256"],
                "train_indices_sha256": manifest["splits"]["train"]["source_indices_sha256"],
                "validation_indices_sha256": manifest["splits"]["validation"][
                    "source_indices_sha256"
                ],
                "train_rows": manifest["splits"]["train"]["count"],
                "validation_rows": manifest["splits"]["validation"]["count"],
            }
        )
    if len({entry["validation_indices_sha256"] for entry in entries}) != len(entries):
        raise AssertionError("registered seeds did not produce distinct validation splits")
    registry: dict[str, Any] = {
        "schema_version": MULTISEED_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "banking77_multiseed_manifest_registry",
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "dataset_config_sha256": sha256_file(dataset_config_path),
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "manifest_entries": entries,
        "model_access_boundary": {
            "permitted_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_splits": list(PROHIBITED_MODEL_SPLITS),
            "official_test_metrics_computed": False,
            "integrity_preparation_read_official_test_for_duplicate_quarantine": True,
        },
    }
    registry["registry_sha256"] = stable_json_sha256(registry)
    validate_manifest_registry(
        registry,
        config=config,
        dataset_config_sha256=sha256_file(dataset_config_path),
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    write_json_artifact(registry, config.registry_path)
    return registry


def validate_manifest_registry(
    registry: Mapping[str, Any],
    *,
    config: MultiSeedExperimentConfig,
    dataset_config_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if registry.get("artifact_type") != "banking77_multiseed_manifest_registry":
        raise ValueError("expected a multi-seed manifest registry")
    body = dict(registry)
    expected_hash = body.pop("registry_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("multi-seed registry content hash check failed")
    expected = {
        "seeds": list(config.seeds),
        "dataset_config_sha256": dataset_config_sha256,
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ValueError(f"multi-seed registry has an invalid {key}")
    boundary = registry.get("model_access_boundary", {})
    if boundary.get("permitted_splits") != list(PERMITTED_MODEL_SPLITS):
        raise ValueError("registry permits an invalid model split")
    if boundary.get("prohibited_splits") != list(PROHIBITED_MODEL_SPLITS):
        raise ValueError("registry does not prohibit official test")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("registry cannot contain official-test metrics")
    entries = registry.get("manifest_entries", [])
    if [entry.get("seed") for entry in entries] != list(config.seeds):
        raise ValueError("registry manifest entries do not match registered seeds")
    assert_text_free_artifact(registry)


def train_validation_seed(
    config: MultiSeedExperimentConfig,
    *,
    seed: int,
    train_records: Sequence[BankingRecord],
    validation_records: Sequence[BankingRecord],
    label_names: Sequence[str],
    manifest_sha256: str,
    snapshot: ModelSnapshot,
    checkpoint_directory: Path,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Train one seed and stop exclusively from its validation trajectory."""

    if seed not in config.seeds:
        raise ValueError(f"unregistered seed: {seed}")
    _validate_development_records(train_records, validation_records)
    seed_everything(seed)
    torch.set_num_threads(config.thread_limit)
    device, runtime = select_device(config.encoder.device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    train_data = TokenizedBankingDataset(
        train_records,
        tokenizer,
        label_to_id=label_to_id,
        max_length=config.encoder.max_length,
    )
    validation_data = TokenizedBankingDataset(
        validation_records,
        tokenizer,
        label_to_id=label_to_id,
        max_length=config.encoder.max_length,
    )
    model, parameter_counts = build_lora_model(
        snapshot,
        config.candidate,
        label_names=label_names,
        attention_implementation=config.attention_implementation,
    )
    model.to(device)
    train_loader = DataLoader(
        train_data,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
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
        lr=config.candidate.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    maximum_epochs = config.training.early_stopping.maximum_epochs
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * maximum_epochs
    warmup_updates = int(total_updates * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_updates, num_training_steps=total_updates
    )
    tracker = EarlyStoppingTracker(config.training.early_stopping)
    epoch_results: list[dict[str, Any]] = []
    best_sort_key: tuple[float, float, float, int] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stopped_early = False
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
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
            model,
            validation_loader,
            validation_records,
            label_names=label_names,
            device=device,
        )
        sort_key = (-metrics["macro_f1"], -metrics["accuracy"], metrics["log_loss"], epoch)
        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_metrics = metrics
            best_epoch = epoch
            model.save_pretrained(checkpoint_directory, safe_serialization=True)
        should_stop = tracker.observe(epoch, metrics["macro_f1"])
        epoch_results.append(
            {
                "epoch": epoch,
                "mean_training_loss": _metric_float(train_loss),
                "validation": _aggregate_metrics(metrics),
                "early_stopping": {
                    "significant_best_macro_f1": _metric_float(tracker.significant_best),
                    "epochs_without_min_delta": tracker.stale_epochs,
                    "stop_after_epoch": should_stop,
                },
            }
        )
        if should_stop:
            stopped_early = True
            break
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    if best_metrics is None:
        raise AssertionError("multi-seed run did not produce validation metrics")
    checkpoint_hashes = hash_checkpoint_files(checkpoint_directory)
    artifact: dict[str, Any] = {
        "schema_version": MULTISEED_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "multiseed_lora_validation_run",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seed": seed,
        "dataset_manifest_sha256": manifest_sha256,
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "encoder": {
            "repository": config.encoder.repository,
            "revision": config.encoder.revision,
            "files_sha256": dict(sorted(snapshot.file_sha256.items())),
        },
        "candidate": asdict(config.candidate),
        "parameters": parameter_counts,
        "data_boundary": {
            "loaded_model_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "official_test_access_history": config.official_test_access_history,
            "train_rows": len(train_records),
            "validation_rows": len(validation_records),
        },
        "token_length_audit": {
            "train": token_length_audit(
                train_records, tokenizer, max_length=config.encoder.max_length
            ),
            "validation": token_length_audit(
                validation_records, tokenizer, max_length=config.encoder.max_length
            ),
        },
        "early_stopping_policy": asdict(config.training.early_stopping),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.candidate.learning_rate,
            "weight_decay": config.training.weight_decay,
            "scheduler": "linear_with_warmup",
            "planned_optimizer_updates": total_updates,
            "warmup_updates": warmup_updates,
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "epochs_completed": len(epoch_results),
        "stopped_early": stopped_early,
        "stopping_reason": (
            "validation_patience_exhausted" if stopped_early else "maximum_epochs_reached"
        ),
        "epochs": epoch_results,
        "training_seconds": _metric_float(training_seconds),
        "checkpoint": {
            "path": str(checkpoint_directory),
            "files_sha256": checkpoint_hashes,
            "total_bytes": sum(
                (checkpoint_directory / name).stat().st_size for name in checkpoint_hashes
            ),
        },
        "runtime_device": runtime.to_dict(),
        "software": software_versions(),
        "claim_notice": (
            "Validation-only post-test exploratory evidence; no official-test performance "
            "or confirmatory improvement claim is permitted."
        ),
    }
    artifact["run_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    assert_record_text_absent(artifact, (*train_records, *validation_records))
    return artifact


def validate_seed_run(
    artifact: Mapping[str, Any],
    *,
    config: MultiSeedExperimentConfig,
    manifest_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
    model_files_sha256: Mapping[str, str],
    verify_checkpoint: bool,
) -> None:
    if artifact.get("artifact_type") != "multiseed_lora_validation_run":
        raise ValueError("expected a multi-seed validation run")
    body = dict(artifact)
    expected_hash = body.pop("run_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("multi-seed run content hash check failed")
    expected = {
        "claim_scope": "post_test_exploratory",
        "dataset_manifest_sha256": manifest_sha256,
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"multi-seed run has an invalid {key}")
    if artifact.get("seed") not in config.seeds:
        raise ValueError("multi-seed run uses an unregistered seed")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("loaded_model_splits") != list(PERMITTED_MODEL_SPLITS):
        raise ValueError("multi-seed run loaded an invalid model split")
    if boundary.get("prohibited_model_splits") != list(PROHIBITED_MODEL_SPLITS):
        raise ValueError("multi-seed run did not prohibit official test")
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("multi-seed run accessed official test")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("multi-seed run computed official-test metrics")
    if artifact.get("encoder", {}).get("files_sha256") != dict(sorted(model_files_sha256.items())):
        raise ValueError("multi-seed run uses different base-model files")
    checkpoint = artifact.get("checkpoint", {})
    if verify_checkpoint:
        observed = hash_checkpoint_files(Path(checkpoint["path"]))
        if observed != checkpoint.get("files_sha256"):
            raise ValueError("multi-seed checkpoint hash verification failed")
    assert_text_free_artifact(artifact)


def aggregate_validation_runs(
    config: MultiSeedExperimentConfig,
    runs: Sequence[Mapping[str, Any]],
    *,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if [run.get("seed") for run in runs] != list(config.seeds):
        raise ValueError("validation runs must appear in registered seed order")
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "top_3_accuracy",
        "mean_max_confidence_uncalibrated",
    )
    aggregated_metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [float(run["best_validation_metrics"][name]) for run in runs]
        mean = statistics.fmean(values)
        std = statistics.stdev(values)
        half_width = float(student_t.ppf(0.975, df=len(values) - 1)) * std / math.sqrt(len(values))
        aggregated_metrics[name] = {
            "values_by_seed": {
                str(run["seed"]): _metric_float(value)
                for run, value in zip(runs, values, strict=True)
            },
            "mean": _metric_float(mean),
            "sample_standard_deviation": _metric_float(std),
            "confidence_interval_95": [
                _metric_float(mean - half_width),
                _metric_float(mean + half_width),
            ],
        }
    artifact: dict[str, Any] = {
        "schema_version": MULTISEED_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "multiseed_lora_validation_aggregate",
        "experiment_name": config.experiment_name,
        "claim_scope": config.claim_scope,
        "contains_message_text": False,
        "seeds": list(config.seeds),
        "manifest_registry_sha256": registry_sha256,
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "data_boundary": {
            "loaded_model_splits": list(PERMITTED_MODEL_SPLITS),
            "prohibited_model_splits": list(PROHIBITED_MODEL_SPLITS),
            "test_split_loaded": False,
            "official_test_metrics_computed": False,
            "official_test_access_history": config.official_test_access_history,
        },
        "run_hashes": {str(run["seed"]): run["run_sha256"] for run in runs},
        "best_epochs": {str(run["seed"]): run["best_epoch"] for run in runs},
        "epochs_completed": {str(run["seed"]): run["epochs_completed"] for run in runs},
        "stopping_reasons": {str(run["seed"]): run["stopping_reason"] for run in runs},
        "validation_metrics": aggregated_metrics,
        "statistical_notice": (
            "The t intervals use only three validation seeds and are descriptive, not "
            "confirmatory or representative of production uncertainty."
        ),
        "claim_notice": (
            "Do not compare these revised validation results with previously observed "
            "official-test "
            "scores as evidence of a test improvement."
        ),
    }
    artifact["aggregate_sha256"] = stable_json_sha256(artifact)
    validate_validation_aggregate(
        artifact,
        config=config,
        registry_sha256=registry_sha256,
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    return artifact


def validate_validation_aggregate(
    artifact: Mapping[str, Any],
    *,
    config: MultiSeedExperimentConfig,
    registry_sha256: str,
    config_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> None:
    if artifact.get("artifact_type") != "multiseed_lora_validation_aggregate":
        raise ValueError("expected a multi-seed validation aggregate")
    body = dict(artifact)
    expected_hash = body.pop("aggregate_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("multi-seed aggregate content hash check failed")
    expected = {
        "claim_scope": "post_test_exploratory",
        "seeds": list(config.seeds),
        "manifest_registry_sha256": registry_sha256,
        "multiseed_config_sha256": config_sha256,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"multi-seed aggregate has an invalid {key}")
    boundary = artifact.get("data_boundary", {})
    if boundary.get("test_split_loaded") is not False:
        raise ValueError("multi-seed aggregate cannot contain test access")
    if boundary.get("official_test_metrics_computed") is not False:
        raise ValueError("multi-seed aggregate cannot contain test metrics")
    assert_text_free_artifact(artifact)


def _validate_development_records(
    train_records: Sequence[BankingRecord], validation_records: Sequence[BankingRecord]
) -> None:
    if not train_records or not validation_records:
        raise ValueError("train and validation records cannot be empty")
    if any(
        record.source_split != "official_train" for record in (*train_records, *validation_records)
    ):
        raise ValueError("Module 6 model records must come only from official_train")
    if {record.source_index for record in train_records} & {
        record.source_index for record in validation_records
    }:
        raise ValueError("Module 6 train and validation indices overlap")


def _train_epoch(
    model: torch.nn.Module,
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
    total_loss = 0.0
    for batch_index, batch in enumerate(loader, start=1):
        moved = {key: value.to(device) for key, value in batch.items()}
        loss = model(**moved).loss
        total_loss += float(loss.detach().cpu())
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
    return total_loss / len(loader)


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
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "scikit_learn": version("scikit-learn"),
        "scipy": version("scipy"),
    }
