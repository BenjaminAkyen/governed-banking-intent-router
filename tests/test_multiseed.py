from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.data import sha256_file, stable_json_sha256, validate_manifest
from governed_banking.multiseed import (
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    MultiSeedExperimentConfig,
    aggregate_validation_runs,
    assert_model_split_permitted,
    validate_manifest_registry,
    validate_seed_run,
    validate_validation_aggregate,
)


def _config() -> MultiSeedExperimentConfig:
    return MultiSeedExperimentConfig.from_yaml(Path("configs/multiseed_lora.yaml"))


def _manifest_implementation_hashes() -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "prepare_multiseed_manifests.py": Path("scripts/prepare_multiseed_manifests.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _training_implementation_hashes() -> dict[str, str]:
    paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "run_multiseed_lora.py": Path("scripts/run_multiseed_lora.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def test_registered_multiseed_protocol_is_validation_only() -> None:
    config = _config()

    assert config.seeds == (17, 42, 73)
    assert config.claim_scope == "post_test_exploratory"
    assert config.permitted_model_splits == ("train", "validation")
    assert config.prohibited_model_splits == ("test",)
    assert config.official_test_access_history == "observed_in_module_5"
    assert config.official_test_evaluation is False
    assert config.confirmatory_claims_permitted is False
    assert config.training.early_stopping == EarlyStoppingConfig(
        minimum_epochs=4,
        maximum_epochs=8,
        patience=2,
        min_delta=0.002,
        monitored_metric="macro_f1",
    )
    assert config.candidate.rank == 8
    assert config.candidate.learning_rate == 0.0002


def test_model_split_gate_fails_closed_for_test_and_unknown_names() -> None:
    assert_model_split_permitted("train")
    assert_model_split_permitted("validation")

    with pytest.raises(ValueError, match="prohibits"):
        assert_model_split_permitted("test")
    with pytest.raises(ValueError, match="prohibits"):
        assert_model_split_permitted("development")


def test_early_stopping_uses_minimum_epochs_patience_and_delta() -> None:
    tracker = EarlyStoppingTracker(
        EarlyStoppingConfig(4, 8, patience=2, min_delta=0.002, monitored_metric="macro_f1")
    )

    decisions = [
        tracker.observe(epoch, score)
        for epoch, score in enumerate((0.50, 0.60, 0.601, 0.6015), start=1)
    ]

    assert decisions == [False, False, False, True]
    assert tracker.significant_best == 0.60
    assert tracker.stale_epochs == 2


def test_early_stopping_resets_after_cumulative_material_improvement() -> None:
    tracker = EarlyStoppingTracker(
        EarlyStoppingConfig(4, 8, patience=2, min_delta=0.002, monitored_metric="macro_f1")
    )

    decisions = [
        tracker.observe(epoch, score)
        for epoch, score in enumerate((0.50, 0.501, 0.5031, 0.504, 0.5062), start=1)
    ]

    assert decisions == [False, False, False, False, False]
    assert tracker.significant_best == 0.5062
    assert tracker.stale_epochs == 0


def test_aggregate_reports_validation_statistics_without_test_metrics() -> None:
    config = _config()
    runs = [
        {
            "seed": seed,
            "run_sha256": str(seed) * 64,
            "best_epoch": 5,
            "epochs_completed": 6,
            "stopping_reason": "validation_patience_exhausted",
            "best_validation_metrics": {
                "accuracy": value,
                "balanced_accuracy": value,
                "macro_f1": value,
                "weighted_f1": value,
                "log_loss": 1 - value,
                "top_3_accuracy": value,
                "mean_max_confidence_uncalibrated": value,
            },
        }
        for seed, value in zip(config.seeds, (0.89, 0.90, 0.91), strict=True)
    ]
    implementation = {"multiseed.py": "a" * 64}

    artifact = aggregate_validation_runs(
        config,
        runs,
        registry_sha256="b" * 64,
        config_sha256="c" * 64,
        implementation_sha256=implementation,
    )

    assert artifact["validation_metrics"]["macro_f1"]["mean"] == 0.9
    assert artifact["validation_metrics"]["macro_f1"]["sample_standard_deviation"] == 0.01
    assert artifact["data_boundary"]["test_split_loaded"] is False
    assert artifact["data_boundary"]["official_test_metrics_computed"] is False
    assert "test_metrics" not in artifact


def test_committed_multiseed_manifests_and_registry_when_present() -> None:
    config = _config()
    if not config.registry_path.is_file():
        pytest.skip("Module 6 manifest registry has not been generated yet")
    registry = json.loads(config.registry_path.read_text(encoding="utf-8"))
    validate_manifest_registry(
        registry,
        config=config,
        dataset_config_sha256=sha256_file(Path("configs/dataset.yaml")),
        config_sha256=sha256_file(Path("configs/multiseed_lora.yaml")),
        implementation_sha256=_manifest_implementation_hashes(),
    )
    validation_hashes = set()
    for entry in registry["manifest_entries"]:
        manifest = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        validate_manifest(manifest)
        assert manifest["policy"]["seed"] == entry["seed"]
        assert manifest["manifest_sha256"] == entry["manifest_sha256"]
        assert manifest["splits"]["test"]["source_indices"] == list(range(3_080))
        validation_hashes.add(entry["validation_indices_sha256"])
    assert len(validation_hashes) == 3


def test_committed_multiseed_training_evidence_when_present() -> None:
    aggregate_path = Path("reports/multiseed-lora/validation-aggregate.json")
    if not aggregate_path.is_file():
        pytest.skip("Module 6 training evidence has not been generated yet")
    config = _config()
    registry = json.loads(config.registry_path.read_text(encoding="utf-8"))
    entries = {entry["seed"]: entry for entry in registry["manifest_entries"]}
    implementation = _training_implementation_hashes()
    model_hashes: dict[str, str] | None = None
    runs = []
    for seed in config.seeds:
        path = Path(f"reports/multiseed-lora/seed-{seed}-validation.json")
        run = json.loads(path.read_text(encoding="utf-8"))
        model_hashes = model_hashes or run["encoder"]["files_sha256"]
        validate_seed_run(
            run,
            config=config,
            manifest_sha256=entries[seed]["manifest_sha256"],
            config_sha256=sha256_file(Path("configs/multiseed_lora.yaml")),
            implementation_sha256=implementation,
            model_files_sha256=model_hashes,
            verify_checkpoint=False,
        )
        assert run["data_boundary"]["test_split_loaded"] is False
        runs.append(run)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_validation_aggregate(
        aggregate,
        config=config,
        registry_sha256=registry["registry_sha256"],
        config_sha256=sha256_file(Path("configs/multiseed_lora.yaml")),
        implementation_sha256=implementation,
    )
    assert aggregate["run_hashes"] == {str(run["seed"]): run["run_sha256"] for run in runs}
    aggregate_body = dict(aggregate)
    aggregate_hash = aggregate_body.pop("aggregate_sha256")
    assert stable_json_sha256(aggregate_body) == aggregate_hash
