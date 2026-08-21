from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from governed_banking.calibration import (
    BootstrapConfig,
    CalibrationConfig,
    TemperatureConfig,
    assert_calibration_split_permitted,
    bootstrap_calibration_deltas,
    calibration_metrics,
    fit_temperature,
    partition_calibration_records,
    probabilities_from_logits,
    validate_calibration_aggregate,
    validate_calibration_registry,
    validate_seed_calibration_report,
)
from governed_banking.data import BankingRecord, sha256_file

LABELS = ("alpha", "beta", "gamma", "delta")


def _config() -> CalibrationConfig:
    return CalibrationConfig.from_yaml(Path("configs/calibration.yaml"))


def _records() -> tuple[BankingRecord, ...]:
    rows = [
        BankingRecord(
            "official_train",
            index,
            f"{label} calibration example {example}",
            label,
        )
        for index, (label, example) in enumerate(
            (label, example) for label in LABELS for example in range(8)
        )
    ]
    rows.extend(
        (
            BankingRecord("official_train", 100, "Repeated alpha", "alpha"),
            BankingRecord("official_train", 101, " repeated   ALPHA ", "alpha"),
        )
    )
    return tuple(rows)


def _partition_implementation_hashes() -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_calibration_splits.py": Path("scripts/prepare_calibration_splits.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _calibration_implementation_hashes() -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "run_temperature_scaling.py": Path("scripts/run_temperature_scaling.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def test_registered_calibration_protocol_has_disjoint_roles_and_no_test_access() -> None:
    config = _config()

    assert config.seeds == (17, 42, 73)
    assert config.claim_scope == "post_selection_post_test_exploratory"
    assert config.source_split == "validation"
    assert config.source_role_history == "used_for_module6_checkpoint_selection"
    assert config.partition.development_role == "temperature_fit"
    assert config.partition.assessment_role == "calibration_assessment"
    assert config.partition.development_fraction == 0.5
    assert config.ece_bins == 15
    assert config.bootstrap.resamples == 2000
    assert config.permitted_model_splits == ("validation",)
    assert config.prohibited_model_splits == ("train", "test")
    assert config.official_test_evaluation is False
    assert config.independent_model_evaluation is False


def test_calibration_split_gate_fails_closed() -> None:
    assert_calibration_split_permitted("validation")

    with pytest.raises(ValueError, match="prohibits"):
        assert_calibration_split_permitted("train")
    with pytest.raises(ValueError, match="prohibits"):
        assert_calibration_split_permitted("test")
    with pytest.raises(ValueError, match="prohibits"):
        assert_calibration_split_permitted("unknown")


def test_partition_is_deterministic_stratified_and_group_safe() -> None:
    records = _records()

    first_fit, first_assessment = partition_calibration_records(
        records,
        label_names=LABELS,
        development_fraction=0.5,
        random_seed=7017,
    )
    second_fit, second_assessment = partition_calibration_records(
        records,
        label_names=LABELS,
        development_fraction=0.5,
        random_seed=7017,
    )

    assert [record.source_index for record in first_fit] == [
        record.source_index for record in second_fit
    ]
    assert [record.source_index for record in first_assessment] == [
        record.source_index for record in second_assessment
    ]
    assert not {record.source_index for record in first_fit} & {
        record.source_index for record in first_assessment
    }
    assert {record.category for record in first_fit} == set(LABELS)
    assert {record.category for record in first_assessment} == set(LABELS)
    repeated_membership = [
        {100, 101}.issubset({record.source_index for record in partition})
        for partition in (first_fit, first_assessment)
    ]
    assert repeated_membership.count(True) == 1


def test_temperature_fit_reduces_nll_for_overconfident_synthetic_logits() -> None:
    generator = np.random.default_rng(42)
    latent_logits = generator.normal(size=(4000, 4))
    true_probabilities = probabilities_from_logits(latent_logits, temperature=2.0)
    labels = np.asarray(
        [generator.choice(4, p=probabilities) for probabilities in true_probabilities],
        dtype=np.int64,
    )

    fitted = fit_temperature(
        latent_logits,
        labels,
        TemperatureConfig(0.05, 10.0, absolute_tolerance=1e-8, maximum_iterations=500),
    )

    assert 1.7 < fitted["temperature"] < 2.3
    assert fitted["calibrated_nll"] < fitted["raw_nll"]
    assert fitted["temperature_at_registered_boundary"] is False


def test_calibration_metrics_and_positive_temperature_preserve_predictions() -> None:
    logits = np.asarray([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [2.5, 2.0, 0.0], [0.0, 1.0, 4.0]])
    labels = np.asarray([0, 1, 1, 2])
    raw = probabilities_from_logits(logits)
    calibrated = probabilities_from_logits(logits, temperature=1.8)

    raw_metrics = calibration_metrics(labels, raw, bins=5)
    calibrated_metrics = calibration_metrics(labels, calibrated, bins=5)

    assert np.array_equal(raw.argmax(axis=1), calibrated.argmax(axis=1))
    assert raw_metrics["count"] == 4
    assert len(raw_metrics["reliability_bins"]) == 5
    assert 0 <= raw_metrics["expected_calibration_error"] <= 1
    assert calibrated_metrics["accuracy"] == raw_metrics["accuracy"]
    assert calibrated_metrics["macro_f1"] == raw_metrics["macro_f1"]


def test_bootstrap_reports_calibrated_minus_raw_intervals() -> None:
    logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [1.0, 2.0], [3.0, 0.0], [0.0, 3.0]])
    labels = np.asarray([0, 1, 1, 0, 0, 1])
    raw = probabilities_from_logits(logits)
    calibrated = probabilities_from_logits(logits, temperature=1.5)

    result = bootstrap_calibration_deltas(
        labels,
        raw,
        calibrated,
        bins=3,
        config=BootstrapConfig(100, 0.95, 9000),
        random_seed=9017,
    )

    assert result["resamples"] == 100
    assert result["delta_definition"] == "calibrated_minus_raw"
    assert set(result["intervals"]) == {
        "negative_log_likelihood",
        "expected_calibration_error",
        "maximum_calibration_error",
        "multiclass_brier_score",
        "signed_confidence_gap",
    }


def test_committed_calibration_registry_when_present() -> None:
    config = _config()
    if not config.partition.registry_path.is_file():
        pytest.skip("Module 7 calibration registry has not been generated yet")
    registry = json.loads(config.partition.registry_path.read_text(encoding="utf-8"))
    validate_calibration_registry(
        registry,
        config=config,
        config_sha256=sha256_file(Path("configs/calibration.yaml")),
        implementation_sha256=_partition_implementation_hashes(),
    )
    for entry in registry["entries"]:
        fit = entry["temperature_fit"]
        assessment = entry["calibration_assessment"]
        assert not set(fit["source_indices"]) & set(assessment["source_indices"])
        assert entry["normalized_text_group_overlap"] == 0


def test_committed_calibration_evidence_when_present() -> None:
    aggregate_path = Path("reports/calibration/temperature-scaling-aggregate.json")
    if not aggregate_path.is_file():
        pytest.skip("Module 7 calibration evidence has not been generated yet")
    config = _config()
    registry = json.loads(config.partition.registry_path.read_text(encoding="utf-8"))
    implementation = _calibration_implementation_hashes()
    reports = []
    for seed in config.seeds:
        report = json.loads(
            Path(f"reports/calibration/seed-{seed}-temperature-scaling.json").read_text(
                encoding="utf-8"
            )
        )
        validate_seed_calibration_report(
            report,
            config=config,
            registry_sha256=registry["registry_sha256"],
            config_sha256=sha256_file(Path("configs/calibration.yaml")),
            implementation_sha256=implementation,
        )
        assert report["prediction_invariance"]["changed_class_predictions"] == 0
        assert "development_metrics" not in report
        assert set(report["assessment_metrics"]) == {"raw", "calibrated", "calibrated_minus_raw"}
        reports.append(report)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_calibration_aggregate(
        aggregate,
        config=config,
        registry_sha256=registry["registry_sha256"],
        config_sha256=sha256_file(Path("configs/calibration.yaml")),
        implementation_sha256=implementation,
    )
    assert aggregate["report_hashes"] == {
        str(report["seed"]): report["report_sha256"] for report in reports
    }
