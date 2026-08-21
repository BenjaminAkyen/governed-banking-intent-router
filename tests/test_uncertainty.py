from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from governed_banking.data import BankingRecord, sha256_file
from governed_banking.uncertainty import (
    UncertaintyConfig,
    assert_uncertainty_source_permitted,
    load_possible_ood_fixture,
    partition_known_records,
    partition_possible_ood_records,
    risk_coverage_evidence,
    select_signal_and_threshold,
    selection_metrics,
    uncertainty_signals,
    validate_seed_uncertainty_report,
    validate_uncertainty_aggregate,
    validate_uncertainty_registry,
)


def _config() -> UncertaintyConfig:
    return UncertaintyConfig.from_yaml(Path("configs/uncertainty.yaml"))


def _known_records() -> tuple[BankingRecord, ...]:
    labels = ("alpha", "beta", "gamma", "delta")
    records = [
        BankingRecord("official_train", index, f"{label} example {number}", label)
        for index, (label, number) in enumerate(
            (label, number) for label in labels for number in range(8)
        )
    ]
    records.extend(
        (
            BankingRecord("official_train", 100, "Repeated alpha", "alpha"),
            BankingRecord("official_train", 101, " repeated ALPHA ", "alpha"),
        )
    )
    return tuple(records)


def _partition_implementation_hashes() -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_uncertainty_roles.py": Path("scripts/prepare_uncertainty_roles.py"),
        "uncertainty.py": Path("src/governed_banking/uncertainty.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _run_implementation_hashes() -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "run_uncertainty_evaluation.py": Path("scripts/run_uncertainty_evaluation.py"),
        "uncertainty.py": Path("src/governed_banking/uncertainty.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def test_registered_uncertainty_protocol_has_separate_roles_and_no_test_access() -> None:
    config = _config()

    assert config.seeds == (17, 42, 73)
    assert config.claim_scope == "post_calibration_post_test_exploratory"
    assert config.known_partition.development_role == "threshold_known_development"
    assert config.known_partition.assessment_role == "selective_known_assessment"
    assert config.possible_ood.development_role == "threshold_ood_development"
    assert config.possible_ood.assessment_role == "possible_ood_assessment"
    assert config.signals == (
        "max_probability",
        "top_two_margin",
        "inverse_normalized_entropy",
    )
    assert config.selection.target_ood_recall == 0.90
    assert config.selection.target_selective_risk == 0.05
    assert config.selection.minimum_known_coverage == 0.70
    assert config.official_test_evaluation is False
    assert config.independent_model_evaluation is False


def test_uncertainty_source_gate_fails_closed() -> None:
    assert_uncertainty_source_permitted("validation")
    assert_uncertainty_source_permitted("synthetic_fixture")

    for prohibited in ("train", "test", "unknown"):
        with pytest.raises(ValueError, match="prohibits"):
            assert_uncertainty_source_permitted(prohibited)


def test_synthetic_possible_ood_fixture_is_diverse_unique_and_group_safe() -> None:
    config = _config()
    records = load_possible_ood_fixture(config.possible_ood.fixture_path)
    first_development, first_assessment = partition_possible_ood_records(
        records,
        development_fraction=config.possible_ood.development_fraction,
        random_seed=config.possible_ood.random_seed,
    )
    second_development, second_assessment = partition_possible_ood_records(
        records,
        development_fraction=config.possible_ood.development_fraction,
        random_seed=config.possible_ood.random_seed,
    )

    assert len(records) == 96
    assert len({record.domain for record in records}) == 12
    assert [record.fixture_id for record in first_development] == [
        record.fixture_id for record in second_development
    ]
    assert [record.fixture_id for record in first_assessment] == [
        record.fixture_id for record in second_assessment
    ]
    assert not {record.scenario_group for record in first_development} & {
        record.scenario_group for record in first_assessment
    }
    assert len(first_development) == len(first_assessment) == 48


def test_known_partition_is_deterministic_and_normalized_group_safe() -> None:
    records = _known_records()
    first_development, first_assessment = partition_known_records(
        records, development_fraction=0.5, random_seed=11017
    )
    second_development, second_assessment = partition_known_records(
        records, development_fraction=0.5, random_seed=11017
    )

    assert [record.source_index for record in first_development] == [
        record.source_index for record in second_development
    ]
    assert [record.source_index for record in first_assessment] == [
        record.source_index for record in second_assessment
    ]
    repeated_membership = [
        {100, 101}.issubset({record.source_index for record in partition})
        for partition in (first_development, first_assessment)
    ]
    assert repeated_membership.count(True) == 1


def test_uncertainty_signals_have_registered_direction() -> None:
    probabilities = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.40, 0.35, 0.25],
        ]
    )

    signals = uncertainty_signals(probabilities)

    assert signals["max_probability"][0] > signals["max_probability"][1]
    assert signals["top_two_margin"][0] > signals["top_two_margin"][1]
    assert signals["inverse_normalized_entropy"][0] > signals["inverse_normalized_entropy"][1]


def test_development_only_selection_avoids_reject_everything_solution() -> None:
    config = _config()
    known_scores = np.asarray([0.99, 0.97, 0.95, 0.93, 0.91, 0.88, 0.82, 0.76, 0.70, 0.62])
    known_correct = np.asarray([True, True, True, True, True, True, True, False, True, False])
    ood_scores = np.asarray([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    known_signals = {signal: known_scores for signal in config.signals}
    ood_signals = {signal: ood_scores for signal in config.signals}

    selection = select_signal_and_threshold(
        known_signals, known_correct, ood_signals, config=config
    )

    assert selection["development_constraints_feasible"] is True
    assert selection["selected_signal"] == "max_probability"
    assert selection["selected_development_metrics"]["known_coverage"] >= 0.70
    assert selection["selected_development_metrics"]["selective_risk"] <= 0.05
    assert selection["selected_development_metrics"]["possible_ood_recall"] >= 0.90


def test_selection_and_risk_coverage_metrics_are_well_formed() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    correct = np.asarray([True, True, False, False])
    ood_scores = np.asarray([0.1, 0.2, 0.75])

    metrics = selection_metrics(scores, correct, ood_scores, threshold=0.7)
    curve = risk_coverage_evidence(scores, correct)

    assert metrics["known_coverage"] == 0.75
    assert metrics["selective_risk"] == pytest.approx(1 / 3)
    assert metrics["possible_ood_recall"] == pytest.approx(2 / 3)
    assert 0 <= curve["aurc"] <= 1
    assert len(curve["curve"]) == 10


def test_committed_uncertainty_registry_when_present() -> None:
    config = _config()
    if not config.possible_ood.registry_path.is_file():
        pytest.skip("Module 8 uncertainty registry has not been generated yet")
    registry = json.loads(config.possible_ood.registry_path.read_text(encoding="utf-8"))
    validate_uncertainty_registry(
        registry,
        config=config,
        config_sha256=sha256_file(Path("configs/uncertainty.yaml")),
        implementation_sha256=_partition_implementation_hashes(),
    )


def test_committed_uncertainty_evidence_when_present() -> None:
    aggregate_path = Path("reports/uncertainty/selective-ood-aggregate.json")
    if not aggregate_path.is_file():
        pytest.skip("Module 8 uncertainty evidence has not been generated yet")
    config = _config()
    registry = json.loads(config.possible_ood.registry_path.read_text(encoding="utf-8"))
    implementation = _run_implementation_hashes()
    reports = []
    for seed in config.seeds:
        report = json.loads(
            Path(f"reports/uncertainty/seed-{seed}-selective-ood.json").read_text(encoding="utf-8")
        )
        validate_seed_uncertainty_report(
            report,
            config=config,
            registry_sha256=registry["registry_sha256"],
            config_sha256=sha256_file(Path("configs/uncertainty.yaml")),
            implementation_sha256=implementation,
        )
        assert report["data_boundary"]["test_split_loaded"] is False
        reports.append(report)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_uncertainty_aggregate(
        aggregate,
        config=config,
        registry_sha256=registry["registry_sha256"],
        config_sha256=sha256_file(Path("configs/uncertainty.yaml")),
        implementation_sha256=implementation,
    )
    assert aggregate["report_hashes"] == {
        str(report["seed"]): report["report_sha256"] for report in reports
    }
