#!/usr/bin/env python3
"""Select uncertainty controls on development roles and assess locked controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from governed_banking.calibration import (
    CalibrationConfig,
    select_partition_records,
    validate_calibration_registry,
    validate_seed_calibration_report,
)
from governed_banking.data import load_manifest_split, sha256_file, validate_manifest
from governed_banking.frozen_baseline import resolve_model_snapshot
from governed_banking.multiseed import MultiSeedExperimentConfig
from governed_banking.uncertainty import (
    UncertaintyConfig,
    aggregate_uncertainty_reports,
    assert_uncertainty_source_permitted,
    build_seed_uncertainty_report,
    extract_uncertainty_logits,
    load_possible_ood_fixture,
    select_known_partition,
    select_ood_partition,
    validate_seed_uncertainty_report,
    validate_uncertainty_aggregate,
    validate_uncertainty_registry,
    write_uncertainty_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/uncertainty.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument(
        "--report-pattern",
        default="reports/uncertainty/seed-{seed}-selective-ood.json",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("reports/uncertainty/selective-ood-aggregate.json"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def uncertainty_partition_implementation_hashes() -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_uncertainty_roles.py": Path("scripts/prepare_uncertainty_roles.py"),
        "uncertainty.py": Path("src/governed_banking/uncertainty.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def uncertainty_run_implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "run_uncertainty_evaluation.py": script_path,
        "uncertainty.py": Path("src/governed_banking/uncertainty.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def calibration_partition_implementation_hashes() -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_calibration_splits.py": Path("scripts/prepare_calibration_splits.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def calibration_run_implementation_hashes() -> dict[str, str]:
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


def main() -> None:
    args = parse_args()
    if args.report_pattern.count("{seed}") != 1:
        raise ValueError("--report-pattern must contain exactly one {seed}")
    config = UncertaintyConfig.from_yaml(args.config)
    calibration_config = CalibrationConfig.from_yaml(config.calibration_config_path)
    multiseed_config = MultiSeedExperimentConfig.from_yaml(calibration_config.multiseed_config_path)
    config_sha256 = sha256_file(args.config)
    implementation = uncertainty_run_implementation_hashes(Path(__file__))
    registry = json.loads(config.possible_ood.registry_path.read_text(encoding="utf-8"))
    validate_uncertainty_registry(
        registry,
        config=config,
        config_sha256=config_sha256,
        implementation_sha256=uncertainty_partition_implementation_hashes(),
    )
    known_entries = {entry["seed"]: entry for entry in registry["known_entries"]}
    assert_uncertainty_source_permitted("synthetic_fixture")
    possible_ood_records = load_possible_ood_fixture(config.possible_ood.fixture_path)
    ood_development = select_ood_partition(
        possible_ood_records,
        registry["possible_ood"][config.possible_ood.development_role],
    )
    ood_assessment = select_ood_partition(
        possible_ood_records,
        registry["possible_ood"][config.possible_ood.assessment_role],
    )
    calibration_registry = json.loads(config.calibration_registry_path.read_text(encoding="utf-8"))
    validate_calibration_registry(
        calibration_registry,
        config=calibration_config,
        config_sha256=sha256_file(config.calibration_config_path),
        implementation_sha256=calibration_partition_implementation_hashes(),
    )
    calibration_entries = {entry["seed"]: entry for entry in calibration_registry["entries"]}
    snapshot = resolve_model_snapshot(
        multiseed_config.encoder,
        cache_directory=args.model_cache,
        offline=args.offline,
    )
    reports: list[dict[str, Any]] = []
    for seed in config.seeds:
        report_path = Path(args.report_pattern.format(seed=seed))
        if args.resume and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_seed_uncertainty_report(
                report,
                config=config,
                registry_sha256=registry["registry_sha256"],
                config_sha256=config_sha256,
                implementation_sha256=implementation,
            )
            reports.append(report)
            continue
        manifest_path = config.manifest_path(seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        calibration_report = json.loads(
            config.calibration_report_path(seed).read_text(encoding="utf-8")
        )
        validate_seed_calibration_report(
            calibration_report,
            config=calibration_config,
            registry_sha256=calibration_registry["registry_sha256"],
            config_sha256=sha256_file(config.calibration_config_path),
            implementation_sha256=calibration_run_implementation_hashes(),
        )
        entry = known_entries[seed]
        if calibration_report["report_sha256"] != entry["source_calibration_report_sha256"]:
            raise ValueError(f"uncertainty registry uses a different calibration report: {seed}")
        assert_uncertainty_source_permitted("validation")
        validation_records = load_manifest_split(
            args.dataset_config, args.raw_dir, manifest_path, "validation"
        )
        calibration_assessment = select_partition_records(
            validation_records,
            calibration_entries[seed][calibration_config.partition.assessment_role],
        )
        known_development = select_known_partition(
            calibration_assessment, entry[config.known_partition.development_role]
        )
        known_assessment = select_known_partition(
            calibration_assessment, entry[config.known_partition.assessment_role]
        )
        all_known = (*known_development, *known_assessment)
        all_ood = (*ood_development, *ood_assessment)
        known_logits, known_labels, ood_logits, extraction = extract_uncertainty_logits(
            all_known,
            all_ood,
            snapshot=snapshot,
            checkpoint_directory=config.checkpoint_path(seed),
            label_names=manifest["label_names"],
            max_length=multiseed_config.encoder.max_length,
            batch_size=multiseed_config.evaluation_batch_size,
            attention_implementation=multiseed_config.attention_implementation,
            device_name=multiseed_config.encoder.device,
        )
        if (
            extraction["checkpoint_files_sha256"]
            != calibration_report["extraction"]["checkpoint_files_sha256"]
        ):
            raise ValueError(f"checkpoint differs from calibrated Module 7 model: seed {seed}")
        known_development_count = len(known_development)
        ood_development_count = len(ood_development)
        report = build_seed_uncertainty_report(
            config,
            seed=seed,
            temperature=calibration_report["temperature_fit"]["temperature"],
            known_development_logits=known_logits[:known_development_count],
            known_development_labels=known_labels[:known_development_count],
            known_assessment_logits=known_logits[known_development_count:],
            known_assessment_labels=known_labels[known_development_count:],
            ood_development_logits=ood_logits[:ood_development_count],
            ood_assessment_logits=ood_logits[ood_development_count:],
            ood_assessment_records=ood_assessment,
            partition_entry=entry,
            ood_partition=registry["possible_ood"],
            extraction_metadata=extraction,
            source_calibration_report_sha256=calibration_report["report_sha256"],
            source_manifest_sha256=manifest["manifest_sha256"],
            registry_sha256=registry["registry_sha256"],
            config_sha256=config_sha256,
            implementation_sha256=implementation,
        )
        validate_seed_uncertainty_report(
            report,
            config=config,
            registry_sha256=registry["registry_sha256"],
            config_sha256=config_sha256,
            implementation_sha256=implementation,
        )
        write_uncertainty_artifact(report, report_path)
        reports.append(report)
        metrics = report["assessment"]["metrics"]
        print(
            json.dumps(
                {
                    "seed": seed,
                    "signal": report["assessment"]["selected_signal"],
                    "threshold": report["assessment"]["locked_threshold"],
                    "known_coverage": metrics["known_coverage"],
                    "selective_risk": metrics["selective_risk"],
                    "possible_ood_recall": metrics["possible_ood_recall"],
                    "gate_passed": report["acceptance_gate"]["all_passed"],
                    "test_split_loaded": report["data_boundary"]["test_split_loaded"],
                }
            ),
            flush=True,
        )
    aggregate = aggregate_uncertainty_reports(
        config,
        reports,
        registry_sha256=registry["registry_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    validate_uncertainty_aggregate(
        aggregate,
        config=config,
        registry_sha256=registry["registry_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_uncertainty_artifact(aggregate, args.aggregate_report)
    print(
        json.dumps(
            {
                "aggregate_report": str(args.aggregate_report),
                "mean_known_coverage": aggregate["assessment_metrics"]["known_coverage"]["mean"],
                "mean_selective_risk": aggregate["assessment_metrics"]["selective_risk"]["mean"],
                "mean_possible_ood_recall": aggregate["assessment_metrics"]["possible_ood_recall"][
                    "mean"
                ],
                "all_seeds_passed": aggregate["acceptance_gate"]["all_seeds_passed"],
                "official_test_metrics_computed": aggregate["data_boundary"][
                    "official_test_metrics_computed"
                ],
                "claim_scope": aggregate["claim_scope"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
