#!/usr/bin/env python3
"""Fit temperatures on development rows and assess calibration on disjoint rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from governed_banking.calibration import (
    CalibrationConfig,
    aggregate_calibration_reports,
    assert_calibration_split_permitted,
    build_seed_calibration_report,
    extract_logits,
    select_partition_records,
    validate_calibration_aggregate,
    validate_calibration_registry,
    validate_seed_calibration_report,
    write_calibration_artifact,
)
from governed_banking.data import load_manifest_split, sha256_file, validate_manifest
from governed_banking.frozen_baseline import resolve_model_snapshot
from governed_banking.multiseed import MultiSeedExperimentConfig, validate_seed_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument(
        "--report-pattern",
        default="reports/calibration/seed-{seed}-temperature-scaling.json",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("reports/calibration/temperature-scaling-aggregate.json"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def partition_implementation_hashes() -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_calibration_splits.py": Path("scripts/prepare_calibration_splits.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def module6_implementation_hashes() -> dict[str, str]:
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


def calibration_implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "run_temperature_scaling.py": script_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    if args.report_pattern.count("{seed}") != 1:
        raise ValueError("--report-pattern must contain exactly one {seed}")
    config = CalibrationConfig.from_yaml(args.config)
    multiseed_config = MultiSeedExperimentConfig.from_yaml(config.multiseed_config_path)
    config_sha256 = sha256_file(args.config)
    implementation = calibration_implementation_hashes(Path(__file__))
    registry = json.loads(config.partition.registry_path.read_text(encoding="utf-8"))
    validate_calibration_registry(
        registry,
        config=config,
        config_sha256=config_sha256,
        implementation_sha256=partition_implementation_hashes(),
    )
    entries = {entry["seed"]: entry for entry in registry["entries"]}
    snapshot = resolve_model_snapshot(
        multiseed_config.encoder,
        cache_directory=args.model_cache,
        offline=args.offline,
    )
    reports: list[dict[str, Any]] = []
    for seed in config.seeds:
        entry = entries[seed]
        report_path = Path(args.report_pattern.format(seed=seed))
        if args.resume and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_seed_calibration_report(
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
        source_run = json.loads(config.validation_report_path(seed).read_text(encoding="utf-8"))
        validate_seed_run(
            source_run,
            config=multiseed_config,
            manifest_sha256=manifest["manifest_sha256"],
            config_sha256=sha256_file(config.multiseed_config_path),
            implementation_sha256=module6_implementation_hashes(),
            model_files_sha256=snapshot.file_sha256,
            verify_checkpoint=True,
        )
        if source_run["run_sha256"] != entry["source_run_sha256"]:
            raise ValueError(f"calibration registry uses a different Module 6 run for seed {seed}")
        assert_calibration_split_permitted(config.source_split)
        validation_records = load_manifest_split(
            args.dataset_config,
            args.raw_dir,
            manifest_path,
            config.source_split,
        )
        development_records = select_partition_records(validation_records, entry["temperature_fit"])
        assessment_records = select_partition_records(
            validation_records, entry["calibration_assessment"]
        )
        logits, labels, extraction = extract_logits(
            validation_records,
            snapshot=snapshot,
            checkpoint_directory=config.checkpoint_path(seed),
            label_names=manifest["label_names"],
            max_length=multiseed_config.encoder.max_length,
            batch_size=multiseed_config.evaluation_batch_size,
            attention_implementation=multiseed_config.attention_implementation,
            device_name=multiseed_config.encoder.device,
        )
        if extraction["checkpoint_files_sha256"] != source_run["checkpoint"]["files_sha256"]:
            raise ValueError(f"checkpoint hashes differ from Module 6 run for seed {seed}")
        positions = {record.source_index: index for index, record in enumerate(validation_records)}
        development_positions = [positions[record.source_index] for record in development_records]
        assessment_positions = [positions[record.source_index] for record in assessment_records]
        report = build_seed_calibration_report(
            config,
            seed=seed,
            development_logits=logits[development_positions],
            development_labels=labels[development_positions],
            assessment_logits=logits[assessment_positions],
            assessment_labels=labels[assessment_positions],
            partition_entry=entry,
            extraction_metadata=extraction,
            source_run_sha256=source_run["run_sha256"],
            source_manifest_sha256=manifest["manifest_sha256"],
            registry_sha256=registry["registry_sha256"],
            config_sha256=config_sha256,
            implementation_sha256=implementation,
        )
        validate_seed_calibration_report(
            report,
            config=config,
            registry_sha256=registry["registry_sha256"],
            config_sha256=config_sha256,
            implementation_sha256=implementation,
        )
        write_calibration_artifact(report, report_path)
        reports.append(report)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "temperature": report["temperature_fit"]["temperature"],
                    "assessment_raw_ece": report["assessment_metrics"]["raw"][
                        "expected_calibration_error"
                    ],
                    "assessment_calibrated_ece": report["assessment_metrics"]["calibrated"][
                        "expected_calibration_error"
                    ],
                    "gate_passed": report["acceptance_gate"]["all_passed"],
                    "test_split_loaded": report["data_boundary"]["test_split_loaded"],
                }
            ),
            flush=True,
        )
    aggregate = aggregate_calibration_reports(
        config,
        reports,
        registry_sha256=registry["registry_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    validate_calibration_aggregate(
        aggregate,
        config=config,
        registry_sha256=registry["registry_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_calibration_artifact(aggregate, args.aggregate_report)
    print(
        json.dumps(
            {
                "aggregate_report": str(args.aggregate_report),
                "mean_raw_ece": aggregate["assessment_metrics"]["expected_calibration_error"][
                    "raw"
                ]["mean"],
                "mean_calibrated_ece": aggregate["assessment_metrics"][
                    "expected_calibration_error"
                ]["calibrated"]["mean"],
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
