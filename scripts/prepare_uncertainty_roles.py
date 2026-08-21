#!/usr/bin/env python3
"""Create immutable Module 8 development and assessment role registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from governed_banking.calibration import (
    CalibrationConfig,
    select_partition_records,
    validate_calibration_registry,
    validate_seed_calibration_report,
)
from governed_banking.data import load_manifest_split, sha256_file, validate_manifest
from governed_banking.uncertainty import (
    UncertaintyConfig,
    build_uncertainty_registry,
    load_possible_ood_fixture,
    validate_uncertainty_registry,
    write_uncertainty_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/uncertainty.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    return parser.parse_args()


def uncertainty_partition_implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_uncertainty_roles.py": script_path,
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
    config = UncertaintyConfig.from_yaml(args.config)
    calibration_config = CalibrationConfig.from_yaml(config.calibration_config_path)
    calibration_registry = json.loads(config.calibration_registry_path.read_text(encoding="utf-8"))
    validate_calibration_registry(
        calibration_registry,
        config=calibration_config,
        config_sha256=sha256_file(config.calibration_config_path),
        implementation_sha256=calibration_partition_implementation_hashes(),
    )
    calibration_entries = {entry["seed"]: entry for entry in calibration_registry["entries"]}
    known_sources = {}
    report_hashes = {}
    manifest_hashes = {}
    for seed in config.seeds:
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
        validation_records = load_manifest_split(
            args.dataset_config, args.raw_dir, manifest_path, "validation"
        )
        known_sources[seed] = select_partition_records(
            validation_records,
            calibration_entries[seed][calibration_config.partition.assessment_role],
        )
        report_hashes[seed] = calibration_report["report_sha256"]
        manifest_hashes[seed] = manifest["manifest_sha256"]
    possible_ood_records = load_possible_ood_fixture(config.possible_ood.fixture_path)
    implementation = uncertainty_partition_implementation_hashes(Path(__file__))
    config_sha256 = sha256_file(args.config)
    registry = build_uncertainty_registry(
        config,
        known_sources,
        possible_ood_records,
        source_calibration_report_sha256=report_hashes,
        source_manifest_sha256=manifest_hashes,
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    validate_uncertainty_registry(
        registry,
        config=config,
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_uncertainty_artifact(registry, config.possible_ood.registry_path)
    print(
        json.dumps(
            {
                "registry": str(config.possible_ood.registry_path),
                "registry_sha256": registry["registry_sha256"],
                "known_partitions": [
                    {
                        "seed": entry["seed"],
                        "development": entry[config.known_partition.development_role]["count"],
                        "assessment": entry[config.known_partition.assessment_role]["count"],
                        "overlap": entry["source_index_overlap"],
                    }
                    for entry in registry["known_entries"]
                ],
                "possible_ood_partitions": {
                    "development": registry["possible_ood"][config.possible_ood.development_role][
                        "count"
                    ],
                    "assessment": registry["possible_ood"][config.possible_ood.assessment_role][
                        "count"
                    ],
                    "scenario_group_overlap": registry["possible_ood"]["scenario_group_overlap"],
                },
                "boundary": registry["model_access_boundary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
