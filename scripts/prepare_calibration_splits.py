#!/usr/bin/env python3
"""Partition Module 6 validation rows into temperature-fit and assessment roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from governed_banking.calibration import (
    CalibrationConfig,
    assert_calibration_split_permitted,
    build_calibration_registry,
    validate_calibration_registry,
    write_calibration_artifact,
)
from governed_banking.data import load_manifest_split, sha256_file, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    return parser.parse_args()


def implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "calibration.py": Path("src/governed_banking/calibration.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "prepare_calibration_splits.py": script_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    config = CalibrationConfig.from_yaml(args.config)
    config_sha256 = sha256_file(args.config)
    implementation = implementation_hashes(Path(__file__))
    source_records = {}
    labels_by_seed = {}
    manifest_hashes = {}
    run_hashes = {}
    for seed in config.seeds:
        manifest_path = config.manifest_path(seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["policy"]["seed"] != seed:
            raise ValueError(f"manifest seed mismatch for seed {seed}")
        run = json.loads(config.validation_report_path(seed).read_text(encoding="utf-8"))
        if (
            run.get("seed") != seed
            or run.get("data_boundary", {}).get("test_split_loaded") is not False
        ):
            raise ValueError(f"source validation report is invalid for seed {seed}")
        assert_calibration_split_permitted(config.source_split)
        source_records[seed] = load_manifest_split(
            args.dataset_config,
            args.raw_dir,
            manifest_path,
            config.source_split,
        )
        labels_by_seed[seed] = manifest["label_names"]
        manifest_hashes[seed] = manifest["manifest_sha256"]
        run_hashes[seed] = run["run_sha256"]
    registry = build_calibration_registry(
        config,
        source_records,
        label_names_by_seed=labels_by_seed,
        source_manifest_sha256=manifest_hashes,
        source_run_sha256=run_hashes,
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    validate_calibration_registry(
        registry,
        config=config,
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_calibration_artifact(registry, config.partition.registry_path)
    print(
        json.dumps(
            {
                "registry": str(config.partition.registry_path),
                "registry_sha256": registry["registry_sha256"],
                "partitions": [
                    {
                        "seed": entry["seed"],
                        "temperature_fit_rows": entry["temperature_fit"]["count"],
                        "calibration_assessment_rows": entry["calibration_assessment"]["count"],
                        "source_index_overlap": entry["source_index_overlap"],
                    }
                    for entry in registry["entries"]
                ],
                "boundary": registry["model_access_boundary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
