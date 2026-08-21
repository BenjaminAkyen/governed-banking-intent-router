#!/usr/bin/env python3
"""Run post-test exploratory LoRA training with validation-only stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from governed_banking.baseline import write_json_artifact
from governed_banking.data import (
    load_manifest_split,
    sha256_file,
    validate_manifest,
)
from governed_banking.frozen_baseline import resolve_model_snapshot
from governed_banking.multiseed import (
    MultiSeedExperimentConfig,
    aggregate_validation_runs,
    assert_model_split_permitted,
    train_validation_seed,
    validate_manifest_registry,
    validate_seed_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/multiseed_lora.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("artifacts/multiseed-lora"))
    parser.add_argument(
        "--report-pattern",
        default="reports/multiseed-lora/seed-{seed}-validation.json",
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("reports/multiseed-lora/validation-aggregate.json"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a seed only when its report and local checkpoint fully validate",
    )
    return parser.parse_args()


def manifest_implementation_hashes() -> dict[str, str]:
    paths = {
        "data.py": Path("src/governed_banking/data.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "prepare_multiseed_manifests.py": Path("scripts/prepare_multiseed_manifests.py"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def training_implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "multiseed.py": Path("src/governed_banking/multiseed.py"),
        "run_multiseed_lora.py": script_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_registered_context(
    args: argparse.Namespace,
) -> tuple[
    MultiSeedExperimentConfig,
    dict[str, Any],
    str,
    dict[str, str],
    Any,
]:
    config = MultiSeedExperimentConfig.from_yaml(args.config)
    config_sha256 = sha256_file(args.config)
    registry = json.loads(config.registry_path.read_text(encoding="utf-8"))
    validate_manifest_registry(
        registry,
        config=config,
        dataset_config_sha256=sha256_file(args.dataset_config),
        config_sha256=config_sha256,
        implementation_sha256=manifest_implementation_hashes(),
    )
    implementation = training_implementation_hashes(Path(__file__))
    snapshot = resolve_model_snapshot(
        config.encoder, cache_directory=args.model_cache, offline=args.offline
    )
    return config, registry, config_sha256, implementation, snapshot


def main() -> None:
    args = parse_args()
    if args.report_pattern.count("{seed}") != 1:
        raise ValueError("--report-pattern must contain exactly one {seed}")
    config, registry, config_sha256, implementation, snapshot = load_registered_context(args)
    registry_entries = {entry["seed"]: entry for entry in registry["manifest_entries"]}
    runs: list[dict[str, Any]] = []
    for seed in config.seeds:
        entry = registry_entries[seed]
        manifest_path = config.manifest_path(seed)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if manifest["policy"]["seed"] != seed:
            raise ValueError(f"manifest seed mismatch for seed {seed}")
        if manifest["manifest_sha256"] != entry["manifest_sha256"]:
            raise ValueError(f"registry hash mismatch for seed {seed}")
        report_path = Path(args.report_pattern.format(seed=seed))
        checkpoint_path = args.checkpoint_root / f"seed-{seed}"
        if args.resume and report_path.is_file():
            run = json.loads(report_path.read_text(encoding="utf-8"))
            validate_seed_run(
                run,
                config=config,
                manifest_sha256=manifest["manifest_sha256"],
                config_sha256=config_sha256,
                implementation_sha256=implementation,
                model_files_sha256=snapshot.file_sha256,
                verify_checkpoint=True,
            )
        else:
            # Only these two explicit loads exist; there is no test-evaluation mode or argument.
            assert_model_split_permitted("train")
            train_records = load_manifest_split(
                args.dataset_config, args.raw_dir, manifest_path, "train"
            )
            assert_model_split_permitted("validation")
            validation_records = load_manifest_split(
                args.dataset_config, args.raw_dir, manifest_path, "validation"
            )
            run = train_validation_seed(
                config,
                seed=seed,
                train_records=train_records,
                validation_records=validation_records,
                label_names=manifest["label_names"],
                manifest_sha256=manifest["manifest_sha256"],
                snapshot=snapshot,
                checkpoint_directory=checkpoint_path,
                config_sha256=config_sha256,
                implementation_sha256=implementation,
            )
            write_json_artifact(run, report_path)
        runs.append(run)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "best_epoch": run["best_epoch"],
                    "epochs_completed": run["epochs_completed"],
                    "validation_macro_f1": run["best_validation_metrics"]["macro_f1"],
                    "stopping_reason": run["stopping_reason"],
                    "test_split_loaded": run["data_boundary"]["test_split_loaded"],
                }
            ),
            flush=True,
        )
    aggregate = aggregate_validation_runs(
        config,
        runs,
        registry_sha256=registry["registry_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation,
    )
    write_json_artifact(aggregate, args.aggregate_report)
    print(
        json.dumps(
            {
                "aggregate_report": str(args.aggregate_report),
                "seeds": list(config.seeds),
                "validation_macro_f1": aggregate["validation_metrics"]["macro_f1"],
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
