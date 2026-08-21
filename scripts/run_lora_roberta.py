#!/usr/bin/env python3
"""Train, select and locked-test the registered LoRA-RoBERTa candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from governed_banking.data import (
    load_manifest_split,
    sha256_file,
    stable_json_sha256,
    validate_manifest,
)
from governed_banking.frozen_baseline import ModelSnapshot, resolve_model_snapshot
from governed_banking.lora_baseline import (
    LoraExperimentConfig,
    evaluate_locked_lora,
    train_and_select_lora,
    validate_lora_evaluation_artifact,
    validate_lora_selection_artifact,
    write_lora_predictions,
    write_lora_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_roberta.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/manifests/banking77-seed-42.json"),
    )
    parser.add_argument("--model-cache", type=Path, default=Path("artifacts/huggingface"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("artifacts/lora-roberta/candidates")
    )
    parser.add_argument(
        "--selection-report", type=Path, default=Path("reports/lora-roberta/selection.json")
    )
    parser.add_argument(
        "--evaluation-report", type=Path, default=Path("reports/lora-roberta/test.json")
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/lora-roberta/test-predictions.jsonl"),
    )
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def _context(
    args: argparse.Namespace,
) -> tuple[LoraExperimentConfig, dict[str, Any], str, dict[str, str], ModelSnapshot]:
    config = LoraExperimentConfig.from_yaml(args.config)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    implementation_paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "lora_baseline.py": Path("src/governed_banking/lora_baseline.py"),
        "run_lora_roberta.py": Path(__file__),
    }
    implementation_sha256 = {name: sha256_file(path) for name, path in implementation_paths.items()}
    snapshot = resolve_model_snapshot(
        config.encoder, cache_directory=args.model_cache, offline=args.offline
    )
    return config, manifest, sha256_file(args.config), implementation_sha256, snapshot


def run_selection(
    args: argparse.Namespace,
    config: LoraExperimentConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
    snapshot: ModelSnapshot,
) -> dict[str, Any]:
    train_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "train"
    )
    validation_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "validation"
    )
    selection = train_and_select_lora(
        config,
        train_records,
        validation_records,
        label_names=manifest["label_names"],
        snapshot=snapshot,
        checkpoint_root=args.checkpoint_root,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    write_lora_report(selection, args.selection_report)
    return selection


def run_evaluation(
    args: argparse.Namespace,
    config: LoraExperimentConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
    snapshot: ModelSnapshot,
) -> dict[str, Any]:
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    validate_lora_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
        verify_checkpoint=True,
    )

    # The official test records are loaded only after selection and checkpoint hashes validate.
    test_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "test"
    )
    evaluation, predictions = evaluate_locked_lora(
        config,
        selection,
        test_records,
        label_names=manifest["label_names"],
        snapshot=snapshot,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    predictions_sha256 = write_lora_predictions(predictions, args.predictions)
    if predictions_sha256 != evaluation["test_predictions_sha256"]:
        raise AssertionError("written LoRA predictions differ from evaluation evidence")
    body = dict(evaluation)
    body.pop("evaluation_sha256")
    body["prediction_artifact"] = {
        "path": str(args.predictions),
        "sha256": predictions_sha256,
        "rows": len(predictions),
    }
    body["evaluation_sha256"] = stable_json_sha256(body)
    validate_lora_evaluation_artifact(
        body,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
    )
    write_lora_report(body, args.evaluation_report)
    return body


def main() -> None:
    args = parse_args()
    config, manifest, config_sha256, implementation_sha256, snapshot = _context(args)
    selection: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    if args.phase in {"select", "run"}:
        selection = run_selection(
            args, config, manifest, config_sha256, implementation_sha256, snapshot
        )
    if args.phase in {"evaluate", "run"}:
        evaluation = run_evaluation(
            args, config, manifest, config_sha256, implementation_sha256, snapshot
        )
    locked = selection or json.loads(args.selection_report.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "phase": args.phase,
        "model": f"{config.encoder.repository}@{config.encoder.revision}",
        "selected_candidate": locked["selected_candidate"],
        "selection_report": str(args.selection_report),
    }
    if evaluation is not None:
        metrics = evaluation["test_result"]["metrics"]
        summary["evaluation_report"] = str(args.evaluation_report)
        summary["test_metrics"] = {
            name: metrics[name]
            for name in ("accuracy", "macro_f1", "weighted_f1", "log_loss", "top_3_accuracy")
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
