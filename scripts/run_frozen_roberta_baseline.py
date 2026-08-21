#!/usr/bin/env python3
"""Run frozen RoBERTa embedding selection and locked-test evaluation."""

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
from governed_banking.frozen_baseline import (
    FrozenBaselineConfig,
    ModelSnapshot,
    evaluate_locked_frozen_baseline,
    extract_or_load_embeddings,
    resolve_model_snapshot,
    select_frozen_baseline,
    validate_frozen_evaluation_artifact,
    validate_frozen_selection_artifact,
    write_classifier,
    write_frozen_predictions,
    write_frozen_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_roberta.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/manifests/banking77-seed-42.json"),
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/huggingface"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("artifacts/embeddings/frozen-roberta"),
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=Path("reports/frozen-roberta/selection.json"),
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("reports/frozen-roberta/test.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/frozen-roberta/test-predictions.jsonl"),
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("artifacts/frozen-roberta/classifier.joblib"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def _context(
    args: argparse.Namespace,
) -> tuple[
    FrozenBaselineConfig,
    dict[str, Any],
    str,
    dict[str, str],
    ModelSnapshot,
]:
    config = FrozenBaselineConfig.from_yaml(args.config)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    implementation_paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "device.py": Path("src/governed_banking/device.py"),
        "frozen_baseline.py": Path("src/governed_banking/frozen_baseline.py"),
        "run_frozen_roberta_baseline.py": Path(__file__),
    }
    implementation_sha256 = {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    snapshot = resolve_model_snapshot(
        config.encoder,
        cache_directory=args.model_cache,
        offline=args.offline,
    )
    return config, manifest, sha256_file(args.config), implementation_sha256, snapshot


def _embedding_bundle(
    args: argparse.Namespace,
    config: FrozenBaselineConfig,
    manifest: dict[str, Any],
    snapshot: ModelSnapshot,
    implementation_sha256: dict[str, str],
    split_name: str,
):
    records = load_manifest_split(
        args.dataset_config,
        args.raw_dir,
        args.dataset_manifest,
        split_name,
    )
    bundle = extract_or_load_embeddings(
        records,
        split_name=split_name,
        source_indices_sha256=manifest["splits"][split_name]["source_indices_sha256"],
        config=config,
        snapshot=snapshot,
        cache_directory=args.embedding_cache,
        implementation_sha256=implementation_sha256,
        force=args.force_embeddings,
    )
    return records, bundle


def run_selection(
    args: argparse.Namespace,
    config: FrozenBaselineConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
    snapshot: ModelSnapshot,
) -> dict[str, Any]:
    train_records, train_embeddings = _embedding_bundle(
        args, config, manifest, snapshot, implementation_sha256, "train"
    )
    validation_records, validation_embeddings = _embedding_bundle(
        args, config, manifest, snapshot, implementation_sha256, "validation"
    )
    selection = select_frozen_baseline(
        config,
        train_records,
        validation_records,
        train_embeddings,
        validation_embeddings,
        label_names=manifest["label_names"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
    )
    write_frozen_report(selection, args.selection_report)
    return selection


def run_evaluation(
    args: argparse.Namespace,
    config: FrozenBaselineConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
    snapshot: ModelSnapshot,
) -> dict[str, Any]:
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    validate_frozen_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
    )

    # Test records and embeddings remain inaccessible until the selection lock validates.
    train_records, train_embeddings = _embedding_bundle(
        args, config, manifest, snapshot, implementation_sha256, "train"
    )
    test_records, test_embeddings = _embedding_bundle(
        args, config, manifest, snapshot, implementation_sha256, "test"
    )
    classifier, evaluation, predictions = evaluate_locked_frozen_baseline(
        config,
        selection,
        train_records,
        test_records,
        train_embeddings,
        test_embeddings,
        label_names=manifest["label_names"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
    )
    prediction_sha256 = write_frozen_predictions(predictions, args.predictions)
    if prediction_sha256 != evaluation["test_predictions_sha256"]:
        raise AssertionError("written predictions do not match frozen evaluation evidence")
    classifier_sha256 = write_classifier(classifier, args.classifier)

    body = dict(evaluation)
    body.pop("evaluation_sha256")
    body["prediction_artifact"] = {
        "path": str(args.predictions),
        "sha256": prediction_sha256,
        "rows": len(predictions),
    }
    body["local_classifier_artifact"] = {
        "path": str(args.classifier),
        "sha256": classifier_sha256,
        "load_only_if_trusted": True,
    }
    body["evaluation_sha256"] = stable_json_sha256(body)
    validate_frozen_evaluation_artifact(
        body,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
        model_files_sha256=snapshot.file_sha256,
    )
    write_frozen_report(body, args.evaluation_report)
    return body


def main() -> None:
    args = parse_args()
    config, manifest, config_sha256, implementation_sha256, snapshot = _context(args)
    selection: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    if args.phase in {"select", "run"}:
        selection = run_selection(
            args,
            config,
            manifest,
            config_sha256,
            implementation_sha256,
            snapshot,
        )
    if args.phase in {"evaluate", "run"}:
        evaluation = run_evaluation(
            args,
            config,
            manifest,
            config_sha256,
            implementation_sha256,
            snapshot,
        )

    selected_candidate = (
        selection or json.loads(args.selection_report.read_text(encoding="utf-8"))
    )["selected_candidate"]
    summary: dict[str, Any] = {
        "phase": args.phase,
        "encoder": f"{config.encoder.repository}@{config.encoder.revision}",
        "selected_candidate": selected_candidate,
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
