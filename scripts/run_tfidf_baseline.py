#!/usr/bin/env python3
"""Select and evaluate the leakage-safe TF-IDF logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from governed_banking.baseline import (
    BaselineConfig,
    evaluate_locked_baseline,
    select_baseline,
    validate_evaluation_artifact,
    validate_selection_artifact,
    write_json_artifact,
    write_model,
    write_prediction_jsonl,
)
from governed_banking.data import (
    load_manifest_split,
    sha256_file,
    stable_json_sha256,
    validate_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "evaluate", "run"), nargs="?", default="run")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_tfidf.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/banking77"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("data/manifests/banking77-seed-42.json"),
    )
    parser.add_argument(
        "--selection-report",
        type=Path,
        default=Path("reports/baseline/tfidf-logreg-selection.json"),
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("reports/baseline/tfidf-logreg-test.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/baseline/tfidf-logreg-test-predictions.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/baseline/tfidf-logreg.joblib"),
    )
    return parser.parse_args()


def _context(
    args: argparse.Namespace,
) -> tuple[BaselineConfig, dict[str, Any], str, dict[str, str]]:
    config = BaselineConfig.from_yaml(args.config)
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    implementation_paths = {
        "baseline.py": Path("src/governed_banking/baseline.py"),
        "data.py": Path("src/governed_banking/data.py"),
        "run_tfidf_baseline.py": Path(__file__),
    }
    implementation_sha256 = {
        name: sha256_file(path) for name, path in implementation_paths.items()
    }
    return config, manifest, sha256_file(args.config), implementation_sha256


def run_selection(
    args: argparse.Namespace,
    config: BaselineConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
) -> dict[str, Any]:
    train_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "train"
    )
    validation_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "validation"
    )
    selection = select_baseline(
        config,
        train_records,
        validation_records,
        label_names=manifest["label_names"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    write_json_artifact(selection, args.selection_report)
    return selection


def run_evaluation(
    args: argparse.Namespace,
    config: BaselineConfig,
    manifest: dict[str, Any],
    config_sha256: str,
    implementation_sha256: dict[str, str],
) -> dict[str, Any]:
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    validate_selection_artifact(
        selection,
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )

    # Test data is deliberately loaded only after the selection artifact passes every lock check.
    train_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "train"
    )
    test_records = load_manifest_split(
        args.dataset_config, args.raw_dir, args.dataset_manifest, "test"
    )
    model, evaluation, predictions = evaluate_locked_baseline(
        config,
        selection,
        train_records,
        test_records,
        label_names=manifest["label_names"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    prediction_sha256 = write_prediction_jsonl(predictions, args.predictions)
    if prediction_sha256 != evaluation["test_predictions_sha256"]:
        raise AssertionError("written prediction artifact does not match evaluation evidence")
    model_sha256 = write_model(model, args.model)

    evaluation_body = dict(evaluation)
    evaluation_body.pop("evaluation_sha256")
    evaluation_body["prediction_artifact"] = {
        "path": str(args.predictions),
        "sha256": prediction_sha256,
        "rows": len(predictions),
    }
    evaluation_body["local_model_artifact"] = {
        "path": str(args.model),
        "sha256": model_sha256,
        "load_only_if_trusted": True,
    }
    evaluation_body["evaluation_sha256"] = stable_json_sha256(evaluation_body)
    validate_evaluation_artifact(
        evaluation_body,
        selection_sha256=selection["selection_sha256"],
        dataset_manifest_sha256=manifest["manifest_sha256"],
        config_sha256=config_sha256,
        implementation_sha256=implementation_sha256,
    )
    write_json_artifact(evaluation_body, args.evaluation_report)
    return evaluation_body


def main() -> None:
    args = parse_args()
    config, manifest, config_sha256, implementation_sha256 = _context(args)
    selection: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    if args.phase in {"select", "run"}:
        selection = run_selection(
            args, config, manifest, config_sha256, implementation_sha256
        )
    if args.phase in {"evaluate", "run"}:
        evaluation = run_evaluation(
            args, config, manifest, config_sha256, implementation_sha256
        )

    summary: dict[str, Any] = {
        "phase": args.phase,
        "selection_report": str(args.selection_report),
        "selected_candidate": (
            selection or json.loads(args.selection_report.read_text(encoding="utf-8"))
        )["selected_candidate"],
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
