#!/usr/bin/env python3
"""Compare locked LoRA predictions with both prior Module 3 and 4 baselines."""

from __future__ import annotations

import json
from pathlib import Path

from governed_banking.comparison import (
    compare_predictions,
    load_prediction_rows,
    write_comparison,
)
from governed_banking.data import sha256_file, stable_json_sha256


def main() -> None:
    lora_report = json.loads(Path("reports/lora-roberta/test.json").read_text(encoding="utf-8"))
    lora_rows = load_prediction_rows(Path("reports/lora-roberta/test-predictions.jsonl"))
    implementation = {
        "comparison.py": sha256_file(Path("src/governed_banking/comparison.py")),
        "compare_module5_models.py": sha256_file(Path(__file__)),
    }
    references = (
        ("tfidf_logistic_regression", "baseline"),
        ("frozen_roberta_mean_logistic_regression", "frozen-roberta"),
    )
    comparisons = []
    for reference_name, directory in references:
        report_name = "tfidf-logreg-test.json" if directory == "baseline" else "test.json"
        reference_report = json.loads(
            Path(f"reports/{directory}/{report_name}").read_text(encoding="utf-8")
        )
        prediction_name = (
            "tfidf-logreg-test-predictions.jsonl"
            if directory == "baseline"
            else "test-predictions.jsonl"
        )
        reference_rows = load_prediction_rows(Path(f"reports/{directory}/{prediction_name}"))
        comparisons.append(
            compare_predictions(
                reference_rows,
                lora_rows,
                reference_name=reference_name,
                candidate_name="lora_adapted_roberta",
                reference_evaluation_sha256=reference_report["evaluation_sha256"],
                candidate_evaluation_sha256=lora_report["evaluation_sha256"],
                implementation_sha256=implementation,
                reference_macro_f1=reference_report["test_result"]["metrics"]["macro_f1"],
                candidate_macro_f1=lora_report["test_result"]["metrics"]["macro_f1"],
            )
        )
    artifact = {
        "schema_version": 1,
        "artifact_type": "module5_model_comparisons",
        "contains_message_text": False,
        "implementation_sha256": implementation,
        "comparisons": comparisons,
    }
    artifact["evidence_sha256"] = stable_json_sha256(artifact)
    write_comparison(artifact, Path("reports/lora-roberta/paired-comparisons.json"))
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
