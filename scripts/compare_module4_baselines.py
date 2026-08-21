#!/usr/bin/env python3
"""Create the paired Module 4 comparison against the TF-IDF baseline."""

from __future__ import annotations

import json
from pathlib import Path

from governed_banking.comparison import (
    compare_predictions,
    load_prediction_rows,
    write_comparison,
)
from governed_banking.data import sha256_file


def main() -> None:
    tfidf_report_path = Path("reports/baseline/tfidf-logreg-test.json")
    frozen_report_path = Path("reports/frozen-roberta/test.json")
    tfidf_predictions_path = Path("reports/baseline/tfidf-logreg-test-predictions.jsonl")
    frozen_predictions_path = Path("reports/frozen-roberta/test-predictions.jsonl")
    destination = Path("reports/frozen-roberta/paired-vs-tfidf.json")

    tfidf_report = json.loads(tfidf_report_path.read_text(encoding="utf-8"))
    frozen_report = json.loads(frozen_report_path.read_text(encoding="utf-8"))
    artifact = compare_predictions(
        load_prediction_rows(tfidf_predictions_path),
        load_prediction_rows(frozen_predictions_path),
        reference_name="tfidf_word_char_c4",
        candidate_name="frozen_roberta_mean_c1024",
        reference_evaluation_sha256=tfidf_report["evaluation_sha256"],
        candidate_evaluation_sha256=frozen_report["evaluation_sha256"],
        implementation_sha256={
            "comparison.py": sha256_file(Path("src/governed_banking/comparison.py")),
            "compare_module4_baselines.py": sha256_file(Path(__file__)),
        },
        reference_macro_f1=tfidf_report["test_result"]["metrics"]["macro_f1"],
        candidate_macro_f1=frozen_report["test_result"]["metrics"]["macro_f1"],
    )
    write_comparison(artifact, destination)
    print(json.dumps(artifact["correctness"] | artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
