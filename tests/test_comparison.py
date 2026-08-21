from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.comparison import compare_predictions
from governed_banking.data import sha256_file, stable_json_sha256


def _row(index: int, true_label: str, predicted_label: str) -> dict[str, object]:
    return {
        "source_index": index,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence_uncalibrated": 0.5,
    }


def test_paired_comparison_counts_complementary_errors() -> None:
    reference = [
        _row(0, "a", "a"),
        _row(1, "a", "a"),
        _row(2, "b", "a"),
        _row(3, "b", "a"),
    ]
    candidate = [
        _row(0, "a", "a"),
        _row(1, "a", "b"),
        _row(2, "b", "b"),
        _row(3, "b", "a"),
    ]

    artifact = compare_predictions(
        reference,
        candidate,
        reference_name="reference",
        candidate_name="candidate",
        reference_evaluation_sha256="a" * 64,
        candidate_evaluation_sha256="b" * 64,
        implementation_sha256={"comparison.py": "c" * 64},
        reference_macro_f1=0.6,
        candidate_macro_f1=0.5,
    )

    assert artifact["correctness"] == {
        "both_correct": 1,
        "reference_only_correct": 1,
        "candidate_only_correct": 1,
        "both_wrong": 1,
        "discordant_rows": 2,
        "prediction_disagreements": 2,
    }
    assert artifact["metrics"]["exact_mcnemar_two_sided_p_value"] == 1.0


def test_paired_comparison_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="aligned"):
        compare_predictions(
            [_row(0, "a", "a")],
            [_row(1, "a", "a")],
            reference_name="reference",
            candidate_name="candidate",
            reference_evaluation_sha256="a" * 64,
            candidate_evaluation_sha256="b" * 64,
            implementation_sha256={"comparison.py": "c" * 64},
            reference_macro_f1=1.0,
            candidate_macro_f1=1.0,
        )


def test_committed_paired_comparison_is_hash_valid() -> None:
    artifact = json.loads(
        Path("reports/frozen-roberta/paired-vs-tfidf.json").read_text(encoding="utf-8")
    )
    expected_hash = artifact["comparison_sha256"]
    body = dict(artifact)
    body.pop("comparison_sha256")

    assert stable_json_sha256(body) == expected_hash
    assert artifact["implementation_sha256"] == {
        "compare_module4_baselines.py": sha256_file(
            Path("scripts/compare_module4_baselines.py")
        ),
        "comparison.py": sha256_file(Path("src/governed_banking/comparison.py")),
    }
    assert artifact["correctness"]["candidate_only_correct"] == 135
    assert artifact["metrics"]["exact_mcnemar_two_sided_p_value"] == 0.1176530802
