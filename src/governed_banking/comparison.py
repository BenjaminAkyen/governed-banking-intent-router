"""Paired, text-free comparison of index-aligned classifier predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from governed_banking.baseline import assert_text_free_artifact, write_json_artifact
from governed_banking.data import stable_json_sha256


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compare_predictions(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
    reference_evaluation_sha256: str,
    candidate_evaluation_sha256: str,
    implementation_sha256: Mapping[str, str],
    reference_macro_f1: float,
    candidate_macro_f1: float,
) -> dict[str, Any]:
    """Compare models on identical rows with an exact paired correctness test."""

    if not reference_rows or len(reference_rows) != len(candidate_rows):
        raise ValueError("prediction artifacts must have the same non-zero row count")
    required_keys = {"source_index", "true_label", "predicted_label"}
    if any(not required_keys.issubset(row) for row in (*reference_rows, *candidate_rows)):
        raise ValueError("prediction rows are missing required fields")
    reference_identity = [
        (row["source_index"], row["true_label"]) for row in reference_rows
    ]
    candidate_identity = [
        (row["source_index"], row["true_label"]) for row in candidate_rows
    ]
    if reference_identity != candidate_identity:
        raise ValueError("prediction artifacts are not index-and-label aligned")

    both_correct = 0
    reference_only = 0
    candidate_only = 0
    both_wrong = 0
    disagreements = 0
    by_intent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"support": 0, "reference_correct": 0, "candidate_correct": 0}
    )
    for reference, candidate in zip(reference_rows, candidate_rows, strict=True):
        true_label = str(reference["true_label"])
        reference_correct = reference["predicted_label"] == true_label
        candidate_correct = candidate["predicted_label"] == true_label
        by_intent[true_label]["support"] += 1
        by_intent[true_label]["reference_correct"] += int(reference_correct)
        by_intent[true_label]["candidate_correct"] += int(candidate_correct)
        both_correct += int(reference_correct and candidate_correct)
        reference_only += int(reference_correct and not candidate_correct)
        candidate_only += int(candidate_correct and not reference_correct)
        both_wrong += int(not reference_correct and not candidate_correct)
        disagreements += int(reference["predicted_label"] != candidate["predicted_label"])

    discordant = reference_only + candidate_only
    exact_p_value = (
        float(binomtest(reference_only, discordant, p=0.5).pvalue) if discordant else 1.0
    )
    intent_differences = [
        {
            "intent": intent,
            "support": counts["support"],
            "reference_correct": counts["reference_correct"],
            "candidate_correct": counts["candidate_correct"],
            "candidate_minus_reference_correct": (
                counts["candidate_correct"] - counts["reference_correct"]
            ),
        }
        for intent, counts in sorted(by_intent.items())
    ]
    candidate_advantages = sorted(
        intent_differences,
        key=lambda item: (-item["candidate_minus_reference_correct"], item["intent"]),
    )[:10]
    reference_advantages = sorted(
        intent_differences,
        key=lambda item: (item["candidate_minus_reference_correct"], item["intent"]),
    )[:10]

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "paired_baseline_comparison",
        "contains_message_text": False,
        "reference_model": reference_name,
        "candidate_model": candidate_name,
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
        "inputs": {
            "reference_evaluation_sha256": reference_evaluation_sha256,
            "candidate_evaluation_sha256": candidate_evaluation_sha256,
            "reference_predictions_sha256": stable_json_sha256(reference_rows),
            "candidate_predictions_sha256": stable_json_sha256(candidate_rows),
        },
        "row_count": len(reference_rows),
        "correctness": {
            "both_correct": both_correct,
            "reference_only_correct": reference_only,
            "candidate_only_correct": candidate_only,
            "both_wrong": both_wrong,
            "discordant_rows": discordant,
            "prediction_disagreements": disagreements,
        },
        "metrics": {
            "reference_macro_f1": round(float(reference_macro_f1), 10),
            "candidate_macro_f1": round(float(candidate_macro_f1), 10),
            "candidate_minus_reference_macro_f1": round(
                float(candidate_macro_f1 - reference_macro_f1), 10
            ),
            "exact_mcnemar_two_sided_p_value": round(exact_p_value, 10),
        },
        "candidate_advantage_intents": candidate_advantages,
        "reference_advantage_intents": reference_advantages,
        "interpretation_guardrail": (
            "A non-significant paired result does not establish model equivalence."
        ),
    }
    artifact["comparison_sha256"] = stable_json_sha256(artifact)
    assert_text_free_artifact(artifact)
    return artifact


def write_comparison(value: Mapping[str, Any], destination: Path) -> None:
    write_json_artifact(value, destination)
