#!/usr/bin/env python3
"""Validate and register the Module 13 synthetic robustness pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from governed_banking.data import (
    DatasetConfig,
    read_banking_csv,
    sha256_file,
    stable_json_sha256,
    validate_manifest,
)
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.robustness import (
    REGISTERED_FAMILIES,
    ROBUSTNESS_MANIFEST_SCHEMA_VERSION,
    RobustnessEvaluationConfig,
    find_internal_leakage,
    find_leakage,
    load_robustness_cases,
    summarize_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/robustness_evaluation.yaml")
    )
    parser.add_argument("--privacy-config", type=Path, default=Path("configs/privacy.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RobustnessEvaluationConfig.from_yaml(args.config)
    taxonomy_manifest = json.loads(
        config.taxonomy_manifest_path.read_text(encoding="utf-8")
    )
    validate_manifest(taxonomy_manifest)
    labels = taxonomy_manifest.get("label_names", [])
    if not isinstance(labels, list) or len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("taxonomy manifest does not contain 77 unique labels")

    privacy_config = PrivacyConfig.from_yaml(args.privacy_config)
    cases = load_robustness_cases(
        config.pack_path,
        allowed_intents=labels,
        allowed_pii_types=privacy_config.detector_order,
    )
    summary = summarize_cases(cases)
    missing_families = sorted(set(REGISTERED_FAMILIES) - set(summary["primary_family_counts"]))
    underfilled_families = sorted(
        family
        for family, count in summary["primary_family_counts"].items()
        if count < config.minimum_cases_per_family
    )

    pii_mismatches = []
    for case in cases:
        result = redact_pii(privacy_config, case.text)
        observed = tuple(sorted(result.pii_type_counts))
        expected = tuple(sorted(case.expected_pii_types))
        if observed != expected:
            pii_mismatches.append(
                {"case_id": case.case_id, "expected": list(expected), "observed": list(observed)}
            )

    internal_findings = find_internal_leakage(
        cases,
        ngram_size=config.near_duplicate_ngram_size,
        jaccard_threshold=config.near_duplicate_jaccard_threshold,
        minimum_characters=config.minimum_near_duplicate_characters,
    )
    dataset_config = DatasetConfig.from_yaml(config.dataset_config_path)
    train_path = config.raw_directory / dataset_config.files["official_train"]
    test_path = config.raw_directory / dataset_config.files["official_test"]
    for logical_name, path in (("official_train", train_path), ("official_test", test_path)):
        if not path.exists():
            raise FileNotFoundError(
                f"verified {logical_name} source is required for the Module 13 leakage gate: {path}"
            )
        if sha256_file(path) != dataset_config.expected_sha256[logical_name]:
            raise ValueError(f"{logical_name} source hash differs from the dataset registration")
    official_train = read_banking_csv(
        train_path,
        source_split="official_train",
        expected_rows=dataset_config.expected_train_rows,
    )
    official_test = read_banking_csv(
        test_path,
        source_split="official_test",
        expected_rows=dataset_config.expected_test_rows,
    )
    banking_findings = find_leakage(
        cases,
        {
            "banking77_official_train": [
                (f"row-{record.source_index:05d}", record.text) for record in official_train
            ],
            "banking77_official_test": [
                (f"row-{record.source_index:05d}", record.text) for record in official_test
            ],
        },
        ngram_size=config.near_duplicate_ngram_size,
        jaccard_threshold=config.near_duplicate_jaccard_threshold,
        minimum_characters=config.minimum_near_duplicate_characters,
    )
    banking_match_counts = Counter(finding.match_type for finding in banking_findings)
    internal_match_counts = Counter(finding.match_type for finding in internal_findings)

    gates = {
        "all_registered_families_present": not missing_families,
        "minimum_cases_per_family_met": not underfilled_families,
        "zero_internal_exact_duplicates": internal_match_counts["exact"] == 0,
        "zero_internal_near_duplicates": internal_match_counts["near_duplicate"] == 0,
        "zero_banking77_exact_matches": banking_match_counts["exact"] == 0,
        "zero_banking77_near_duplicates": banking_match_counts["near_duplicate"] == 0,
        "all_pii_expectations_matched": not pii_mismatches,
        "all_cases_require_escalation": summary["all_cases_require_escalation"] is True,
        "no_suggestion_actions_registered": "suggest_queue"
        not in summary["expected_routing_action_counts"],
    }
    if not all(gates.values()):
        failed = sorted(key for key, value in gates.items() if not value)
        details = {
            "failed_gates": failed,
            "internal_findings": [value.to_dict() for value in internal_findings],
            "banking77_findings": [value.to_dict() for value in banking_findings],
            "pii_mismatches": pii_mismatches,
        }
        raise AssertionError(json.dumps(details, indent=2, sort_keys=True))

    manifest: dict[str, Any] = {
        "schema_version": ROBUSTNESS_MANIFEST_SCHEMA_VERSION,
        "artifact_type": "module13_synthetic_robustness_pack_manifest",
        "pack_version": config.pack_version,
        "claim_scope": "synthetic_stress_test_not_production_validation",
        "pack_sha256": config.expected_pack_sha256,
        "config_sha256": sha256_file(config.config_path),
        "taxonomy_manifest_sha256": taxonomy_manifest["manifest_sha256"],
        "privacy_config_sha256": privacy_config.config_sha256,
        "licence": {
            "spdx_id": "MIT",
            "copyright": "Copyright (c) 2026 Benjamin Akyen",
            "case_level_licence_present": True,
        },
        "provenance": {
            "origin": "project_authored_synthetic",
            "creation_method": "human_directed_ai_assisted_scenario_authoring",
            "case_level_provenance_present": True,
            "contains_customer_data": False,
            "derived_from_production_data": False,
            "derived_from_banking77_text": False,
        },
        "coverage": summary,
        "leakage_evidence": {
            "normalization": "unicode_nfkc_casefold_whitespace_v1",
            "near_duplicate_method": "character_5gram_jaccard",
            "near_duplicate_jaccard_threshold": config.near_duplicate_jaccard_threshold,
            "minimum_near_duplicate_characters": config.minimum_near_duplicate_characters,
            "internal_exact_match_count": internal_match_counts["exact"],
            "internal_near_duplicate_count": internal_match_counts["near_duplicate"],
            "banking77_exact_match_count": banking_match_counts["exact"],
            "banking77_near_duplicate_count": banking_match_counts["near_duplicate"],
            "banking77_source_commit": dataset_config.commit,
            "banking77_source_hashes": {
                "official_train": dataset_config.expected_sha256["official_train"],
                "official_test": dataset_config.expected_sha256["official_test"],
            },
            "banking77_rows_scanned": len(official_train) + len(official_test),
            "finding_metadata": [],
        },
        "privacy_expectations": {
            "registered_detector_count": len(privacy_config.detector_order),
            "detectors_exercised": sorted(summary["expected_pii_type_case_counts"]),
            "mismatch_count": len(pii_mismatches),
        },
        "data_boundary": {
            "fixture_is_synthetic": True,
            "customer_data_access": False,
            "production_data_access": False,
            "official_test_used_for_model_scoring": False,
            "banking77_text_used_only_for_leakage_detection": True,
            "production_validation": False,
        },
        "acceptance_gate": gates,
        "limitations": [
            "Synthetic cases do not estimate production accuracy, safety, fairness, drift, "
            "or attack resistance.",
            "BANKING77 train and test text were read only to detect fixture leakage; no model "
            "score was computed on those rows.",
            "The character-ngram near-duplicate check detects high lexical overlap but not "
            "every semantic derivation.",
            "Representative claims require lawfully obtained, appropriately governed "
            "real-world evaluation data.",
        ],
    }
    manifest["acceptance_gate"]["all_passed"] = all(gates.values())
    manifest["manifest_sha256"] = stable_json_sha256(manifest)
    _write_json(manifest, config.manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(config.manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "case_count": len(cases),
                "banking77_rows_scanned": len(official_train) + len(official_test),
                "all_gates_passed": manifest["acceptance_gate"]["all_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
