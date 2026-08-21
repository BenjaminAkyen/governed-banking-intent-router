from __future__ import annotations

import json
from pathlib import Path

from governed_banking.baseline import assert_text_free_artifact
from governed_banking.data import DatasetConfig, sha256_file, stable_json_sha256, validate_manifest
from governed_banking.privacy import PrivacyConfig, redact_pii
from governed_banking.robustness import (
    REGISTERED_FAMILIES,
    RobustnessEvaluationConfig,
    find_internal_leakage,
    load_robustness_cases,
    summarize_cases,
)

CONFIG_PATH = Path("configs/robustness_evaluation.yaml")
REPORT_PATH = Path("reports/robustness/module13-lora-mps-assessment.json")


def _config() -> RobustnessEvaluationConfig:
    return RobustnessEvaluationConfig.from_yaml(CONFIG_PATH)


def _taxonomy() -> list[str]:
    value = json.loads(Path("data/manifests/banking77-seed-42.json").read_text(encoding="utf-8"))
    validate_manifest(value)
    return value["label_names"]


def _cases():
    config = _config()
    privacy = PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))
    return load_robustness_cases(
        config.pack_path,
        allowed_intents=_taxonomy(),
        allowed_pii_types=privacy.detector_order,
    )


def _manifest() -> dict[str, object]:
    return json.loads(Path("data/robustness/v1/manifest.json").read_text(encoding="utf-8"))


def _report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_pack_schema_has_balanced_family_coverage_and_case_level_governance() -> None:
    cases = _cases()
    summary = summarize_cases(cases)

    assert len(cases) == 60
    assert summary["primary_family_counts"] == {
        family: 6 for family in sorted(REGISTERED_FAMILIES)
    }
    assert summary["all_cases_require_escalation"] is True
    assert summary["out_of_scope_case_count"] == 6
    assert summary["ambiguous_label_case_count"] == 23
    assert all(case.expected_routing_action != "suggest_queue" for case in cases)


def test_pack_has_no_internal_exact_or_registered_near_duplicates() -> None:
    config = _config()
    findings = find_internal_leakage(
        _cases(),
        ngram_size=config.near_duplicate_ngram_size,
        jaccard_threshold=config.near_duplicate_jaccard_threshold,
        minimum_characters=config.minimum_near_duplicate_characters,
    )

    assert findings == ()


def test_pack_manifest_is_self_hashing_and_records_full_banking77_leakage_gate() -> None:
    manifest = _manifest()
    expected = manifest.pop("manifest_sha256")
    dataset = DatasetConfig.from_yaml(Path("configs/dataset.yaml"))

    assert stable_json_sha256(manifest) == expected
    assert manifest["acceptance_gate"]["all_passed"] is True
    assert manifest["leakage_evidence"]["banking77_rows_scanned"] == 13083
    assert manifest["leakage_evidence"]["banking77_exact_match_count"] == 0
    assert manifest["leakage_evidence"]["banking77_near_duplicate_count"] == 0
    assert manifest["leakage_evidence"]["banking77_source_commit"] == dataset.commit
    assert manifest["leakage_evidence"]["banking77_source_hashes"] == {
        "official_train": dataset.expected_sha256["official_train"],
        "official_test": dataset.expected_sha256["official_test"],
    }
    assert manifest["data_boundary"]["banking77_text_used_only_for_leakage_detection"] is True
    assert manifest["data_boundary"]["official_test_used_for_model_scoring"] is False


def test_pack_manifest_is_bound_to_current_implementation_and_configuration() -> None:
    manifest = _manifest()
    assert manifest["config_sha256"] == sha256_file(CONFIG_PATH)
    assert manifest["implementation_sha256"] == {
        "build_robustness_pack.py": sha256_file(Path("scripts/build_robustness_pack.py")),
        "data.py": sha256_file(Path("src/governed_banking/data.py")),
        "privacy.py": sha256_file(Path("src/governed_banking/privacy.py")),
        "robustness.py": sha256_file(Path("src/governed_banking/robustness.py")),
    }


def test_all_registered_pii_expectations_match_the_real_redactor() -> None:
    privacy = PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))

    for case in _cases():
        observed = redact_pii(privacy, case.text)
        assert tuple(sorted(observed.pii_type_counts)) == tuple(sorted(case.expected_pii_types))


def test_real_mps_report_is_self_hashing_text_free_and_pack_bound() -> None:
    report = _report()
    expected = report.pop("report_sha256")
    config = _config()
    manifest = _manifest()

    assert stable_json_sha256(report) == expected
    assert_text_free_artifact(report)
    assert report["source_evidence"]["config_sha256"] == sha256_file(CONFIG_PATH)
    assert report["source_evidence"]["pack_sha256"] == config.expected_pack_sha256
    assert report["source_evidence"]["pack_manifest_sha256"] == manifest["manifest_sha256"]
    assert report["runtime"]["requested"] == "mps"
    assert report["runtime"]["selected"] == "mps"
    assert report["runtime"]["real_hardware_observed"] is True
    assert report["data_boundary"]["official_test_access"] is False
    assert report["data_boundary"]["production_validation"] is False


def test_report_contains_no_original_or_redacted_case_values() -> None:
    report_text = json.dumps(_report(), sort_keys=True)
    privacy = PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))

    for case in _cases():
        redacted = redact_pii(privacy, case.text).redacted_text
        assert case.text not in report_text
        assert redacted not in report_text


def test_failed_preregistered_gates_are_preserved_without_relaxation() -> None:
    report = _report()

    assert report["metrics"]["in_scope_acceptable_intent_rate"] == 0.6851851852
    assert report["metrics"]["expected_security_routing_recall"] == 0.7857142857
    assert report["metrics"]["routing_action_match_rate"] == 0.9
    assert report["metrics"]["pii_expectation_match_rate"] == 1.0
    assert report["metrics"]["suggest_queue_count"] == 0
    assert report["assessment_gate"] == {
        "all_passed": False,
        "all_pii_expectations_matched": True,
        "expected_security_routing_recall": False,
        "in_scope_acceptable_intent_rate": False,
        "overall_routing_action_match_rate": False,
        "text_free_report": True,
        "zero_suggestion_actions": True,
    }
    assert report["model_role"] == "module10_lora_research_service_not_champion"
    assert report["champion_model_evaluated"] is False
