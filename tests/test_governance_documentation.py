from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/governance/module17.yaml"
RISK_PATH = PROJECT_ROOT / "governance/risk-register.yaml"
APPROVAL_PATH = PROJECT_ROOT / "governance/change-approvals/module17.yaml"


def _yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_governance_contract_preserves_confirmed_research_boundary() -> None:
    config = _yaml(CONFIG_PATH)
    assumptions = config["operating_assumptions"]
    assert isinstance(assumptions, dict)

    assert config["accountable_organisation"] == "INNETWORK Technology Limited"
    assert config["release_target"] == "v0.2.0"
    assert config["release_classification"] == "research_preview"
    assert config["approval_status"] == "pending_release_approval"
    assert config["production_approved"] is False
    assert assumptions["operating_mode"] == "shadow_review_only"
    assert assumptions["autonomous_routing_permitted"] is False
    assert assumptions["banking_actions_permitted"] is False
    assert assumptions["real_customer_data_permitted"] is False
    assert assumptions["public_internet_origin_permitted"] is False
    assert assumptions["current_champion"] == "tfidf-word-char-c4"
    assert assumptions["currently_served_shadow_model"] == "lora-roberta-r8-revised"
    assert assumptions["champion_service_aligned"] is False


def test_all_mandatory_governance_documents_exist() -> None:
    config = _yaml(CONFIG_PATH)
    documents = config["mandatory_documents"]
    assert isinstance(documents, dict)
    assert len(documents) == 13

    for path in documents.values():
        assert isinstance(path, str)
        resolved = PROJECT_ROOT / path
        assert resolved.is_file(), path
        assert resolved.stat().st_size > 500, path


def test_risk_register_scores_and_evidence_are_consistent() -> None:
    register = _yaml(RISK_PATH)
    risks = register["risks"]
    assert isinstance(risks, list)
    assert len(risks) == 13
    assert [risk["id"] for risk in risks] == [f"R-{index:03d}" for index in range(1, 14)]

    expected_bands = {range(1, 5): "low", range(5, 10): "medium", range(10, 17): "high"}
    for risk in risks:
        assert isinstance(risk, dict)
        for state in ("inherent", "residual"):
            score = risk[state]
            assert score["score"] == score["likelihood"] * score["impact"]
            expected = "critical"
            for score_range, band in expected_bands.items():
                if score["score"] in score_range:
                    expected = band
                    break
            assert score["band"] == expected
        assert str(risk["owner_role"]) in {
            "accountable_system_owner",
            "model_risk_reviewer",
            "security_and_privacy_reviewer",
            "service_operator",
        }
        for evidence_path in risk["evidence"]:
            assert (PROJECT_ROOT / evidence_path).exists(), evidence_path


def test_release_blockers_match_known_negative_evidence() -> None:
    config = _yaml(CONFIG_PATH)
    blockers = set(config["release_blockers"])

    assert {
        "representative_external_evaluation_missing",
        "uncertainty_operating_point_gates_failed",
        "robustness_classification_and_security_routing_gates_failed",
        "served_model_is_not_registered_champion",
        "identity_provider_not_configured",
        "release_change_approval_pending",
    } <= blockers
    assert "publication_hygiene_failed_on_committed_notebook_output" not in blockers


def test_change_record_never_fabricates_approval() -> None:
    record = _yaml(APPROVAL_PATH)
    approval = record["approval"]
    release = record["release_effect"]

    assert approval["status"] == "pending_release_approval"
    assert approval["recorded_approvals"] == []
    assert release["release_blocked_until_approved"] is True
    assert release["production_approval_created"] is False
    assert record["model_artifact_changed"] is False
    assert record["routing_policy_changed"] is False
    assert (
        record["verification_observations"]["overall"]
        == "release_blocked_pending_safety_evidence_and_accountable_approval"
    )


def test_model_card_matches_champion_registry() -> None:
    registry = _yaml(PROJECT_ROOT / "configs/champion_challenger.yaml")
    card = (PROJECT_ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")

    assert registry["current_champion_id"] in card
    assert registry["service_alignment"]["currently_served_model_id"] in card
    assert "0.9053" in card
    assert "68.52%" in card
    assert "78.57%" in card
    assert "No model is approved for production" in card


def test_threat_model_has_required_structure_and_stable_ids() -> None:
    path = PROJECT_ROOT / "docs/governance/governed-banking-intent-router-threat-model.md"
    document = path.read_text(encoding="utf-8")
    required_sections = (
        "## Executive summary",
        "## Scope and assumptions",
        "## System model",
        "## Assets and security objectives",
        "## Attacker model",
        "## Entry points and attack surfaces",
        "## Top abuse paths",
        "## Threat model table",
        "## Criticality calibration",
        "## Focus paths for security review",
    )
    for section in required_sections:
        assert section in document
    identifiers = sorted(set(re.findall(r"TM-\d{3}", document)))
    assert identifiers == [f"TM-{index:03d}" for index in range(1, 10)]
    assert "```mermaid" in document
    assert "pull_request_target" not in document


def test_nist_mapping_is_complete_at_function_level_and_not_a_certification() -> None:
    document = (PROJECT_ROOT / "docs/governance/NIST_AI_RMF_MAPPING.md").read_text(encoding="utf-8")

    for function in ("## GOVERN", "## MAP", "## MEASURE", "## MANAGE"):
        assert function in document
    assert "not a certification" in document
    assert "AI RMF 1.0 is being revised" in document


def test_operational_documents_preserve_fail_closed_and_no_real_data_rules() -> None:
    combined = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/governance/INTENDED_USE.md",
            "docs/governance/HUMAN_OVERSIGHT.md",
            "docs/governance/INCIDENT_RESPONSE.md",
            "docs/governance/ROLLBACK_PROCEDURE.md",
            "docs/governance/CHANGE_APPROVAL.md",
            "docs/governance/MONITORING_PLAN.md",
            "docs/governance/VALIDATION_REVALIDATION_POLICY.md",
        )
    )

    normalized = " ".join(combined.casefold().split())
    assert "real customer" in normalized
    assert "human review" in normalized
    assert "hot mutation" in normalized
    assert "metadata-only" in normalized
    assert "production approval" in normalized
