import json
from pathlib import Path

import pytest

from governed_banking.champion import (
    ChampionChallengerConfig,
    PairedIntervals,
    SeedComparison,
    evaluate_registered_gates,
    validate_registry_report,
)
from governed_banking.data import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/champion_challenger.yaml"
REPORT_PATH = PROJECT_ROOT / "reports/champion/champion-registry.json"


def _comparison(
    *,
    champion_macro_f1: float = 0.90,
    challenger_macro_f1: float = 0.91,
    champion_ece: float = 0.04,
    challenger_ece: float = 0.04,
    champion_risk: float = 0.06,
    challenger_risk: float = 0.06,
    security_delta: float = 0.0,
) -> SeedComparison:
    return SeedComparison(
        champion_macro_f1=champion_macro_f1,
        challenger_macro_f1=challenger_macro_f1,
        champion_ece=champion_ece,
        challenger_ece=challenger_ece,
        champion_selective_risk=champion_risk,
        challenger_selective_risk=challenger_risk,
        champion_known_coverage=0.90,
        challenger_known_coverage=0.90,
        challenger_possible_ood_recall=0.95,
        champion_security_intent_f1=0.90,
        challenger_security_intent_f1=0.90 + security_delta,
    )


def test_registry_binds_real_evidence_and_retains_tfidf() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    validate_registry_report(report, config=config)
    assert config.current_champion_id == "tfidf-word-char-c4"
    assert report["current_decision"]["action"] == "retain_champion"
    assert report["promotion_readiness"]["eligible_challenger_evaluations"] == 0
    assert report["historical_ranking_context_only"][0]["model_id"] == (
        "tfidf-word-char-c4"
    )
    assert report["implementation_sha256"] == {
        "build_champion_registry.py": sha256_file(
            PROJECT_ROOT / "scripts/build_champion_registry.py"
        ),
        "champion.py": sha256_file(PROJECT_ROOT / "src/governed_banking/champion.py"),
    }


def test_registered_evidence_never_becomes_promotion_eligible() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)

    assert all(
        evidence.promotion_eligible is False
        for model in config.models
        for evidence in model.evidence
    )
    assert config.data_boundary["external_evaluation_lock_status"] == "missing"
    assert config.service_alignment["champion_aligned"] is False


def test_superiority_route_requires_human_approval_after_all_gates_pass() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)
    comparisons = {seed: _comparison() for seed in (17, 42, 73)}

    result = evaluate_registered_gates(
        config,
        comparisons_by_seed=comparisons,
        intervals=PairedIntervals(
            macro_f1_delta=(0.005, 0.015),
            ece_delta=(-0.001, 0.001),
            selective_risk_delta=(-0.001, 0.001),
        ),
        privacy_tests_passed=True,
        routing_tests_passed=True,
        audit_tests_passed=True,
        matched_coverage=True,
    )

    assert result["classification_superiority_passed"] is True
    assert result["eligible_for_human_approval"] is True
    assert result["automatic_promotion_permitted"] is False
    assert result["decision"] == "await_human_approval"


def test_noninferior_calibration_route_can_pass_without_classification_superiority() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)
    comparisons = {
        seed: _comparison(
            challenger_macro_f1=0.899,
            champion_ece=0.04,
            challenger_ece=0.025,
        )
        for seed in (17, 42, 73)
    }

    result = evaluate_registered_gates(
        config,
        comparisons_by_seed=comparisons,
        intervals=PairedIntervals(
            macro_f1_delta=(-0.004, 0.002),
            ece_delta=(-0.020, -0.010),
            selective_risk_delta=(-0.001, 0.001),
        ),
        privacy_tests_passed=True,
        routing_tests_passed=True,
        audit_tests_passed=True,
        matched_coverage=True,
    )

    assert result["classification_superiority_passed"] is False
    assert result["classification_noninferiority_passed"] is True
    assert result["calibration_improvement_passed"] is True
    assert result["operational_improvement_route_passed"] is True
    assert result["eligible_for_human_approval"] is True


def test_safety_veto_retains_champion_even_when_macro_f1_is_higher() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)
    comparisons = {seed: _comparison(security_delta=-0.03) for seed in (17, 42, 73)}

    result = evaluate_registered_gates(
        config,
        comparisons_by_seed=comparisons,
        intervals=PairedIntervals(
            macro_f1_delta=(0.005, 0.015),
            ece_delta=(-0.001, 0.001),
            selective_risk_delta=(-0.001, 0.001),
        ),
        privacy_tests_passed=True,
        routing_tests_passed=True,
        audit_tests_passed=True,
        matched_coverage=True,
    )

    assert result["classification_superiority_passed"] is True
    assert result["security_intent_veto_passed"] is False
    assert result["eligible_for_human_approval"] is False
    assert result["decision"] == "retain_champion"


def test_gate_evaluation_rejects_incomplete_seed_evidence() -> None:
    config = ChampionChallengerConfig.from_yaml(CONFIG_PATH)

    with pytest.raises(ValueError, match="exactly seeds 17, 42 and 73"):
        evaluate_registered_gates(
            config,
            comparisons_by_seed={17: _comparison(), 42: _comparison()},
            intervals=PairedIntervals(
                macro_f1_delta=(0.005, 0.015),
                ece_delta=(-0.001, 0.001),
                selective_risk_delta=(-0.001, 0.001),
            ),
            privacy_tests_passed=True,
            routing_tests_passed=True,
            audit_tests_passed=True,
            matched_coverage=True,
        )
