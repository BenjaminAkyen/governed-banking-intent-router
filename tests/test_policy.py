from __future__ import annotations

from pathlib import Path

import pytest

from governed_banking.policy import (
    RoutingInput,
    RoutingPolicyConfig,
    route_request,
)


def _config() -> RoutingPolicyConfig:
    return RoutingPolicyConfig.from_yaml(Path("configs/routing_policy.yaml"))


def _input(
    *,
    intent: str = "card_arrival",
    seed: int = 42,
    signal: str | None = "max_probability",
    score: float | None = 0.95,
    pii: dict[str, int] | None = None,
    redaction_succeeded: bool = True,
) -> RoutingInput:
    return RoutingInput(
        predicted_intent=intent,
        model_seed=seed,
        uncertainty_signal=signal,
        uncertainty_score=score,
        pii_type_counts=pii or {},
        redaction_succeeded=redaction_succeeded,
    )


def test_policy_is_bound_to_failed_module8_evidence_and_shadow_mode() -> None:
    config = _config()

    assert config.operating_mode == "shadow_review_only"
    assert config.uncertainty_status == "failed_registered_gates"
    assert config.uncertainty_use == "review_signal_only"
    assert config.uncertainty_may_authorize_suggestion is False
    assert len(config.allowed_intents) == 77
    assert set(config.thresholds) == {17, 42, 73}


@pytest.mark.parametrize(
    "intent",
    [
        "card_payment_not_recognised",
        "cash_withdrawal_not_recognised",
        "compromised_card",
        "direct_debit_payment_not_recognised",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "passcode_forgotten",
        "pin_blocked",
    ],
)
def test_security_intents_always_override_to_security_queue(intent: str) -> None:
    decision = route_request(_config(), _input(intent=intent, score=0.99))

    assert decision.action == "security_queue"
    assert decision.queue == "security_operations"
    assert "SECURITY_INTENT_OVERRIDE" in decision.reason_codes
    assert decision.uncertainty_authorized_suggestion is False


def test_exposed_authentication_secret_overrides_standard_intent() -> None:
    decision = route_request(_config(), _input(pii={"authentication_secret": 1}, score=0.99))

    assert decision.action == "security_queue"
    assert decision.reason_codes[0] == "EXPOSED_AUTHENTICATION_SECRET"


def test_above_experimental_threshold_still_requires_review() -> None:
    decision = route_request(_config(), _input(score=0.99))

    assert decision.action == "human_review"
    assert decision.uncertainty_observation == "at_or_above_experimental_threshold"
    assert "EXPERIMENTAL_UNCERTAINTY_NOT_AUTHORIZED" in decision.reason_codes
    assert decision.uncertainty_authorized_suggestion is False


def test_below_experimental_threshold_requires_review() -> None:
    decision = route_request(_config(), _input(score=0.10))

    assert decision.action == "human_review"
    assert decision.uncertainty_observation == "below_experimental_threshold"
    assert "EXPERIMENTAL_UNCERTAINTY_BELOW_THRESHOLD" in decision.reason_codes


@pytest.mark.parametrize(
    ("seed", "signal", "score"),
    [(999, "max_probability", 0.9), (42, "wrong_signal", 0.9), (42, None, None)],
)
def test_invalid_uncertainty_metadata_fails_to_review(
    seed: int, signal: str | None, score: float | None
) -> None:
    decision = route_request(_config(), _input(seed=seed, signal=signal, score=score))

    assert decision.action == "human_review"
    assert decision.uncertainty_observation == "invalid_or_missing"
    assert "UNCERTAINTY_METADATA_INVALID" in decision.reason_codes


def test_redaction_failure_and_sensitive_pii_require_review() -> None:
    decision = route_request(
        _config(),
        _input(
            pii={"payment_card": 1},
            redaction_succeeded=False,
        ),
    )

    assert decision.action == "human_review"
    assert "REDACTION_FAILURE" in decision.reason_codes
    assert "SENSITIVE_PII_REVIEW" in decision.reason_codes


def test_unsupported_intent_fails_closed() -> None:
    decision = route_request(_config(), _input(intent="make_me_a_loan"))

    assert decision.action == "human_review"
    assert "UNSUPPORTED_INTENT" in decision.reason_codes


def test_current_policy_never_emits_suggest_queue_for_any_registered_intent() -> None:
    config = _config()

    decisions = [route_request(config, _input(intent=intent)) for intent in config.allowed_intents]

    assert all(decision.action != "suggest_queue" for decision in decisions)
    assert {decision.action for decision in decisions} == {"human_review", "security_queue"}


def test_routing_is_deterministic_and_reason_order_is_stable() -> None:
    config = _config()
    value = _input(intent="unknown", score=0.1, pii={"iban": 1}, redaction_succeeded=False)

    first = route_request(config, value)
    second = route_request(config, value)

    assert first == second
    assert first.reason_codes == (
        "REDACTION_FAILURE",
        "UNSUPPORTED_INTENT",
        "SENSITIVE_PII_REVIEW",
        "EXPERIMENTAL_UNCERTAINTY_BELOW_THRESHOLD",
        "SHADOW_MODE_REQUIRES_REVIEW",
    )


def test_invalid_pii_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="PII counts"):
        route_request(_config(), _input(pii={"email": -1}))
