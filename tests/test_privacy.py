from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.privacy import PrivacyConfig, input_size_bucket, redact_pii


def _config() -> PrivacyConfig:
    return PrivacyConfig.from_yaml(Path("configs/privacy.yaml"))


def _cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path("data/fixtures/pii-redaction-cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["case_id"]))
def test_registered_synthetic_redaction_cases(case: dict[str, object]) -> None:
    result = redact_pii(_config(), str(case["input"]))

    assert result.redacted_text == case["expected_redacted"]
    assert dict(result.pii_type_counts) == case["expected_counts"]
    assert result.redaction_applied is case["expected_detection"]
    assert all(not hasattr(finding, "value") for finding in result.findings)


def test_redaction_is_deterministic_and_text_idempotent() -> None:
    config = _config()
    message = "Email alex@example.test and use card 4111 1111 1111 1111."

    first = redact_pii(config, message)
    second = redact_pii(config, message)
    repeated = redact_pii(config, first.redacted_text)

    assert first == second
    assert repeated.redacted_text == first.redacted_text
    assert "alex@example.test" not in repr(first)
    assert "4111 1111 1111 1111" not in repr(first)


def test_overlapping_numeric_patterns_prefer_valid_payment_card() -> None:
    result = redact_pii(_config(), "Test card: 4111 1111 1111 1111")

    assert result.redacted_text == "Test card: [PAYMENT_CARD]"
    assert result.pii_type_counts == {"payment_card": 1}


def test_long_authentication_secret_is_fully_redacted() -> None:
    secret = "x" * 1000
    result = redact_pii(_config(), f"Password: {secret}")

    assert result.redacted_text == "Password: [AUTHENTICATION_SECRET]"
    assert secret not in repr(result)


def test_out_of_scope_twenty_digit_number_is_not_partially_redacted() -> None:
    number = "41111111111111111111"
    result = redact_pii(_config(), f"Reference {number}")

    assert result.redacted_text == f"Reference {number}"
    assert result.pii_type_counts == {}


@pytest.mark.parametrize("message", ["", "   ", "contains\x00null"])
def test_blank_or_null_bearing_messages_are_rejected(message: str) -> None:
    with pytest.raises(ValueError):
        redact_pii(_config(), message)


def test_oversized_messages_are_rejected() -> None:
    config = _config()

    with pytest.raises(ValueError, match="character limit"):
        redact_pii(config, "x" * (config.maximum_input_characters + 1))


@pytest.mark.parametrize(
    ("characters", "bucket"),
    [(0, "le_32"), (32, "le_32"), (33, "le_64"), (4096, "le_4096"), (4097, "gt_4096")],
)
def test_input_size_bucket_avoids_exact_length_logging(characters: int, bucket: str) -> None:
    assert input_size_bucket(characters) == bucket


def test_privacy_configuration_keeps_limitations_explicit() -> None:
    config = _config()

    assert config.policy_version == "module9-structured-pii-v1"
    assert config.residual_scan_required is True
    assert "authentication_secret" in config.detector_order
