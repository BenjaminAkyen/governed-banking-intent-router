from __future__ import annotations

import json
import os
import stat
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from governed_banking.audit import (
    AuditConfig,
    AuditSink,
    ModelAuditContext,
    PrivacyAuditContext,
    build_audit_event,
    validate_audit_event,
)
from governed_banking.policy import RoutingInput, RoutingPolicyConfig, route_request
from governed_banking.privacy import PrivacyConfig, redact_pii


def _configs() -> tuple[AuditConfig, RoutingPolicyConfig, PrivacyConfig]:
    return (
        AuditConfig.from_yaml(Path("configs/audit.yaml")),
        RoutingPolicyConfig.from_yaml(Path("configs/routing_policy.yaml")),
        PrivacyConfig.from_yaml(Path("configs/privacy.yaml")),
    )


def _event() -> tuple[AuditConfig, dict[str, object], str, str]:
    audit_config, routing_config, privacy_config = _configs()
    original = "Email alex@example.test about my card"
    redaction = redact_pii(privacy_config, original)
    routing_input = RoutingInput(
        predicted_intent="card_arrival",
        model_seed=42,
        uncertainty_signal="max_probability",
        uncertainty_score=0.99,
        pii_type_counts=redaction.pii_type_counts,
        redaction_succeeded=True,
    )
    decision = route_request(routing_config, routing_input)
    event = build_audit_event(
        audit_config,
        routing_config,
        ModelAuditContext("a" * 64),
        routing_input,
        redaction,
        decision,
        event_id="6ba7b810-9dad-4b11-80b4-00c04fd430c8",
        occurred_at=datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
    )
    return audit_config, event, original, redaction.redacted_text


def test_event_is_allowlisted_metadata_without_message_or_message_hash() -> None:
    config, event, original, redacted = _event()
    serialized = json.dumps(event, sort_keys=True)

    validate_audit_event(event, config)
    assert event["data_classification"] == "metadata_only"
    assert original not in serialized
    assert redacted not in serialized
    assert "message_hash" not in serialized
    assert "input_character_count" not in serialized
    assert event["privacy"]["input_size_bucket"] == "le_64"


@pytest.mark.parametrize("field", ["message", "redacted_text", "message_hash", "payload"])
def test_free_form_or_reversible_fields_are_rejected(field: str) -> None:
    config, event, _, _ = _event()
    tampered = deepcopy(event)
    tampered[field] = "must not be logged"

    with pytest.raises(ValueError):
        validate_audit_event(tampered, config)


def test_experimental_uncertainty_cannot_be_promoted_in_audit() -> None:
    config, event, _, _ = _event()
    tampered = deepcopy(event)
    tampered["routing"]["uncertainty_authorized_suggestion"] = True

    with pytest.raises(ValueError, match="cannot authorize"):
        validate_audit_event(tampered, config)


@pytest.mark.skipif(os.name != "posix", reason="local audit permission contract is POSIX-only")
def test_audit_sink_round_trip_and_permissions(tmp_path: Path) -> None:
    config, event, _, _ = _event()
    sink = AuditSink(tmp_path, config)

    sink.append(event)

    assert sink.read_validated() == [event]
    assert stat.S_IMODE(sink.path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(sink.path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="local audit symlink contract is POSIX-only")
def test_audit_sink_rejects_symbolic_link_file(tmp_path: Path) -> None:
    config, event, _, _ = _event()
    sink = AuditSink(tmp_path, config)
    sink.path.parent.mkdir(parents=True)
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    sink.path.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        sink.append(event)


def test_event_builder_rejects_pii_count_mismatch() -> None:
    audit_config, routing_config, privacy_config = _configs()
    redaction = redact_pii(privacy_config, "Email alex@example.test")
    routing_input = RoutingInput(
        predicted_intent="card_arrival",
        model_seed=42,
        uncertainty_signal="max_probability",
        uncertainty_score=0.99,
        pii_type_counts={},
        redaction_succeeded=True,
    )
    decision = route_request(routing_config, routing_input)

    with pytest.raises(ValueError, match="PII counts"):
        build_audit_event(
            audit_config,
            routing_config,
            ModelAuditContext("a" * 64),
            routing_input,
            redaction,
            decision,
        )


def test_redaction_failure_can_be_audited_without_text() -> None:
    audit_config, routing_config, privacy_config = _configs()
    routing_input = RoutingInput(
        predicted_intent="card_arrival",
        model_seed=42,
        uncertainty_signal="max_probability",
        uncertainty_score=0.99,
        pii_type_counts={},
        redaction_succeeded=False,
    )
    decision = route_request(routing_config, routing_input)
    privacy = PrivacyAuditContext.failed(privacy_config, input_character_count=5000)

    event = build_audit_event(
        audit_config,
        routing_config,
        ModelAuditContext("a" * 64),
        routing_input,
        privacy,
        decision,
    )

    assert event["privacy"]["redaction_succeeded"] is False
    assert event["privacy"]["input_size_bucket"] == "gt_4096"
    assert "REDACTION_FAILURE" in event["routing"]["reason_codes"]
