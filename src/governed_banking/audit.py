"""Strict metadata-only audit events and a symlink-resistant local JSONL sink."""

from __future__ import annotations

import json
import math
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from governed_banking.data import sha256_file
from governed_banking.policy import (
    REASON_ORDER,
    RoutingDecision,
    RoutingInput,
    RoutingPolicyConfig,
)
from governed_banking.privacy import (
    REGISTERED_DETECTORS,
    PrivacyConfig,
    RedactionResult,
    input_size_bucket,
)

AUDIT_CONFIG_SCHEMA_VERSION = 1
AUDIT_EVENT_SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = {
    "schema_version",
    "event_id",
    "occurred_at",
    "data_classification",
    "model",
    "privacy",
    "routing",
}
MODEL_KEYS = {
    "artifact_sha256",
    "seed",
    "predicted_intent",
    "uncertainty_signal",
    "uncertainty_score",
}
PRIVACY_KEYS = {
    "policy_version",
    "policy_sha256",
    "redaction_succeeded",
    "redaction_applied",
    "pii_type_counts",
    "input_size_bucket",
}
ROUTING_KEYS = {
    "policy_version",
    "policy_sha256",
    "operating_mode",
    "action",
    "queue",
    "reason_codes",
    "uncertainty_observation",
    "uncertainty_authorized_suggestion",
}
PROHIBITED_TEXT_KEYS = {
    "content",
    "input",
    "message",
    "message_hash",
    "message_text",
    "payload",
    "prompt",
    "query",
    "raw_text",
    "redacted_text",
    "response",
    "text",
}


@dataclass(frozen=True)
class AuditConfig:
    config_sha256: str
    audit_schema_version: int
    sink_directory: Path
    sink_filename: str
    directory_mode: int
    file_mode: int
    maximum_event_bytes: int
    maximum_events_on_read: int

    @classmethod
    def from_yaml(cls, path: Path) -> AuditConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "audit_schema_version",
            "sink_directory",
            "sink_filename",
            "directory_mode",
            "file_mode",
            "maximum_event_bytes",
            "maximum_events_on_read",
            "allow_message_hash",
            "allow_free_form_fields",
            "required_timezone",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError("audit configuration fields differ from registration")
        if raw.get("schema_version") != AUDIT_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported audit configuration schema")
        if raw.get("audit_schema_version") != AUDIT_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported audit event schema")
        sink_directory = _relative_path(raw.get("sink_directory"), "sink_directory")
        sink_filename = _safe_filename(raw.get("sink_filename"))
        directory_mode = _mode(raw.get("directory_mode"), "directory_mode")
        file_mode = _mode(raw.get("file_mode"), "file_mode")
        if directory_mode != 0o700 or file_mode != 0o600:
            raise ValueError("audit sink permissions must remain 0700/0600")
        maximum_event_bytes = _bounded_int(
            raw.get("maximum_event_bytes"), "maximum_event_bytes", 1024, 65536
        )
        maximum_events_on_read = _bounded_int(
            raw.get("maximum_events_on_read"), "maximum_events_on_read", 1, 100000
        )
        if raw.get("allow_message_hash") is not False:
            raise ValueError("message hashes are prohibited because low-entropy text is reversible")
        if raw.get("allow_free_form_fields") is not False:
            raise ValueError("free-form audit fields are prohibited")
        if raw.get("required_timezone") != "UTC":
            raise ValueError("audit timestamps must use UTC")
        return cls(
            config_sha256=sha256_file(path),
            audit_schema_version=AUDIT_EVENT_SCHEMA_VERSION,
            sink_directory=sink_directory,
            sink_filename=sink_filename,
            directory_mode=directory_mode,
            file_mode=file_mode,
            maximum_event_bytes=maximum_event_bytes,
            maximum_events_on_read=maximum_events_on_read,
        )


@dataclass(frozen=True)
class ModelAuditContext:
    artifact_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "artifact_sha256")


@dataclass(frozen=True)
class PrivacyAuditContext:
    policy_version: str
    policy_sha256: str
    redaction_succeeded: bool
    redaction_applied: bool
    pii_type_counts: Mapping[str, int]
    input_character_count: int

    @classmethod
    def from_redaction(cls, result: RedactionResult) -> PrivacyAuditContext:
        return cls(
            policy_version=result.privacy_policy_version,
            policy_sha256=result.privacy_policy_sha256,
            redaction_succeeded=True,
            redaction_applied=result.redaction_applied,
            pii_type_counts=dict(result.pii_type_counts),
            input_character_count=result.input_character_count,
        )

    @classmethod
    def failed(cls, config: PrivacyConfig, *, input_character_count: int) -> PrivacyAuditContext:
        return cls(
            policy_version=config.policy_version,
            policy_sha256=config.config_sha256,
            redaction_succeeded=False,
            redaction_applied=False,
            pii_type_counts={},
            input_character_count=input_character_count,
        )


def build_audit_event(
    audit_config: AuditConfig,
    routing_config: RoutingPolicyConfig,
    model: ModelAuditContext,
    routing_input: RoutingInput,
    privacy: PrivacyAuditContext | RedactionResult,
    decision: RoutingDecision,
    *,
    event_id: str | None = None,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an allowlisted event; message and redacted text are intentionally unavailable."""

    privacy_context = (
        PrivacyAuditContext.from_redaction(privacy)
        if isinstance(privacy, RedactionResult)
        else privacy
    )
    if dict(routing_input.pii_type_counts) != dict(privacy_context.pii_type_counts):
        raise ValueError("routing PII counts must come from the supplied redaction result")
    if routing_input.redaction_succeeded is not privacy_context.redaction_succeeded:
        raise ValueError("routing and privacy redaction outcomes disagree")
    if decision.policy_version != routing_config.policy_version:
        raise ValueError("decision policy version differs from the registered policy")
    if decision.policy_sha256 != routing_config.config_sha256:
        raise ValueError("decision policy hash differs from the registered policy")
    timestamp = occurred_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise ValueError("occurred_at must be timezone-aware UTC")
    identifier = event_id or str(uuid.uuid4())
    event = {
        "schema_version": audit_config.audit_schema_version,
        "event_id": identifier,
        "occurred_at": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "data_classification": "metadata_only",
        "model": {
            "artifact_sha256": model.artifact_sha256,
            "seed": routing_input.model_seed,
            "predicted_intent": routing_input.predicted_intent,
            "uncertainty_signal": routing_input.uncertainty_signal,
            "uncertainty_score": routing_input.uncertainty_score,
        },
        "privacy": {
            "policy_version": privacy_context.policy_version,
            "policy_sha256": privacy_context.policy_sha256,
            "redaction_succeeded": privacy_context.redaction_succeeded,
            "redaction_applied": privacy_context.redaction_applied,
            "pii_type_counts": dict(privacy_context.pii_type_counts),
            "input_size_bucket": input_size_bucket(privacy_context.input_character_count),
        },
        "routing": {
            "policy_version": decision.policy_version,
            "policy_sha256": decision.policy_sha256,
            "operating_mode": routing_config.operating_mode,
            "action": decision.action,
            "queue": decision.queue,
            "reason_codes": list(decision.reason_codes),
            "uncertainty_observation": decision.uncertainty_observation,
            "uncertainty_authorized_suggestion": decision.uncertainty_authorized_suggestion,
        },
    }
    validate_audit_event(event, audit_config)
    return event


def validate_audit_event(event: Mapping[str, Any], config: AuditConfig) -> None:
    if not isinstance(event, dict) or set(event) != TOP_LEVEL_KEYS:
        raise ValueError("audit event top-level fields differ from the allowlist")
    if _contains_prohibited_key(event):
        raise ValueError("audit event contains a prohibited text-bearing field")
    if event.get("schema_version") != config.audit_schema_version:
        raise ValueError("audit event schema version differs from configuration")
    _uuid4(event.get("event_id"))
    _utc_timestamp(event.get("occurred_at"))
    if event.get("data_classification") != "metadata_only":
        raise ValueError("audit event must be classified as metadata_only")
    model = _exact_mapping(event, "model", MODEL_KEYS)
    _sha256(model.get("artifact_sha256"), "model.artifact_sha256")
    _bounded_int(model.get("seed"), "model.seed", 0, 2**31 - 1)
    _bounded_string(model.get("predicted_intent"), "model.predicted_intent", 1, 128)
    signal = model.get("uncertainty_signal")
    if signal is not None:
        _bounded_string(signal, "model.uncertainty_signal", 1, 64)
    score = model.get("uncertainty_score")
    if score is not None and (
        not isinstance(score, int | float)
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("model.uncertainty_score must be null or between zero and one")
    privacy = _exact_mapping(event, "privacy", PRIVACY_KEYS)
    _bounded_string(privacy.get("policy_version"), "privacy.policy_version", 1, 128)
    _sha256(privacy.get("policy_sha256"), "privacy.policy_sha256")
    if not isinstance(privacy.get("redaction_succeeded"), bool):
        raise ValueError("privacy.redaction_succeeded must be boolean")
    if not isinstance(privacy.get("redaction_applied"), bool):
        raise ValueError("privacy.redaction_applied must be boolean")
    counts = privacy.get("pii_type_counts")
    if not isinstance(counts, dict) or not set(counts) <= set(REGISTERED_DETECTORS):
        raise ValueError("privacy.pii_type_counts contains unregistered types")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in counts.values()
    ):
        raise ValueError("privacy PII counts must be positive integers")
    if bool(counts) != privacy.get("redaction_applied"):
        raise ValueError("privacy redaction flag and PII counts disagree")
    if not privacy.get("redaction_succeeded") and privacy.get("redaction_applied"):
        raise ValueError("failed redaction cannot claim that redaction was applied")
    if privacy.get("input_size_bucket") not in {
        "le_32",
        "le_64",
        "le_128",
        "le_256",
        "le_512",
        "le_1024",
        "le_2048",
        "le_4096",
        "gt_4096",
    }:
        raise ValueError("privacy.input_size_bucket is invalid")
    routing = _exact_mapping(event, "routing", ROUTING_KEYS)
    _bounded_string(routing.get("policy_version"), "routing.policy_version", 1, 128)
    _sha256(routing.get("policy_sha256"), "routing.policy_sha256")
    if routing.get("operating_mode") != "shadow_review_only":
        raise ValueError("audit event must retain the Module 9 shadow boundary")
    if routing.get("action") not in {"security_queue", "human_review"}:
        raise ValueError("audit action is not allowed in shadow mode")
    _bounded_string(routing.get("queue"), "routing.queue", 1, 128)
    reasons = routing.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(reason not in REASON_ORDER for reason in reasons)
        or reasons != [reason for reason in REASON_ORDER if reason in set(reasons)]
    ):
        raise ValueError("routing reason codes are invalid or out of order")
    if routing.get("uncertainty_observation") not in {
        "invalid_or_missing",
        "below_experimental_threshold",
        "at_or_above_experimental_threshold",
    }:
        raise ValueError("routing uncertainty observation is invalid")
    if routing.get("uncertainty_authorized_suggestion") is not False:
        raise ValueError("experimental uncertainty cannot authorize a suggestion")
    payload = serialize_audit_event(event, maximum_bytes=config.maximum_event_bytes)
    if len(payload) > config.maximum_event_bytes:
        raise ValueError("audit event exceeds its configured byte limit")


def serialize_audit_event(event: Mapping[str, Any], *, maximum_bytes: int) -> bytes:
    payload = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    if b"\n" in payload or len(payload) + 1 > maximum_bytes:
        raise ValueError("serialized audit event is invalid or too large")
    return payload + b"\n"


class AuditSink:
    """Append-only local evidence sink with fixed path and restrictive permissions."""

    def __init__(self, project_root: Path, config: AuditConfig) -> None:
        self._root = project_root.resolve(strict=True)
        self._config = config
        self.path = self._root / config.sink_directory / config.sink_filename
        if not self.path.is_relative_to(self._root):
            raise ValueError("audit path escapes the project root")

    def append(self, event: Mapping[str, Any]) -> None:
        validate_audit_event(event, self._config)
        payload = serialize_audit_event(event, maximum_bytes=self._config.maximum_event_bytes)
        self._prepare_directory()
        if self.path.is_symlink():
            raise ValueError("audit file cannot be a symbolic link")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, self._config.file_mode)
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode) or file_status.st_uid != os.getuid():
                raise PermissionError("audit sink must be a regular file owned by this user")
            os.fchmod(descriptor, self._config.file_mode)
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("partial audit-event write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_validated(self) -> list[dict[str, Any]]:
        if self.path.is_symlink():
            raise ValueError("audit file cannot be a symbolic link")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        events: list[dict[str, Any]] = []
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode) or file_status.st_uid != os.getuid():
                raise PermissionError("audit sink must be a regular file owned by this user")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while True:
                    line = handle.readline(self._config.maximum_event_bytes + 1)
                    if not line:
                        break
                    if len(line) > self._config.maximum_event_bytes or not line.endswith(b"\n"):
                        raise ValueError("audit sink contains an oversized or incomplete event")
                    event = json.loads(line)
                    validate_audit_event(event, self._config)
                    events.append(event)
                    if len(events) > self._config.maximum_events_on_read:
                        raise ValueError("audit sink exceeds the configured read limit")
        finally:
            os.close(descriptor)
        return events

    def _prepare_directory(self) -> None:
        current = self._root
        for part in self._config.sink_directory.parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError("audit directory cannot traverse a symbolic link")
        self.path.parent.mkdir(mode=self._config.directory_mode, parents=True, exist_ok=True)
        if self.path.parent.is_symlink() or not self.path.parent.is_dir():
            raise ValueError("audit sink parent must be a real directory")
        os.chmod(self.path.parent, self._config.directory_mode)


def _contains_prohibited_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in PROHIBITED_TEXT_KEYS or _contains_prohibited_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_prohibited_key(item) for item in value)
    return False


def _exact_mapping(parent: Mapping[str, Any], key: str, keys: set[str]) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"audit {key} fields differ from the allowlist")
    return value


def _relative_path(value: Any, name: str) -> Path:
    result = Path(str(value or ""))
    if not str(value or "").strip() or result.is_absolute() or ".." in result.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return result


def _safe_filename(value: Any) -> str:
    result = str(value or "").strip()
    if not result or result in {".", ".."} or Path(result).name != result:
        raise ValueError("sink_filename must be a plain filename")
    return result


def _mode(value: Any, name: str) -> int:
    if (
        not isinstance(value, str)
        or len(value) != 4
        or any(char not in "01234567" for char in value)
    ):
        raise ValueError(f"{name} must be a four-digit octal string")
    return int(value, 8)


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _bounded_string(value: Any, name: str, minimum: int, maximum: int) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum or value.strip() != value:
        raise ValueError(f"{name} must be a bounded non-blank string")
    return value


def _sha256(value: Any, name: str) -> str:
    result = _bounded_string(value, name, 64, 64)
    if any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _uuid4(value: Any) -> str:
    result = _bounded_string(value, "event_id", 36, 36)
    parsed = uuid.UUID(result)
    if parsed.version != 4 or str(parsed) != result:
        raise ValueError("event_id must be a canonical UUID4")
    return result


def _utc_timestamp(value: Any) -> datetime:
    result = _bounded_string(value, "occurred_at", 20, 32)
    if not result.endswith("Z"):
        raise ValueError("occurred_at must use the UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("occurred_at is not valid ISO-8601") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("occurred_at must use UTC")
    return parsed
