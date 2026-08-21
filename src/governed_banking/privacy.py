"""Bounded structured-PII detection and pre-inference redaction."""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from governed_banking.data import sha256_file

PRIVACY_SCHEMA_VERSION = 1
REGISTERED_DETECTORS = (
    "authentication_secret",
    "email",
    "iban",
    "national_insurance_number",
    "payment_card",
    "uk_sort_code",
    "uk_bank_account",
    "phone",
    "ip_address",
    "date_of_birth",
    "uk_postcode",
)
REGISTERED_REPLACEMENTS = {name: f"[{name.upper()}]" for name in REGISTERED_DETECTORS}


@dataclass(frozen=True)
class PrivacyConfig:
    policy_version: str
    config_sha256: str
    maximum_input_characters: int
    reject_null_bytes: bool
    residual_scan_required: bool
    detector_order: tuple[str, ...]
    replacements: Mapping[str, str]

    @classmethod
    def from_yaml(cls, path: Path) -> PrivacyConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != PRIVACY_SCHEMA_VERSION:
            raise ValueError("unsupported privacy configuration schema")
        detector_order = tuple(raw.get("detector_order", []))
        if detector_order != REGISTERED_DETECTORS:
            raise ValueError("privacy detector order differs from registration")
        replacements = raw.get("replacements")
        if replacements != REGISTERED_REPLACEMENTS:
            raise ValueError("privacy replacement tokens differ from registration")
        maximum = raw.get("maximum_input_characters")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 256 <= maximum <= 16384:
            raise ValueError("maximum_input_characters must be between 256 and 16384")
        if raw.get("reject_null_bytes") is not True:
            raise ValueError("privacy controls must reject null bytes")
        if raw.get("residual_scan_required") is not True:
            raise ValueError("privacy controls must require a residual scan")
        limitations = raw.get("limitations")
        if limitations != {
            "detects_free_form_names": False,
            "detects_contextual_identifiers": False,
            "iban_scope": "gb_format",
            "production_dlp_replacement": False,
        }:
            raise ValueError("privacy limitations must remain explicit")
        return cls(
            policy_version=_non_blank(raw.get("policy_version"), "policy_version"),
            config_sha256=sha256_file(path),
            maximum_input_characters=maximum,
            reject_null_bytes=True,
            residual_scan_required=True,
            detector_order=detector_order,
            replacements=dict(replacements),
        )


@dataclass(frozen=True)
class PIIFinding:
    pii_type: str
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    findings: tuple[PIIFinding, ...]
    pii_type_counts: Mapping[str, int]
    redaction_applied: bool
    input_character_count: int
    privacy_policy_version: str
    privacy_policy_sha256: str


@dataclass(frozen=True)
class _Detector:
    name: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool]


def _always(_: str) -> bool:
    return True


DETECTORS = {
    "authentication_secret": _Detector(
        "authentication_secret",
        re.compile(
            r"(?i)\b(?:password|passcode|secret|api[ _-]?key|access[ _-]?token|auth[ _-]?token)"
            r"\s*(?:is|=|:)\s*[\"']?(?P<value>[A-Za-z0-9_./+@#$%!?=-]{4,4096})"
        ),
        _always,
    ),
    "email": _Detector(
        "email",
        re.compile(
            r"(?i)(?<![A-Z0-9._%+-])(?P<value>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63})(?![A-Z0-9._%+-])"
        ),
        _always,
    ),
    "iban": _Detector(
        "iban",
        re.compile(
            r"(?i)(?<![A-Z0-9])(?P<value>GB\d{2}\s?[A-Z]{4}(?:\s?\d{4}){3}\s?\d{2})(?![A-Z0-9])"
        ),
        lambda value: _valid_iban(value),
    ),
    "national_insurance_number": _Detector(
        "national_insurance_number",
        re.compile(
            r"(?i)(?<![A-Z0-9])(?P<value>(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D])(?![A-Z0-9])"
        ),
        _always,
    ),
    "payment_card": _Detector(
        "payment_card",
        re.compile(r"(?<!\d)(?P<value>\d(?:[ -]?\d){12,18})(?![ -]?\d)"),
        lambda value: _valid_luhn(value),
    ),
    "uk_sort_code": _Detector(
        "uk_sort_code",
        re.compile(
            r"(?i)\bsort\s*code\s*(?:is|=|:)?\s*"
            r"(?P<value>(?<!\d)\d{2}[- ]\d{2}[- ]\d{2}(?!\d))"
        ),
        _always,
    ),
    "uk_bank_account": _Detector(
        "uk_bank_account",
        re.compile(
            r"(?i)\b(?:account|acct)\s*(?:number|no\.?|#)\s*(?:is|=|:)?\s*(?P<value>\d{8})(?!\d)"
        ),
        _always,
    ),
    "phone": _Detector(
        "phone",
        re.compile(
            r"(?<![\w+])(?P<value>(?:\+44[\s().-]?\d(?:[\s().-]?\d){9}|0\d(?:[\s().-]?\d){9}))(?!\w)"
        ),
        lambda value: 10 <= len(_digits(value)) <= 12,
    ),
    "ip_address": _Detector(
        "ip_address",
        re.compile(r"(?<![\d.])(?P<value>(?:\d{1,3}\.){3}\d{1,3})(?![\d.])"),
        lambda value: _valid_ip(value),
    ),
    "date_of_birth": _Detector(
        "date_of_birth",
        re.compile(
            r"(?i)\b(?:date\s+of\s+birth|dob|born)\s*(?:is|=|:|on)?\s*(?P<value>(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2})"
        ),
        lambda value: _valid_date(value),
    ),
    "uk_postcode": _Detector(
        "uk_postcode",
        re.compile(
            r"(?i)(?<![A-Z0-9])(?P<value>GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})(?![A-Z0-9])"
        ),
        _always,
    ),
}


def redact_pii(config: PrivacyConfig, message: str) -> RedactionResult:
    """Return a transient redacted representation without retaining matched values."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not message or not message.strip():
        raise ValueError("message cannot be blank")
    if len(message) > config.maximum_input_characters:
        raise ValueError("message exceeds the registered character limit")
    if "\x00" in message:
        raise ValueError("message contains a prohibited null byte")
    candidates: list[tuple[int, int, int, str]] = []
    for priority, detector_name in enumerate(config.detector_order):
        detector = DETECTORS[detector_name]
        for match in detector.pattern.finditer(message):
            start, end = match.span("value")
            value = match.group("value")
            if detector.validator(value):
                candidates.append((start, end, priority, detector_name))
    selected = _resolve_overlaps(candidates)
    parts: list[str] = []
    findings: list[PIIFinding] = []
    cursor = 0
    for start, end, _, detector_name in selected:
        parts.append(message[cursor:start])
        parts.append(config.replacements[detector_name])
        findings.append(PIIFinding(detector_name, start, end))
        cursor = end
    parts.append(message[cursor:])
    redacted = "".join(parts)
    if config.residual_scan_required:
        residual = _detect_pii_types(redacted, config.detector_order)
        if residual:
            raise RuntimeError(f"redaction residual scan failed for types: {sorted(residual)}")
    counts = dict(sorted(Counter(finding.pii_type for finding in findings).items()))
    return RedactionResult(
        redacted_text=redacted,
        findings=tuple(findings),
        pii_type_counts=counts,
        redaction_applied=bool(findings),
        input_character_count=len(message),
        privacy_policy_version=config.policy_version,
        privacy_policy_sha256=config.config_sha256,
    )


def _resolve_overlaps(
    candidates: Sequence[tuple[int, int, int, str]],
) -> tuple[tuple[int, int, int, str], ...]:
    ordered = sorted(candidates, key=lambda row: (row[0], row[2], -(row[1] - row[0])))
    selected: list[tuple[int, int, int, str]] = []
    for candidate in ordered:
        start, end, _, _ = candidate
        if any(
            start < existing_end and end > existing_start
            for existing_start, existing_end, *_ in selected
        ):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda row: row[0]))


def _detect_pii_types(text: str, detector_order: Sequence[str]) -> set[str]:
    detected = set()
    for detector_name in detector_order:
        detector = DETECTORS[detector_name]
        if any(
            detector.validator(match.group("value")) for match in detector.pattern.finditer(text)
        ):
            detected.add(detector_name)
    return detected


def _digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _valid_luhn(value: str) -> bool:
    digits = _digits(value)
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _valid_iban(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(
        str(ord(character) - 55) if character.isalpha() else character for character in rearranged
    )
    remainder = 0
    for character in numeric:
        if not character.isdigit():
            return False
        remainder = (remainder * 10 + int(character)) % 97
    return remainder == 1


def _valid_ip(value: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(value), ipaddress.IPv4Address)
    except ValueError:
        return False


def _valid_date(value: str) -> bool:
    normalized = re.sub(r"[-.]", "/", value)
    try:
        datetime.strptime(normalized, "%d/%m/%Y")
    except ValueError:
        return False
    return True


def input_size_bucket(character_count: int) -> str:
    if (
        not isinstance(character_count, int)
        or isinstance(character_count, bool)
        or character_count < 0
    ):
        raise ValueError("character count must be a non-negative integer")
    boundaries = (32, 64, 128, 256, 512, 1024, 2048, 4096)
    for boundary in boundaries:
        if character_count <= boundary:
            return f"le_{boundary}"
    return "gt_4096"


def _non_blank(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} cannot be blank")
    return result
