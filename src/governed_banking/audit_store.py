"""Pluggable metadata-only audit-store boundary for deployable services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from governed_banking.audit import AuditConfig, AuditSink


@runtime_checkable
class AuditStore(Protocol):
    """Minimal synchronous store contract used after event allowlist validation."""

    def append(self, event: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class LocalJsonlAuditStore:
    """Built-in append-only JSONL store; deployments may inject another implementation."""

    def __init__(self, project_root: Path, config: AuditConfig) -> None:
        self._sink = AuditSink(project_root, config)

    @property
    def path(self) -> Path:
        return self._sink.path

    def append(self, event: dict[str, Any]) -> None:
        self._sink.append(event)

    def close(self) -> None:
        """The local sink opens, fsyncs and closes every append; no handle remains open."""


def require_audit_store(value: object) -> AuditStore:
    """Fail before readiness when an injected store does not implement the contract."""

    if not isinstance(value, AuditStore):
        raise TypeError("audit store must implement append(event) and close()")
    return value
