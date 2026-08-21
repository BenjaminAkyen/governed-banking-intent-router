#!/usr/bin/env python3
"""Record a metadata-only smoke test of the real native Module 14 MPS service."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from governed_banking.audit import AuditConfig
from governed_banking.audit_store import LocalJsonlAuditStore
from governed_banking.data import sha256_file, stable_json_sha256
from governed_banking.deployment_config import DeploymentProfile
from governed_banking.deployment_service import create_deployment_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/deployment/native-mps.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/deployment/module14-native-mps-smoke.json"),
    )
    return parser.parse_args()


def implementation_hashes(script_path: Path) -> dict[str, str]:
    paths = {
        "audit_store.py": Path("src/governed_banking/audit_store.py"),
        "deployment_config.py": Path("src/governed_banking/deployment_config.py"),
        "deployment_service.py": Path("src/governed_banking/deployment_service.py"),
        "run_deployment_smoke.py": script_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = parse_args()
    profile = DeploymentProfile.from_yaml(args.profile)
    if profile.expected_device != "mps" or profile.platform != "native_macos":
        raise ValueError("this evidence script is registered only for native macOS MPS")

    environment = dict(os.environ)
    environment.pop("GOVERNED_BANKING_CONTAINER", None)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    token = secrets.token_urlsafe(32)
    environment[profile.authentication.secret_environment_variable] = token
    synthetic_message = "When will my new bank card arrive?"
    correlation_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory(prefix="module14-audit-") as temporary:
        stores: list[LocalJsonlAuditStore] = []

        def audit_factory(_: Path, config: AuditConfig) -> LocalJsonlAuditStore:
            store = LocalJsonlAuditStore(Path(temporary), config)
            stores.append(store)
            return store

        app = create_deployment_app(
            profile,
            environment=environment,
            audit_store_factory=audit_factory,
        )
        startup_started = time.perf_counter()
        with TestClient(app) as client:
            startup_seconds = time.perf_counter() - startup_started
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            unversioned = client.post(
                "/route",
                json={"message": synthetic_message},
                headers={"Authorization": f"Bearer {token}"},
            )
            unauthorized = client.post(
                "/v1/route",
                json={"message": synthetic_message},
            )
            route_started = time.perf_counter()
            route = client.post(
                "/v1/route",
                json={"message": synthetic_message},
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Correlation-ID": correlation_id,
                },
            )
            route_seconds = time.perf_counter() - route_started
            if not stores:
                raise AssertionError("audit store was not initialized")
            audit_path = stores[0].path
            events = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            loaded = app.state.deployment_runtime.loaded
            if loaded is None:
                raise AssertionError("deployment did not load")
            runtime_metadata: dict[str, Any] = dict(loaded.runtime_metadata)
            checks = {
                "liveness_healthy": live.status_code == 200,
                "readiness_healthy": ready.status_code == 200
                and ready.json().get("selected_device") == "mps",
                "versioned_route_successful": route.status_code == 200,
                "unversioned_route_absent": unversioned.status_code == 404,
                "authentication_required": unauthorized.status_code == 401,
                "canonical_request_id_returned": _is_canonical_uuid(
                    route.headers.get("x-request-id", "")
                ),
                "correlation_id_preserved": route.headers.get("x-correlation-id")
                == correlation_id,
                "release_header_bound": route.headers.get("x-model-release")
                == profile.model_release_id,
                "one_metadata_only_audit_event": len(events) == 1,
                "message_absent_from_response_and_audit": synthetic_message
                not in route.text + json.dumps(events, sort_keys=True),
                "mps_selected_without_fallback": runtime_metadata.get("selected") == "mps",
            }
            audit_directory_mode = f"{stat.S_IMODE(audit_path.parent.stat().st_mode):04o}"
            audit_file_mode = f"{stat.S_IMODE(audit_path.stat().st_mode):04o}"

        checks["graceful_shutdown_completed"] = (
            app.state.deployment_runtime.graceful_shutdown_completed is True
            and app.state.deployment_runtime.phase == "stopped"
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "module14_native_mps_deployment_smoke",
        "claim_scope": "single_process_native_mps_smoke_not_production_validation",
        "contains_message_text": False,
        "contains_redacted_text": False,
        "contains_message_hash": False,
        "deployment_profile_sha256": profile.config_sha256,
        "implementation_sha256": dict(sorted(implementation_hashes(Path(__file__)).items())),
        "runtime": runtime_metadata,
        "observations": {
            "startup_seconds": round(startup_seconds, 4),
            "single_route_seconds": round(route_seconds, 4),
            "audit_event_count": len(events),
            "audit_directory_mode": audit_directory_mode,
            "audit_file_mode": audit_file_mode,
        },
        "checks": dict(sorted(checks.items())),
        "all_checks_passed": all(checks.values()),
        "limitations": [
            "This is one in-process synthetic request on one Mac, not a load, "
            "availability, or production validation.",
            "The Linux CPU container was not built or executed in this environment.",
            "The Linux CUDA container was not built or executed because no NVIDIA "
            "CUDA runtime was available.",
            "The service remains shadow-review-only; Module 13 classification and "
            "safety-routing gates failed.",
        ],
    }
    report["report_sha256"] = stable_json_sha256(report)
    if not report["all_checks_passed"]:
        raise AssertionError("Module 14 native MPS deployment smoke checks failed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "selected_device": runtime_metadata.get("selected"),
                "startup_seconds": report["observations"]["startup_seconds"],
                "single_route_seconds": report["observations"]["single_route_seconds"],
                "all_checks_passed": report["all_checks_passed"],
            },
            indent=2,
        )
    )


def _is_canonical_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


if __name__ == "__main__":
    main()
