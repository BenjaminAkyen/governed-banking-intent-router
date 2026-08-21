from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed_banking.data import sha256_file
from governed_banking.runtime_evidence import (
    RuntimeProfile,
    run_runtime_verification,
    validate_runtime_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = PROJECT_ROOT / "configs/runtime"


def _implementation_paths() -> dict[str, Path]:
    return {
        "accelerator.py": PROJECT_ROOT / "src/governed_banking/accelerator.py",
        "runtime_evidence.py": PROJECT_ROOT / "src/governed_banking/runtime_evidence.py",
        "verify_accelerator.py": PROJECT_ROOT / "scripts/verify_accelerator.py",
    }


@pytest.mark.parametrize("backend", ["auto", "cuda", "mps", "cpu"])
def test_runtime_profiles_are_strict_and_hash_bound(backend: str) -> None:
    path = PROFILE_DIRECTORY / f"{backend}.yaml"
    profile = RuntimeProfile.from_yaml(path)

    assert profile.device_preference == backend
    assert profile.profile_name == f"module11-{backend}-fp32"
    assert profile.config_sha256 == sha256_file(path)
    assert profile.to_dict()["config_path"] == f"configs/runtime/{backend}.yaml"
    assert profile.allow_mps_cpu_fallback is False
    assert profile.require_accelerator is (backend in {"cuda", "mps"})


def test_cpu_runtime_report_uses_real_hardware_and_is_self_hashing(tmp_path: Path) -> None:
    profile = RuntimeProfile.from_yaml(PROFILE_DIRECTORY / "cpu.yaml")
    report_path = tmp_path / "cpu-runtime.json"

    report = run_runtime_verification(
        profile,
        report_path=report_path,
        implementation_paths=_implementation_paths(),
    )

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted == report
    validate_runtime_report(persisted, profile=profile)
    assert report["runtime"]["selected"] == "cpu"
    assert report["verification"]["cpu_verified"] is True
    assert report["verification"]["cuda_verified"] is False
    assert report["verification"]["mps_verified"] is False
    assert report["probe"]["sum"] == 4704.0
    assert report["data_boundary"]["official_test_access"] is False


def test_runtime_report_tampering_is_detected(tmp_path: Path) -> None:
    profile = RuntimeProfile.from_yaml(PROFILE_DIRECTORY / "cpu.yaml")
    report = run_runtime_verification(
        profile,
        report_path=tmp_path / "cpu-runtime.json",
        implementation_paths=_implementation_paths(),
    )
    report["runtime"]["selected"] = "cuda"

    with pytest.raises(ValueError, match="content hash"):
        validate_runtime_report(report, profile=profile)


def test_registered_cpu_report_matches_current_implementation() -> None:
    profile = RuntimeProfile.from_yaml(PROFILE_DIRECTORY / "cpu.yaml")
    report = json.loads(
        (PROJECT_ROOT / "reports/runtime/cpu-runtime.json").read_text(encoding="utf-8")
    )

    validate_runtime_report(report, profile=profile)
    assert report["implementation_sha256"] == {
        name: sha256_file(path) for name, path in sorted(_implementation_paths().items())
    }
    assert report["runtime"]["selected"] == "cpu"
    assert report["verification"]["real_hardware_observed"] is True
