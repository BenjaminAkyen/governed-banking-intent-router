"""Versioned real-hardware runtime profiles and evidence for Module 11."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from governed_banking.accelerator import (
    AcceleratorPreference,
    empty_accelerator_cache,
    seed_accelerator,
    select_accelerator,
    synchronize_accelerator,
)
from governed_banking.data import sha256_file, stable_json_sha256

RUNTIME_PROFILE_SCHEMA_VERSION = 1
RUNTIME_REPORT_SCHEMA_VERSION = 1
RUNTIME_CLAIM_SCOPE = "hardware_runtime_verification_not_model_quality"
RUNTIME_PRECISION = "float32"
PROFILE_KEYS = {
    "schema_version",
    "profile_name",
    "device_preference",
    "cuda_device_index",
    "require_accelerator",
    "allow_mps_cpu_fallback",
    "precision",
    "claim_scope",
}


@dataclass(frozen=True)
class RuntimeProfile:
    """Strict accelerator profile that cannot silently change an explicit request."""

    schema_version: int
    profile_name: str
    device_preference: AcceleratorPreference
    cuda_device_index: int
    require_accelerator: bool
    allow_mps_cpu_fallback: bool
    precision: str
    claim_scope: str
    config_path: Path
    config_sha256: str

    @classmethod
    def from_yaml(cls, path: Path) -> RuntimeProfile:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != PROFILE_KEYS:
            raise ValueError("runtime profile fields differ from the registered schema")
        if raw.get("schema_version") != RUNTIME_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported runtime profile schema")

        name = raw.get("profile_name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"module11-(?:auto|cuda|mps|cpu)-fp32", name) is None
        ):
            raise ValueError("runtime profile name is invalid")
        preference = raw.get("device_preference")
        if preference not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device_preference must be auto, cuda, mps or cpu")
        expected_name = f"module11-{preference}-fp32"
        if name != expected_name:
            raise ValueError("profile name and device preference differ")

        cuda_device_index = raw.get("cuda_device_index")
        if (
            not isinstance(cuda_device_index, int)
            or isinstance(cuda_device_index, bool)
            or cuda_device_index < 0
        ):
            raise ValueError("cuda_device_index must be a non-negative integer")
        require_accelerator = raw.get("require_accelerator")
        if not isinstance(require_accelerator, bool):
            raise ValueError("require_accelerator must be boolean")
        if require_accelerator is not (preference in {"cuda", "mps"}):
            raise ValueError("only explicit accelerator profiles may require an accelerator")
        if raw.get("allow_mps_cpu_fallback") is not False:
            raise ValueError("registered profiles prohibit MPS CPU fallback")
        if raw.get("precision") != RUNTIME_PRECISION:
            raise ValueError("Module 11 runtime verification requires float32")
        if raw.get("claim_scope") != RUNTIME_CLAIM_SCOPE:
            raise ValueError("runtime claim scope differs from registration")

        return cls(
            schema_version=RUNTIME_PROFILE_SCHEMA_VERSION,
            profile_name=name,
            device_preference=preference,
            cuda_device_index=cuda_device_index,
            require_accelerator=require_accelerator,
            allow_mps_cpu_fallback=False,
            precision=RUNTIME_PRECISION,
            claim_scope=RUNTIME_CLAIM_SCOPE,
            config_path=path,
            config_sha256=sha256_file(path),
        )

    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["config_path"] = _portable_repository_path(self.config_path).as_posix()
        return body


def run_runtime_verification(
    profile: RuntimeProfile,
    *,
    report_path: Path,
    implementation_paths: dict[str, Path],
    seed: int = 42,
) -> dict[str, Any]:
    """Run a real tensor probe and atomically write a self-hashing runtime report."""

    if profile.device_preference == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("registered MPS verification prohibits PYTORCH_ENABLE_MPS_FALLBACK=1")
    if not implementation_paths:
        raise ValueError("implementation_paths must not be empty")

    device, metadata = select_accelerator(
        profile.device_preference,
        cuda_device_index=profile.cuda_device_index,
    )
    if profile.require_accelerator and device.type == "cpu":
        raise RuntimeError("runtime profile requires a real accelerator")

    seed_accelerator(seed, device)
    probe = _run_tensor_probe(device)
    synchronize_accelerator(device)
    empty_accelerator_cache(device)

    report: dict[str, Any] = {
        "schema_version": RUNTIME_REPORT_SCHEMA_VERSION,
        "artifact_type": "module11_real_hardware_runtime_report",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_scope": profile.claim_scope,
        "profile": profile.to_dict(),
        "runtime": metadata.to_dict(),
        "probe": probe,
        "seed": seed,
        "data_boundary": {
            "model_inference_performed": False,
            "official_test_access": False,
            "customer_data_access": False,
        },
        "verification": {
            "real_hardware_observed": metadata.real_hardware_observed,
            "cuda_verified": device.type == "cuda",
            "mps_verified": device.type == "mps",
            "cpu_verified": device.type == "cpu",
            "requested_backend_selected": (
                profile.device_preference == "auto"
                or profile.device_preference == metadata.selected
            ),
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(implementation_paths.items())
        },
    }
    report["report_sha256"] = stable_json_sha256(report)
    validate_runtime_report(report, profile=profile)
    _atomic_write_json(report_path, report)
    return report


def validate_runtime_report(report: dict[str, Any], *, profile: RuntimeProfile) -> None:
    """Validate report integrity and the non-model-quality claim boundary."""

    body = dict(report)
    expected_hash = body.pop("report_sha256", None)
    if expected_hash != stable_json_sha256(body):
        raise ValueError("runtime report content hash check failed")
    if report.get("schema_version") != RUNTIME_REPORT_SCHEMA_VERSION:
        raise ValueError("runtime report schema differs from registration")
    if report.get("artifact_type") != "module11_real_hardware_runtime_report":
        raise ValueError("runtime report artifact type is invalid")
    if report.get("claim_scope") != RUNTIME_CLAIM_SCOPE:
        raise ValueError("runtime report claim scope is invalid")
    if report.get("profile") != profile.to_dict():
        raise ValueError("runtime report profile differs from the selected configuration")

    runtime = report.get("runtime", {})
    if runtime.get("requested") != profile.device_preference:
        raise ValueError("runtime report requested backend differs from profile")
    if profile.device_preference != "auto" and runtime.get("selected") != profile.device_preference:
        raise ValueError("explicit runtime profile changed backend")
    if runtime.get("real_hardware_observed") is not True:
        raise ValueError("runtime report must describe observed hardware")

    boundary = report.get("data_boundary", {})
    if boundary != {
        "model_inference_performed": False,
        "official_test_access": False,
        "customer_data_access": False,
    }:
        raise ValueError("runtime report crossed its data boundary")
    probe = report.get("probe", {})
    if probe.get("dtype") != RUNTIME_PRECISION or probe.get("all_finite") is not True:
        raise ValueError("runtime tensor probe is invalid")
    if not math.isfinite(float(probe.get("sum", math.nan))):
        raise ValueError("runtime tensor probe sum is invalid")


def _run_tensor_probe(device: torch.device) -> dict[str, Any]:
    source = torch.arange(1, 17, dtype=torch.float32, device=device).reshape(4, 4)
    result = source @ source.transpose(0, 1)
    synchronize_accelerator(device)
    host_result = result.detach().cpu()
    return {
        "operation": "float32_matrix_multiplication",
        "input_shape": [4, 4],
        "output_shape": list(host_result.shape),
        "dtype": str(host_result.dtype).removeprefix("torch."),
        "all_finite": bool(torch.isfinite(host_result).all().item()),
        "sum": float(host_result.sum().item()),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def _portable_repository_path(path: Path) -> Path:
    if not path.is_absolute():
        return path
    for candidate in path.parents:
        if (candidate / "pyproject.toml").is_file():
            return path.relative_to(candidate)
    raise ValueError("runtime configuration must be inside the project repository")
