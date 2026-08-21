from __future__ import annotations

from pathlib import Path

import pytest

from governed_banking.data import sha256_file
from governed_banking.deployment_config import DeploymentProfile

PROFILE_DIRECTORY = Path("configs/deployment")


@pytest.mark.parametrize(
    ("filename", "platform", "device", "authentication_mode"),
    [
        ("native-mps.yaml", "native_macos", "mps", "development_bearer"),
        ("linux-cpu.yaml", "linux_container", "cpu", "trusted_gateway"),
        ("linux-cuda.yaml", "linux_container", "cuda", "trusted_gateway"),
    ],
)
def test_deployment_profiles_are_hash_bound_and_device_explicit(
    filename: str,
    platform: str,
    device: str,
    authentication_mode: str,
) -> None:
    path = PROFILE_DIRECTORY / filename
    profile = DeploymentProfile.from_yaml(path)

    assert profile.config_sha256 == sha256_file(path)
    assert profile.platform == platform
    assert profile.expected_device == device
    assert profile.authentication.mode == authentication_mode
    assert profile.model_release_id == "module10-lora-seed42-research"
    assert profile.model_artifact_sha256 == (
        "b78b2dafce23a633c86b962cb672b8b91c8e07c5308debf124a73ffbcb21cca8"
    )


def test_native_mps_is_not_registered_as_a_container_profile() -> None:
    profile = DeploymentProfile.from_yaml(PROFILE_DIRECTORY / "native-mps.yaml")

    assert profile.platform == "native_macos"
    assert profile.container_required is False
    assert profile.bind_host == "127.0.0.1"
    assert profile.rollback_reference_required is False


@pytest.mark.parametrize("filename", ["linux-cpu.yaml", "linux-cuda.yaml"])
def test_container_profiles_require_gateway_auth_and_immutable_rollback(filename: str) -> None:
    profile = DeploymentProfile.from_yaml(PROFILE_DIRECTORY / filename)

    assert profile.platform == "linux_container"
    assert profile.container_required is True
    assert profile.authentication.mode == "trusted_gateway"
    assert profile.authentication.secret_environment_variable == (
        "GOVERNED_BANKING_GATEWAY_ASSERTION"
    )
    assert profile.rollback_strategy == "immutable_process_or_container_revision"
    assert profile.rollback_reference_required is True


def test_development_and_deployment_secrets_use_different_environment_variables() -> None:
    native = DeploymentProfile.from_yaml(PROFILE_DIRECTORY / "native-mps.yaml")
    container = DeploymentProfile.from_yaml(PROFILE_DIRECTORY / "linux-cpu.yaml")

    assert native.authentication.secret_environment_variable == (
        "GOVERNED_BANKING_DEV_API_TOKEN"
    )
    assert container.authentication.secret_environment_variable == (
        "GOVERNED_BANKING_GATEWAY_ASSERTION"
    )
    assert (
        native.authentication.secret_environment_variable
        != container.authentication.secret_environment_variable
    )
