from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from governed_banking.deployment_config import DeploymentProfile
from governed_banking.observability_config import (
    ALLOWED_ATTRIBUTE_KEYS,
    METRIC_NAMES,
    PROHIBITED_ATTRIBUTE_KEYS,
    SPAN_NAMES,
    ObservabilityConfig,
    registered_redaction_categories,
)


def _deployment() -> DeploymentProfile:
    return DeploymentProfile.from_yaml(Path("configs/deployment/native-mps.yaml"))


def test_registered_observability_config_is_hash_bound_and_privacy_minimal() -> None:
    config = ObservabilityConfig.from_yaml(
        Path("configs/observability.yaml"), deployment_profile=_deployment()
    )
    assert config.metric_names == METRIC_NAMES
    assert config.span_names == SPAN_NAMES
    assert config.allowed_attribute_keys == ALLOWED_ATTRIBUTE_KEYS
    assert config.prohibited_attribute_keys == PROHIBITED_ATTRIBUTE_KEYS
    assert set(config.allowed_attribute_keys).isdisjoint(config.prohibited_attribute_keys)
    assert not any(
        fragment in key
        for key in config.allowed_attribute_keys
        for fragment in ("message", "text", "token", "user", "request_id", "hash")
    )
    assert len(registered_redaction_categories()) == 11
    assert config.minimum_observations == 20
    assert config.rolling_window_observations == 100
    assert config.reference_distribution["human_review"] == pytest.approx(5 / 6)
    assert config.reference_distribution["security_queue"] == pytest.approx(1 / 6)


def test_all_three_module14_profiles_are_registered() -> None:
    config = ObservabilityConfig.from_yaml(Path("configs/observability.yaml"))
    names = {path.name for path in config.registered_deployment_profiles}
    assert names == {"native-mps.yaml", "linux-cpu.yaml", "linux-cuda.yaml"}


def test_unregistered_deployment_hash_fails_closed() -> None:
    deployment = _deployment()
    tampered = replace(deployment, config_sha256="0" * 64)
    with pytest.raises(ValueError, match="not registered"):
        ObservabilityConfig.from_yaml(
            Path("configs/observability.yaml"), deployment_profile=tampered
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("instrumentation", "automatic_http_instrumentation", True, "weakens"),
        ("instrumentation", "exception_stacktraces", True, "weakens"),
        ("instrumentation", "baggage_propagation", True, "weakens"),
        ("export", "exporter_headers_allowed", True, "export boundary"),
        ("claims", "message_hashes_collected", True, "claims overstate"),
        ("routing_distribution", "alerting_threshold_approved", True, "claim boundary"),
    ],
)
def test_privacy_and_claim_weakening_is_rejected(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    source = Path("configs/observability.yaml")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw[section][field] = value
    candidate = tmp_path / "observability.yaml"
    candidate.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ObservabilityConfig.from_yaml(candidate, project_root=Path.cwd())
