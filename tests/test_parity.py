from pathlib import Path

import pytest
import torch

from governed_banking.api import ServiceConfig
from governed_banking.data import sha256_file
from governed_banking.parity import CLAIM_SCOPE, PredictionParityConfig
from governed_banking.portable_inference import PortableLoRAPredictor
from governed_banking.runtime_evidence import RuntimeProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARITY_CONFIG = PROJECT_ROOT / "configs/prediction_parity.yaml"


def test_registered_parity_contract_is_hash_bound_and_text_safe() -> None:
    config = PredictionParityConfig.from_yaml(PARITY_CONFIG)

    assert config.reference_backend == "mps"
    assert config.candidate_backend == "cuda"
    assert config.probability_absolute_tolerance == 0.001
    assert config.config_sha256 == sha256_file(PARITY_CONFIG)
    assert tuple(config.runtime_profiles) == ("mps", "cuda")
    for backend, registered in config.runtime_profiles.items():
        assert sha256_file(registered.path) == registered.expected_sha256
        assert RuntimeProfile.from_yaml(registered.path).device_preference == backend


def test_parity_contract_does_not_claim_model_quality() -> None:
    config = PredictionParityConfig.from_yaml(PARITY_CONFIG)

    assert CLAIM_SCOPE == "cross_device_numerical_and_decision_parity_not_model_quality"
    assert "test" not in config.fixture_path.name
    assert config.fixture_path.is_relative_to(config.project_root)


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a host without CUDA")
def test_portable_predictor_fails_before_loading_when_explicit_cuda_is_unavailable() -> None:
    config = PredictionParityConfig.from_yaml(PARITY_CONFIG)
    service = ServiceConfig.from_yaml(config.legacy_service_config_path)
    cuda_profile = RuntimeProfile.from_yaml(config.runtime_profiles["cuda"].path)

    with pytest.raises(RuntimeError, match="no MPS or CPU fallback is permitted"):
        PortableLoRAPredictor(service.predictor, cuda_profile)
