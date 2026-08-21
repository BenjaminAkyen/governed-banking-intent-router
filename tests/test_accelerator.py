from __future__ import annotations

import pytest
import torch

from governed_banking.accelerator import (
    empty_accelerator_cache,
    seed_accelerator,
    select_accelerator,
    synchronize_accelerator,
)


def test_explicit_cpu_records_real_runtime_metadata() -> None:
    device, metadata = select_accelerator("cpu")

    assert device == torch.device("cpu")
    assert metadata.requested == "cpu"
    assert metadata.selected == "cpu"
    assert metadata.real_hardware_observed is True
    assert metadata.torch_version == torch.__version__
    assert metadata.memory_semantics == "system_physical_memory"
    assert metadata.memory_total_bytes is None or metadata.memory_total_bytes > 0


def test_auto_uses_real_cuda_then_mps_then_cpu_priority() -> None:
    device, metadata = select_accelerator("auto")

    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"
    assert metadata.selected == device.type
    assert metadata.real_hardware_observed is True


def test_explicit_cuda_selects_real_device_or_fails_without_fallback() -> None:
    if torch.cuda.is_available():
        device, metadata = select_accelerator("cuda")
        assert device.type == "cuda"
        assert metadata.selected == "cuda"
        assert metadata.accelerator_name
        assert metadata.memory_total_bytes is not None
        assert metadata.memory_total_bytes > 0
    else:
        with pytest.raises(RuntimeError, match="no MPS or CPU fallback"):
            select_accelerator("cuda")


def test_explicit_mps_selects_real_device_or_fails_without_fallback() -> None:
    if torch.backends.mps.is_available():
        device, metadata = select_accelerator("mps")
        assert device.type == "mps"
        assert metadata.selected == "mps"
        assert metadata.accelerator_name
        assert metadata.memory_total_bytes is not None
        assert metadata.memory_total_bytes > 0
    else:
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            select_accelerator("mps")


def test_runtime_operations_dispatch_only_to_real_selected_devices() -> None:
    available_preferences = ["cpu"]
    if torch.cuda.is_available():
        available_preferences.append("cuda")
    if torch.backends.mps.is_available():
        available_preferences.append("mps")

    for preference in available_preferences:
        device, _ = select_accelerator(preference)
        seed_accelerator(42, device)
        synchronize_accelerator(device)
        empty_accelerator_cache(device)


@pytest.mark.parametrize("preferred", ["gpu", "metal", "CUDA", ""])
def test_invalid_preference_is_rejected(preferred: str) -> None:
    with pytest.raises(ValueError, match="auto, cuda, mps, cpu"):
        select_accelerator(preferred)


def test_invalid_seed_and_cuda_index_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        seed_accelerator(-1, torch.device("cpu"))
    with pytest.raises(ValueError, match="non-negative"):
        select_accelerator("cpu", cuda_device_index=-1)
