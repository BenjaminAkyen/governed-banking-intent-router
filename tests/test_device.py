from __future__ import annotations

import pytest
import torch

from governed_banking.device import seed_everything, select_device


def test_auto_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device, metadata = select_device("auto")

    assert device.type == "cpu"
    assert metadata.selected == "cpu"
    assert metadata.mps_built is True
    assert metadata.mps_available is False


def test_explicit_mps_fails_fast_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS was requested"):
        select_device("mps")


def test_cpu_can_be_selected_explicitly() -> None:
    device, metadata = select_device("cpu")

    assert device.type == "cpu"
    assert metadata.requested == "cpu"


def test_invalid_device_preference_is_rejected() -> None:
    with pytest.raises(ValueError, match="preferred"):
        select_device("cuda")


def test_seed_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
