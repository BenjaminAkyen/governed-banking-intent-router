"""Runtime device selection for Apple Silicon and CPU environments."""

from __future__ import annotations

import platform
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeDevice:
    """Serializable description of the selected compute device."""

    requested: str
    selected: str
    mps_built: bool
    mps_available: bool
    torch_version: str
    python_version: str
    platform: str

    def to_dict(self) -> dict[str, str | bool]:
        """Return metadata suitable for an experiment manifest."""

        return asdict(self)


def select_device(preferred: str = "auto") -> tuple[torch.device, RuntimeDevice]:
    """Select MPS when available, otherwise fall back to CPU.

    Explicitly requesting MPS fails fast instead of silently running an expensive
    experiment on CPU. The automatic mode is suitable for notebooks and CI.
    """

    if preferred not in {"auto", "mps", "cpu"}:
        raise ValueError("preferred must be one of: auto, mps, cpu")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()

    if preferred == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested but is unavailable. Verify the Apple Silicon Python "
            "environment, macOS version and PyTorch installation."
        )

    selected = "mps" if preferred in {"auto", "mps"} and mps_available else "cpu"
    metadata = RuntimeDevice(
        requested=preferred,
        selected=selected,
        mps_built=mps_built,
        mps_available=mps_available,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    return torch.device(selected), metadata


def seed_everything(seed: int) -> None:
    """Seed supported random number generators for repeatable experiments."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
