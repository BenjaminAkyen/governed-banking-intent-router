"""Additive cross-platform accelerator runtime for Module 11 and later modules."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

AcceleratorPreference = Literal["auto", "cuda", "mps", "cpu"]


@dataclass(frozen=True)
class AcceleratorMetadata:
    """Serializable real-runtime metadata with explicit memory semantics."""

    requested: str
    selected: str
    device: str
    device_index: int | None
    accelerator_name: str
    real_hardware_observed: bool
    machine: str
    processor: str
    operating_system: str
    operating_system_release: str
    python_version: str
    torch_version: str
    cuda_build_version: str | None
    cudnn_version: int | None
    cuda_available: bool
    cuda_device_count: int
    mps_built: bool
    mps_available: bool
    compute_capability: str | None
    memory_total_bytes: int | None
    memory_free_bytes: int | None
    memory_allocated_bytes: int | None
    memory_reserved_bytes: int | None
    memory_semantics: str

    def to_dict(self) -> dict[str, str | int | bool | None]:
        """Return stable manifest-ready metadata."""

        return asdict(self)


def select_accelerator(
    preferred: AcceleratorPreference | str = "auto",
    *,
    cuda_device_index: int = 0,
) -> tuple[torch.device, AcceleratorMetadata]:
    """Select CUDA, MPS or CPU without silently changing an explicit request."""

    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("preferred must be one of: auto, cuda, mps, cpu")
    if not isinstance(cuda_device_index, int) or isinstance(cuda_device_index, bool):
        raise TypeError("cuda_device_index must be an integer")
    if cuda_device_index < 0:
        raise ValueError("cuda_device_index must be non-negative")

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())

    if preferred == "cuda":
        _require_cuda(cuda_available, cuda_device_count, cuda_device_index)
        selected = "cuda"
    elif preferred == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was explicitly requested but is unavailable; no CPU fallback is permitted"
            )
        selected = "mps"
    elif preferred == "cpu":
        selected = "cpu"
    elif cuda_available:
        _require_cuda(cuda_available, cuda_device_count, cuda_device_index)
        selected = "cuda"
    elif mps_available:
        selected = "mps"
    else:
        selected = "cpu"

    device = (
        torch.device("cuda", cuda_device_index)
        if selected == "cuda"
        else torch.device(selected)
    )
    metadata = _runtime_metadata(
        requested=str(preferred),
        device=device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_built=mps_built,
        mps_available=mps_available,
    )
    return device, metadata


def seed_accelerator(seed: int, device: torch.device) -> None:
    """Seed host libraries and the selected real accelerator."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    _require_selected_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def synchronize_accelerator(device: torch.device) -> None:
    """Wait for work on the selected accelerator; CPU is already synchronous here."""

    _require_selected_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_accelerator_cache(device: torch.device) -> None:
    """Release unoccupied cache owned by the selected accelerator backend."""

    _require_selected_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _require_cuda(available: bool, device_count: int, device_index: int) -> None:
    if not available:
        raise RuntimeError(
            "CUDA was explicitly requested but torch.cuda.is_available() is false; "
            "no MPS or CPU fallback is permitted"
        )
    if device_index >= device_count:
        raise RuntimeError(
            f"CUDA device index {device_index} is unavailable; detected {device_count} device(s)"
        )


def _require_selected_device(device: torch.device) -> None:
    if device.type not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be cuda, mps or cpu")
    if device.type == "cuda":
        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        _require_cuda(bool(torch.cuda.is_available()), count, device.index or 0)
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("selected MPS device is no longer available")


def _runtime_metadata(
    *,
    requested: str,
    device: torch.device,
    cuda_available: bool,
    cuda_device_count: int,
    mps_built: bool,
    mps_available: bool,
) -> AcceleratorMetadata:
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        accelerator_name = str(properties.name)
        compute_capability = f"{properties.major}.{properties.minor}"
        allocated_bytes = int(torch.cuda.memory_allocated(index))
        reserved_bytes = int(torch.cuda.memory_reserved(index))
        memory_semantics = "dedicated_cuda_device_memory"
        cudnn_version = torch.backends.cudnn.version()
    elif device.type == "mps":
        index = 0
        total_bytes = int(torch.mps.recommended_max_memory())
        free_bytes = None
        accelerator_name = _mps_accelerator_name()
        compute_capability = None
        allocated_bytes = int(torch.mps.current_allocated_memory())
        reserved_bytes = int(torch.mps.driver_allocated_memory())
        memory_semantics = "metal_recommended_max_working_set_unified_memory"
        cudnn_version = None
    else:
        index = None
        total_bytes = _physical_memory_bytes()
        free_bytes = None
        accelerator_name = platform.processor() or platform.machine() or "cpu"
        compute_capability = None
        allocated_bytes = None
        reserved_bytes = None
        memory_semantics = "system_physical_memory"
        cudnn_version = None

    return AcceleratorMetadata(
        requested=requested,
        selected=device.type,
        device=str(device),
        device_index=index,
        accelerator_name=accelerator_name,
        real_hardware_observed=True,
        machine=platform.machine(),
        processor=platform.processor(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_build_version=torch.version.cuda,
        cudnn_version=cudnn_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_built=mps_built,
        mps_available=mps_available,
        compute_capability=compute_capability,
        memory_total_bytes=total_bytes,
        memory_free_bytes=free_bytes,
        memory_allocated_bytes=allocated_bytes,
        memory_reserved_bytes=reserved_bytes,
        memory_semantics=memory_semantics,
    )


def _mps_accelerator_name() -> str:
    if platform.system() != "Darwin":
        return platform.processor() or platform.machine() or "mps"
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(result.stdout)
        displays = payload.get("SPDisplaysDataType", [])
        for display in displays:
            if not isinstance(display, dict):
                continue
            name = display.get("sppci_model") or display.get("_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return platform.processor() or platform.machine() or "mps"


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    total = page_size * physical_pages
    return total if total > 0 else None
