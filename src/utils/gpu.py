"""GPU device management utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_percent: float | None  # Requires pynvml for accurate reading
    compute_capability: tuple[int, int]


class GPUManager:
    """Manages GPU device selection and memory.

    For research workloads (1-6 agents), we optimize for:
    - Fast model swapping
    - Predictable memory usage
    - Easy device selection
    """

    def __init__(
        self,
        allowed_devices: list[int] | None = None,
        memory_fraction: float = 0.9,
    ):
        """Initialize GPU manager.

        Args:
            allowed_devices: List of CUDA device indices to use (None = all)
            memory_fraction: Maximum fraction of GPU memory to use
        """
        self.memory_fraction = memory_fraction

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - running on CPU")
            self.available_devices: list[int] = []
            self.allowed_devices: list[int] = []
            return

        # Get available devices
        num_devices = torch.cuda.device_count()
        self.available_devices = list(range(num_devices))

        # Filter to allowed devices
        if allowed_devices is not None:
            self.allowed_devices = [d for d in allowed_devices if d in self.available_devices]
        else:
            self.allowed_devices = self.available_devices.copy()

        logger.info(f"GPU Manager initialized: {len(self.allowed_devices)} devices available")

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return len(self.allowed_devices) > 0

    @property
    def default_device(self) -> str:
        """Get the default device string."""
        if self.has_gpu:
            return f"cuda:{self.allowed_devices[0]}"
        return "cpu"

    def get_device(self, device: str | int | None = None) -> torch.device:
        """Get a torch device, validating it's allowed.

        Args:
            device: Device specification (None = default)
                - None: Use default device
                - int: CUDA device index
                - str: "cuda:0", "cpu", etc.

        Returns:
            torch.device object
        """
        if device is None:
            return torch.device(self.default_device)

        if isinstance(device, int):
            if device not in self.allowed_devices:
                raise ValueError(f"Device {device} not in allowed devices: {self.allowed_devices}")
            return torch.device(f"cuda:{device}")

        # Parse string device
        torch_device = torch.device(device)

        if torch_device.type == "cuda":
            idx = torch_device.index or 0
            if idx not in self.allowed_devices:
                raise ValueError(
                    f"Device cuda:{idx} not in allowed devices: {self.allowed_devices}"
                )

        return torch_device

    def get_gpu_info(self, device_idx: int | None = None) -> GPUInfo | list[GPUInfo]:
        """Get information about GPU(s).

        Args:
            device_idx: Specific device index, or None for all allowed devices

        Returns:
            GPUInfo for single device, or list for all devices
        """
        if not self.has_gpu:
            return []

        if device_idx is not None:
            return self._get_single_gpu_info(device_idx)

        return [self._get_single_gpu_info(idx) for idx in self.allowed_devices]

    def _get_single_gpu_info(self, device_idx: int) -> GPUInfo:
        """Get info for a single GPU."""
        props = torch.cuda.get_device_properties(device_idx)

        # Get memory info
        torch.cuda.set_device(device_idx)
        total_memory = props.total_memory / (1024**3)  # Convert to GB
        free_memory, total_memory_check = torch.cuda.mem_get_info(device_idx)
        free_memory_gb = free_memory / (1024**3)
        used_memory_gb = total_memory - free_memory_gb

        return GPUInfo(
            index=device_idx,
            name=props.name,
            total_memory_gb=total_memory,
            free_memory_gb=free_memory_gb,
            used_memory_gb=used_memory_gb,
            utilization_percent=None,  # Would need pynvml
            compute_capability=(props.major, props.minor),
        )

    def get_best_device(self, required_memory_gb: float = 0) -> str:
        """Get the best available device based on free memory.

        Args:
            required_memory_gb: Minimum required free memory

        Returns:
            Device string (e.g., "cuda:0")
        """
        if not self.has_gpu:
            return "cpu"

        best_device = None
        best_free_memory: float = -1.0

        for idx in self.allowed_devices:
            info = self._get_single_gpu_info(idx)

            # Check if enough memory
            if info.free_memory_gb >= required_memory_gb:
                if info.free_memory_gb > best_free_memory:
                    best_free_memory = info.free_memory_gb
                    best_device = idx

        if best_device is None:
            logger.warning(f"No GPU with {required_memory_gb}GB free memory, using first available")
            return f"cuda:{self.allowed_devices[0]}"

        return f"cuda:{best_device}"

    def clear_cache(self, device: int | str | None = None) -> None:
        """Clear CUDA cache for device(s).

        Args:
            device: Specific device, or None for all
        """
        if not self.has_gpu:
            return

        if device is None:
            for idx in self.allowed_devices:
                with torch.cuda.device(idx):
                    torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache on all devices")
        else:
            if isinstance(device, str):
                device = int(device.split(":")[-1])
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache on device {device}")

    def estimate_model_memory(
        self,
        num_params: int,
        dtype: str = "float16",
    ) -> float:
        """Estimate memory required for a model.

        This is a rough estimate - actual memory varies by architecture.

        Args:
            num_params: Number of model parameters
            dtype: Data type (float16, bfloat16, float32)

        Returns:
            Estimated memory in GB
        """
        bytes_per_param = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "int8": 1,
        }

        param_bytes = bytes_per_param.get(dtype, 2)
        base_memory = num_params * param_bytes / (1024**3)

        # Add ~20% overhead for activations, optimizer states, etc.
        return base_memory * 1.2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        gpu_list = self.get_gpu_info() if self.has_gpu else []
        if isinstance(gpu_list, GPUInfo):
            gpu_list = [gpu_list]

        return {
            "has_gpu": self.has_gpu,
            "default_device": self.default_device,
            "allowed_devices": self.allowed_devices,
            "memory_fraction": self.memory_fraction,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "total_memory_gb": round(gpu.total_memory_gb, 2),
                    "free_memory_gb": round(gpu.free_memory_gb, 2),
                    "used_memory_gb": round(gpu.used_memory_gb, 2),
                    "compute_capability": f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                }
                for gpu in gpu_list
            ],
        }
