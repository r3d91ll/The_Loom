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
        """
        Create a GPUManager configured with which CUDA devices may be used and the maximum fraction of GPU memory to consume.
        
        Parameters:
            allowed_devices (list[int] | None): Specific CUDA device indices to allow. If None, all detected CUDA devices are allowed.
            memory_fraction (float): Fraction (0.0-1.0) of each GPU's memory that the manager should consider available for workloads.
        
        Behavior:
            - If CUDA is not available, logs a warning and sets both `available_devices` and `allowed_devices` to empty lists.
            - When CUDA is available, detects all CUDA devices, populates `available_devices`, and sets `allowed_devices` to the intersection of the provided list and detected devices (or all detected devices if `allowed_devices` is None).
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
        """
        Return whether at least one GPU device is allowed for use.
        
        Returns:
            `True` if there is at least one allowed GPU device, `False` otherwise.
        """
        return len(self.allowed_devices) > 0

    @property
    def default_device(self) -> str:
        """
        Selects the default device used for computations.
        
        Returns:
            str: `"cuda:<index>"` for the first allowed GPU, `"cpu"` if no GPUs are available.
        """
        if self.has_gpu:
            return f"cuda:{self.allowed_devices[0]}"
        return "cpu"

    def get_device(self, device: str | int | None = None) -> torch.device:
        """
        Resolve and validate a device specification against the manager's allowed devices.
        
        Parameters:
            device: Device specification to resolve. If None, the manager's default device is used.
                Accepts an integer index (interpreted as a CUDA device), a device string (e.g., "cuda:0", "cpu"),
                or None. CUDA device selections are validated against the manager's allowed_devices.
        
        Returns:
            torch.device: The resolved torch device.
        
        Raises:
            ValueError: If a specified CUDA device index is not present in allowed_devices.
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
        """
        Retrieve GPU information for a specific allowed device or for all allowed devices.
        
        Args:
            device_idx (int | None): Index of an allowed GPU to query. If None, returns information for all allowed devices.
        
        Returns:
            GPUInfo: Information for the specified device;
            list[GPUInfo]: Information for all allowed devices;
            an empty list if no GPUs are available.
        """
        if not self.has_gpu:
            return []

        if device_idx is not None:
            return self._get_single_gpu_info(device_idx)

        return [self._get_single_gpu_info(idx) for idx in self.allowed_devices]

    def _get_single_gpu_info(self, device_idx: int) -> GPUInfo:
        """
        Return detailed information about a single GPU device identified by index.
        
        Parameters:
            device_idx (int): Index of the CUDA device to query.
        
        Returns:
            GPUInfo: Information for the specified GPU including index, name, total_memory_gb, free_memory_gb, used_memory_gb, utilization_percent (`None` if unavailable), and compute_capability as a (major, minor) tuple.
        """
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
        """
        Selects the best available device based on free GPU memory.
        
        Parameters:
            required_memory_gb (float): Minimum required free memory in gigabytes.
        
        Returns:
            str: Device identifier — `"cpu"` if no GPUs are available, otherwise `"cuda:<index>"`. If no allowed GPU has at least `required_memory_gb` free, returns the first allowed GPU as `"cuda:<index>"`.
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
        """
        Clear CUDA memory cache for a specific device or for all allowed devices.
        
        Parameters:
            device (int | str | None): Device index (e.g., 0), or device string like "cuda:0". If None, clears cache on all allowed devices.
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
        """
        Estimate model memory usage in gigabytes.
        
        Estimates total memory for model parameters plus a typical ~20% overhead for activations/optimizer states based on the number of parameters and parameter data type.
        
        Parameters:
            num_params (int): Total number of model parameters.
            dtype (str): Parameter data type; common values include "float16", "bfloat16", "float32", and "int8".
        
        Returns:
            Estimated memory in gigabytes as a float.
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
        """
        Serialize the GPUManager state and per-GPU information into a dictionary for API responses.
        
        Returns:
            result (dict[str, Any]): Dictionary containing:
                - has_gpu (bool): Whether any GPU is allowed/available.
                - default_device (str): The default device string (e.g., "cuda:0" or "cpu").
                - allowed_devices (list[int]): List of allowed GPU indices.
                - memory_fraction (float): Configured memory fraction for allocations.
                - gpus (list[dict]): Per-GPU dictionaries with keys:
                    - index (int): GPU index.
                    - name (str): GPU name.
                    - total_memory_gb (float): Total memory in gigabytes (rounded to 2 decimals).
                    - free_memory_gb (float): Free memory in gigabytes (rounded to 2 decimals).
                    - used_memory_gb (float): Used memory in gigabytes (rounded to 2 decimals).
                    - compute_capability (str): Compute capability formatted as "major.minor".
        """
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