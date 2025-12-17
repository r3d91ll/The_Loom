"""Tensor serialization utilities for JSON responses."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ..extraction.hidden_states import HiddenStateResult


def tensor_to_list(tensor: torch.Tensor | np.ndarray) -> list[float]:
    """Convert tensor to list for JSON serialization.

    Args:
        tensor: PyTorch tensor or numpy array

    Returns:
        Flattened list of floats
    """
    if isinstance(tensor, torch.Tensor):
        # Convert bfloat16 to float32 first since numpy doesn't support bf16
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        result: list[float] = tensor.cpu().detach().numpy().flatten().tolist()
        return result
    result_np: list[float] = np.asarray(tensor).flatten().tolist()
    return result_np


def tensor_to_base64(tensor: torch.Tensor | np.ndarray, dtype: str = "float32") -> str:
    """Convert tensor to base64-encoded bytes.

    More efficient for large tensors than JSON lists.

    Args:
        tensor: PyTorch tensor or numpy array
        dtype: Target dtype for encoding

    Returns:
        Base64-encoded string
    """
    if isinstance(tensor, torch.Tensor):
        # Convert bfloat16 to float32 first since numpy doesn't support bf16
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        arr = tensor.cpu().detach().numpy()
    else:
        arr = np.asarray(tensor)

    # Convert to specified dtype
    arr = arr.astype(np.dtype(dtype))

    # Encode to base64
    return base64.b64encode(arr.tobytes()).decode("ascii")


def base64_to_array(encoded: str, shape: tuple[int, ...], dtype: str = "float32") -> np.ndarray:
    """Decode base64-encoded tensor back to numpy array.

    Args:
        encoded: Base64-encoded string
        shape: Target shape for the array
        dtype: Data type of the encoded data

    Returns:
        Numpy array with specified shape
    """
    decoded = base64.b64decode(encoded)
    arr = np.frombuffer(decoded, dtype=np.dtype(dtype))
    return arr.reshape(shape)


def serialize_hidden_states(
    hidden_states: Mapping[int, HiddenStateResult | torch.Tensor | np.ndarray],
    format: str = "list",
) -> dict[str, Any]:
    """Serialize hidden states for JSON response.

    Args:
        hidden_states: Mapping of layer index to hidden state
        format: Serialization format ("list" or "base64")

    Returns:
        JSON-serializable dictionary
    """
    result: dict[str, Any] = {}

    for layer_idx, state in hidden_states.items():
        layer_key = str(layer_idx)  # JSON keys must be strings

        if isinstance(state, HiddenStateResult):
            vector = state.vector
            shape = state.shape
            dtype = state.dtype
        elif isinstance(state, torch.Tensor):
            # Convert bfloat16 to float32 first since numpy doesn't support bf16
            if state.dtype == torch.bfloat16:
                vector = state.cpu().detach().float().numpy()
                dtype = "float32"
            else:
                vector = state.cpu().detach().numpy()
                dtype = str(state.dtype).replace("torch.", "")
            shape = tuple(vector.shape)
        else:
            vector = np.asarray(state)
            shape = tuple(vector.shape)
            dtype = str(vector.dtype)

        if format == "list":
            result[layer_key] = {
                "data": vector.flatten().tolist(),
                "shape": list(shape),
                "dtype": dtype,
            }
        elif format == "base64":
            # Use float32 for base64 to ensure compatibility
            result[layer_key] = {
                "data": tensor_to_base64(vector, "float32"),
                "shape": list(shape),
                "dtype": "float32",
                "encoding": "base64",
            }
        else:
            raise ValueError(f"Unknown format: {format}. Use: list, base64")

    return result


def deserialize_hidden_states(
    data: dict[str, Any],
) -> dict[int, np.ndarray]:
    """Deserialize hidden states from JSON response.

    Args:
        data: Serialized hidden states dictionary

    Returns:
        Mapping of layer index to numpy array
    """
    result: dict[int, np.ndarray] = {}

    for layer_key, state_data in data.items():
        layer_idx = int(layer_key)
        shape = tuple(state_data["shape"])
        dtype = state_data.get("dtype", "float32")

        if state_data.get("encoding") == "base64":
            arr = base64_to_array(state_data["data"], shape, dtype)
        else:
            arr = np.array(state_data["data"], dtype=np.dtype(dtype)).reshape(shape)

        result[layer_idx] = arr

    return result
