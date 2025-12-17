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
    """
    Encode a tensor or NumPy array as a base64 ASCII string of its raw bytes after casting to a specified dtype.
    
    Parameters:
        tensor (torch.Tensor | numpy.ndarray): Input tensor or array to encode.
        dtype (str): NumPy dtype name to cast the array to before encoding (e.g., "float32"). Defaults to "float32".
    
    Returns:
        str: Base64-encoded ASCII string containing the array's raw bytes after casting to `dtype`.
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
    """
    Convert a base64-encoded buffer into a NumPy array with the given shape and dtype.
    
    Parameters:
        encoded (str): Base64-encoded ASCII string containing the array bytes.
        shape (tuple[int, ...]): Desired shape of the resulting array.
        dtype (str): NumPy dtype name for interpreting the bytes (e.g., "float32"). Defaults to "float32".
    
    Returns:
        np.ndarray: Array reshaped to `shape` and having the specified `dtype`.
    """
    decoded = base64.b64decode(encoded)
    arr = np.frombuffer(decoded, dtype=np.dtype(dtype))
    return arr.reshape(shape)


def serialize_hidden_states(
    hidden_states: Mapping[int, HiddenStateResult | torch.Tensor | np.ndarray],
    format: str = "list",
) -> dict[str, Any]:
    """
    Serialize a mapping of layer indices to hidden-state tensors or arrays into a JSON-friendly dictionary.
    
    Parameters:
        hidden_states (Mapping[int, HiddenStateResult | torch.Tensor | np.ndarray]):
            Mapping from layer index to a hidden state. Each value may be a HiddenStateResult (providing .vector, .shape, .dtype),
            a torch.Tensor, or any array-like object convertible to a NumPy array.
        format (str):
            Serialization format: "list" to produce a flattened Python list of numbers under the "data" key, or "base64" to produce a
            base64-encoded string of the tensor bytes (dtype "float32") with an additional "encoding": "base64" field.
    
    Returns:
        dict[str, Any]:
            Dictionary keyed by stringified layer indices. Each value is a dict containing:
              - "shape": list of integers describing the array shape,
              - "dtype": the data type string,
              - "data": either a flattened list of numbers (for "list") or a base64 string (for "base64").
            When "base64" format is used, the returned dtype will be "float32" and "encoding" will be "base64".
    
    Raises:
        ValueError: If `format` is not "list" or "base64".
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
    """
    Convert a JSON-serializable mapping of serialized hidden states into a mapping of layer indices to NumPy arrays.
    
    Parameters:
        data (dict[str, Any]): Mapping where keys are stringified layer indices and values are dicts with at least:
            - "shape": list/tuple of ints describing the array shape
            - "data": either a flat list of values or a base64-encoded string
            Optional fields:
            - "dtype": NumPy dtype as string (defaults to "float32")
            - "encoding": if set to "base64", "data" is treated as a base64-encoded byte buffer
    
    Returns:
        dict[int, numpy.ndarray]: Mapping from integer layer index to a NumPy array reconstructed with the provided shape and dtype.
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