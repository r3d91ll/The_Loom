"""Hidden state extraction and analysis utilities.

This module provides utilities for extracting and processing hidden states
from transformer models - the core capability for conveyance measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class HiddenStateResult:
    """Container for extracted hidden state with metadata."""

    vector: np.ndarray  # The hidden state as numpy array
    shape: tuple[int, ...]
    layer: int  # Which layer (-1 = last)
    dtype: str  # Original dtype as string
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_list(self) -> list[float]:
        """
        Return the hidden state vector as a flat list of floats suitable for JSON serialization.
        
        Returns:
            list[float]: Flattened hidden-state values.
        """
        result: list[float] = self.vector.flatten().tolist()
        return result

    def l2_normalize(self) -> HiddenStateResult:
        """
        Produce a new HiddenStateResult whose vector is scaled to have unit L2 norm.
        
        If the original vector has zero L2 norm, the vector is left unchanged. The returned object's metadata is updated with "normalized": True.
        
        Returns:
            HiddenStateResult: New instance with an L2-normalized vector (or the original vector if its norm is zero) and updated metadata.
        """
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            normalized = self.vector / norm
        else:
            normalized = self.vector
        return HiddenStateResult(
            vector=normalized,
            shape=self.shape,
            layer=self.layer,
            dtype=self.dtype,
            metadata={**self.metadata, "normalized": True},
        )


def extract_hidden_states(
    hidden_states_dict: dict[int, torch.Tensor],
    normalize: bool = False,
) -> dict[int, HiddenStateResult]:
    """
    Convert a mapping of layer indices to tensor/array-like hidden states into HiddenStateResult objects.
    
    Parameters:
        hidden_states_dict (dict[int, torch.Tensor | array-like]): Mapping from layer index to a tensor or array-like hidden state. Each value will be converted to a NumPy array and any leading batch dimension will be removed via squeeze.
        normalize (bool): If True, return L2-normalized vectors in the resulting HiddenStateResult objects.
    
    Returns:
        dict[int, HiddenStateResult]: Mapping from layer index to the corresponding HiddenStateResult containing the (optionally normalized) vector, its shape, layer index, dtype, and metadata.
    """
    results: dict[int, HiddenStateResult] = {}

    for layer_idx, tensor in hidden_states_dict.items():
        # Convert to numpy (convert bfloat16 to float32 first since numpy doesn't support bf16)
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.bfloat16:
                vector = tensor.cpu().float().numpy()
                dtype_str = "float32"
            else:
                vector = tensor.cpu().numpy()
                dtype_str = str(tensor.dtype).replace("torch.", "")
        else:
            vector = np.array(tensor)
            dtype_str = str(vector.dtype)

        result = HiddenStateResult(
            vector=vector.squeeze(),  # Remove batch dimension if present
            shape=tuple(vector.shape),
            layer=layer_idx,
            dtype=dtype_str,
        )

        if normalize:
            result = result.l2_normalize()

        results[layer_idx] = result

    return results


def compute_d_eff(
    embeddings: np.ndarray,
    variance_threshold: float = 0.90,
) -> int:
    """
    Compute the effective dimensionality (D_eff) of a set of embeddings using PCA.
    
    D_eff is the smallest number of principal components whose cumulative variance meets or exceeds the given variance_threshold. Each embedding row is L2-normalized and mean-centered before variance is computed.
    
    Parameters:
        embeddings (np.ndarray): Array of shape [n_samples, hidden_dim] or a 1-D array [hidden_dim].
        variance_threshold (float): Cumulative variance fraction to reach (e.g., 0.90).
    
    Returns:
        int: Number of dimensions required to capture at least `variance_threshold` of the variance (clamped to the feature dimension).
    """
    # Handle single vector case
    if embeddings.ndim == 1:
        # Single vector - return dimension count (can't compute variance)
        return embeddings.shape[0]

    if embeddings.shape[0] < 2:
        # Need at least 2 samples for variance
        return embeddings.shape[1]

    # L2 normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    normalized = embeddings / norms

    # Center the data
    centered = normalized - normalized.mean(axis=0)

    # Compute covariance matrix
    n_samples = centered.shape[0]
    cov = centered.T @ centered / (n_samples - 1)

    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]  # Sort descending

    # Handle numerical issues
    eigenvalues = np.maximum(eigenvalues, 0)

    # Cumulative variance ratio
    total_var = eigenvalues.sum()
    if total_var == 0:
        return 1

    cumvar = np.cumsum(eigenvalues) / total_var

    # Count dimensions below threshold
    d_eff = int(np.searchsorted(cumvar, variance_threshold) + 1)

    return min(d_eff, embeddings.shape[1])


def compute_beta(
    input_d_eff: int,
    output_d_eff: int,
) -> float:
    """
    Compute the collapse indicator beta measuring the relative change in effective dimensionality.
    
    Parameters:
        input_d_eff (int): Effective dimensionality before processing.
        output_d_eff (int): Effective dimensionality after processing.
    
    Returns:
        float: The ratio input_d_eff / output_d_eff; returns `float('inf')` if output_d_eff is 0 to indicate complete collapse.
    """
    if output_d_eff == 0:
        return float("inf")  # Complete collapse
    return input_d_eff / output_d_eff


def compute_geometric_alignment(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """
    Measure geometric alignment between two embeddings by computing their cosine similarity.
    
    Inputs are flattened to 1-D; if either vector has zero L2 norm, the function returns 0.0.
    
    Parameters:
        embedding_a (np.ndarray): First embedding; will be flattened before computation.
        embedding_b (np.ndarray): Second embedding; will be flattened before computation.
    
    Returns:
        float: Cosine similarity in [-1, 1]; `0.0` if either input has zero L2 norm.
    """
    # Flatten if needed
    a = embedding_a.flatten()
    b = embedding_b.flatten()

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def analyze_hidden_state(
    hidden_state: HiddenStateResult,
) -> dict[str, Any]:
    """
    Compute diagnostic statistics for a HiddenStateResult.
    
    Parameters:
        hidden_state (HiddenStateResult): Container holding the hidden-state vector and related metadata.
    
    Returns:
        dict[str, Any]: Mapping of computed metrics including:
            - `shape`: original vector shape.
            - `layer`: layer index.
            - `dtype`: original data type as a string.
            - `mean`, `std`, `min`, `max`: basic summary statistics.
            - `l2_norm`: Euclidean norm of the flattened vector.
            - `sparsity`: fraction of elements with absolute value less than 1e-6.
            - `percentile_25`, `percentile_50`, `percentile_75`: quartile values when the vector contains at least one element.
    """
    vector = hidden_state.vector.flatten()

    # Basic statistics
    analysis = {
        "shape": hidden_state.shape,
        "layer": hidden_state.layer,
        "dtype": hidden_state.dtype,
        "mean": float(np.mean(vector)),
        "std": float(np.std(vector)),
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
        "l2_norm": float(np.linalg.norm(vector)),
        "sparsity": float(np.mean(np.abs(vector) < 1e-6)),  # Fraction near zero
    }

    # Distribution metrics
    if len(vector) > 0:
        analysis["percentile_25"] = float(np.percentile(vector, 25))
        analysis["percentile_50"] = float(np.percentile(vector, 50))
        analysis["percentile_75"] = float(np.percentile(vector, 75))

    return analysis