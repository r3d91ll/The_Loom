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
        """Convert to list for JSON serialization."""
        result: list[float] = self.vector.flatten().tolist()
        return result

    def l2_normalize(self) -> HiddenStateResult:
        """Return L2-normalized version (critical for geometric analysis)."""
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
    """Convert hidden state tensors to HiddenStateResult objects.

    Args:
        hidden_states_dict: Mapping of layer index to tensor
        normalize: Whether to L2 normalize the vectors

    Returns:
        Mapping of layer index to HiddenStateResult
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
    """Compute effective dimensionality via PCA.

    D_eff is the primary metric in the Conveyance Framework - it measures
    the semantic richness of representations.

    CRITICAL: L2 normalize embeddings first to prevent magnitude artifacts
    from dominating variance.

    Args:
        embeddings: Array of shape [n_samples, hidden_dim] or [hidden_dim]
        variance_threshold: Cumulative variance threshold (default 0.90)

    Returns:
        Number of dimensions capturing the specified variance
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
    """Compute collapse indicator beta.

    Beta is a DIAGNOSTIC metric (not optimization target) in the
    Conveyance Framework. High beta indicates dimensional collapse.

    Beta = D_eff_input / D_eff_output

    Target: beta < 2.0 (healthy)
    Warning: beta > 2.5 (concerning)

    Args:
        input_d_eff: Effective dimensionality before processing
        output_d_eff: Effective dimensionality after processing

    Returns:
        Beta collapse indicator
    """
    if output_d_eff == 0:
        return float("inf")  # Complete collapse
    return input_d_eff / output_d_eff


def compute_geometric_alignment(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:
    """Compute cosine similarity between two embeddings.

    This measures geometric alignment between agent representations -
    a key metric for bilateral conveyance measurement.

    Args:
        embedding_a: First embedding vector
        embedding_b: Second embedding vector

    Returns:
        Cosine similarity in [-1, 1]
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
    """Compute diagnostic metrics for a hidden state.

    Returns a dictionary of metrics useful for research analysis.
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
