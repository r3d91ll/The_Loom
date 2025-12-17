"""Tests for hidden state extraction utilities."""

import numpy as np
import pytest
import torch

from src.extraction.hidden_states import (
    HiddenStateResult,
    analyze_hidden_state,
    compute_beta,
    compute_d_eff,
    compute_geometric_alignment,
    extract_hidden_states,
)


class TestHiddenStateResult:
    """Tests for HiddenStateResult dataclass."""

    def test_creation(self):
        vector = np.array([1.0, 2.0, 3.0, 4.0])
        result = HiddenStateResult(
            vector=vector,
            shape=(4,),
            layer=-1,
            dtype="float32",
        )
        assert result.layer == -1
        assert result.shape == (4,)
        np.testing.assert_array_equal(result.vector, vector)

    def test_to_list(self):
        vector = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = HiddenStateResult(
            vector=vector,
            shape=(2, 2),
            layer=-1,
            dtype="float32",
        )
        assert result.to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_l2_normalize(self):
        vector = np.array([3.0, 4.0])  # Norm = 5
        result = HiddenStateResult(
            vector=vector,
            shape=(2,),
            layer=-1,
            dtype="float32",
        )
        normalized = result.l2_normalize()

        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(normalized.vector, expected)
        assert normalized.metadata.get("normalized") is True

    def test_l2_normalize_zero_vector(self):
        vector = np.array([0.0, 0.0])
        result = HiddenStateResult(
            vector=vector,
            shape=(2,),
            layer=-1,
            dtype="float32",
        )
        normalized = result.l2_normalize()
        np.testing.assert_array_equal(normalized.vector, vector)


class TestExtractHiddenStates:
    """Tests for extract_hidden_states function."""

    def test_extract_from_torch_tensor(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0]])
        hidden_dict = {-1: tensor}

        results = extract_hidden_states(hidden_dict)

        assert -1 in results
        assert results[-1].layer == -1
        np.testing.assert_array_almost_equal(results[-1].vector, [1.0, 2.0, 3.0])

    def test_extract_with_normalization(self):
        tensor = torch.tensor([[3.0, 4.0]])
        hidden_dict = {-1: tensor}

        results = extract_hidden_states(hidden_dict, normalize=True)

        expected = np.array([0.6, 0.8])
        np.testing.assert_array_almost_equal(results[-1].vector, expected)

    def test_extract_multiple_layers(self):
        hidden_dict = {
            -1: torch.tensor([[1.0, 0.0]]),
            -2: torch.tensor([[0.0, 1.0]]),
        }

        results = extract_hidden_states(hidden_dict)

        assert len(results) == 2
        assert -1 in results
        assert -2 in results


class TestComputeDEff:
    """Tests for effective dimensionality computation."""

    def test_single_vector_returns_dim(self):
        vector = np.random.randn(256)
        d_eff = compute_d_eff(vector)
        assert d_eff == 256

    def test_single_sample_returns_dim(self):
        embeddings = np.random.randn(1, 128)
        d_eff = compute_d_eff(embeddings)
        assert d_eff == 128

    def test_uniform_variance(self):
        # When variance is uniform across all dimensions, D_eff should be high
        # Note: compute_d_eff L2-normalizes data, which projects points onto
        # a unit sphere and changes variance structure, so we use a lower threshold
        np.random.seed(42)
        n_samples = 100
        n_dims = 50
        embeddings = np.random.randn(n_samples, n_dims)

        d_eff = compute_d_eff(embeddings)
        # After L2 normalization, D_eff is lower than raw dimensions
        # but should still capture significant variance
        assert d_eff > 25

    def test_collapsed_dimensions(self):
        # When data lies in a low-dimensional subspace, D_eff should be low
        np.random.seed(42)
        n_samples = 100
        n_dims = 50

        # Create data in a 5D subspace embedded in 50D
        low_dim = np.random.randn(n_samples, 5)
        projection = np.random.randn(5, n_dims)
        embeddings = low_dim @ projection

        d_eff = compute_d_eff(embeddings)
        # D_eff should be close to actual dimensionality (5)
        assert d_eff <= 10

    def test_variance_threshold(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 100)

        d_eff_90 = compute_d_eff(embeddings, variance_threshold=0.90)
        d_eff_99 = compute_d_eff(embeddings, variance_threshold=0.99)

        # Higher threshold should require more dimensions
        assert d_eff_99 >= d_eff_90


class TestComputeBeta:
    """Tests for beta collapse indicator."""

    def test_no_collapse(self):
        # Input and output have same D_eff
        beta = compute_beta(input_d_eff=50, output_d_eff=50)
        assert beta == 1.0

    def test_moderate_collapse(self):
        # Output has half the D_eff
        beta = compute_beta(input_d_eff=50, output_d_eff=25)
        assert beta == 2.0

    def test_severe_collapse(self):
        beta = compute_beta(input_d_eff=100, output_d_eff=10)
        assert beta == 10.0

    def test_zero_output(self):
        beta = compute_beta(input_d_eff=50, output_d_eff=0)
        assert beta == float("inf")


class TestGeometricAlignment:
    """Tests for geometric alignment (cosine similarity)."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        alignment = compute_geometric_alignment(a, a)
        assert alignment == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        alignment = compute_geometric_alignment(a, b)
        assert alignment == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        alignment = compute_geometric_alignment(a, b)
        assert alignment == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        alignment = compute_geometric_alignment(a, b)
        assert alignment == 0.0


class TestAnalyzeHiddenState:
    """Tests for hidden state analysis."""

    def test_basic_analysis(self):
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = HiddenStateResult(
            vector=vector,
            shape=(5,),
            layer=-1,
            dtype="float32",
        )

        analysis = analyze_hidden_state(result)

        assert analysis["shape"] == (5,)
        assert analysis["layer"] == -1
        assert analysis["mean"] == pytest.approx(3.0)
        assert analysis["min"] == pytest.approx(1.0)
        assert analysis["max"] == pytest.approx(5.0)
        assert "l2_norm" in analysis
        assert "sparsity" in analysis

    def test_percentiles(self):
        vector = np.arange(100).astype(float)
        result = HiddenStateResult(
            vector=vector,
            shape=(100,),
            layer=-1,
            dtype="float32",
        )

        analysis = analyze_hidden_state(result)

        assert "percentile_25" in analysis
        assert "percentile_50" in analysis
        assert "percentile_75" in analysis
