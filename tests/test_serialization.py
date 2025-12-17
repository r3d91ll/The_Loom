"""Tests for tensor serialization utilities."""

import numpy as np
import pytest
import torch

from src.extraction.hidden_states import HiddenStateResult
from src.utils.serialization import (
    base64_to_array,
    deserialize_hidden_states,
    serialize_hidden_states,
    tensor_to_base64,
    tensor_to_list,
)


class TestTensorToList:
    """Tests for tensor_to_list function."""

    def test_torch_tensor_1d(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_list(tensor)
        assert result == [1.0, 2.0, 3.0]

    def test_torch_tensor_2d(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_to_list(tensor)
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = tensor_to_list(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_cuda_tensor(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        result = tensor_to_list(tensor)
        assert result == [1.0, 2.0, 3.0]


class TestBase64Encoding:
    """Tests for base64 encoding/decoding."""

    def test_roundtrip_1d(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = tensor_to_base64(original)
        decoded = base64_to_array(encoded, (3,), "float32")
        np.testing.assert_array_almost_equal(original, decoded)

    def test_roundtrip_2d(self):
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        encoded = tensor_to_base64(original)
        decoded = base64_to_array(encoded, (2, 2), "float32")
        np.testing.assert_array_almost_equal(original, decoded)

    def test_roundtrip_torch_tensor(self):
        original = torch.tensor([1.0, 2.0, 3.0, 4.0])
        encoded = tensor_to_base64(original)
        decoded = base64_to_array(encoded, (4,), "float32")
        np.testing.assert_array_almost_equal(original.numpy(), decoded)

    def test_large_tensor(self):
        original = np.random.randn(1000).astype(np.float32)
        encoded = tensor_to_base64(original)
        decoded = base64_to_array(encoded, (1000,), "float32")
        np.testing.assert_array_almost_equal(original, decoded)


class TestSerializeHiddenStates:
    """Tests for serialize_hidden_states function."""

    def test_serialize_list_format(self):
        hidden_states = {
            -1: HiddenStateResult(
                vector=np.array([1.0, 2.0, 3.0]),
                shape=(3,),
                layer=-1,
                dtype="float32",
            ),
        }

        result = serialize_hidden_states(hidden_states, format="list")

        assert "-1" in result
        assert result["-1"]["data"] == [1.0, 2.0, 3.0]
        assert result["-1"]["shape"] == [3]
        assert result["-1"]["dtype"] == "float32"

    def test_serialize_base64_format(self):
        hidden_states = {
            -1: HiddenStateResult(
                vector=np.array([1.0, 2.0, 3.0]),
                shape=(3,),
                layer=-1,
                dtype="float32",
            ),
        }

        result = serialize_hidden_states(hidden_states, format="base64")

        assert "-1" in result
        assert isinstance(result["-1"]["data"], str)
        assert result["-1"]["encoding"] == "base64"

    def test_serialize_torch_tensor(self):
        hidden_states = {
            -1: torch.tensor([1.0, 2.0, 3.0]),
        }

        result = serialize_hidden_states(hidden_states, format="list")

        assert "-1" in result
        assert result["-1"]["data"] == [1.0, 2.0, 3.0]

    def test_serialize_multiple_layers(self):
        hidden_states = {
            -1: np.array([1.0, 2.0]),
            -2: np.array([3.0, 4.0]),
        }

        result = serialize_hidden_states(hidden_states, format="list")

        assert "-1" in result
        assert "-2" in result


class TestDeserializeHiddenStates:
    """Tests for deserialize_hidden_states function."""

    def test_deserialize_list_format(self):
        data = {
            "-1": {
                "data": [1.0, 2.0, 3.0],
                "shape": [3],
                "dtype": "float32",
            },
        }

        result = deserialize_hidden_states(data)

        assert -1 in result
        np.testing.assert_array_almost_equal(result[-1], np.array([1.0, 2.0, 3.0]))

    def test_deserialize_base64_format(self):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = tensor_to_base64(original)

        data = {
            "-1": {
                "data": encoded,
                "shape": [3],
                "dtype": "float32",
                "encoding": "base64",
            },
        }

        result = deserialize_hidden_states(data)

        assert -1 in result
        np.testing.assert_array_almost_equal(result[-1], original)

    def test_roundtrip(self):
        original = {
            -1: HiddenStateResult(
                vector=np.array([1.0, 2.0, 3.0, 4.0]),
                shape=(4,),
                layer=-1,
                dtype="float32",
            ),
        }

        serialized = serialize_hidden_states(original, format="list")
        deserialized = deserialize_hidden_states(serialized)

        np.testing.assert_array_almost_equal(original[-1].vector, deserialized[-1])
