"""Integration tests with real models.

These tests actually load models and validate hidden state extraction.
Run with: pytest tests/test_integration.py -v -s

Requires:
- GPU with sufficient VRAM (8GB+ recommended)
- Models will be downloaded on first run

Mark as slow to skip in normal test runs:
    pytest -m "not integration"
"""

import json
import time
from pathlib import Path

import pytest
import torch

# Skip all tests in this module if no GPU available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration tests"),
]

# Test models - using smaller variants for faster testing
TEST_MODELS = {
    "embedding": "BAAI/bge-small-en-v1.5",  # 33M params, ~130MB
    "generative_small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params
}

# Directory for saving example outputs
EXAMPLES_DIR = Path(__file__).parent.parent / "examples" / "outputs"


@pytest.fixture(scope="module")
def examples_dir():
    """Create examples directory for saving outputs."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_DIR


@pytest.fixture(scope="module")
def app_client():
    """Create test client with real model loading.

    Cleanup is handled in fixture teardown to avoid relying on test execution order.
    """
    from fastapi.testclient import TestClient
    from src.transport.http import create_http_app
    from src.config import Config

    config = Config()
    app = create_http_app(config)

    with TestClient(app) as client:
        yield client

        # Teardown: unload all models after tests complete
        print("\n=== Fixture Cleanup: Unloading Models ===")
        try:
            response = client.get("/models")
            if response.status_code == 200:
                data = response.json()
                for model in data.get("loaded_models", []):
                    model_id = model["model_id"].replace("/", "--")
                    client.delete(f"/models/{model_id}")
                    print(f"  Unloaded: {model['model_id']}")
            print("=== Cleanup Complete ===")
        except Exception as e:
            print(f"  Cleanup warning: {e}")


class TestHealthAndSetup:
    """Basic health checks before model tests."""

    def test_health_endpoint(self, app_client):
        """Verify server health endpoint works."""
        response = app_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_info" in data

        print("\n=== Health Check ===")
        print(json.dumps(data, indent=2))

    def test_list_loaders(self, app_client):
        """Verify loader registry is populated."""
        response = app_client.get("/loaders")
        assert response.status_code == 200

        data = response.json()
        assert "loaders" in data
        assert "fallback_order" in data
        assert len(data["fallback_order"]) >= 3

        print("\n=== Available Loaders ===")
        print(json.dumps(data, indent=2))


class TestEmbeddingModel:
    """Test hidden state extraction with embedding models."""

    @pytest.fixture(scope="class")
    def loaded_embedding_model(self, app_client):
        """Load embedding model once for all tests in class."""
        model_id = TEST_MODELS["embedding"]

        print(f"\n=== Loading Embedding Model: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_embed_basic(self, app_client, loaded_embedding_model, examples_dir):
        """Test basic embedding extraction."""
        model_id = TEST_MODELS["embedding"]

        response = app_client.post("/embed", json={
            "model": model_id,
            "text": "The quick brown fox jumps over the lazy dog.",
            "pooling": "mean",
            "normalize": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "embedding" in data
        assert "shape" in data
        assert "metadata" in data

        # Validate embedding
        embedding = data["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Validate shape matches embedding length
        assert data["shape"][-1] == len(embedding)

        # Validate normalization (L2 norm should be ~1.0)
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        assert 0.99 < norm < 1.01, f"Expected normalized vector, got norm={norm}"

        print("\n=== Embedding Output ===")
        print(f"Shape: {data['shape']}")
        print(f"Embedding (first 10): {embedding[:10]}")
        print(f"L2 norm: {norm:.6f}")
        print(f"Metadata: {data['metadata']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "text": "The quick brown fox jumps over the lazy dog.",
                "pooling": "mean",
                "normalize": True,
            },
            "response": data,
            "validation": {
                "embedding_length": len(embedding),
                "l2_norm": norm,
                "is_normalized": True,
            }
        }
        with open(examples_dir / "embed_basic.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_analyze_endpoint(self, app_client, loaded_embedding_model, examples_dir):
        """Test analyze endpoint with D_eff computation."""
        model_id = TEST_MODELS["embedding"]

        response = app_client.post("/analyze", json={
            "model": model_id,
            "text": "Machine learning models can extract semantic representations.",
            "pooling": "mean",
        })

        assert response.status_code == 200
        data = response.json()

        # Validate analysis fields
        assert "mean" in data
        assert "std" in data
        assert "min" in data
        assert "max" in data
        assert "embedding_shape" in data

        print("\n=== Analysis Output ===")
        print(json.dumps(data, indent=2))

        # Save example
        with open(examples_dir / "analyze_output.json", "w") as f:
            json.dump(data, f, indent=2)


class TestGenerativeModel:
    """Test hidden state extraction with generative models."""

    @pytest.fixture(scope="class")
    def loaded_generative_model(self, app_client):
        """Load generative model once for all tests in class."""
        model_id = TEST_MODELS["generative_small"]

        print(f"\n=== Loading Generative Model: {model_id} ===")
        start = time.time()

        response = app_client.post("/models/load", json={
            "model": model_id,
            "dtype": "float16",
        })

        load_time = time.time() - start
        assert response.status_code == 200, f"Failed to load model: {response.text}"

        data = response.json()
        print(f"Loaded in {load_time:.2f}s")
        print(json.dumps(data, indent=2))

        yield data

        # Cleanup
        app_client.delete(f"/models/{model_id.replace('/', '--')}")

    def test_generate_with_hidden_states(self, app_client, loaded_generative_model, examples_dir):
        """Test generation with hidden state extraction."""
        model_id = TEST_MODELS["generative_small"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "The meaning of life is",
            "max_tokens": 20,
            "temperature": 0.7,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "text" in data
        assert "token_count" in data
        assert "hidden_states" in data
        assert "metadata" in data

        # Validate hidden states
        hidden_states = data["hidden_states"]
        assert hidden_states is not None
        assert "-1" in hidden_states

        layer_data = hidden_states["-1"]
        assert "data" in layer_data
        assert "shape" in layer_data
        assert "dtype" in layer_data

        # Validate shape matches model hidden size
        expected_hidden_size = loaded_generative_model["hidden_size"]
        assert layer_data["shape"][-1] == expected_hidden_size

        print("\n=== Generation with Hidden States ===")
        print(f"Generated text: {data['text']}")
        print(f"Token count: {data['token_count']}")
        print(f"Hidden state shape: {layer_data['shape']}")
        print(f"Hidden state dtype: {layer_data['dtype']}")
        print(f"Hidden state (first 10 values): {layer_data['data'][:10]}")
        print(f"Metadata: {data['metadata']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "prompt": "The meaning of life is",
                "max_tokens": 20,
                "temperature": 0.7,
                "return_hidden_states": True,
                "hidden_state_layers": [-1],
            },
            "response": {
                "text": data["text"],
                "token_count": data["token_count"],
                "hidden_states": {
                    "-1": {
                        "shape": layer_data["shape"],
                        "dtype": layer_data["dtype"],
                        "data_preview": layer_data["data"][:20],
                        "data_length": len(layer_data["data"]),
                    }
                },
                "metadata": data["metadata"],
            },
            "validation": {
                "hidden_size_matches": layer_data["shape"][-1] == expected_hidden_size,
                "expected_hidden_size": expected_hidden_size,
            }
        }
        with open(examples_dir / "generate_hidden_states.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_multiple_layers(self, app_client, loaded_generative_model, examples_dir):
        """Test extraction from multiple layers."""
        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_generative_model["num_layers"]

        # Request last 3 layers
        layers_to_request = [-1, -2, -3]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Hello, how are you?",
            "max_tokens": 10,
            "temperature": 0.5,
            "return_hidden_states": True,
            "hidden_state_layers": layers_to_request,
        })

        assert response.status_code == 200
        data = response.json()

        hidden_states = data["hidden_states"]
        assert hidden_states is not None

        # Validate all requested layers present
        for layer_idx in layers_to_request:
            layer_key = str(layer_idx)
            assert layer_key in hidden_states, f"Layer {layer_idx} missing from response"

            layer_data = hidden_states[layer_key]
            assert "data" in layer_data
            assert "shape" in layer_data

        print("\n=== Multiple Layer Extraction ===")
        print(f"Requested layers: {layers_to_request}")
        print(f"Model has {num_layers} layers")
        for layer_idx in layers_to_request:
            layer_data = hidden_states[str(layer_idx)]
            print(f"  Layer {layer_idx}: shape={layer_data['shape']}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "hidden_state_layers": layers_to_request,
            },
            "response": {
                "layers_returned": list(hidden_states.keys()),
                "layer_shapes": {k: v["shape"] for k, v in hidden_states.items()},
            },
            "validation": {
                "all_layers_present": all(str(l) in hidden_states for l in layers_to_request),
                "model_num_layers": num_layers,
            }
        }
        with open(examples_dir / "generate_multiple_layers.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_all_layers(self, app_client, loaded_generative_model, examples_dir):
        """Test extraction from ALL layers using 'all' keyword."""
        model_id = TEST_MODELS["generative_small"]
        num_layers = loaded_generative_model["num_layers"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Test",
            "max_tokens": 5,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": "all",
        })

        assert response.status_code == 200
        data = response.json()

        hidden_states = data["hidden_states"]
        assert hidden_states is not None

        # Should have all layers
        assert len(hidden_states) == num_layers, \
            f"Expected {num_layers} layers, got {len(hidden_states)}"

        print("\n=== All Layers Extraction ===")
        print(f"Model layers: {num_layers}")
        print(f"Layers returned: {len(hidden_states)}")
        print(f"Layer keys: {sorted(hidden_states.keys(), key=lambda x: int(x))}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "hidden_state_layers": "all",
            },
            "response": {
                "num_layers_returned": len(hidden_states),
                "layer_keys": sorted(hidden_states.keys(), key=lambda x: int(x)),
                "first_layer_shape": hidden_states[str(-num_layers)]["shape"],
                "last_layer_shape": hidden_states["-1"]["shape"],
            },
            "validation": {
                "expected_layers": num_layers,
                "actual_layers": len(hidden_states),
                "matches": len(hidden_states) == num_layers,
            }
        }
        with open(examples_dir / "generate_all_layers.json", "w") as f:
            json.dump(example, f, indent=2)

    def test_generate_full_sequence(self, app_client, loaded_generative_model, examples_dir):
        """Test full sequence hidden states for manifold analysis."""
        model_id = TEST_MODELS["generative_small"]

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "One two three",
            "max_tokens": 5,
            "temperature": 0.1,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
            "return_full_sequence": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Check for sequence hidden states
        if data.get("sequence_hidden_states"):
            seq_states = data["sequence_hidden_states"]
            assert "-1" in seq_states

            layer_data = seq_states["-1"]
            assert "data" in layer_data
            assert "shape" in layer_data

            # Shape should be [num_tokens, hidden_size]
            shape = layer_data["shape"]
            assert len(shape) == 2
            num_tokens, hidden_size = shape

            print("\n=== Full Sequence Hidden States ===")
            print(f"Shape: [{num_tokens} tokens, {hidden_size} hidden_size]")
            print(f"Total values: {num_tokens * hidden_size}")

            # Save example
            example = {
                "request": {
                    "model": model_id,
                    "return_full_sequence": True,
                },
                "response": {
                    "shape": shape,
                    "interpretation": f"{num_tokens} tokens × {hidden_size} hidden dimensions",
                    "use_case": "Manifold/boundary object analysis - track semantic evolution across tokens",
                },
            }
            with open(examples_dir / "generate_full_sequence.json", "w") as f:
                json.dump(example, f, indent=2)
        else:
            print("\n=== Full Sequence Hidden States ===")
            print("Note: return_full_sequence not supported by this model/loader")


class TestBatchOperations:
    """Test batch endpoints."""

    def test_batch_embed(self, app_client, examples_dir):
        """Test batch embedding extraction."""
        model_id = TEST_MODELS["embedding"]

        # First ensure model is loaded
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        texts = [
            "First sentence about machine learning.",
            "Second sentence about natural language processing.",
            "Third sentence about neural networks.",
        ]

        response = app_client.post("/embed/batch", json={
            "model": model_id,
            "texts": texts,
            "pooling": "mean",
            "normalize": True,
        })

        assert response.status_code == 200
        data = response.json()

        # Validate structure
        assert "embeddings" in data
        assert "shapes" in data
        assert "total_time_ms" in data
        assert "texts_processed" in data

        # Validate counts
        assert len(data["embeddings"]) == len(texts)
        assert len(data["shapes"]) == len(texts)
        assert data["texts_processed"] == len(texts)

        print("\n=== Batch Embedding ===")
        print(f"Texts processed: {data['texts_processed']}")
        print(f"Total time: {data['total_time_ms']:.2f}ms")
        print(f"Time per text: {data['total_time_ms'] / len(texts):.2f}ms")
        for i, shape in enumerate(data["shapes"]):
            print(f"  Text {i+1} shape: {shape}")

        # Save example
        example = {
            "request": {
                "model": model_id,
                "texts": texts,
                "pooling": "mean",
                "normalize": True,
            },
            "response": {
                "texts_processed": data["texts_processed"],
                "total_time_ms": data["total_time_ms"],
                "shapes": data["shapes"],
                "embeddings_preview": [emb[:5] for emb in data["embeddings"]],
            }
        }
        with open(examples_dir / "batch_embed.json", "w") as f:
            json.dump(example, f, indent=2)


class TestOutputFormat:
    """Validate output format is correct and easily parseable."""

    def test_hidden_state_format_list(self, app_client, examples_dir):
        """Validate list format is correct JSON arrays."""
        model_id = TEST_MODELS["embedding"]
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        response = app_client.post("/embed", json={
            "model": model_id,
            "text": "Test text",
            "pooling": "mean",
        })

        assert response.status_code == 200
        data = response.json()

        embedding = data["embedding"]

        # Validate it's a flat list of floats
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)

        # Validate JSON round-trip works
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["embedding"] == embedding

        print("\n=== List Format Validation ===")
        print(f"Type: {type(embedding).__name__}")
        print(f"Length: {len(embedding)}")
        print(f"Element type: {type(embedding[0]).__name__}")
        print(f"JSON round-trip: OK")

    def test_hidden_state_format_base64(self, app_client, examples_dir):
        """Validate base64 format for efficient transfer."""
        model_id = TEST_MODELS["generative_small"]
        app_client.post("/models/load", json={"model": model_id, "dtype": "float16"})

        response = app_client.post("/generate", json={
            "model": model_id,
            "prompt": "Test",
            "max_tokens": 5,
            "return_hidden_states": True,
            "hidden_state_layers": [-1],
            "hidden_state_format": "base64",
        })

        assert response.status_code == 200
        data = response.json()

        if data.get("hidden_states"):
            layer_data = data["hidden_states"]["-1"]

            assert "data" in layer_data
            assert "shape" in layer_data
            assert "dtype" in layer_data
            assert layer_data.get("encoding") == "base64"

            # Validate it's a base64 string
            import base64
            decoded = base64.b64decode(layer_data["data"])

            # Map dtype string to bytes per element
            dtype_to_bytes = {
                "float16": 2,
                "float32": 4,
                "float64": 8,
                "bfloat16": 2,
            }
            dtype_str = layer_data["dtype"].replace("torch.", "")
            bytes_per_element = dtype_to_bytes.get(dtype_str, 4)  # default float32

            # Validate size matches shape
            expected_elements = 1
            for dim in layer_data["shape"]:
                expected_elements *= dim
            expected_bytes = expected_elements * bytes_per_element
            assert len(decoded) == expected_bytes

            print("\n=== Base64 Format Validation ===")
            print(f"Shape: {layer_data['shape']}")
            print(f"Dtype: {dtype_str} ({bytes_per_element} bytes/element)")
            print(f"Encoded length: {len(layer_data['data'])} chars")
            print(f"Decoded bytes: {len(decoded)}")
            print(f"Expected bytes: {expected_bytes}")
            print(f"Decoding: OK")

            # Map dtype for numpy
            numpy_dtype = {
                "float16": "np.float16",
                "float32": "np.float32",
                "float64": "np.float64",
                "bfloat16": "np.float16",  # numpy doesn't have bfloat16, use float16
            }.get(dtype_str, "np.float32")

            # Save example
            example = {
                "format": "base64",
                "shape": layer_data["shape"],
                "dtype": dtype_str,
                "bytes_per_element": bytes_per_element,
                "encoded_sample": layer_data["data"][:100] + "...",
                "decoding_instructions": [
                    "1. Base64 decode the 'data' field",
                    f"2. Interpret as {dtype_str} array ({bytes_per_element} bytes/element)",
                    "3. Reshape according to 'shape' field",
                ],
                "python_example": f"""
import base64
import numpy as np

decoded = base64.b64decode(layer_data['data'])
array = np.frombuffer(decoded, dtype={numpy_dtype})
array = array.reshape(layer_data['shape'])
""".strip(),
            }
            with open(examples_dir / "hidden_state_base64_format.json", "w") as f:
                json.dump(example, f, indent=2)


