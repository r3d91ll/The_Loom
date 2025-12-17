"""Tests for HTTP server endpoints."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from src.config import Config
from src.loaders.base import EmbeddingOutput, GenerationOutput, LoadedModel
from src.transport.http import create_http_app


@pytest.fixture
def mock_config():
    """
    Create a default Config instance for tests.
    
    Returns:
        Config: A new Config object initialized with default settings suitable for unit tests.
    """
    return Config()


@pytest.fixture
def mock_loaded_model():
    """Create a mock loaded model."""
    return LoadedModel(
        model=MagicMock(),
        tokenizer=MagicMock(),
        model_id="test-model",
        device=torch.device("cpu"),
        dtype=torch.float32,
        hidden_size=768,
        num_layers=12,
        loader_type="transformers",
        metadata={"load_time_seconds": 1.0},
    )


@pytest.fixture
def mock_generation_output():
    """
    Create a mock GenerationOutput representing a short example generation.
    
    The returned object contains a short text "Hello, world!", token ids [1, 2, 3], a single-layer hidden state mapping for key -1, and metadata with inference timing and token counts.
    
    Returns:
        GenerationOutput: A mock generation result with `text`, `token_ids`, `hidden_states`, `attention_weights` (None), and `metadata`.
    """
    return GenerationOutput(
        text="Hello, world!",
        token_ids=[1, 2, 3],
        hidden_states={-1: torch.randn(1, 768)},
        attention_weights=None,
        metadata={
            "inference_time_ms": 100.0,
            "tokens_generated": 3,
            "tokens_per_second": 30.0,
        },
    )


@pytest.fixture
def mock_embedding_output():
    """
    Constructs a mock EmbeddingOutput with a random 768-dimensional embedding and basic metadata for tests.
    
    Returns:
        EmbeddingOutput: contains
            - embedding (torch.Tensor): a randomly-initialized 1D tensor of length 768,
            - shape (tuple): the embedding shape (768,),
            - metadata (dict): includes "pooling" set to "last_token" and "inference_time_ms" with a sample value.
    """
    return EmbeddingOutput(
        embedding=torch.randn(768),
        shape=(768,),
        metadata={
            "pooling": "last_token",
            "inference_time_ms": 50.0,
        },
    )


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, mock_config):
        with patch("src.transport.http.GPUManager") as mock_gpu:
            mock_gpu.return_value.to_dict.return_value = {
                "has_gpu": False,
                "default_device": "cpu",
            }

            app = create_http_app(mock_config)
            client = TestClient(app)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "gpu_info" in data


class TestModelsEndpoints:
    """Tests for model management endpoints."""

    def test_list_models_empty(self, mock_config):
        with patch("src.transport.http.GPUManager"):
            app = create_http_app(mock_config)
            client = TestClient(app)

            response = client.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert data["loaded_models"] == []

    def test_load_model(self, mock_config, mock_loaded_model):
        with patch("src.transport.http.GPUManager"):
            with patch("src.transport.http.LoaderRegistry") as mock_registry:
                mock_registry.return_value.load.return_value = mock_loaded_model

                app = create_http_app(mock_config)

                client = TestClient(app)

                response = client.post(
                    "/models/load",
                    json={"model": "test-model", "device": "cpu"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["model_id"] == "test-model"
                assert data["hidden_size"] == 768


class TestGenerateEndpoint:
    """Tests for /generate endpoint."""

    def test_generate_with_hidden_states(
        self,
        mock_config,
        mock_loaded_model,
        mock_generation_output,
    ):
        with patch("src.transport.http.GPUManager"):
            with patch("src.transport.http.LoaderRegistry") as mock_registry:
                mock_registry.return_value.load.return_value = mock_loaded_model
                mock_registry.return_value.generate.return_value = mock_generation_output

                app = create_http_app(mock_config)

                client = TestClient(app)

                response = client.post(
                    "/generate",
                    json={
                        "model": "test-model",
                        "prompt": "Hello",
                        "max_tokens": 10,
                        "return_hidden_states": True,
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["text"] == "Hello, world!"
                assert data["token_count"] == 3
                assert "hidden_states" in data
                assert "-1" in data["hidden_states"]

    def test_generate_without_hidden_states(
        self,
        mock_config,
        mock_loaded_model,
    ):
        output = GenerationOutput(
            text="Response text",
            token_ids=[1, 2],
            hidden_states=None,
            attention_weights=None,
            metadata={"inference_time_ms": 50.0},
        )

        with patch("src.transport.http.GPUManager"):
            with patch("src.transport.http.LoaderRegistry") as mock_registry:
                mock_registry.return_value.load.return_value = mock_loaded_model
                mock_registry.return_value.generate.return_value = output

                app = create_http_app(mock_config)

                client = TestClient(app)

                response = client.post(
                    "/generate",
                    json={
                        "model": "test-model",
                        "prompt": "Hello",
                        "return_hidden_states": False,
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["hidden_states"] is None


class TestEmbedEndpoint:
    """Tests for /embed endpoint."""

    def test_embed_basic(
        self,
        mock_config,
        mock_loaded_model,
        mock_embedding_output,
    ):
        with patch("src.transport.http.GPUManager"):
            with patch("src.transport.http.LoaderRegistry") as mock_registry:
                mock_registry.return_value.load.return_value = mock_loaded_model
                mock_registry.return_value.embed.return_value = mock_embedding_output

                app = create_http_app(mock_config)

                client = TestClient(app)

                response = client.post(
                    "/embed",
                    json={
                        "model": "test-model",
                        "text": "Hello world",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert "embedding" in data
                assert len(data["embedding"]) == 768
                assert data["shape"] == [768]

    def test_embed_with_normalization(
        self,
        mock_config,
        mock_loaded_model,
    ):
        # Create output with known embedding
        embedding = torch.tensor([3.0, 4.0])  # Norm = 5
        output = EmbeddingOutput(
            embedding=embedding,
            shape=(2,),
            metadata={},
        )

        with patch("src.transport.http.GPUManager"):
            with patch("src.transport.http.LoaderRegistry") as mock_registry:
                mock_registry.return_value.load.return_value = mock_loaded_model
                mock_registry.return_value.embed.return_value = output

                app = create_http_app(mock_config)

                client = TestClient(app)

                response = client.post(
                    "/embed",
                    json={
                        "model": "test-model",
                        "text": "Hello",
                        "normalize": True,
                    },
                )

                assert response.status_code == 200
                data = response.json()
                # Check normalization: [3/5, 4/5] = [0.6, 0.8]
                assert abs(data["embedding"][0] - 0.6) < 0.01
                assert abs(data["embedding"][1] - 0.8) < 0.01


class TestRequestValidation:
    """Tests for request validation."""

    def test_generate_missing_model(self, mock_config):
        with patch("src.transport.http.GPUManager"):
            app = create_http_app(mock_config)
            client = TestClient(app)

            response = client.post(
                "/generate",
                json={"prompt": "Hello"},  # Missing model
            )

            assert response.status_code == 422  # Validation error

    def test_generate_invalid_temperature(self, mock_config):
        with patch("src.transport.http.GPUManager"):
            app = create_http_app(mock_config)
            client = TestClient(app)

            response = client.post(
                "/generate",
                json={
                    "model": "test",
                    "prompt": "Hello",
                    "temperature": 5.0,  # Out of range
                },
            )

            assert response.status_code == 422

    def test_embed_missing_text(self, mock_config):
        with patch("src.transport.http.GPUManager"):
            app = create_http_app(mock_config)
            client = TestClient(app)

            response = client.post(
                "/embed",
                json={"model": "test"},  # Missing text
            )

            assert response.status_code == 422