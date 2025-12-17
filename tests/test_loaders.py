"""Tests for multi-loader functionality."""

from unittest.mock import MagicMock

import pytest
import torch

from src.loaders.base import EmbeddingOutput, GenerationOutput, LoadedModel
from src.loaders.custom_loader import (
    CUSTOM_MODEL_REGISTRY,
    CustomLoader,
    CustomModelConfig,
    register_custom_model,
)
from src.loaders.registry import LoaderRegistry
from src.loaders.sentence_transformers_loader import (
    SentenceTransformersLoader,
)
from src.loaders.transformers_loader import TransformersLoader


class TestTransformersLoader:
    """Tests for TransformersLoader."""

    def test_name(self):
        loader = TransformersLoader()
        assert loader.name == "transformers"

    def test_can_load_standard_models(self):
        loader = TransformersLoader()

        # Should handle standard HuggingFace models
        assert loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert loader.can_load("mistralai/Mistral-7B-Instruct-v0.2")
        assert loader.can_load("Qwen/Qwen2-7B-Instruct")
        assert loader.can_load("microsoft/phi-2")

    def test_can_load_excludes_sentence_transformers(self):
        loader = TransformersLoader()

        # Should NOT handle sentence-transformers models
        assert not loader.can_load("sentence-transformers/all-MiniLM-L6-v2")


class TestSentenceTransformersLoader:
    """Tests for SentenceTransformersLoader."""

    def test_name(self):
        loader = SentenceTransformersLoader()
        assert loader.name == "sentence_transformers"

    def test_can_load_embedding_models(self):
        loader = SentenceTransformersLoader()

        # Should handle known embedding model patterns
        assert loader.can_load("sentence-transformers/all-MiniLM-L6-v2")
        assert loader.can_load("BAAI/bge-small-en-v1.5")
        assert loader.can_load("intfloat/e5-small-v2")
        assert loader.can_load("thenlper/gte-small")
        assert loader.can_load("nomic-ai/nomic-embed-text-v1")

    def test_can_load_excludes_instruct_models(self):
        loader = SentenceTransformersLoader()

        # Should NOT handle instruction-tuned models
        assert not loader.can_load("BAAI/bge-instruct")
        assert not loader.can_load("model-chat")

    def test_can_load_standard_models_returns_false(self):
        loader = SentenceTransformersLoader()

        # Should NOT handle standard decoder models
        assert not loader.can_load("meta-llama/Llama-3.1-8B-Instruct")
        assert not loader.can_load("mistralai/Mistral-7B-v0.1")


class TestCustomLoader:
    """Tests for CustomLoader."""

    def test_name(self):
        loader = CustomLoader()
        assert loader.name == "custom"

    def test_can_load_without_config(self):
        loader = CustomLoader()

        # Without registered configs, should return False
        assert not loader.can_load("random-model-id")

    def test_can_load_with_config(self):
        config = CustomModelConfig(
            model_id_pattern="my-org/custom-model",
            model_factory=lambda *args, **kwargs: MagicMock(),
            hidden_size=2048,
            num_layers=24,
        )

        loader = CustomLoader(custom_configs={"my-org/custom-model": config})

        assert loader.can_load("my-org/custom-model")
        assert not loader.can_load("other-model")

    def test_get_config(self):
        config = CustomModelConfig(
            model_id_pattern="my-org/.*",  # Regex pattern
            model_factory=lambda *args, **kwargs: MagicMock(),
        )

        loader = CustomLoader(custom_configs={"my-org/.*": config})

        # Exact match
        result = loader.get_config("my-org/.*")
        assert result is not None

    def test_register_custom_model(self):
        # Clear any existing registrations
        original = CUSTOM_MODEL_REGISTRY.copy()
        CUSTOM_MODEL_REGISTRY.clear()

        try:
            config = CustomModelConfig(
                model_id_pattern="test/model",
                model_factory=lambda *args, **kwargs: MagicMock(),
            )
            register_custom_model(config)

            assert "test/model" in CUSTOM_MODEL_REGISTRY
        finally:
            # Restore original
            CUSTOM_MODEL_REGISTRY.clear()
            CUSTOM_MODEL_REGISTRY.update(original)


class TestLoaderRegistry:
    """Tests for LoaderRegistry."""

    def test_init_default_loaders(self):
        registry = LoaderRegistry()

        assert "mistral" in registry.loaders
        assert "qwen" in registry.loaders
        assert "transformers" in registry.loaders
        assert "sentence_transformers" in registry.loaders
        assert "custom" in registry.loaders

    def test_fallback_order(self):
        registry = LoaderRegistry()

        assert registry.fallback_order == [
            "mistral",
            "qwen",
            "transformers",
            "sentence_transformers",
            "custom",
        ]

    def test_get_loader_auto_detection(self):
        registry = LoaderRegistry()

        # Standard model -> transformers
        name, loader = registry.get_loader("meta-llama/Llama-3.1-8B")
        assert name == "transformers"
        assert isinstance(loader, TransformersLoader)

        # Embedding model -> sentence_transformers
        name, loader = registry.get_loader("sentence-transformers/all-MiniLM-L6-v2")
        assert name == "sentence_transformers"
        assert isinstance(loader, SentenceTransformersLoader)

    def test_get_loader_with_config_override(self):
        registry = LoaderRegistry(
            loader_configs={
                "my-model": {"loader": "sentence_transformers"},
            }
        )

        name, loader = registry.get_loader("my-model")
        assert name == "sentence_transformers"

    def test_get_model_config_defaults(self):
        registry = LoaderRegistry()

        config = registry.get_model_config("some-model")

        assert config["device"] == "cuda:0"
        assert config["dtype"] == "auto"
        assert config["trust_remote_code"] is True

    def test_get_model_config_with_override(self):
        registry = LoaderRegistry(
            loader_configs={
                "special-model": {
                    "device": "cuda:1",
                    "dtype": "float16",
                },
            }
        )

        config = registry.get_model_config("special-model")

        assert config["device"] == "cuda:1"
        assert config["dtype"] == "float16"

    def test_list_loaders(self):
        registry = LoaderRegistry()

        loaders = registry.list_loaders()

        assert "transformers" in loaders
        assert loaders["transformers"]["name"] == "transformers"
        assert loaders["transformers"]["type"] == "TransformersLoader"

    def test_probe_model(self):
        registry = LoaderRegistry()

        # Probe a standard model
        result = registry.probe_model("meta-llama/Llama-3.1-8B")

        assert result["model_id"] == "meta-llama/Llama-3.1-8B"
        assert result["detected_loader"] == "transformers"
        assert result["selected_loader"] == "transformers"
        assert result["can_load"]["transformers"] is True

    def test_probe_model_with_config(self):
        registry = LoaderRegistry(
            loader_configs={
                "forced-model": {"loader": "custom"},
            }
        )

        result = registry.probe_model("forced-model")

        assert result["configured_loader"] == "custom"
        assert result["selected_loader"] == "custom"

    def test_register_loader(self):
        registry = LoaderRegistry()

        # Create a mock loader
        mock_loader = MagicMock()
        mock_loader.name = "mock_loader"
        mock_loader.can_load.return_value = False

        registry.register_loader("mock", mock_loader)

        assert "mock" in registry.loaders
        # Should be inserted before 'custom'
        assert registry.fallback_order.index("mock") < registry.fallback_order.index("custom")


class TestLoaderRegistryIntegration:
    """Integration tests for LoaderRegistry with mocked models."""

    @pytest.fixture
    def mock_loaded_model(self):
        """
        Create a LoadedModel stub for tests.
        
        Returns:
            LoadedModel: A mock LoadedModel with MagicMock `model` and `tokenizer`, `model_id` "test-model", CPU `device`, `dtype` float32, `hidden_size` 768, `num_layers` 12, `loader_type` "transformers", and `metadata` {"loader_name": "transformers"}.
        """
        return LoadedModel(
            model=MagicMock(),
            tokenizer=MagicMock(),
            model_id="test-model",
            device=torch.device("cpu"),
            dtype=torch.float32,
            hidden_size=768,
            num_layers=12,
            loader_type="transformers",
            metadata={"loader_name": "transformers"},
        )

    def test_generate_uses_correct_loader(self, mock_loaded_model):
        registry = LoaderRegistry()

        # Mock the transformers loader's generate method
        mock_output = GenerationOutput(
            text="Generated text",
            token_ids=[1, 2, 3],
            hidden_states={-1: torch.randn(1, 768)},
            attention_weights=None,
            metadata={},
        )
        registry.loaders["transformers"].generate = MagicMock(return_value=mock_output)

        result = registry.generate(mock_loaded_model, "Hello")

        assert result.text == "Generated text"
        registry.loaders["transformers"].generate.assert_called_once()

    def test_embed_uses_correct_loader(self, mock_loaded_model):
        registry = LoaderRegistry()

        # Mock the transformers loader's embed method
        mock_output = EmbeddingOutput(
            embedding=torch.randn(768),
            shape=(768,),
            metadata={},
        )
        registry.loaders["transformers"].embed = MagicMock(return_value=mock_output)

        result = registry.embed(mock_loaded_model, "Hello")

        assert result.shape == (768,)
        registry.loaders["transformers"].embed.assert_called_once()


class TestAutoDetectionPatterns:
    """Tests for model pattern auto-detection."""

    def test_llama_models(self):
        registry = LoaderRegistry()

        llama_models = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B",
        ]

        for model in llama_models:
            name, _ = registry.get_loader(model)
            assert name == "transformers", f"Expected transformers for {model}"

    def test_mistral_models(self):
        registry = LoaderRegistry()

        mistral_models = [
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-v0.1",
        ]

        for model in mistral_models:
            name, _ = registry.get_loader(model)
            assert name == "transformers", f"Expected transformers for {model}"

    def test_embedding_models(self):
        registry = LoaderRegistry()

        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
        ]

        for model in embedding_models:
            name, _ = registry.get_loader(model)
            assert name == "sentence_transformers", f"Expected sentence_transformers for {model}"