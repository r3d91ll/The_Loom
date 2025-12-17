"""Loader registry with auto-detection for model loading.

The registry manages multiple loaders and automatically selects the best
loader for each model based on:
1. Explicit configuration (model_overrides in config)
2. Loader can_load() checks (pattern matching)
3. Fallback chain (mistral -> qwen -> transformers -> sentence_transformers -> custom)

This enables seamless loading of diverse models without manual loader selection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    StreamingOutput,
    StreamingToken,
)
from .custom_loader import CustomLoader, CustomModelConfig
from .mistral_loader import MistralLoader
from .qwen_loader import QwenLoader
from .sentence_transformers_loader import SentenceTransformersLoader
from .transformers_loader import TransformersLoader

logger = logging.getLogger(__name__)


class LoaderRegistry:
    """Central registry managing model loaders with auto-detection.

    The registry provides:
    1. Automatic loader selection based on model ID patterns
    2. Explicit loader override via configuration
    3. Fallback chain for unknown models
    4. Unified interface for loading, generation, and embedding

    Usage:
        registry = LoaderRegistry()
        loaded = registry.load("meta-llama/Llama-3.1-8B-Instruct")
        output = registry.generate(loaded, "Hello")
    """

    def __init__(
        self,
        loader_configs: dict[str, dict[str, Any]] | None = None,
        custom_model_configs: dict[str, CustomModelConfig] | None = None,
    ):
        """Initialize the registry with configured loaders.

        Loaders are checked in priority order:
        1. mistral - Specialized loader for Mistral models
        2. qwen - Specialized loader for Qwen models
        3. transformers - General HuggingFace transformers (~80% of models)
        4. sentence_transformers - Embedding models (BGE, E5, SBERT)
        5. custom - Edge cases and research models

        Args:
            loader_configs: Per-model loader configuration overrides
                Example: {"model-id": {"loader": "transformers", "dtype": "float16"}}
            custom_model_configs: Custom model configurations for CustomLoader
        """
        self.loader_configs = loader_configs or {}

        # Initialize loaders in priority order
        self.loaders: dict[str, ModelLoader] = {
            "mistral": MistralLoader(),
            "qwen": QwenLoader(),
            "transformers": TransformersLoader(),
            "sentence_transformers": SentenceTransformersLoader(),
            "custom": CustomLoader(custom_model_configs),
        }

        # Fallback order for auto-detection
        # Specialized loaders checked first, then transformers as general fallback
        self.fallback_order = [
            "mistral",
            "qwen",
            "transformers",
            "sentence_transformers",
            "custom",
        ]

        logger.info(
            f"LoaderRegistry initialized with {len(self.loaders)} loaders: "
            f"{list(self.loaders.keys())}"
        )

    def register_loader(self, name: str, loader: ModelLoader) -> None:
        """
        Add a model loader to the registry and ensure it appears in the fallback order.
        
        If the loader name is not already present in the fallback order, it is inserted before the 'custom' entry so the loader will be considered prior to the custom fallback.
        
        Parameters:
            name (str): Unique identifier for the loader.
            loader (ModelLoader): Loader instance to register.
        """
        self.loaders[name] = loader
        if name not in self.fallback_order:
            # Insert before 'custom' (last resort)
            idx = self.fallback_order.index("custom")
            self.fallback_order.insert(idx, name)
        logger.info(f"Registered loader: {name}")

    def get_loader(self, model_id: str) -> tuple[str, ModelLoader]:
        """Get the appropriate loader for a model.

        Selection priority:
        1. Explicit configuration in loader_configs
        2. First loader where can_load() returns True
        3. Fallback to transformers loader (most general)

        Args:
            model_id: Model identifier

        Returns:
            Tuple of (loader_name, loader_instance)
        """
        # Check explicit configuration
        if model_id in self.loader_configs:
            config = self.loader_configs[model_id]
            loader_name = config.get("loader", "auto")
            if loader_name != "auto" and loader_name in self.loaders:
                logger.debug(f"Using configured loader '{loader_name}' for {model_id}")
                return loader_name, self.loaders[loader_name]

        # Auto-detect: check each loader's can_load()
        for loader_name in self.fallback_order:
            loader = self.loaders[loader_name]
            if loader.can_load(model_id):
                logger.debug(f"Auto-detected loader '{loader_name}' for {model_id}")
                return loader_name, loader

        # Fallback to transformers (most general)
        logger.debug(f"Falling back to 'transformers' loader for {model_id}")
        return "transformers", self.loaders["transformers"]

    def get_model_config(self, model_id: str) -> dict[str, Any]:
        """
        Return the configuration for a model with defaults overridden by any per-model settings.
        
        Returns:
            A dictionary containing resolved model configuration values (e.g., `device`, `dtype`, `trust_remote_code`), with any per-model overrides applied.
        """
        base_config = {
            "device": "cuda:0",
            "dtype": "auto",
            "trust_remote_code": True,
        }

        if model_id in self.loader_configs:
            base_config.update(self.loader_configs[model_id])

        return base_config

    def load(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        loader_name: str | None = None,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """
        Selects an appropriate loader and loads the specified model into memory.
        
        Parameters:
            model_id (str): Model identifier (e.g., HuggingFace ID or local path).
            device (str | None): Device override; if None the model config default is used.
            dtype (str | None): Dtype override; if None the model config default is used.
            loader_name (str | None): If provided, forces that loader; if None the registry auto-detects a loader and falls back to the default.
            quantization (str | None): Quantization mode to apply (e.g., "4bit", "8bit", "gptq", "awq").
            **kwargs: Additional loader-specific options forwarded to the selected loader.
        
        Returns:
            LoadedModel: The loaded model. The returned object's metadata contains "loader_name" set to the loader that performed the load.
        """
        # Get model-specific config
        model_config = self.get_model_config(model_id)

        # Apply overrides
        if device is not None:
            model_config["device"] = device
        if dtype is not None:
            model_config["dtype"] = dtype

        # Get loader
        if loader_name is not None and loader_name in self.loaders:
            selected_name = loader_name
            loader = self.loaders[loader_name]
        else:
            selected_name, loader = self.get_loader(model_id)

        logger.info(f"Loading {model_id} with '{selected_name}' loader")

        # Load the model
        loaded = loader.load(
            model_id,
            device=model_config["device"],
            dtype=model_config["dtype"],
            trust_remote_code=model_config.get("trust_remote_code", True),
            quantization=quantization,
            **kwargs,
        )

        # Store which loader was used
        loaded.metadata["loader_name"] = selected_name

        return loaded

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        **kwargs: Any,
    ) -> GenerationOutput:
        """
        Generate text from a loaded model using the loader recorded on the model.
        
        Parameters:
            loaded_model (LoadedModel): A model instance previously returned by `load`.
            prompt (str): The input prompt to generate from.
            **kwargs: Additional generation options passed through to the underlying loader.
        
        Returns:
            GenerationOutput: The generated text and any available hidden states.
        """
        loader_name = loaded_model.metadata.get("loader_name", loaded_model.loader_type)
        loader = self.loaders.get(loader_name, self.loaders["transformers"])

        return loader.generate(loaded_model, prompt, **kwargs)

    def generate_stream(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[StreamingToken | StreamingOutput]:
        """
        Stream generation tokens produced by the loader associated with the provided LoadedModel.
        
        Parameters:
            loaded_model (LoadedModel): Previously loaded model whose associated loader will produce the stream.
            prompt (str): Input prompt to generate from.
            **kwargs: Generation parameters forwarded to the loader.
        
        Yields:
            StreamingToken: Emitted for each generated token.
            StreamingOutput: Emitted once after the token stream completes.
        """
        loader_name = loaded_model.metadata.get("loader_name", loaded_model.loader_type)
        loader = self.loaders.get(loader_name, self.loaders["transformers"])

        yield from loader.generate_stream(loaded_model, prompt, **kwargs)

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """
        Compute an embedding for the given text using the loader associated with the provided loaded model.
        
        Parameters:
            loaded_model (LoadedModel): A previously loaded model whose associated loader will perform the embedding.
            text (str): Text to embed.
            **kwargs: Optional embedding parameters forwarded to the underlying loader.
        
        Returns:
            EmbeddingOutput: The computed embedding (embedding tensor and any loader-specific metadata).
        """
        loader_name = loaded_model.metadata.get("loader_name", loaded_model.loader_type)
        loader = self.loaders.get(loader_name, self.loaders["transformers"])

        return loader.embed(loaded_model, text, **kwargs)

    def list_loaders(self) -> dict[str, dict[str, Any]]:
        """
        Provide a mapping of registered loader identifiers to basic metadata.
        
        Returns:
            dict[str, dict[str, Any]]: Mapping from registry key to a metadata dict containing:
                - "name": the loader's configured name (loader.name)
                - "type": the loader's class name
        """
        return {
            name: {
                "name": loader.name,
                "type": type(loader).__name__,
            }
            for name, loader in self.loaders.items()
        }

    def probe_model(self, model_id: str) -> dict[str, Any]:
        """
        Determine which registered loader would be used for a given model identifier without loading the model.
        
        Parameters:
            model_id (str): The model identifier to probe.
        
        Returns:
            dict[str, Any]: A dictionary with the following keys:
                - model_id: The probed model identifier.
                - configured_loader: Loader name explicitly configured for this model, or None.
                - detected_loader: First loader (by fallback order) whose `can_load` returned True, or None.
                - can_load: Mapping of loader name to the boolean result of that loader's `can_load(model_id)`.
                - selected_loader: The loader name that would be selected (configured loader unless set to "auto", otherwise detected_loader, otherwise "transformers").
        """
        results: dict[str, Any] = {
            "model_id": model_id,
            "configured_loader": None,
            "detected_loader": None,
            "can_load": {},
        }

        # Check configuration
        if model_id in self.loader_configs:
            results["configured_loader"] = self.loader_configs[model_id].get("loader")

        # Check each loader
        for name in self.fallback_order:
            loader = self.loaders[name]
            can_load = loader.can_load(model_id)
            results["can_load"][name] = can_load
            if can_load and results["detected_loader"] is None:
                results["detected_loader"] = name

        # Final selection
        if results["configured_loader"] and results["configured_loader"] != "auto":
            results["selected_loader"] = results["configured_loader"]
        elif results["detected_loader"]:
            results["selected_loader"] = results["detected_loader"]
        else:
            results["selected_loader"] = "transformers"  # Fallback

        return results


# Global registry instance (initialized lazily)
_registry: LoaderRegistry | None = None


def get_registry() -> LoaderRegistry:
    """
    Provide the global LoaderRegistry singleton, creating it on first access.
    
    Returns:
        registry (LoaderRegistry): The global LoaderRegistry instance; created and stored if it did not already exist.
    """
    global _registry
    if _registry is None:
        _registry = LoaderRegistry()
    return _registry


def set_registry(registry: LoaderRegistry) -> None:
    """
    Set the global LoaderRegistry used by get_registry and related helpers.
    
    Parameters:
        registry (LoaderRegistry): The registry instance to assign as the global singleton.
    """
    global _registry
    _registry = registry


def create_registry_from_config(config: Any) -> LoaderRegistry:
    """
    Create a LoaderRegistry from an application configuration.
    
    Parameters:
        config (Any): Application config object that may contain a `model_overrides`
            attribute mapping model IDs to per-model loader configuration.
    
    Returns:
        LoaderRegistry: A registry initialized with `loader_configs` taken from
        `config.model_overrides` (uses an empty dict if the attribute is absent).
    """
    loader_configs = getattr(config, "model_overrides", {})
    return LoaderRegistry(loader_configs=loader_configs)