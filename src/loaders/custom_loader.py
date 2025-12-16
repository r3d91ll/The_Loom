"""Custom model loader for edge cases and research models.

This loader provides a flexible framework for loading models that don't fit
standard patterns. It supports:
- Custom model classes
- Non-standard architectures
- Research/experimental models
- Models requiring special initialization

Coverage: ~5% of models (edge cases, research models)
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch
from transformers import AutoTokenizer

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    resolve_dtype,
)

logger = logging.getLogger(__name__)


class ModelFactory(Protocol):
    """Protocol for custom model factory functions."""

    def __call__(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs: Any,
    ) -> Any:
        """Load and return a model instance."""
        ...


class TokenizerFactory(Protocol):
    """Protocol for custom tokenizer factory functions."""

    def __call__(self, model_id: str, **kwargs: Any) -> Any:
        """Load and return a tokenizer instance."""
        ...


@dataclass
class CustomModelConfig:
    """Configuration for a custom model.

    This allows registering custom loading logic for specific models
    that don't work with standard loaders.
    """

    model_id_pattern: str  # Regex or exact match
    model_factory: ModelFactory | str  # Callable or import path
    tokenizer_factory: TokenizerFactory | str | None = None  # Optional custom tokenizer
    hidden_size: int | None = None  # Override if can't be detected
    num_layers: int | None = None  # Override if can't be detected
    generation_supported: bool = True
    embedding_supported: bool = True
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


# Registry of custom model configurations
CUSTOM_MODEL_REGISTRY: dict[str, CustomModelConfig] = {}


def register_custom_model(config: CustomModelConfig) -> None:
    """Register a custom model configuration.

    Example:
        register_custom_model(CustomModelConfig(
            model_id_pattern="my-org/custom-model",
            model_factory=my_custom_loader,
            hidden_size=2048,
        ))
    """
    CUSTOM_MODEL_REGISTRY[config.model_id_pattern] = config
    logger.info(f"Registered custom model: {config.model_id_pattern}")


def _resolve_callable(factory: Callable[..., Any] | str) -> Callable[..., Any]:
    """Resolve a callable from a string import path or return as-is."""
    if isinstance(factory, str):
        module_path, func_name = factory.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return cast(Callable[..., Any], getattr(module, func_name))
    return factory


class CustomLoader(ModelLoader):
    """Flexible loader for edge cases and research models.

    This loader serves as a safety net for models that don't work with
    TransformersLoader or SentenceTransformersLoader. It provides:

    1. Registry-based loading: Pre-configured custom loaders
    2. Fallback loading: Attempts various loading strategies
    3. Manual configuration: Direct model/tokenizer factory specification

    Use Cases:
    - Models with non-standard architectures
    - Research models requiring custom initialization
    - Models needing specific dtype/quantization handling
    - Older or experimental model formats
    """

    def __init__(self, custom_configs: dict[str, CustomModelConfig] | None = None):
        """Initialize with optional custom configurations.

        Args:
            custom_configs: Additional custom model configs to merge with registry
        """
        self.configs = {**CUSTOM_MODEL_REGISTRY}
        if custom_configs:
            self.configs.update(custom_configs)

    @property
    def name(self) -> str:
        return "custom"

    def can_load(self, model_id: str) -> bool:
        """Check if we have a custom config for this model.

        The custom loader is a fallback - it returns False by default
        and relies on explicit registration or being called as last resort.
        """
        return self.get_config(model_id) is not None

    def get_config(self, model_id: str) -> CustomModelConfig | None:
        """Get custom config for a model if registered."""
        # Exact match first
        if model_id in self.configs:
            return self.configs[model_id]

        # Pattern match
        import re

        for pattern, config in self.configs.items():
            if re.match(pattern, model_id):
                return config

        return None

    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        model_factory: ModelFactory | str | None = None,
        tokenizer_factory: TokenizerFactory | str | None = None,
        trust_remote_code: bool = True,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model using custom or fallback logic.

        Args:
            model_id: Model identifier
            device: Target device
            dtype: Data type
            model_factory: Custom model loading function (overrides registry)
            tokenizer_factory: Custom tokenizer loading function
            trust_remote_code: Allow remote code execution
            quantization: Not directly supported (passed to fallback loader if used)
            **kwargs: Additional arguments

        Returns:
            LoadedModel with the custom model
        """
        if quantization:
            logger.warning(
                f"Quantization '{quantization}' requested for CustomLoader. "
                "Support depends on the specific model factory used."
            )
        logger.info(f"Loading model {model_id} with custom loader on {device}")
        start_time = time.time()

        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        # Get config from registry if available
        config = self.get_config(model_id)

        # Determine factories to use
        if model_factory is None and config is not None:
            model_factory = config.model_factory
        if tokenizer_factory is None and config is not None:
            tokenizer_factory = config.tokenizer_factory

        # Merge kwargs with config extras
        if config is not None:
            kwargs = {**config.extra_kwargs, **kwargs}

        # Load model
        if model_factory is not None:
            factory = _resolve_callable(model_factory)
            model = factory(
                model_id,
                device=torch_device,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            # Fallback: try various loading strategies
            model = self._fallback_load_model(
                model_id, torch_device, torch_dtype, trust_remote_code, **kwargs
            )

        # Load tokenizer
        if tokenizer_factory is not None:
            factory = _resolve_callable(tokenizer_factory)
            tokenizer = factory(model_id, trust_remote_code=trust_remote_code)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Determine hidden size and num layers
        hidden_size = self._get_hidden_size(model, config)
        num_layers = self._get_num_layers(model, config)

        load_time = time.time() - start_time
        logger.info(
            f"Model loaded in {load_time:.2f}s - "
            f"hidden_size={hidden_size}, num_layers={num_layers}"
        )

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            device=torch_device,
            dtype=torch_dtype,
            hidden_size=hidden_size,
            num_layers=num_layers,
            loader_type=self.name,
            metadata={
                "load_time_seconds": load_time,
                "trust_remote_code": trust_remote_code,
                "custom_config": config.model_id_pattern if config else None,
            },
        )

    def _fallback_load_model(
        self,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
        **kwargs: Any,
    ) -> Any:
        """Try various loading strategies as fallback."""
        from transformers import AutoModel, AutoModelForCausalLM

        strategies = [
            # Try CausalLM first (most common for generation)
            lambda: AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=str(device),
                trust_remote_code=trust_remote_code,
                output_hidden_states=True,
                **kwargs,
            ),
            # Then generic AutoModel
            lambda: AutoModel.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=str(device),
                trust_remote_code=trust_remote_code,
                output_hidden_states=True,
                **kwargs,
            ),
        ]

        last_error = None
        for strategy in strategies:
            try:
                model = strategy()
                model.eval()
                return model
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"Failed to load model {model_id} with any strategy. " f"Last error: {last_error}"
        )

    def _get_hidden_size(self, model: Any, config: CustomModelConfig | None) -> int:
        """Extract or estimate hidden size."""
        if config is not None and config.hidden_size is not None:
            return config.hidden_size

        # Try various attributes
        for attr in ["config.hidden_size", "config.n_embd", "config.d_model"]:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return int(obj)
            except AttributeError:
                continue

        logger.warning("Could not determine hidden size, defaulting to 4096")
        return 4096

    def _get_num_layers(self, model: Any, config: CustomModelConfig | None) -> int:
        """Extract or estimate number of layers."""
        if config is not None and config.num_layers is not None:
            return config.num_layers

        for attr in ["config.num_hidden_layers", "config.n_layer", "config.num_layers"]:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return int(obj)
            except AttributeError:
                continue

        logger.warning("Could not determine num_layers, defaulting to 32")
        return 32

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        return_attention: bool = False,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text with custom model.

        Uses standard HuggingFace generate() interface. May not work for
        all custom models - check the model's documentation.
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        # Check if model supports generation
        if not hasattr(model, "generate"):
            return GenerationOutput(
                text="[CustomLoader: Model does not support generate()]",
                token_ids=[],
                hidden_states=None,
                attention_weights=None,
                metadata={"error": "no_generate_method"},
            )

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "output_hidden_states": return_hidden_states,
            "output_attentions": return_attention,
            "return_dict_in_generate": True,
            **kwargs,
        }

        start_time = time.time()

        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]
            except Exception as e:
                return GenerationOutput(
                    text=f"[CustomLoader: Generation failed: {e}]",
                    token_ids=[],
                    hidden_states=None,
                    attention_weights=None,
                    metadata={"error": str(e)},
                )

        inference_time = time.time() - start_time

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if available
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        if return_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden_states_dict = self._extract_hidden_states(
                outputs.hidden_states,
                hidden_state_layers,
                loaded_model.num_layers,
            )

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            hidden_states=hidden_states_dict,
            attention_weights=None,  # Simplified for custom loader
            metadata={
                "inference_time_ms": inference_time * 1000,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": len(generated_ids) / inference_time
                if inference_time > 0
                else 0,
                "model_id": loaded_model.model_id,
            },
        )

    def _extract_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states from generation output."""
        result: dict[int, torch.Tensor] = {}

        if not hidden_states:
            return result

        final_step = hidden_states[-1]

        for layer_idx in layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            if 0 <= actual_idx < len(final_step):
                layer_hidden = final_step[actual_idx][:, -1, :].cpu()
                result[layer_idx] = layer_hidden

        return result

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Extract embedding from custom model.

        Uses a forward pass with output_hidden_states=True and applies
        the specified pooling strategy.
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        inference_time = time.time() - start_time

        # Get last layer hidden states
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            last_hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        else:
            raise ValueError("Model output does not contain hidden states")

        # Apply pooling
        if pooling == "last_token":
            attention_mask = inputs.attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            embedding = last_hidden[torch.arange(batch_size, device=device), seq_lengths]
        elif pooling == "mean":
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            embedding = (last_hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        elif pooling == "first_token":
            embedding = last_hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        embedding = embedding.cpu().squeeze(0)

        return EmbeddingOutput(
            embedding=embedding,
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": inputs.input_ids.shape[1],
                "model_id": loaded_model.model_id,
            },
        )
