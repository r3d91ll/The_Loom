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
        """
        Load and return a model instance for the given model identifier, placed on the specified device and using the specified dtype.
        
        Parameters:
            model_id (str): Identifier or path of the model to load.
            device (torch.device): Target device for the loaded model.
            dtype (torch.dtype): Desired tensor dtype for model weights.
            **kwargs: Additional loader-specific keyword arguments forwarded to the underlying model factory or loader.
        
        Returns:
            Any: The instantiated model object ready for use on the specified device.
        """
        ...


class TokenizerFactory(Protocol):
    """Protocol for custom tokenizer factory functions."""

    def __call__(self, model_id: str, **kwargs: Any) -> Any:
        """
        Load a tokenizer for the specified model identifier.
        
        Parameters:
            model_id (str): Identifier or path of the model whose tokenizer should be loaded.
            **kwargs: Additional keyword arguments forwarded to the tokenizer loading implementation (e.g., cache_dir, trust_remote_code).
        
        Returns:
            Any: A tokenizer instance compatible with the model.
        """
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
    """
    Resolve a callable from either a dotted import path string or return the callable unchanged.
    
    Parameters:
        factory (Callable[..., Any] | str): A callable or a string in the form "module.submodule.callable_name".
    
    Returns:
        Callable[..., Any]: The callable object. If `factory` was a string, the named attribute is imported and returned; otherwise `factory` is returned as-is.
    """
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
        """
        Initialize the loader and merge any provided custom model configurations with the global registry.
        
        Parameters:
            custom_configs (dict[str, CustomModelConfig] | None): Optional mapping of model_id patterns to CustomModelConfig instances to add or override entries from CUSTOM_MODEL_REGISTRY. Entries in `custom_configs` take precedence over the global registry.
        """
        self.configs = {**CUSTOM_MODEL_REGISTRY}
        if custom_configs:
            self.configs.update(custom_configs)

    @property
    def name(self) -> str:
        """
        Return the loader's canonical name.
        
        Returns:
            loader_name (str): The string "custom".
        """
        return "custom"

    def can_load(self, model_id: str) -> bool:
        """
        Determine whether a custom configuration exists for the given model identifier.
        
        Returns:
            `True` if a matching custom configuration is registered for `model_id`, `False` otherwise.
        """
        return self.get_config(model_id) is not None

    def get_config(self, model_id: str) -> CustomModelConfig | None:
        """
        Retrieve a registered custom model configuration matching the given model identifier.
        
        Searches for a configuration by exact model_id match first, then by treating registered keys as regular-expression patterns and returning the first pattern that matches.
        
        Returns:
            CustomModelConfig | None: The matching configuration if found, `None` otherwise.
        """
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
        """
        Load a model and its tokenizer using registered custom factories or fallback HuggingFace strategies.
        
        If a matching CustomModelConfig exists and no explicit factory is provided, the config's factories and extra_kwargs are used. The tokenizer's pad token will be set to the eos token if missing.
        
        Parameters:
            model_id: Identifier of the model to load.
            device: Target device string (e.g., "cuda:0").
            dtype: Data type hint (e.g., "auto", "float16").
            model_factory: Optional callable or import path used to load the model; if None, a fallback loader is attempted.
            tokenizer_factory: Optional callable or import path used to load the tokenizer; if None, AutoTokenizer.from_pretrained is used.
            trust_remote_code: Whether to allow execution of remote model code when loading.
            quantization: Optional quantization specifier; not directly handled by the loader and only meaningful to specific factories or fallback loaders.
            **kwargs: Additional keyword arguments passed to the model/tokenizer factories; merged with config.extra_kwargs when a config is present.
        
        Returns:
            LoadedModel: An object containing the loaded model and tokenizer plus metadata (including load_time_seconds, trust_remote_code, and the matched custom_config pattern when applicable).
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
        """
        Attempt to load a model using multiple Hugging Face Transformer loading strategies and return the first successful model.
        
        Attempts common strategies (e.g., causal LM first, then generic model) and returns the loaded model on success.
        
        Returns:
            The loaded model instance.
        
        Raises:
            RuntimeError: If all loading strategies fail; message includes the last encountered error.
        """
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
        """
        Determine the model's hidden size, using an override from `config` when provided.
        
        Parameters:
            model (Any): Model object to inspect for hidden-size attributes.
            config (CustomModelConfig | None): Optional config whose `hidden_size` takes precedence if set.
        
        Returns:
            int: Hidden size (e.g., embedding dimension). If not found on the model or in `config`, returns 4096.
        """
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
        """
        Determine the model's number of transformer layers, using an explicit config override when available.
        
        If `config.num_layers` is provided, that value is returned. Otherwise the function examines common model config attributes (such as `num_hidden_layers`, `n_layer`, or `num_layers`) and returns the first one found. If none are present, it logs a warning and returns 32.
        
        Parameters:
            model (Any): Loaded model instance whose configuration will be inspected.
            config (CustomModelConfig | None): Optional custom config that may contain a `num_layers` override.
        
        Returns:
            int: The determined number of layers, or 32 if it cannot be inferred.
        """
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
        """
        Generate text from a loaded model using the HuggingFace-style generation interface.
        
        Generates tokens for the provided prompt and returns decoded text, optional extracted hidden states for requested layers, and timing/throughput metadata. If the model lacks a `generate` method or generation raises an exception, the function returns a GenerationOutput containing an error message in metadata rather than raising.
        
        Parameters:
            loaded_model (LoadedModel): LoadedModel containing model, tokenizer, device, and model metadata.
            prompt (str): Input text prompt to generate from.
            max_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature; values <= 0 are treated as 1.0. When > 0, sampling is enabled.
            return_hidden_states (bool): Whether to attempt extraction of hidden states from generation outputs.
            hidden_state_layers (list[int] | None): List of layer indices to extract hidden states from; supports negative indices (e.g., -1 for last layer). If None, defaults to [-1].
            return_attention (bool): Whether to request attention matrices from the model (returned value is simplified/None for this loader).
            **kwargs: Additional keyword arguments forwarded to the model's `generate` call.
        
        Returns:
            GenerationOutput: Object containing:
              - `text`: decoded generated text string.
              - `token_ids`: list of generated token ids.
              - `hidden_states`: mapping of requested layer index to tensor (or None).
              - `attention_weights`: None for this loader (attention extraction is simplified).
              - `metadata`: dictionary with `inference_time_ms`, `tokens_generated`, `tokens_per_second`, `model_id`, and on error an `error` key with the error message.
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
            raise NotImplementedError(
                f"Model {loaded_model.model_id} does not support text generation "
                "(no generate() method). Use embed() for embedding extraction instead."
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
                raise RuntimeError(
                    f"Generation failed for model {loaded_model.model_id}: {e}"
                ) from e

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
        """
        Map requested layer indices to the final-step hidden-state tensors for the last token.
        
        Parameters:
            hidden_states (tuple): Sequence of per-layer hidden-state tuples returned by the model; expects the final generation step at index -1.
            layers (list[int]): List of layer indices to extract. Negative indices are interpreted relative to `num_layers` (e.g., -1 means last layer).
            num_layers (int): Total number of layers in the model; used to resolve negative indices.
        
        Returns:
            dict[int, torch.Tensor]: A mapping from each requested layer index (as provided in `layers`) to the corresponding hidden-state tensor
            for the last token of the final generation step, moved to CPU. Layers that are out of range are omitted; returns an empty dict if no hidden states are available.
        """
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
        """
        Compute a pooled embedding for the given text using the provided loaded model.
        
        Performs a forward pass with hidden states enabled, selects the last layer's hidden states (or last_hidden_state), and applies the specified pooling strategy.
        
        Parameters:
            loaded_model (LoadedModel): LoadedModel containing model, tokenizer, device, and metadata.
            text (str): Input text to embed.
            pooling (str): Pooling strategy to apply to token embeddings. One of:
                - "last_token": use the embedding of the last real token (based on attention mask).
                - "mean": mean-pool token embeddings weighted by attention mask.
                - "first_token": use the first token embedding.
            **kwargs: Reserved for future options (ignored).
        
        Returns:
            EmbeddingOutput: Contains:
                - embedding: CPU tensor of the resulting embedding (batch dimension removed).
                - shape: tuple describing the embedding shape.
                - metadata: dict with keys including "pooling", "inference_time_ms", "input_tokens", and "model_id".
        
        Raises:
            ValueError: If the model outputs neither `hidden_states` nor `last_hidden_state`, or if an unknown pooling strategy is requested.
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
            mask_sum = attention_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
            embedding = (last_hidden * attention_mask).sum(dim=1) / mask_sum
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