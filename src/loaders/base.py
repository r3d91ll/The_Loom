"""Base classes for model loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class StreamingToken:
    """A single token from streaming generation."""

    token: str
    token_id: int
    is_finished: bool = False
    finish_reason: str | None = None  # "stop", "length", "error"


@dataclass
class StreamingOutput:
    """Final output from streaming generation (sent at end)."""

    text: str
    token_ids: list[int]
    token_count: int
    hidden_states: dict[int, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationOutput:
    """Output from model generation including hidden states."""

    text: str
    token_ids: list[int]
    hidden_states: dict[int, torch.Tensor] | None = None  # layer_idx -> tensor [hidden_size]
    attention_weights: dict[int, torch.Tensor] | None = None  # layer_idx -> tensor
    # Full sequence hidden states for manifold construction
    # Shape: layer_idx -> tensor [num_tokens, hidden_size]
    sequence_hidden_states: dict[int, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingOutput:
    """Output from embedding extraction."""

    embedding: torch.Tensor
    shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
    """Container for a loaded model and its components."""

    model: PreTrainedModel | Any
    tokenizer: PreTrainedTokenizer | Any
    model_id: str
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    num_layers: int
    loader_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> LoadedModel:
        """
        Move the contained model to the specified device and update the container's device attribute.
        
        Parameters:
            device (torch.device | str): Target device or device string (e.g., "cuda:0", "cpu").
        
        Returns:
            LoadedModel: The same LoadedModel instance after the model has been moved to the target device.
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.model = self.model.to(device)  # type: ignore[arg-type]
        self.device = device
        return self


class ModelLoader(ABC):
    """Abstract base class for model loaders.

    Each loader handles a specific type of model architecture or loading strategy.
    The loader is responsible for:
    - Loading model and tokenizer
    - Generating text with hidden state extraction
    - Extracting embeddings
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this loader type."""
        ...

    @abstractmethod
    def can_load(self, model_id: str) -> bool:
        """
        Determine whether this loader is able to load the specified model.
        
        Parameters:
            model_id (str): HuggingFace model ID or local filesystem path identifying the model.
        
        Returns:
            `True` if the loader can load the model, `False` otherwise.
        """
        ...

    @abstractmethod
    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        **kwargs: Any,
    ) -> LoadedModel:
        """
        Load a model and its tokenizer into a LoadedModel container.
        
        Parameters:
            model_id (str): HuggingFace model identifier or local filesystem path.
            device (str): Target device for the model (e.g., "cuda:0" or "cpu").
            dtype (str): Data type selection; one of "auto", "float16", "bfloat16", or "float32".
            **kwargs: Loader-specific options passed through to the implementation.
        
        Returns:
            LoadedModel: Container holding the loaded model, tokenizer, model_id, device, dtype, model dimensions, loader_type, and any loader metadata.
        """
        ...

    @abstractmethod
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
        Produce generated text for a prompt and optionally include model hidden states and attention weights.
        
        Parameters:
            loaded_model (LoadedModel): Container holding the model and tokenizer for generation.
            prompt (str): Input prompt text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature controlling randomness.
            return_hidden_states (bool): If true, include hidden states in the result.
            hidden_state_layers (list[int] | None): Specific layer indices to return; use [-1] or -1 to indicate the last layer.
            return_attention (bool): If true, include attention weight tensors in the result.
            **kwargs (Any): Additional backend-specific generation options.
        
        Returns:
            GenerationOutput: Generated text, token ids, and optionally hidden_states and attention_weights along with metadata.
        """
        ...

    @abstractmethod
    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """
        Compute an embedding vector for the given text using the provided loaded model.
        
        Parameters:
            loaded_model (LoadedModel): Container with model and tokenizer to use for embedding.
            text (str): Input text to convert into an embedding.
            pooling (str): Pooling strategy to produce a single vector from token-level representations. Supported values: "last_token", "mean", "cls".
            **kwargs: Additional backend-specific options passed to the loader implementation.
        
        Returns:
            EmbeddingOutput: Contains the embedding tensor, its shape, and any backend metadata.
        """
        ...

    def generate_stream(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = False,
        hidden_state_layers: list[int] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamingToken | StreamingOutput]:
        """
        Produce streaming token events for a generation request, falling back to the synchronous `generate` implementation.
        
        This default implementation calls `generate`, yields a StreamingToken for each produced token (decoded via the loader's tokenizer), and then yields a final StreamingOutput containing the full text, token ids, token count, optional hidden states, and metadata.
        
        Parameters:
            loaded_model: Container holding the model and tokenizer used for decoding.
            prompt: Input prompt text to generate from.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature for generation.
            return_hidden_states: If true, include hidden states in the final StreamingOutput.
            hidden_state_layers: Specific layer indices to include in hidden states (`-1` denotes the last layer). If None, layer selection is backend-dependent.
            **kwargs: Additional generation options forwarded to the underlying `generate` method.
        
        Yields:
            StreamingToken for each generated token (with `is_finished` true and `finish_reason="stop"` on the last token), followed by a single StreamingOutput summarizing the complete generation.
        """
        # Default: fall back to non-streaming
        output = self.generate(
            loaded_model=loaded_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_hidden_states=return_hidden_states,
            hidden_state_layers=hidden_state_layers,
            **kwargs,
        )

        # Yield tokens one at a time
        for i, token_id in enumerate(output.token_ids):
            token_text = loaded_model.tokenizer.decode([token_id])
            is_last = i == len(output.token_ids) - 1
            yield StreamingToken(
                token=token_text,
                token_id=token_id,
                is_finished=is_last,
                finish_reason="stop" if is_last else None,
            )

        # Yield final output with hidden states
        yield StreamingOutput(
            text=output.text,
            token_ids=output.token_ids,
            token_count=len(output.token_ids),
            hidden_states=output.hidden_states,
            metadata=output.metadata,
        )


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """
    Resolve a dtype name to a torch.dtype, using the device when dtype_str is "auto".
    
    Parameters:
        dtype_str (str): One of "auto", "float16", "bfloat16", or "float32".
        device (torch.device): Target device used to decide "auto" resolution.
    
    Returns:
        torch.dtype: The resolved torch dtype. For "auto", returns `torch.bfloat16` on CUDA devices with major capability >= 8, `torch.float16` on other CUDA devices, and `torch.float32` on non-CUDA devices.
    """
    if dtype_str == "auto":
        # Use bfloat16 on modern CUDA devices, float16 otherwise
        if device.type == "cuda":
            capability = torch.cuda.get_device_capability(device)
            if capability[0] >= 8:  # Ampere or newer
                return torch.bfloat16
            return torch.float16
        return torch.float32

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Use: {list(dtype_map.keys())}")

    return dtype_map[dtype_str]