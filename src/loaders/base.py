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
        """Move model to a different device."""
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
        """Check if this loader can handle the given model.

        Args:
            model_id: HuggingFace model ID or local path

        Returns:
            True if this loader can load the model
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
        """Load a model and tokenizer.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on (e.g., "cuda:0", "cpu")
            dtype: Data type (auto, float16, bfloat16, float32)
            **kwargs: Additional loader-specific arguments

        Returns:
            LoadedModel container with model, tokenizer, and metadata
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
        """Generate text and optionally extract hidden states.

        Args:
            loaded_model: Previously loaded model container
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Whether to return hidden states
            hidden_state_layers: Which layers to return (-1 = last)
            return_attention: Whether to return attention weights
            **kwargs: Additional generation arguments

        Returns:
            GenerationOutput with text, tokens, and optionally hidden states
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
        """Extract embedding for text.

        Args:
            loaded_model: Previously loaded model container
            text: Input text to embed
            pooling: Pooling strategy (last_token, mean, cls)
            **kwargs: Additional arguments

        Returns:
            EmbeddingOutput with embedding tensor and metadata
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
        """Generate text with streaming token output.

        Yields tokens as they're generated, then optionally yields
        a StreamingOutput with hidden states at the end.

        Default implementation falls back to non-streaming generate.

        Args:
            loaded_model: Previously loaded model container
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Whether to return hidden states at end
            hidden_state_layers: Which layers to return (-1 = last)
            **kwargs: Additional generation arguments

        Yields:
            StreamingToken for each token, then StreamingOutput at end
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
    """Resolve dtype string to torch.dtype.

    Args:
        dtype_str: One of "auto", "float16", "bfloat16", "float32"
        device: Target device (affects auto resolution)

    Returns:
        Resolved torch.dtype
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
