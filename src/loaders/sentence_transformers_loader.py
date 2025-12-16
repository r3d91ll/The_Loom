"""SentenceTransformers model loader for embedding models and specialized variants.

This loader handles models that work best with the sentence-transformers library,
including:
- SBERT models (sentence-transformers/*)
- Some DeepSeek variants
- Instructor models
- E5 models
- BGE models
- Research embedding models

Coverage: ~15% of models (primarily embedding-focused)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

# Known model patterns that work well with sentence-transformers
SENTENCE_TRANSFORMER_PATTERNS = [
    "sentence-transformers/",
    "BAAI/bge-",
    "intfloat/e5-",
    "intfloat/multilingual-e5-",
    "hkunlp/instructor-",
    "thenlper/gte-",
    "jinaai/jina-embeddings-",
    "nomic-ai/nomic-embed-",
]

# Models that should NOT use sentence-transformers even if they match patterns
EXCLUDED_PATTERNS = [
    "-instruct",  # Instruction-tuned models usually need transformers
    "-chat",
]


class SentenceTransformersLoader(ModelLoader):
    """Model loader using sentence-transformers library.

    Best for:
    - Embedding models (SBERT, BGE, E5, etc.)
    - Models with custom pooling strategies
    - Models that require specific tokenization

    Note: This loader is primarily for embedding extraction. Generation
    capability is limited and falls back to a simple approach.
    """

    @property
    def name(self) -> str:
        return "sentence_transformers"

    def can_load(self, model_id: str) -> bool:
        """Check if this loader should handle the model.

        Returns True for known sentence-transformer models and embedding models.
        """
        model_lower = model_id.lower()

        # Check exclusions first
        for pattern in EXCLUDED_PATTERNS:
            if pattern in model_lower:
                return False

        # Check known patterns
        for pattern in SENTENCE_TRANSFORMER_PATTERNS:
            if pattern.lower() in model_lower:
                return True

        return False

    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model using sentence-transformers.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load on
            dtype: Data type (note: sentence-transformers has limited dtype support)
            trust_remote_code: Allow remote code execution
            quantization: Not supported for sentence-transformers (ignored with warning)
            **kwargs: Additional arguments passed to SentenceTransformer

        Returns:
            LoadedModel with the sentence-transformer model
        """
        if quantization:
            logger.warning(
                f"Quantization '{quantization}' requested but not supported by "
                "SentenceTransformersLoader. Loading without quantization."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from err

        logger.info(f"Loading model {model_id} with sentence-transformers on {device}")
        start_time = time.time()

        # Resolve device
        torch_device = torch.device(device)

        # Load the model
        model = SentenceTransformer(
            model_id,
            device=str(torch_device),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        # Get model dimensions
        hidden_size = model.get_sentence_embedding_dimension()

        # Estimate number of layers (sentence-transformers abstracts this)
        # We'll try to introspect the underlying model
        num_layers = self._estimate_num_layers(model)

        # Resolve dtype for metadata (sentence-transformers manages this internally)
        torch_dtype = resolve_dtype(dtype, torch_device)

        load_time = time.time() - start_time
        logger.info(
            f"Model loaded in {load_time:.2f}s - "
            f"hidden_size={hidden_size}, estimated_layers={num_layers}"
        )

        return LoadedModel(
            model=model,
            tokenizer=model.tokenizer,  # SentenceTransformer has tokenizer attribute
            model_id=model_id,
            device=torch_device,
            dtype=torch_dtype,
            hidden_size=hidden_size,
            num_layers=num_layers,
            loader_type=self.name,
            metadata={
                "load_time_seconds": load_time,
                "trust_remote_code": trust_remote_code,
                "model_type": "sentence_transformer",
                "max_seq_length": getattr(model, "max_seq_length", None),
            },
        )

    def _estimate_num_layers(self, model: Any) -> int:
        """Estimate number of transformer layers in the model."""
        try:
            # Try to access the underlying transformer
            if hasattr(model, "_first_module"):
                first_module = model._first_module()
                if hasattr(first_module, "auto_model"):
                    config = first_module.auto_model.config
                    return getattr(
                        config,
                        "num_hidden_layers",
                        getattr(config, "n_layer", 12),
                    )
        except Exception:
            pass
        return 12  # Default assumption

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
        """Generate text using sentence-transformer model.

        Note: Sentence-transformers are primarily embedding models. This method
        provides limited generation capability by encoding the prompt and
        returning a placeholder response with the hidden states.

        For actual text generation, use TransformersLoader instead.
        """
        model = loaded_model.model

        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        start_time = time.time()

        # Encode the prompt to get hidden states
        # We use encode with output_value parameter to get embeddings
        with torch.no_grad():
            embedding = model.encode(  # type: ignore[operator]
                prompt,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        inference_time = time.time() - start_time

        # Build hidden states dict
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        if return_hidden_states:
            # Sentence-transformers gives us the final pooled embedding
            # We'll return it as the "last layer" hidden state
            hidden_states_dict = {
                -1: embedding.unsqueeze(0).cpu(),  # Add batch dimension
            }

        # Note: We can't actually generate text with sentence-transformers
        # Return a message indicating this limitation
        generated_text = (
            "[SentenceTransformersLoader: This model is optimized for embeddings, "
            "not text generation. Use TransformersLoader for generation tasks.]"
        )

        return GenerationOutput(
            text=generated_text,
            token_ids=[],  # No tokens generated
            hidden_states=hidden_states_dict,
            attention_weights=None,  # Not available from sentence-transformers
            metadata={
                "inference_time_ms": inference_time * 1000,
                "tokens_generated": 0,
                "tokens_per_second": 0,
                "model_id": loaded_model.model_id,
                "note": "embedding_model_limited_generation",
            },
        )

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "default",
        normalize: bool = False,
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Extract embedding using sentence-transformers.

        This is the primary use case for this loader. Sentence-transformers
        handles tokenization, encoding, and pooling automatically.

        Args:
            loaded_model: Previously loaded model
            text: Text to embed
            pooling: Pooling strategy (sentence-transformers handles this)
                - "default": Use model's built-in pooling
                - Other values are ignored (model uses its own strategy)
            normalize: Whether to L2 normalize (passed to encode)
            **kwargs: Additional arguments

        Returns:
            EmbeddingOutput with embedding tensor
        """
        model = loaded_model.model

        start_time = time.time()

        # Encode using sentence-transformers
        with torch.no_grad():
            embedding = model.encode(  # type: ignore[operator]
                text,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )

        inference_time = time.time() - start_time

        # Ensure tensor is on CPU for serialization
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu()
        else:
            embedding = torch.tensor(embedding)

        return EmbeddingOutput(
            embedding=embedding,
            shape=tuple(embedding.shape),
            metadata={
                "pooling": "sentence_transformer_default",
                "inference_time_ms": inference_time * 1000,
                "model_id": loaded_model.model_id,
                "normalized": normalize,
            },
        )

    def encode_batch(
        self,
        loaded_model: LoadedModel,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple texts efficiently.

        This method leverages sentence-transformers' built-in batching
        for efficient encoding of multiple texts.

        Args:
            loaded_model: Previously loaded model
            texts: List of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize
            show_progress: Show progress bar

        Returns:
            Numpy array of shape [num_texts, embedding_dim]
        """
        model = loaded_model.model

        embeddings: np.ndarray = model.encode(  # type: ignore[operator]
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings
