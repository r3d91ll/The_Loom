"""HuggingFace Transformers model loader with hidden state extraction."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    StreamingOutput,
    StreamingToken,
    resolve_dtype,
)

logger = logging.getLogger(__name__)


class TransformersLoader(ModelLoader):
    """Model loader using HuggingFace Transformers.

    This is the primary loader covering ~80% of models including:
    - LLaMA, Mistral, Qwen, Phi families
    - GPT-2, GPT-Neo, GPT-J
    - CodeLlama, StarCoder
    - Most decoder-only transformer models

    Key capability: Proper hidden state extraction via output_hidden_states=True
    """

    @property
    def name(self) -> str:
        return "transformers"

    def can_load(self, model_id: str) -> bool:
        """Check if model can be loaded with transformers.

        Most models work with transformers, so this returns True by default.
        Specific exclusions can be added for models known to require other loaders.
        """
        model_lower = model_id.lower()

        # Known models that need sentence-transformers (embedding models)
        embedding_patterns = [
            "sentence-transformers/",
            "baai/bge-",
            "intfloat/e5-",
            "intfloat/multilingual-e5-",
            "hkunlp/instructor-",
            "thenlper/gte-",
            "jinaai/jina-embeddings-",
            "nomic-ai/nomic-embed-",
        ]

        for pattern in embedding_patterns:
            if pattern in model_lower:
                return False

        return True

    def load(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a model using HuggingFace Transformers.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load on (cuda:0, cpu, etc.)
            dtype: Data type (auto, float16, bfloat16, float32)
            trust_remote_code: Allow remote code execution for custom architectures
            quantization: Quantization mode (None, "4bit", "8bit", "gptq", "awq")
            **kwargs: Passed to AutoModelForCausalLM.from_pretrained

        Returns:
            LoadedModel with model, tokenizer, and metadata
        """
        quant_info = quantization or "none"
        logger.info(f"Loading model {model_id} on {device} with dtype={dtype}, quant={quant_info}")
        start_time = time.time()

        # Resolve device and dtype
        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build model loading kwargs based on quantization
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "output_hidden_states": True,
        }

        # Apply quantization configuration
        if quantization in ("4bit", "8bit"):
            model_kwargs.update(
                self._get_bitsandbytes_config(quantization, torch_dtype)
            )
            # Use device_map for automatic placement with quantization
            model_kwargs["device_map"] = "auto"
        elif quantization == "gptq":
            # GPTQ models auto-detect quantization config
            model_kwargs["device_map"] = device
            model_kwargs["torch_dtype"] = torch_dtype
            logger.info("Using GPTQ quantization (auto-detected from model)")
        elif quantization == "awq":
            # AWQ models auto-detect quantization config
            model_kwargs["device_map"] = device
            model_kwargs["torch_dtype"] = torch_dtype
            logger.info("Using AWQ quantization (auto-detected from model)")
        else:
            # No quantization
            model_kwargs["device_map"] = device
            model_kwargs["torch_dtype"] = torch_dtype

        # Merge with additional kwargs
        model_kwargs.update(kwargs)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        model.eval()  # Inference mode

        # Extract model config info
        config = model.config
        hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", 4096))
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 32))

        load_time = time.time() - start_time
        logger.info(
            f"Model loaded in {load_time:.2f}s - "
            f"hidden_size={hidden_size}, num_layers={num_layers}, quant={quant_info}"
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
                "model_type": getattr(config, "model_type", "unknown"),
                "quantization": quant_info,
            },
        )

    def _get_bitsandbytes_config(
        self,
        mode: str,
        compute_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Get BitsAndBytes configuration for quantization.

        Args:
            mode: "4bit" or "8bit"
            compute_dtype: Compute dtype for quantized operations

        Returns:
            Dict with quantization_config for model loading
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for 4bit/8bit quantization. "
                "Install with: pip install bitsandbytes"
            ) from e

        if mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization with NF4")
        elif mode == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")
        else:
            raise ValueError(f"Unknown quantization mode: {mode}")

        return {"quantization_config": quantization_config}

    def generate(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        return_attention: bool = False,
        return_full_sequence: bool = False,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Generate text with hidden state extraction.

        This is the core research capability - extracting the geometric
        representation of meaning before it becomes text.

        Args:
            loaded_model: Previously loaded model
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            return_hidden_states: Extract hidden states
            hidden_state_layers: Which layers (-1 = last, None = all requested)
            return_attention: Extract attention weights
            return_full_sequence: Return hidden states for ALL generated tokens,
                not just the last. Creates a [num_tokens, hidden_size] tensor
                that represents the manifold/boundary object geometry.
            top_p: Nucleus sampling probability
            do_sample: Use sampling vs greedy decoding
            **kwargs: Additional generation args

        Returns:
            GenerationOutput with text and optionally hidden states
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Default to last layer only
        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample and temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "output_hidden_states": return_hidden_states,
            "output_attentions": return_attention,
            "return_dict_in_generate": True,
            **kwargs,
        }

        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]

        inference_time = time.time() - start_time

        # Extract generated tokens (excluding input)
        generated_ids = outputs.sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if requested
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        sequence_hidden_states_dict: dict[int, torch.Tensor] | None = None

        if return_hidden_states and hasattr(outputs, "hidden_states"):
            # Always extract last token's hidden state
            hidden_states_dict = self._extract_hidden_states(
                outputs.hidden_states,
                hidden_state_layers,
                loaded_model.num_layers,
            )

            # Optionally extract full sequence for manifold construction
            if return_full_sequence:
                sequence_hidden_states_dict = self._extract_sequence_hidden_states(
                    outputs.hidden_states,
                    hidden_state_layers,
                    loaded_model.num_layers,
                )

        # Extract attention if requested
        attention_dict: dict[int, torch.Tensor] | None = None
        if return_attention and hasattr(outputs, "attentions"):
            attention_dict = self._extract_attention(
                outputs.attentions,
                hidden_state_layers,
                loaded_model.num_layers,
            )

        return GenerationOutput(
            text=generated_text,
            token_ids=generated_ids,
            hidden_states=hidden_states_dict,
            attention_weights=attention_dict,
            sequence_hidden_states=sequence_hidden_states_dict,
            metadata={
                "inference_time_ms": inference_time * 1000,
                "tokens_generated": len(generated_ids),
                "tokens_per_second": len(generated_ids) / inference_time
                if inference_time > 0
                else 0,
                "input_tokens": input_length,
                "temperature": temperature,
                "model_id": loaded_model.model_id,
                "full_sequence": return_full_sequence,
            },
        )

    def _extract_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states from generation output.

        The hidden_states tuple from generate() is structured as:
        - Tuple of generation steps
        - Each step has tuple of (num_layers + 1) tensors (including embeddings)
        - Each tensor is [batch, seq_len, hidden_size]

        For the final hidden state (boundary object), we want the last token's
        representation from the last layer at the final generation step.
        """
        result: dict[int, torch.Tensor] = {}

        # Get the final generation step's hidden states
        # hidden_states is a tuple of tuples: (step, layer)
        if not hidden_states:
            return result

        # The last step contains the most recent hidden states
        final_step = hidden_states[-1]

        for layer_idx in layers:
            # Convert negative indices
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            if 0 <= actual_idx < len(final_step):
                # Extract last token's hidden state from this layer
                # Shape: [batch, seq_len, hidden] -> [batch, hidden]
                layer_hidden = final_step[actual_idx][:, -1, :].cpu()
                result[layer_idx] = layer_hidden

        return result

    def _extract_attention(
        self,
        attentions: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract attention weights from generation output."""
        result: dict[int, torch.Tensor] = {}

        if not attentions:
            return result

        # Similar structure to hidden states
        final_step = attentions[-1]

        for layer_idx in layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

            if 0 <= actual_idx < len(final_step):
                # Shape: [batch, heads, seq, seq] -> keep last query position
                layer_attn = final_step[actual_idx][:, :, -1, :].cpu()
                result[layer_idx] = layer_attn

        return result

    def _extract_sequence_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states for ALL generated tokens (manifold construction).

        This creates the full geometric representation of the generation -
        each token's position in the model's semantic space, forming the
        "boundary object" manifold.

        The hidden_states tuple from generate() is structured as:
        - Tuple of generation steps (one per token generated)
        - Each step has tuple of (num_layers + 1) tensors
        - Each tensor is [batch, seq_len, hidden_size]

        We extract the last token position from each step, giving us
        the newly generated token's representation at each step.

        Returns:
            dict mapping layer_idx to tensor of shape [num_tokens, hidden_size]
        """
        result: dict[int, torch.Tensor] = {}

        if not hidden_states:
            return result

        for layer_idx in layers:
            # Convert negative indices
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            # Collect hidden state for each generation step
            step_vectors = []
            for step in hidden_states:
                if 0 <= actual_idx < len(step):
                    # Get last token's hidden state for this step
                    # Shape: [batch, seq_len, hidden] -> [hidden]
                    token_hidden = step[actual_idx][0, -1, :].cpu()
                    step_vectors.append(token_hidden)

            if step_vectors:
                # Stack into [num_tokens, hidden_size] matrix
                sequence_tensor = torch.stack(step_vectors, dim=0)
                result[layer_idx] = sequence_tensor

        return result

    def embed(
        self,
        loaded_model: LoadedModel,
        text: str,
        pooling: str = "last_token",
        **kwargs: Any,
    ) -> EmbeddingOutput:
        """Extract embedding from text using specified pooling.

        For decoder-only models, the last token embedding is typically
        the best representation (contains accumulated context).

        Args:
            loaded_model: Previously loaded model
            text: Text to embed
            pooling: Pooling strategy
                - "last_token": Use last token's hidden state (recommended)
                - "mean": Mean pool across all tokens
                - "first_token": Use first token (CLS-style)

        Returns:
            EmbeddingOutput with embedding tensor
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        # Tokenize
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

        # Get last layer hidden states: [batch, seq_len, hidden_size]
        last_hidden = outputs.hidden_states[-1]

        # Apply pooling
        if pooling == "last_token":
            # Get last non-padding token
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
            raise ValueError(f"Unknown pooling: {pooling}. Use: last_token, mean, first_token")

        embedding = embedding.cpu()

        return EmbeddingOutput(
            embedding=embedding.squeeze(0),  # Remove batch dim for single input
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": inputs.input_ids.shape[1],
                "model_id": loaded_model.model_id,
            },
        )

    def generate_stream(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        return_hidden_states: bool = False,
        hidden_state_layers: list[int] | None = None,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> Iterator[StreamingToken | StreamingOutput]:
        """Generate text with true streaming token output.

        Uses TextIteratorStreamer for real-time token streaming.

        Args:
            loaded_model: Previously loaded model container
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Whether to return hidden states at end
            hidden_state_layers: Which layers to return (-1 = last)
            top_p: Nucleus sampling probability
            do_sample: Use sampling vs greedy decoding
            **kwargs: Additional generation arguments

        Yields:
            StreamingToken for each token, then StreamingOutput at end
        """
        model = loaded_model.model
        tokenizer = loaded_model.tokenizer
        device = loaded_model.device

        if hidden_state_layers is None:
            hidden_state_layers = [-1]

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer,  # type: ignore[arg-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p if do_sample else 1.0,
            "do_sample": do_sample and temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "output_hidden_states": return_hidden_states,
            "return_dict_in_generate": True,
            "streamer": streamer,
            **kwargs,
        }

        start_time = time.time()

        # Track generated tokens
        generated_tokens: list[int] = []
        generated_text_parts: list[str] = []
        generation_output: Any = None

        # Run generation in background thread
        def generate_thread() -> None:
            nonlocal generation_output
            with torch.no_grad():
                generation_output = model.generate(**inputs, **gen_kwargs)  # type: ignore[operator]

        thread = threading.Thread(target=generate_thread)
        thread.start()

        # Stream tokens
        token_count = 0
        try:
            for text_chunk in streamer:
                if text_chunk:
                    generated_text_parts.append(text_chunk)
                    # Estimate token ID (actual IDs come from final output)
                    token_count += 1
                    yield StreamingToken(
                        token=text_chunk,
                        token_id=token_count,  # Placeholder, updated in final output
                        is_finished=False,
                        finish_reason=None,
                    )
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamingToken(
                token="",
                token_id=-1,
                is_finished=True,
                finish_reason="error",
            )
            thread.join()
            return

        # Wait for generation to complete
        thread.join()
        inference_time = time.time() - start_time

        # Extract actual token IDs and hidden states from final output
        if generation_output is not None:
            generated_tokens = generation_output.sequences[0, input_length:].tolist()

            # Extract hidden states if requested
            hidden_states_dict: dict[int, torch.Tensor] | None = None
            if return_hidden_states and hasattr(generation_output, "hidden_states"):
                hidden_states_dict = self._extract_hidden_states(
                    generation_output.hidden_states,
                    hidden_state_layers,
                    loaded_model.num_layers,
                )

            # Yield final token marker
            yield StreamingToken(
                token="",
                token_id=generated_tokens[-1] if generated_tokens else -1,
                is_finished=True,
                finish_reason="stop",
            )

            # Yield final output with all data
            yield StreamingOutput(
                text="".join(generated_text_parts),
                token_ids=generated_tokens,
                token_count=len(generated_tokens),
                hidden_states=hidden_states_dict,
                metadata={
                    "inference_time_ms": inference_time * 1000,
                    "tokens_generated": len(generated_tokens),
                    "tokens_per_second": len(generated_tokens) / inference_time
                    if inference_time > 0
                    else 0,
                    "input_tokens": input_length,
                    "temperature": temperature,
                    "model_id": loaded_model.model_id,
                    "streaming": True,
                },
            )
