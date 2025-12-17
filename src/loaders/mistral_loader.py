"""Mistral-specific model loader with mistral_common tokenizer support.

This loader handles newer Mistral models that require:
1. The `mistral_common` tokenizer library instead of AutoTokenizer
2. Registration of `ministral3` model type for multimodal models
3. FP8 quantization dequantization support

Supported models:
- mistralai/Devstral-Small-2-24B-Instruct-2512 (FP8, multimodal)
- mistralai/Devstral-Small-2507 (BF16, standard)
- mistralai/Ministral-* models
- Other recent Mistral releases using mistral3/ministral3 architecture
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from transformers import AutoModelForCausalLM

from .base import (
    EmbeddingOutput,
    GenerationOutput,
    LoadedModel,
    ModelLoader,
    StreamingOutput,
    StreamingToken,
    resolve_dtype,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Register ministral3 config on import to allow loading newer Mistral models
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.mistral.configuration_mistral import MistralConfig

    if "ministral3" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING.register("ministral3", MistralConfig)
        logger.debug("Registered 'ministral3' as alias for MistralConfig")
except ImportError:
    logger.warning("Could not register ministral3 config - transformers not available")
except Exception as e:
    logger.warning(f"Could not register ministral3 config: {e}")


class MistralTokenizerWrapper:
    """Wrapper around mistral_common tokenizer to provide HF-compatible interface.

    The mistral_common tokenizer has a different API than HuggingFace tokenizers.
    This wrapper provides the methods expected by the loader and model.
    """

    def __init__(self, model_id: str):
        """Initialize wrapper with mistral_common tokenizer.

        Args:
            model_id: HuggingFace model ID to load tokenizer for
        """
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

        self._tokenizer = MistralTokenizer.from_hf_hub(model_id)
        # Access the underlying tekken tokenizer for special tokens
        self._tekken = self._tokenizer.instruct_tokenizer.tokenizer
        self._vocab_size: int = self._tekken.n_words
        self._pad_token_id: int | None = self._tekken.pad_id
        self._eos_token_id: int | None = self._tekken.eos_id
        self._bos_token_id: int | None = self._tekken.bos_id

    @property
    def pad_token(self) -> str:
        """Return pad token string."""
        return "<pad>"

    @property
    def eos_token(self) -> str:
        """Return EOS token string."""
        return "</s>"

    @property
    def bos_token(self) -> str:
        """Return BOS token string."""
        return "<s>"

    @property
    def pad_token_id(self) -> int | None:
        """Return pad token ID."""
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        """Return EOS token ID."""
        return self._eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        """Return BOS token ID."""
        return self._bos_token_id

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize text, matching HuggingFace tokenizer interface.

        Args:
            text: Text or list of texts to tokenize
            return_tensors: Return format ("pt" for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dict with input_ids and attention_mask
        """
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        all_input_ids = []
        for t in texts:
            # Create a simple user message request
            request = ChatCompletionRequest(messages=[UserMessage(content=t)])
            encoded = self._tokenizer.encode_chat_completion(request)
            all_input_ids.append(encoded.tokens)

        # Handle padding
        if padding and len(all_input_ids) > 1:
            # Determine pad token ID with safe fallback chain
            pad_id = self._pad_token_id
            if pad_id is None:
                pad_id = self._eos_token_id  # Common fallback
            if pad_id is None:
                raise ValueError(
                    "Cannot pad sequences: no pad_token_id or eos_token_id available. "
                    "Either disable padding or use a tokenizer with a defined pad token."
                )

            max_len = max(len(ids) for ids in all_input_ids)
            if max_length is not None:
                max_len = min(max_len, max_length)
            padded_ids = []
            attention_masks = []
            for ids in all_input_ids:
                if truncation and len(ids) > max_len:
                    ids = ids[:max_len]
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
            all_input_ids = padded_ids
        else:
            # Handle single sequence with potential truncation
            if truncation and max_length is not None:
                all_input_ids = [ids[:max_length] for ids in all_input_ids]
            attention_masks = [[1] * len(ids) for ids in all_input_ids]

        result: dict[str, Any] = {
            "input_ids": all_input_ids,
            "attention_mask": attention_masks,
        }

        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(all_input_ids)
            result["attention_mask"] = torch.tensor(attention_masks)

        return result

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to underlying decode

        Returns:
            Decoded text string
        """
        from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # Map skip_special_tokens to SpecialTokenPolicy
        decode_kwargs: dict[str, Any] = {**kwargs}
        if skip_special_tokens:
            decode_kwargs["special_token_policy"] = SpecialTokenPolicy.IGNORE

        # Use the underlying tekken tokenizer for decoding
        return self._tekken.decode(token_ids, **decode_kwargs)

    def to(self, device: torch.device | str) -> MistralTokenizerWrapper:
        """No-op for device placement (tokenizers don't need GPU)."""
        return self


class MistralLoader(ModelLoader):
    """Model loader for newer Mistral models using mistral_common.

    This loader handles Mistral models that require:
    - mistral_common tokenizer (newer models use custom tokenization)
    - ministral3 model type registration
    - FP8 quantization handling (dequantize to BF16 when needed)

    Works with:
    - mistralai/Devstral-Small-2-24B-Instruct-2512
    - mistralai/Devstral-Small-2507
    - mistralai/Ministral-* series
    - Other mistralai/* models with mistral3/ministral3 architecture
    """

    # Patterns that indicate a model needs the Mistral loader
    MISTRAL_PATTERNS: ClassVar[list[str]] = [
        "mistralai/devstral",
        "mistralai/ministral",
        "mistralai/mistral-small",
        "mistralai/codestral",
    ]

    # Models that should NOT use this loader (use standard transformers)
    EXCLUDE_PATTERNS: ClassVar[list[str]] = [
        "mistralai/mistral-7b",
        "mistralai/mixtral",
    ]

    @property
    def name(self) -> str:
        return "mistral"

    def can_load(self, model_id: str) -> bool:
        """Check if this loader should handle the model.

        Returns True for newer Mistral models that need mistral_common.
        """
        model_lower = model_id.lower()

        # Check exclusions first
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern in model_lower:
                return False

        # Check if it matches Mistral patterns
        for pattern in self.MISTRAL_PATTERNS:
            if pattern in model_lower:
                return True

        return False

    def _needs_mistral_tokenizer(self, model_id: str) -> bool:
        """Check if model needs mistral_common tokenizer.

        Some newer Mistral models use a custom tokenizer that requires
        mistral_common library.
        """
        model_lower = model_id.lower()
        # These model patterns need mistral_common tokenizer
        mistral_tokenizer_patterns = [
            "devstral-small-2-24b",  # FP8 multimodal
            "ministral",  # Ministral series
        ]
        return any(p in model_lower for p in mistral_tokenizer_patterns)

    def _has_fp8_quantization(self, model_id: str) -> bool:
        """Check if model has FP8 quantization from config."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            quant_config = getattr(config, "quantization_config", None)
            if quant_config:
                quant_method = getattr(quant_config, "quant_method", None)
                if quant_method == "fp8":
                    return True
        except Exception as e:
            logger.debug(f"Could not check FP8 config for {model_id}: {e}")
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
        """Load a Mistral model.

        Args:
            model_id: HuggingFace model ID
            device: Device to load on
            dtype: Data type (auto, float16, bfloat16, float32)
            trust_remote_code: Allow remote code
            quantization: Quantization mode (ignored for FP8 models)
            **kwargs: Additional arguments

        Returns:
            LoadedModel with model and tokenizer
        """
        logger.info(f"Loading Mistral model {model_id} on {device} with dtype={dtype}")
        start_time = time.time()

        # Resolve device and dtype
        torch_device = torch.device(device)
        torch_dtype = resolve_dtype(dtype, torch_device)

        # Load tokenizer - use mistral_common for newer models
        if self._needs_mistral_tokenizer(model_id):
            logger.info(f"Using mistral_common tokenizer for {model_id}")
            try:
                tokenizer: Any = MistralTokenizerWrapper(model_id)
            except Exception as e:
                logger.warning(
                    f"Failed to load mistral_common tokenizer: {e}. "
                    "Falling back to AutoTokenizer."
                )
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )

        # Ensure padding token exists
        if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
            if hasattr(tokenizer, "eos_token"):
                tokenizer.pad_token = tokenizer.eos_token

        # Check for FP8 quantization
        is_fp8 = self._has_fp8_quantization(model_id)

        # Build model loading kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "output_hidden_states": True,
        }

        if is_fp8:
            logger.info(f"Model {model_id} uses FP8 quantization - loading natively")
            # For FP8 models, we need special handling
            # Try to load with device_map auto for FP8
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch_dtype
            # Don't override quantization config for native FP8 models
        else:
            # Standard loading
            model_kwargs["device_map"] = device
            model_kwargs["torch_dtype"] = torch_dtype

            # Apply explicit quantization if requested
            if quantization in ("4bit", "8bit"):
                model_kwargs.update(self._get_bitsandbytes_config(quantization, torch_dtype))
                model_kwargs["device_map"] = "auto"

        # Merge with additional kwargs
        model_kwargs.update(kwargs)

        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except Exception as e:
            if is_fp8 and "Float8" in str(e):
                logger.warning(
                    f"FP8 loading failed: {e}. "
                    "Native FP8 support may not be available. "
                    "Try using a BF16 variant of this model."
                )
            raise

        model.eval()

        # Extract model config info
        config = model.config
        # Handle multimodal models with text_config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            hidden_size = getattr(text_config, "hidden_size", 4096)
            num_layers = getattr(text_config, "num_hidden_layers", 32)
        else:
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)

        load_time = time.time() - start_time
        model_type = getattr(config, "model_type", "unknown")

        # Determine actual device after loading (device_map="auto" may distribute model)
        device_map = getattr(model, "hf_device_map", None)
        if device_map and isinstance(device_map, dict):
            # Model is distributed; use first device as primary
            first_device = next(iter(device_map.values()))
            actual_device = torch.device(first_device)
            logger.debug(f"Model distributed across devices: {device_map}")
        else:
            actual_device = torch_device

        logger.info(
            f"Mistral model loaded in {load_time:.2f}s - "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"model_type={model_type}, device={actual_device}"
        )

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            device=actual_device,
            dtype=torch_dtype,
            hidden_size=hidden_size,
            num_layers=num_layers,
            loader_type=self.name,
            metadata={
                "load_time_seconds": load_time,
                "trust_remote_code": trust_remote_code,
                "model_type": model_type,
                "is_fp8": is_fp8,
                "uses_mistral_tokenizer": isinstance(tokenizer, MistralTokenizerWrapper),
                "device_map": getattr(model, "hf_device_map", None),
            },
        )

    def _get_bitsandbytes_config(
        self,
        mode: str,
        compute_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Get BitsAndBytes configuration for quantization."""
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
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
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

        Args:
            loaded_model: Previously loaded model
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Extract hidden states
            hidden_state_layers: Which layers (-1 = last)
            return_attention: Extract attention weights
            return_full_sequence: Return hidden states for all tokens
            top_p: Nucleus sampling probability
            do_sample: Use sampling vs greedy
            **kwargs: Additional generation args

        Returns:
            GenerationOutput with text and hidden states
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
        )

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            # Dict-style inputs
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        # Generation config
        gen_kwargs: dict[str, Any] = {
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

        # Extract generated tokens
        generated_ids = outputs.sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract hidden states if requested
        hidden_states_dict: dict[int, torch.Tensor] | None = None
        sequence_hidden_states_dict: dict[int, torch.Tensor] | None = None

        if return_hidden_states and hasattr(outputs, "hidden_states"):
            hidden_states_dict = self._extract_hidden_states(
                outputs.hidden_states,
                hidden_state_layers,
                loaded_model.num_layers,
            )

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
                "loader": self.name,
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

        final_step = attentions[-1]

        for layer_idx in layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx

            if 0 <= actual_idx < len(final_step):
                layer_attn = final_step[actual_idx][:, :, -1, :].cpu()
                result[layer_idx] = layer_attn

        return result

    def _extract_sequence_hidden_states(
        self,
        hidden_states: tuple,
        layers: list[int],
        num_layers: int,
    ) -> dict[int, torch.Tensor]:
        """Extract hidden states for all generated tokens (manifold)."""
        result: dict[int, torch.Tensor] = {}

        if not hidden_states:
            return result

        for layer_idx in layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + 1 + layer_idx

            step_vectors = []
            for step in hidden_states:
                if 0 <= actual_idx < len(step):
                    token_hidden = step[actual_idx][0, -1, :].cpu()
                    step_vectors.append(token_hidden)

            if step_vectors:
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
        """Extract embedding from text.

        Args:
            loaded_model: Previously loaded model
            text: Text to embed
            pooling: Pooling strategy (last_token, mean, first_token)

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
        )

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        start_time = time.time()

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        inference_time = time.time() - start_time

        # Get last layer hidden states
        last_hidden = outputs.hidden_states[-1]

        # Apply pooling
        attention_mask = inputs["attention_mask"]
        if pooling == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            embedding = last_hidden[torch.arange(batch_size, device=device), seq_lengths]
        elif pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            mask_sum = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
            embedding = (last_hidden * mask).sum(dim=1) / mask_sum
        elif pooling == "first_token":
            embedding = last_hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        embedding = embedding.cpu()

        return EmbeddingOutput(
            embedding=embedding.squeeze(0),
            shape=tuple(embedding.shape),
            metadata={
                "pooling": pooling,
                "inference_time_ms": inference_time * 1000,
                "input_tokens": inputs["input_ids"].shape[1],
                "model_id": loaded_model.model_id,
                "loader": self.name,
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
        """Generate text with streaming output.

        Falls back to non-streaming generation and yields tokens.
        Note: This provides buffered pseudo-streaming, not true real-time
        token streaming like TransformersLoader.generate_stream.
        """
        logger.debug("Mistral generate_stream uses buffered output, not true streaming")
        # Use non-streaming generate and yield tokens
        output = self.generate(
            loaded_model=loaded_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            return_hidden_states=return_hidden_states,
            hidden_state_layers=hidden_state_layers,
            top_p=top_p,
            do_sample=do_sample,
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

        # Yield final output
        yield StreamingOutput(
            text=output.text,
            token_ids=output.token_ids,
            token_count=len(output.token_ids),
            hidden_states=output.hidden_states,
            metadata=output.metadata,
        )
