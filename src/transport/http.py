"""HTTP transport layer using FastAPI."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import Config, get_config
from ..extraction.hidden_states import (
    HiddenStateResult,
    analyze_hidden_state,
    extract_hidden_states,
)
from ..loaders.base import LoadedModel, StreamingOutput, StreamingToken
from ..loaders.registry import LoaderRegistry
from ..utils.gpu import GPUManager
from ..utils.metrics import (
    get_metrics,
    is_metrics_available,
    record_embedding,
    record_generation,
    record_model_load,
    record_request,
    set_models_loaded,
)
from ..utils.serialization import serialize_hidden_states, tensor_to_base64, tensor_to_list

logger = logging.getLogger(__name__)


def _expand_hidden_state_layers(
    layers: list[int] | str,
    num_layers: int,
) -> list[int]:
    """Expand hidden_state_layers to a concrete list of layer indices.

    Args:
        layers: Either a list of layer indices, or "all" for every layer
        num_layers: Total number of layers in the model

    Returns:
        List of layer indices (using negative indexing)
    """
    if isinstance(layers, str):
        if layers.lower() == "all":
            # Return all layers as negative indices: [-num_layers, ..., -1]
            return list(range(-num_layers, 0))
        else:
            raise ValueError(f"Unknown layer specification: {layers}. Use 'all' or a list of indices.")
    return layers


def _apply_chat_template(
    messages: list[dict[str, str]],
    tokenizer: Any,
    model_id: str,
) -> str:
    """Apply chat template to convert messages to a prompt string.

    This is the critical bridge between OpenAI-style chat format and
    raw prompt format. Uses the tokenizer's built-in chat template
    if available, otherwise falls back to a simple concatenation.

    Args:
        messages: List of {"role": str, "content": str} dicts
        tokenizer: HuggingFace tokenizer with optional chat_template
        model_id: Model identifier for logging

    Returns:
        Formatted prompt string ready for generation
    """
    # Check if tokenizer has chat template support
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            # Convert Pydantic models to dicts if needed
            msg_dicts = [
                {"role": m["role"], "content": m["content"]}
                if isinstance(m, dict)
                else {"role": m.role, "content": m.content}
                for m in messages
            ]

            prompt: str = tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug(f"Applied chat template for {model_id}")
            return prompt
        except Exception as e:
            logger.warning(
                f"Chat template failed for {model_id}: {e}. Falling back to simple format."
            )

    # Fallback: Simple concatenation for models without chat templates
    # This handles base models and older models
    parts = []
    for m in messages:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content

        if role == "system":
            parts.append(f"System: {content}\n\n")
        elif role == "user":
            parts.append(f"User: {content}\n\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n\n")
        else:
            parts.append(f"{role}: {content}\n\n")

    # Add generation prompt
    parts.append("Assistant:")
    return "".join(parts)


def _serialize_sequence_hidden_states(
    sequence_hidden_states: dict[int, Any],
    format: str = "list",
) -> dict[str, dict[str, Any]]:
    """Serialize full sequence hidden states for manifold analysis.

    Args:
        sequence_hidden_states: dict mapping layer_idx to tensor [num_tokens, hidden_size]
        format: "list" or "base64"

    Returns:
        JSON-serializable dict with sequence data
    """
    import torch

    result: dict[str, dict[str, Any]] = {}

    for layer_idx, tensor in sequence_hidden_states.items():
        layer_key = str(layer_idx)

        # Handle bfloat16 conversion
        if hasattr(tensor, "dtype") and tensor.dtype == torch.bfloat16:
            tensor = tensor.float()

        if hasattr(tensor, "cpu"):
            arr = tensor.cpu().detach().numpy()
        else:
            import numpy as np
            arr = np.asarray(tensor)

        shape = list(arr.shape)
        dtype = "float32"

        if format == "list":
            # Return as nested list [num_tokens][hidden_size]
            result[layer_key] = {
                "data": arr.tolist(),
                "shape": shape,
                "dtype": dtype,
            }
        elif format == "base64":
            result[layer_key] = {
                "data": tensor_to_base64(tensor, "float32"),
                "shape": shape,
                "dtype": dtype,
                "encoding": "base64",
            }
        else:
            raise ValueError(f"Unknown format: {format}")

    return result


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(default=True, description="Return hidden states")
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    return_attention: bool = Field(default=False, description="Return attention weights")
    return_full_sequence: bool = Field(
        default=False,
        description="Return hidden states for ALL tokens (manifold/boundary object). "
        "Creates [num_tokens, hidden_size] tensor for geometric analysis.",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(
        default=None,
        description="Force specific loader (auto-detect if None): transformers, sentence_transformers, custom",
    )


class HiddenStateResponse(BaseModel):
    """Hidden state data in response."""

    data: list[float] | str  # list for 'list' format, str for 'base64'
    shape: list[int]
    dtype: str
    encoding: str | None = None  # 'base64' if base64 encoded


class SequenceHiddenStateResponse(BaseModel):
    """Full sequence hidden states for manifold construction."""

    data: list[list[float]] | str  # [num_tokens, hidden_size] or base64
    shape: list[int]  # [num_tokens, hidden_size]
    dtype: str
    encoding: str | None = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str
    token_count: int
    hidden_states: dict[str, HiddenStateResponse] | None = None
    attention_weights: dict[str, Any] | None = None
    # Full sequence hidden states for manifold/boundary object analysis
    sequence_hidden_states: dict[str, SequenceHiddenStateResponse] | None = None
    metadata: dict[str, Any]


class StreamingGenerateRequest(BaseModel):
    """Request model for streaming text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompt: str = Field(..., description="Input prompt text")
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(
        default=False, description="Return hidden states in final event"
    )
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(default=None, description="Force specific loader")


class BatchGenerateRequest(BaseModel):
    """Request model for batch text generation."""

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    prompts: list[str] = Field(..., description="List of input prompts", min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=1, le=8192, description="Max tokens per generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability")
    return_hidden_states: bool = Field(default=False, description="Return hidden states")
    hidden_state_layers: list[int] | str = Field(
        default=[-1],
        description="Which layers to return: list of indices (-1 = last), or 'all' for every layer",
    )
    hidden_state_format: str = Field(
        default="list", description="Format for hidden states: list or base64"
    )
    loader: str | None = Field(default=None, description="Force specific loader")


class BatchGenerateResponse(BaseModel):
    """Response model for batch text generation."""

    results: list[GenerateResponse]
    total_tokens: int
    total_time_ms: float
    prompts_processed: int


class BatchEmbedRequest(BaseModel):
    """Request model for batch embedding extraction."""

    model: str = Field(..., description="Model ID")
    texts: list[str] = Field(..., description="List of texts to embed", min_length=1, max_length=100)
    pooling: str = Field(default="last_token", description="Pooling: last_token, mean, first_token")
    normalize: bool = Field(default=False, description="L2 normalize the embeddings")


class BatchEmbedResponse(BaseModel):
    """Response model for batch embedding extraction."""

    embeddings: list[list[float]]
    shapes: list[list[int]]
    total_time_ms: float
    texts_processed: int


class EmbedRequest(BaseModel):
    """Request model for embedding extraction."""

    model: str = Field(..., description="Model ID")
    text: str = Field(..., description="Text to embed")
    pooling: str = Field(default="last_token", description="Pooling: last_token, mean, first_token")
    normalize: bool = Field(default=False, description="L2 normalize the embedding")


class EmbedResponse(BaseModel):
    """Response model for embedding extraction."""

    embedding: list[float]
    shape: list[int]
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: str | None
    gpu_info: dict[str, Any]
    config: dict[str, Any]


# ============================================================================
# OpenAI-Compatible Chat Completions (for WeaverCode integration)
# ============================================================================


class ChatMessage(BaseModel):
    """A single chat message with role and content."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with hidden state support.

    This endpoint is designed for WeaverCode integration, providing the
    messages-based API that WeaverCode expects while exposing hidden states
    for conveyance measurement.
    """

    model: str = Field(..., description="Model ID (HuggingFace or local path)")
    messages: list[ChatMessage] = Field(
        ..., description="List of chat messages", min_length=1
    )
    max_tokens: int = Field(
        default=256, ge=1, le=8192, description="Max tokens to generate"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    return_hidden_states: bool = Field(
        default=True, description="Return hidden states for conveyance measurement"
    )
    stream: bool = Field(default=False, description="Stream responses (not yet implemented)")
    loader: str | None = Field(default=None, description="Force specific loader")


class ChatCompletionUsage(BaseModel):
    """Token usage statistics matching OpenAI format."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionHiddenState(BaseModel):
    """Hidden state in WeaverCode-expected format.

    This is the geometric boundary object - the final hidden state
    before lm_head projection, representing meaning as geometry.
    """

    final: list[float] = Field(description="Final layer hidden state vector")
    shape: list[int] = Field(description="Tensor shape [batch, hidden_dim]")
    layer: int = Field(default=-1, description="Layer index (-1 = last)")
    dtype: str = Field(default="float32", description="Data type")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible response with hidden state extraction.

    Extends standard chat completion response with hidden_state field
    for conveyance measurement in WeaverCode.
    """

    text: str = Field(description="Generated text")
    usage: ChatCompletionUsage = Field(description="Token usage statistics")
    hidden_state: ChatCompletionHiddenState | None = Field(
        default=None, description="Hidden state for conveyance measurement"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generation metadata"
    )


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""

    model: str = Field(..., description="Model ID to load")
    device: str | None = Field(default=None, description="Device to load on (e.g., cuda:0)")
    dtype: str = Field(default="auto", description="Data type: auto, float16, bfloat16, float32")
    loader: str | None = Field(
        default=None,
        description="Force specific loader (auto-detect if None): transformers, sentence_transformers, custom",
    )
    quantization: str | None = Field(
        default=None,
        description="Quantization mode: 4bit, 8bit, gptq, awq (requires appropriate packages)",
    )


class ModelLoadResponse(BaseModel):
    """Response model for model loading."""

    model_id: str
    device: str
    dtype: str
    hidden_size: int
    num_layers: int
    load_time_seconds: float
    loader_type: str = Field(description="Which loader was used")
    quantization: str = Field(default="none", description="Quantization mode used")


# ============================================================================
# Model Manager
# ============================================================================


class ModelManager:
    """Manages loaded models with LRU eviction, auto-unload, and multi-loader support."""

    def __init__(
        self,
        gpu_manager: GPUManager,
        registry: LoaderRegistry,
        max_models: int = 3,
        auto_unload_minutes: int = 20,
    ):
        """
        Create a ModelManager that loads and manages models with an LRU eviction policy.

        Parameters:
            gpu_manager: Manager responsible for GPU device allocation and cache management.
            registry: Loader registry used to resolve and load models.
            max_models (int): Maximum number of models to keep loaded simultaneously; older models are evicted when capacity is exceeded.
            auto_unload_minutes (int): Auto-unload models after this many minutes of inactivity (0 = disabled).
        """
        self.gpu_manager = gpu_manager
        self.registry = registry
        self.max_models = max_models
        self.auto_unload_minutes = auto_unload_minutes
        self.loaded_models: dict[str, LoadedModel] = {}
        self.access_order: list[str] = []  # For LRU eviction
        self.last_accessed: dict[str, float] = {}  # Track last access time per model

    def get_or_load(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "auto",
        loader_name: str | None = None,
        quantization: str | None = None,
    ) -> LoadedModel:
        """
        Retrieve a LoadedModel by model_id, loading and caching it if not already present.
        
        Parameters:
            model_id (str): Identifier of the model to retrieve.
            device (str | None): Target device for loading; when None the GPU manager's default device is used.
            dtype (str): Desired numeric dtype for the model (e.g., "float16", "float32", or "auto").
            loader_name (str | None): Specific loader to use; when None the loader will be auto-detected.
            quantization (str | None): Quantization mode to apply (e.g., "4bit", "8bit", "gptq", "awq"), if any.
        
        Returns:
            LoadedModel: The loaded and cached model instance for the requested model_id.
        """
        # Check if already loaded
        if model_id in self.loaded_models:
            # Update access order for LRU
            if model_id in self.access_order:
                self.access_order.remove(model_id)
            self.access_order.append(model_id)
            self.last_accessed[model_id] = time.time()
            return self.loaded_models[model_id]

        # Evict if at capacity
        while len(self.loaded_models) >= self.max_models:
            self._evict_oldest()

        # Resolve device
        if device is None:
            device = self.gpu_manager.default_device

        # Load the model using registry (auto-detects loader if not specified)
        logger.info(f"Loading model: {model_id}")
        loaded = self.registry.load(
            model_id,
            device=device,
            dtype=dtype,
            loader_name=loader_name,
            quantization=quantization,
        )

        self.loaded_models[model_id] = loaded
        self.access_order.append(model_id)
        self.last_accessed[model_id] = time.time()

        return loaded

    def _evict_oldest(self) -> None:
        """
        Remove the least-recently-used model from the manager and clear the GPU cache.

        If no models are loaded, this is a no-op.
        """
        if not self.access_order:
            return

        oldest = self.access_order.pop(0)
        if oldest in self.loaded_models:
            logger.info(f"Evicting model: {oldest}")
            del self.loaded_models[oldest]
            self.last_accessed.pop(oldest, None)
            self.gpu_manager.clear_cache()

    def unload(self, model_id: str) -> bool:
        """
        Unload the specified loaded model from memory and clear related GPU cache.

        Clears the GPU cache and logs the unload when a model is removed.

        Parameters:
            model_id (str): Identifier of the model to unload.

        Returns:
            bool: True if the model was loaded and successfully unloaded, False if the model was not found.
        """
        if model_id not in self.loaded_models:
            return False

        del self.loaded_models[model_id]
        if model_id in self.access_order:
            self.access_order.remove(model_id)
        self.last_accessed.pop(model_id, None)
        self.gpu_manager.clear_cache()
        logger.info(f"Unloaded model: {model_id}")
        return True

    def check_idle_models(self) -> list[str]:
        """
        Check for and unload models that have been idle longer than auto_unload_minutes.

        Returns:
            list[str]: List of model IDs that were unloaded due to inactivity.
        """
        if self.auto_unload_minutes <= 0:
            return []

        unloaded = []
        current_time = time.time()
        timeout_seconds = self.auto_unload_minutes * 60

        # Create a copy of keys to avoid modification during iteration
        for model_id in list(self.loaded_models.keys()):
            last_access = self.last_accessed.get(model_id, current_time)
            idle_seconds = current_time - last_access

            if idle_seconds >= timeout_seconds:
                logger.info(
                    f"Auto-unloading idle model: {model_id} "
                    f"(idle for {idle_seconds / 60:.1f} minutes)"
                )
                self.unload(model_id)
                unloaded.append(model_id)

        return unloaded

    def get_idle_time(self, model_id: str) -> float | None:
        """
        Get how long a model has been idle in seconds.

        Parameters:
            model_id (str): Model identifier.

        Returns:
            float | None: Seconds since last access, or None if model not loaded.
        """
        if model_id not in self.last_accessed:
            return None
        return time.time() - self.last_accessed[model_id]

    def list_loaded(self) -> list[str]:
        """
        Return the identifiers of models currently loaded in the manager.
        
        Returns:
            loaded_models (list[str]): List of model IDs for all models currently loaded.
        """
        return list(self.loaded_models.keys())

    def get_loaded_info(self) -> list[dict[str, Any]]:
        """
        Return a list of dictionaries describing each currently loaded model.

        Each dictionary contains:
        - `model_id`: model identifier string
        - `device`: device identifier (e.g., "cuda:0" or "cpu")
        - `dtype`: data type name (e.g., "float16", "float32")
        - `hidden_size`: hidden dimension size
        - `num_layers`: number of model layers
        - `loader_type`: name of the loader used to load the model
        - `quantization`: quantization metadata value or "none" if not present
        - `idle_seconds`: seconds since last access
        - `auto_unload_at`: remaining seconds until auto-unload (None if disabled)

        Returns:
            list[dict[str, Any]]: List of per-model information dictionaries.
        """
        result = []
        current_time = time.time()
        timeout_seconds = self.auto_unload_minutes * 60 if self.auto_unload_minutes > 0 else None

        for model in self.loaded_models.values():
            idle_seconds = current_time - self.last_accessed.get(model.model_id, current_time)
            auto_unload_in = None
            if timeout_seconds is not None:
                auto_unload_in = max(0, timeout_seconds - idle_seconds)

            result.append({
                "model_id": model.model_id,
                "device": str(model.device),
                "dtype": str(model.dtype).replace("torch.", ""),
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "loader_type": model.loader_type,
                "quantization": model.metadata.get("quantization", "none"),
                "idle_seconds": round(idle_seconds, 1),
                "auto_unload_in": round(auto_unload_in, 1) if auto_unload_in is not None else None,
            })

        return result


# ============================================================================
# Metrics Middleware
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Track request latency and record request metrics.
        
        Skips the /metrics endpoint to avoid recursive metrics collection. Measures the request latency, records the endpoint, HTTP method, status code, and latency via `record_request`, and returns the downstream response.
        
        Returns:
            Response: The HTTP response produced by the next handler.
        """
        # Skip metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            response: Response = await call_next(request)
            return response

        start_time = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start_time

        # Record metrics
        record_request(
            endpoint=request.url.path,
            method=request.method,
            status=response.status_code,
            latency=latency,
        )

        return response


# ============================================================================
# FastAPI Application Factory
# ============================================================================


def create_http_app(config: Config | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application for the hidden-state extraction server.
    
    Initializes CORS and optional metrics middleware, GPU manager, loader registry, and model manager, registers all HTTP endpoints (health, metrics, model management, generation, streaming generation, embedding, analysis, and batch operations), and stores core components on app.state.
    
    Parameters:
        config (Config | None): Optional server configuration; when None the global configuration is used.
    
    Returns:
        FastAPI: A configured FastAPI application ready to serve the transport HTTP API.
    """
    if config is None:
        config = get_config()

    # Initialize components first (needed by lifespan)
    gpu_manager = GPUManager(
        allowed_devices=config.gpu.devices,
        memory_fraction=config.gpu.memory_fraction,
    )

    # Create loader registry with model overrides from config
    registry = LoaderRegistry(loader_configs=config.model_overrides)

    model_manager = ModelManager(
        gpu_manager=gpu_manager,
        registry=registry,
        max_models=config.models.max_loaded,
        auto_unload_minutes=config.models.auto_unload_minutes,
    )

    # Background task state
    auto_unload_task: asyncio.Task[None] | None = None

    async def auto_unload_checker(manager: ModelManager, check_interval: int = 60) -> None:
        """Background task to check and unload idle models."""
        while True:
            try:
                await asyncio.sleep(check_interval)
                unloaded = manager.check_idle_models()
                if unloaded:
                    set_models_loaded(len(manager.list_loaded()))
            except asyncio.CancelledError:
                logger.info("Auto-unload checker stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-unload checker: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage app lifespan - start/stop background tasks."""
        nonlocal auto_unload_task
        # Startup: start the background task if auto-unload is enabled
        if config.models.auto_unload_minutes > 0:
            logger.info(
                f"Starting auto-unload checker (timeout: {config.models.auto_unload_minutes} min)"
            )
            auto_unload_task = asyncio.create_task(
                auto_unload_checker(model_manager, check_interval=60)
            )
        yield
        # Shutdown: cancel the background task
        if auto_unload_task is not None:
            auto_unload_task.cancel()
            try:
                await auto_unload_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="The Loom",
        description="Hidden state extraction server for AI research - part of the Weaver ecosystem",
        version="0.2.0",
        lifespan=lifespan,
    )

    # CORS middleware for browser-based clients
    # Note: allow_credentials=True with allow_origins=["*"] is insecure
    # Configure via config.server.cors_origins and cors_allow_credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=config.server.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics middleware (only if prometheus_client is available)
    if is_metrics_available():
        app.add_middleware(MetricsMiddleware)

    # Store in app state
    app.state.config = config
    app.state.gpu_manager = gpu_manager
    app.state.model_manager = model_manager
    app.state.registry = registry

    # ========================================================================
    # Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """
        Return overall service health and runtime information.
        
        Returns:
            HealthResponse: Contains `status` ("healthy" or other), `model_loaded` (the first loaded model ID or `None`), `gpu_info` (GPU state as a dict), and `config` with `max_loaded_models` and `default_layers`.
        """
        loaded_models = model_manager.list_loaded()
        # Update loaded models gauge
        set_models_loaded(len(loaded_models))
        return HealthResponse(
            status="healthy",
            model_loaded=loaded_models[0] if loaded_models else None,
            gpu_info=gpu_manager.to_dict(),
            config={
                "max_loaded_models": config.models.max_loaded,
                "default_layers": config.hidden_states.default_layers,
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint.

        Returns metrics in Prometheus text format for scraping.
        """
        return Response(
            content=get_metrics(),
            media_type="text/plain; charset=utf-8",
        )

    @app.get("/models")
    async def list_models() -> dict[str, Any]:
        """
        Return information about currently loaded models and the configured maximum.

        Returns:
            info (dict): Dictionary with keys:
                - "loaded_models": a list of dictionaries, each containing details for a loaded model.
                - "max_models": the configured maximum number of models allowed to be loaded.
                - "auto_unload_minutes": minutes of inactivity before auto-unload (0 = disabled).
        """
        return {
            "loaded_models": model_manager.get_loaded_info(),
            "max_models": config.models.max_loaded,
            "auto_unload_minutes": config.models.auto_unload_minutes,
        }

    @app.get("/loaders")
    async def list_loaders() -> dict[str, Any]:
        """
        Provide information about available model loaders and their configured fallback order.
        
        Returns:
            info (dict): A dictionary with two keys:
                - "loaders": mapping of loader names to loader metadata as returned by the loader registry.
                - "fallback_order": list of loader names in the order they will be probed for a given model id.
        """
        return {
            "loaders": registry.list_loaders(),
            "fallback_order": registry.fallback_order,
        }

    @app.get("/loaders/probe/{model_id:path}")
    async def probe_model_loader(model_id: str) -> dict[str, Any]:
        """
        Determine which loader would handle the given model identifier without loading the model.
        
        Returns:
            probe_result (dict): Probe results describing the selected loader and related metadata.
        """
        # Handle URL encoding (-- back to /)
        model_id = model_id.replace("--", "/")
        return registry.probe_model(model_id)

    @app.post("/models/load", response_model=ModelLoadResponse)
    async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
        """
        Load the requested model into memory and return its load metadata.
        
        Returns:
            ModelLoadResponse: Contains the loaded model's identifier, device, dtype (stringified), hidden_size, num_layers, reported load_time_seconds, loader_type, and quantization.
        
        Raises:
            HTTPException: Raised with status code 500 if the model fails to load; the exception's detail contains the error message.
        """
        start_time = time.perf_counter()
        try:
            loaded = model_manager.get_or_load(
                model_id=request.model,
                device=request.device,
                dtype=request.dtype,
                loader_name=request.loader,
                quantization=request.quantization,
            )
            load_time = time.perf_counter() - start_time

            # Record metrics
            record_model_load(
                model=request.model,
                loader=loaded.loader_type,
                quantization=loaded.metadata.get("quantization", "none"),
                latency=load_time,
                success=True,
            )
            set_models_loaded(len(model_manager.list_loaded()))

            return ModelLoadResponse(
                model_id=loaded.model_id,
                device=str(loaded.device),
                dtype=str(loaded.dtype).replace("torch.", ""),
                hidden_size=loaded.hidden_size,
                num_layers=loaded.num_layers,
                load_time_seconds=loaded.metadata.get("load_time_seconds", 0),
                loader_type=loaded.loader_type,
                quantization=loaded.metadata.get("quantization", "none"),
            )
        except Exception as e:
            load_time = time.perf_counter() - start_time
            record_model_load(
                model=request.model,
                loader=request.loader or "auto",
                quantization=request.quantization or "none",
                latency=load_time,
                success=False,
            )
            logger.exception(f"Failed to load model: {request.model}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/models/{model_id}")
    async def unload_model(model_id: str) -> dict[str, Any]:
        """
        Unload a previously loaded model and free its resources.
        
        Parameters:
            model_id (str): Identifier of the model to unload. Instances of "--" in the string are normalized to "/".
        
        Returns:
            result (dict[str, Any]): Dictionary containing "status" set to "unloaded" and the normalized "model_id".
        
        Raises:
            HTTPException: 404 if the specified model is not currently loaded.
        """
        # Handle URL encoding
        model_id = model_id.replace("--", "/")
        success = model_manager.unload(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Model not loaded: {model_id}")
        # Update loaded models gauge
        set_models_loaded(len(model_manager.list_loaded()))
        return {"status": "unloaded", "model_id": model_id}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        """
        Generate text and optionally return serialized hidden states and attention weights.
        
        Parameters:
            request (GenerateRequest): Generation settings including model id, prompt, token limits, sampling params, and flags for returning hidden states or attention.
        
        Returns:
            GenerateResponse: Contains generated text, token_count, optional serialized `hidden_states`, optional serialized `attention_weights`, and `metadata` about the generation.
        
        Raises:
            HTTPException: On failure to load the model or generate (status code 500 with error detail).
        """
        start_time = time.perf_counter()
        try:
            # Get or load model (with optional loader override)
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Expand "all" to full layer list if needed
            hidden_state_layers = _expand_hidden_state_layers(
                request.hidden_state_layers,
                loaded.num_layers,
            )

            # Generate using registry (uses appropriate loader for model)
            output = registry.generate(
                loaded_model=loaded,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                return_hidden_states=request.return_hidden_states,
                hidden_state_layers=hidden_state_layers,
                return_attention=request.return_attention,
                return_full_sequence=request.return_full_sequence,
            )

            # Record generation metrics
            gen_latency = time.perf_counter() - start_time
            record_generation(
                model=request.model,
                tokens=len(output.token_ids),
                latency=gen_latency,
            )

            # Serialize hidden states if present
            hidden_states_response = None
            if output.hidden_states:
                hidden_states_results = extract_hidden_states(output.hidden_states)
                hidden_states_response = serialize_hidden_states(
                    hidden_states_results,
                    format=request.hidden_state_format,
                )

            # Serialize attention if present
            attention_response = None
            if output.attention_weights:
                attention_response = serialize_hidden_states(
                    output.attention_weights,
                    format=request.hidden_state_format,
                )

            # Serialize full sequence hidden states for manifold analysis
            sequence_hidden_states_response: dict[str, SequenceHiddenStateResponse] | None = None
            if output.sequence_hidden_states:
                # Pydantic will coerce the dict to SequenceHiddenStateResponse at validation
                sequence_hidden_states_response = cast(
                    dict[str, SequenceHiddenStateResponse],
                    _serialize_sequence_hidden_states(
                        output.sequence_hidden_states,
                        format=request.hidden_state_format,
                    ),
                )

            return GenerateResponse(
                text=output.text,
                token_count=len(output.token_ids),
                hidden_states=hidden_states_response,
                attention_weights=attention_response,
                sequence_hidden_states=sequence_hidden_states_response,
                metadata=output.metadata,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # OpenAI-Compatible Chat Completions (WeaverCode Integration)
    # ========================================================================

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """OpenAI-compatible chat completion with hidden state extraction.

        This endpoint is designed for WeaverCode integration, providing:
        - Chat message format (messages array with role/content)
        - Chat template application (model-specific formatting)
        - Hidden state extraction in WeaverCode-expected format
        - Token usage breakdown (prompt/completion/total)

        The hidden state returned is the "boundary object" - the geometric
        representation of meaning before lm_head projection. This enables
        conveyance measurement between AI agents.

        Example request:
            {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "return_hidden_states": true
            }

        Example response:
            {
                "text": "Hello! How can I help you today?",
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 12,
                    "total_tokens": 57
                },
                "hidden_state": {
                    "final": [0.123, -0.456, ...],
                    "shape": [1, 2048],
                    "layer": -1,
                    "dtype": "float16"
                },
                "metadata": {...}
            }
        """
        if request.stream:
            raise HTTPException(
                status_code=501,
                detail="Streaming not yet implemented for chat completions. Use stream=false.",
            )

        start_time = time.perf_counter()
        try:
            # Get or load model
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Apply chat template to convert messages to prompt
            prompt = _apply_chat_template(
                request.messages,  # type: ignore[arg-type]
                loaded.tokenizer,
                loaded.model_id,
            )

            # Generate using the existing registry infrastructure
            output = registry.generate(
                loaded_model=loaded,
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                return_hidden_states=request.return_hidden_states,
                hidden_state_layers=[-1],  # WeaverCode only needs final layer
            )

            # Record metrics
            gen_latency = time.perf_counter() - start_time
            record_generation(
                model=request.model,
                tokens=len(output.token_ids),
                latency=gen_latency,
            )

            # Build usage statistics (WeaverCode-expected format)
            prompt_tokens = output.metadata.get("input_tokens", 0)
            completion_tokens = len(output.token_ids)
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

            # Extract hidden state in WeaverCode-expected format
            hidden_state_response: ChatCompletionHiddenState | None = None
            if request.return_hidden_states and output.hidden_states:
                # Get the last layer hidden state
                final_layer_data = output.hidden_states.get(-1)
                if final_layer_data is not None:
                    # Convert to list format
                    hidden_vector = tensor_to_list(final_layer_data)
                    hidden_shape = list(final_layer_data.shape)

                    # Determine dtype string
                    dtype_str = str(final_layer_data.dtype).replace("torch.", "")

                    hidden_state_response = ChatCompletionHiddenState(
                        final=hidden_vector,
                        shape=hidden_shape,
                        layer=-1,
                        dtype=dtype_str,
                    )

            return ChatCompletionResponse(
                text=output.text,
                usage=usage,
                hidden_state=hidden_state_response,
                metadata={
                    "model": loaded.model_id,
                    "latency_ms": gen_latency * 1000,
                    "tokens_per_second": completion_tokens / gen_latency
                    if gen_latency > 0
                    else 0,
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Alias for convenience (some clients expect /v1/generate)
    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate_v1(request: GenerateRequest) -> GenerateResponse:
        """Alias for /generate with /v1/ prefix for API consistency."""
        return await generate(request)

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """
        Produce an embedding for the provided text.
        
        If the underlying model is decoder-only, the embedding is taken from the last token's hidden state (which includes accumulated context). If `request.normalize` is true, the returned embedding is L2-normalized.
        
        Returns:
            EmbedResponse: Object containing `embedding` (list of floats), `shape` (list of ints), and `metadata`.
        """
        start_time = time.perf_counter()
        try:
            # Get or load model
            loaded = model_manager.get_or_load(request.model)

            # Extract embedding using registry (uses appropriate loader)
            output = registry.embed(
                loaded_model=loaded,
                text=request.text,
                pooling=request.pooling,
            )

            # Record embedding metrics
            embed_latency = time.perf_counter() - start_time
            record_embedding(model=request.model, latency=embed_latency)

            # Convert to list
            embedding_list = tensor_to_list(output.embedding)

            # Optionally L2 normalize
            if request.normalize:
                import numpy as np

                arr = np.array(embedding_list)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    embedding_list = (arr / norm).tolist()

            return EmbedResponse(
                embedding=embedding_list,
                shape=list(output.shape),
                metadata=output.metadata,
            )

        except Exception as e:
            logger.exception(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/analyze")
    async def analyze_embedding(request: EmbedRequest) -> dict[str, Any]:
        """
        Compute diagnostic metrics for a single text embedding.
        
        Returns:
            analysis (dict): Analysis results produced by hidden-state analysis. Includes an "embedding_shape" key with the embedding shape as a list and any metadata from the embedding output.
        
        Raises:
            HTTPException: If embedding extraction or analysis fails.
        """
        try:
            loaded = model_manager.get_or_load(request.model)

            # Extract embedding using registry
            output = registry.embed(
                loaded_model=loaded,
                text=request.text,
                pooling=request.pooling,
            )

            # Create HiddenStateResult for analysis

            result = HiddenStateResult(
                vector=output.embedding.numpy(),
                shape=output.shape,
                layer=-1,
                dtype=str(output.embedding.dtype),
            )

            analysis = analyze_hidden_state(result)
            analysis["embedding_shape"] = list(output.shape)
            analysis.update(output.metadata)

            return analysis

        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/generate/stream")
    async def generate_stream(request: StreamingGenerateRequest) -> StreamingResponse:
        """
        Stream generation results as Server-Sent Events.
        
        Emits SSE messages for the stream: `token` events for each generated token, a `done` event with the final output (text, token_count, token_ids, metadata) and optional serialized hidden states, and an `error` event if generation fails.
        
        Returns:
            StreamingResponse: An SSE stream that yields JSON-encoded event data for `token`, `done`, and `error` events.
        """

        async def event_generator() -> AsyncIterator[str]:
            """
            Produce Server-Sent Events (SSE) messages for a streaming generation request, emitting per-token updates and a final completion event, and an error event if streaming fails.
            
            Yields:
                SSE-formatted strings representing one of:
                  - a `token` event with fields `token`, `token_id`, `is_finished`, and `finish_reason`;
                  - a `done` event with final `text`, `token_count`, `token_ids`, `metadata`, and optionally `hidden_states` (serialized according to the request format);
                  - an `error` event with an `error` message when an exception occurs.
            """
            try:
                # Get or load model
                loaded = model_manager.get_or_load(
                    request.model,
                    loader_name=request.loader,
                )

                # Expand "all" to full layer list if needed
                hidden_state_layers = _expand_hidden_state_layers(
                    request.hidden_state_layers,
                    loaded.num_layers,
                )

                # Stream tokens
                for item in registry.generate_stream(
                    loaded_model=loaded,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                ):
                    if isinstance(item, StreamingToken):
                        # Send token event
                        data = {
                            "token": item.token,
                            "token_id": item.token_id,
                            "is_finished": item.is_finished,
                            "finish_reason": item.finish_reason,
                        }
                        yield f"event: token\ndata: {json.dumps(data)}\n\n"

                    elif isinstance(item, StreamingOutput):
                        # Send final output event
                        output_data: dict[str, Any] = {
                            "text": item.text,
                            "token_count": item.token_count,
                            "token_ids": item.token_ids,
                            "metadata": item.metadata,
                        }

                        # Serialize hidden states if present
                        if item.hidden_states:
                            hidden_states_results = extract_hidden_states(item.hidden_states)
                            output_data["hidden_states"] = serialize_hidden_states(
                                hidden_states_results,
                                format=request.hidden_state_format,
                            )

                        yield f"event: done\ndata: {json.dumps(output_data)}\n\n"

            except Exception as e:
                logger.exception(f"Streaming generation failed: {e}")
                error_data = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    @app.post("/generate/batch", response_model=BatchGenerateResponse)
    async def generate_batch(request: BatchGenerateRequest) -> BatchGenerateResponse:
        """
        Generate text for each prompt in the request using a single loaded model.
        
        Processes all prompts with the same model instance and returns per-prompt generation outputs, including optional serialized hidden states.
        
        Parameters:
            request (BatchGenerateRequest): Batch generation request containing the model identifier, list of prompts, and generation options (max_tokens, temperature, top_p, hidden-state flags, and formatting).
        
        Returns:
            BatchGenerateResponse: Object containing:
                - results (list[GenerateResponse]): Generation result for each prompt.
                - total_tokens (int): Sum of token counts across all prompts.
                - total_time_ms (float): Total processing time in milliseconds.
                - prompts_processed (int): Number of prompts processed.
        """
        start_time = time.time()
        results: list[GenerateResponse] = []
        total_tokens = 0

        try:
            # Get or load model once for all prompts
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Expand "all" to full layer list if needed
            hidden_state_layers = _expand_hidden_state_layers(
                request.hidden_state_layers,
                loaded.num_layers,
            )

            # Process each prompt
            for prompt in request.prompts:
                output = registry.generate(
                    loaded_model=loaded,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=hidden_state_layers,
                )

                # Serialize hidden states if present
                hidden_states_response = None
                if output.hidden_states:
                    hidden_states_results = extract_hidden_states(output.hidden_states)
                    hidden_states_response = serialize_hidden_states(
                        hidden_states_results,
                        format=request.hidden_state_format,
                    )

                results.append(
                    GenerateResponse(
                        text=output.text,
                        token_count=len(output.token_ids),
                        hidden_states=hidden_states_response,
                        attention_weights=None,
                        metadata=output.metadata,
                    )
                )
                total_tokens += len(output.token_ids)

            total_time = (time.time() - start_time) * 1000

            return BatchGenerateResponse(
                results=results,
                total_tokens=total_tokens,
                total_time_ms=total_time,
                prompts_processed=len(request.prompts),
            )

        except Exception as e:
            logger.exception(f"Batch generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/embed/batch", response_model=BatchEmbedResponse)
    async def embed_batch(request: BatchEmbedRequest) -> BatchEmbedResponse:
        """
        Generate embeddings for multiple texts using a single loaded model.
        
        Processes the provided texts with the same model and returns their embeddings, per-embedding shapes, total processing time in milliseconds, and the number of texts processed.
        
        Parameters:
            request (BatchEmbedRequest): Batch embedding request containing:
                - model: model identifier to use.
                - texts: list of input strings to embed.
                - pooling: pooling strategy applied to model outputs.
                - normalize: if true, L2-normalize each embedding vector.
        
        Returns:
            BatchEmbedResponse: Contains:
                - embeddings: list of embedding vectors (one list of floats per input text).
                - shapes: list of shapes corresponding to each embedding.
                - total_time_ms: total time spent processing the batch in milliseconds.
                - texts_processed: number of texts processed.
        
        Raises:
            HTTPException: on failure to load the model or compute embeddings (returns HTTP 500).
        """
        import numpy as np

        start_time = time.time()
        embeddings: list[list[float]] = []
        shapes: list[list[int]] = []

        try:
            # Get or load model once for all texts
            loaded = model_manager.get_or_load(request.model)

            # Process each text
            for text in request.texts:
                output = registry.embed(
                    loaded_model=loaded,
                    text=text,
                    pooling=request.pooling,
                )

                # Convert to list
                embedding_list = tensor_to_list(output.embedding)

                # Optionally L2 normalize
                if request.normalize:
                    arr = np.array(embedding_list)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        embedding_list = (arr / norm).tolist()

                embeddings.append(embedding_list)
                shapes.append(list(output.shape))

            total_time = (time.time() - start_time) * 1000

            return BatchEmbedResponse(
                embeddings=embeddings,
                shapes=shapes,
                total_time_ms=total_time,
                texts_processed=len(request.texts),
            )

        except Exception as e:
            logger.exception(f"Batch embedding failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app