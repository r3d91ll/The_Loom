"""HTTP transport layer using FastAPI."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

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
from ..utils.serialization import serialize_hidden_states, tensor_to_list

logger = logging.getLogger(__name__)


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
    hidden_state_layers: list[int] = Field(
        default=[-1], description="Which layers to return (-1 = last)"
    )
    return_attention: bool = Field(default=False, description="Return attention weights")
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


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    text: str
    token_count: int
    hidden_states: dict[str, HiddenStateResponse] | None = None
    attention_weights: dict[str, Any] | None = None
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
    hidden_state_layers: list[int] = Field(
        default=[-1], description="Which layers to return (-1 = last)"
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
    hidden_state_layers: list[int] = Field(
        default=[-1], description="Which layers to return (-1 = last)"
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
    """Manages loaded models with LRU eviction and multi-loader support."""

    def __init__(
        self,
        gpu_manager: GPUManager,
        registry: LoaderRegistry,
        max_models: int = 3,
    ):
        self.gpu_manager = gpu_manager
        self.registry = registry
        self.max_models = max_models
        self.loaded_models: dict[str, LoadedModel] = {}
        self.access_order: list[str] = []  # For LRU eviction

    def get_or_load(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "auto",
        loader_name: str | None = None,
        quantization: str | None = None,
    ) -> LoadedModel:
        """Get a loaded model, loading it if necessary.

        Args:
            model_id: Model identifier
            device: Device to load on (uses default if None)
            dtype: Data type
            loader_name: Force specific loader (auto-detect if None)
            quantization: Quantization mode (4bit, 8bit, gptq, awq)

        Returns:
            LoadedModel instance
        """
        # Check if already loaded
        if model_id in self.loaded_models:
            # Update access order for LRU
            if model_id in self.access_order:
                self.access_order.remove(model_id)
            self.access_order.append(model_id)
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

        return loaded

    def _evict_oldest(self) -> None:
        """Evict the least recently used model."""
        if not self.access_order:
            return

        oldest = self.access_order.pop(0)
        if oldest in self.loaded_models:
            logger.info(f"Evicting model: {oldest}")
            del self.loaded_models[oldest]
            self.gpu_manager.clear_cache()

    def unload(self, model_id: str) -> bool:
        """Unload a specific model."""
        if model_id not in self.loaded_models:
            return False

        del self.loaded_models[model_id]
        if model_id in self.access_order:
            self.access_order.remove(model_id)
        self.gpu_manager.clear_cache()
        logger.info(f"Unloaded model: {model_id}")
        return True

    def list_loaded(self) -> list[str]:
        """List currently loaded models."""
        return list(self.loaded_models.keys())

    def get_loaded_info(self) -> list[dict[str, Any]]:
        """Get detailed info about loaded models."""
        return [
            {
                "model_id": model.model_id,
                "device": str(model.device),
                "dtype": str(model.dtype).replace("torch.", ""),
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "loader_type": model.loader_type,
                "quantization": model.metadata.get("quantization", "none"),
            }
            for model in self.loaded_models.values()
        ]


# ============================================================================
# Metrics Middleware
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request metrics."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Track request latency and status."""
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
    """Create the FastAPI application.

    Args:
        config: Server configuration (uses global config if None)

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = get_config()

    app = FastAPI(
        title="The Loom",
        description="Hidden state extraction server for AI research - part of the Weaver ecosystem",
        version="0.2.0",
    )

    # CORS middleware for browser-based clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics middleware (only if prometheus_client is available)
    if is_metrics_available():
        app.add_middleware(MetricsMiddleware)

    # Initialize components
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
    )

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
        """Health check endpoint."""
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
        """List loaded models with detailed info."""
        return {
            "loaded_models": model_manager.get_loaded_info(),
            "max_models": config.models.max_loaded,
        }

    @app.get("/loaders")
    async def list_loaders() -> dict[str, Any]:
        """List available loaders."""
        return {
            "loaders": registry.list_loaders(),
            "fallback_order": registry.fallback_order,
        }

    @app.get("/loaders/probe/{model_id:path}")
    async def probe_model_loader(model_id: str) -> dict[str, Any]:
        """Probe which loader would handle a model without loading it.

        Useful for debugging loader selection.
        """
        return registry.probe_model(model_id)

    @app.post("/models/load", response_model=ModelLoadResponse)
    async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
        """Load a model into memory."""
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
        """Unload a model from memory."""
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
        """Generate text with optional hidden state extraction.

        This is the core research endpoint - it returns the geometric
        representation (hidden state) alongside the generated text.
        """
        start_time = time.perf_counter()
        try:
            # Get or load model (with optional loader override)
            loaded = model_manager.get_or_load(
                request.model,
                loader_name=request.loader,
            )

            # Generate using registry (uses appropriate loader for model)
            output = registry.generate(
                loaded_model=loaded,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                return_hidden_states=request.return_hidden_states,
                hidden_state_layers=request.hidden_state_layers,
                return_attention=request.return_attention,
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

            return GenerateResponse(
                text=output.text,
                token_count=len(output.token_ids),
                hidden_states=hidden_states_response,
                attention_weights=attention_response,
                metadata=output.metadata,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/embed", response_model=EmbedResponse)
    async def embed(request: EmbedRequest) -> EmbedResponse:
        """Extract embedding for text.

        For decoder-only models, this uses the last token's hidden state
        as the embedding (contains accumulated context).
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
        """Extract embedding and compute diagnostic metrics.

        Returns hidden state analysis including D_eff estimation.
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
        """Generate text with streaming Server-Sent Events.

        Returns SSE stream with events:
        - event: token - For each generated token
        - event: done - Final event with complete output and hidden states
        - event: error - If an error occurs

        Example SSE format:
            event: token
            data: {"token": "Hello", "token_id": 1}

            event: done
            data: {"text": "Hello world", "token_count": 2, "hidden_states": {...}}
        """

        async def event_generator() -> AsyncIterator[str]:
            try:
                # Get or load model
                loaded = model_manager.get_or_load(
                    request.model,
                    loader_name=request.loader,
                )

                # Stream tokens
                for item in registry.generate_stream(
                    loaded_model=loaded,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=request.hidden_state_layers,
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
        """Generate text for multiple prompts in a batch.

        Processes all prompts with the same model efficiently.
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

            # Process each prompt
            for prompt in request.prompts:
                output = registry.generate(
                    loaded_model=loaded,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    return_hidden_states=request.return_hidden_states,
                    hidden_state_layers=request.hidden_state_layers,
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
        """Extract embeddings for multiple texts in a batch.

        Processes all texts with the same model efficiently.
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
