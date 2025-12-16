"""Prometheus metrics for The Loom server.

Provides observability into model server performance including:
- Request latency by endpoint
- Token generation rates
- Model loading/unloading events
- GPU memory usage
- Active model counts
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency
_prometheus_available = False
_metrics_initialized = False

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _prometheus_available = True
except ImportError:
    logger.debug("prometheus_client not installed, metrics will be no-ops")


# ============================================================================
# Metric Definitions (created lazily)
# ============================================================================

# Request metrics
_request_counter: Any = None
_request_latency: Any = None

# Generation metrics
_tokens_generated: Any = None
_generation_latency: Any = None

# Embedding metrics
_embeddings_extracted: Any = None
_embedding_latency: Any = None

# Model metrics
_models_loaded: Any = None
_model_load_latency: Any = None
_model_load_counter: Any = None

# System metrics
_gpu_memory_used: Any = None
_gpu_memory_total: Any = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics (called lazily)."""
    global _metrics_initialized
    global _request_counter, _request_latency
    global _tokens_generated, _generation_latency
    global _embeddings_extracted, _embedding_latency
    global _models_loaded, _model_load_latency, _model_load_counter
    global _gpu_memory_used, _gpu_memory_total

    if _metrics_initialized or not _prometheus_available:
        return

    # Request metrics
    _request_counter = Counter(
        "loom_requests_total",
        "Total HTTP requests",
        ["endpoint", "method", "status"],
    )

    _request_latency = Histogram(
        "loom_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint", "method"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )

    # Generation metrics
    _tokens_generated = Counter(
        "loom_tokens_generated_total",
        "Total tokens generated",
        ["model"],
    )

    _generation_latency = Histogram(
        "loom_generation_latency_seconds",
        "Generation latency in seconds",
        ["model"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )

    # Embedding metrics
    _embeddings_extracted = Counter(
        "loom_embeddings_total",
        "Total embeddings extracted",
        ["model"],
    )

    _embedding_latency = Histogram(
        "loom_embedding_latency_seconds",
        "Embedding extraction latency in seconds",
        ["model"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    # Model metrics
    _models_loaded = Gauge(
        "loom_models_loaded",
        "Number of models currently loaded",
    )

    _model_load_latency = Histogram(
        "loom_model_load_latency_seconds",
        "Model loading latency in seconds",
        ["model", "loader", "quantization"],
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    _model_load_counter = Counter(
        "loom_model_loads_total",
        "Total model load operations",
        ["model", "loader", "quantization", "status"],
    )

    # System metrics
    _gpu_memory_used = Gauge(
        "loom_gpu_memory_used_bytes",
        "GPU memory used in bytes",
        ["device"],
    )

    _gpu_memory_total = Gauge(
        "loom_gpu_memory_total_bytes",
        "Total GPU memory in bytes",
        ["device"],
    )

    _metrics_initialized = True
    logger.info("Prometheus metrics initialized")


def is_metrics_available() -> bool:
    """Check if prometheus_client is available."""
    return _prometheus_available


def get_metrics() -> bytes:
    """
    Retrieve a snapshot of current metrics in Prometheus text exposition format.
    
    Returns:
        bytes: The metrics payload in Prometheus text format. If the Prometheus client is not available, returns a short bytes message indicating it is not installed.
    """
    if not _prometheus_available:
        return b"# prometheus_client not installed\n"

    _init_metrics()
    return cast(bytes, generate_latest(REGISTRY))


# ============================================================================
# Metric Recording Functions
# ============================================================================


def record_request(
    endpoint: str,
    method: str,
    status: int,
    latency: float,
) -> None:
    """
    Record an HTTP request metric for Prometheus.
    
    Parameters:
        endpoint (str): Logical endpoint or route name used as the `endpoint` label.
        method (str): HTTP method (e.g., "GET", "POST") used as the `method` label.
        status (int): HTTP response status code used as the `status` label.
        latency (float): Request latency in seconds to observe on the request latency histogram.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    _request_counter.labels(
        endpoint=endpoint,
        method=method,
        status=str(status),
    ).inc()
    _request_latency.labels(
        endpoint=endpoint,
        method=method,
    ).observe(latency)


def record_generation(
    model: str,
    tokens: int,
    latency: float,
) -> None:
    """
    Record token and latency metrics for a model generation event.
    
    Parameters:
        model (str): Name or identifier of the model that produced the tokens.
        tokens (int): Number of tokens generated.
        latency (float): Generation time in seconds.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    _tokens_generated.labels(model=model).inc(tokens)
    _generation_latency.labels(model=model).observe(latency)


def record_embedding(
    model: str,
    latency: float,
) -> None:
    """
    Record metrics for a single embedding extraction event.
    
    Parameters:
        model (str): Name or identifier of the model used for the embedding.
        latency (float): Elapsed time in seconds for the embedding extraction.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    _embeddings_extracted.labels(model=model).inc()
    _embedding_latency.labels(model=model).observe(latency)


def record_model_load(
    model: str,
    loader: str,
    quantization: str,
    latency: float,
    success: bool,
) -> None:
    """
    Emit Prometheus metrics for a model load attempt.
    
    Parameters:
        model (str): Name or identifier of the model being loaded.
        loader (str): Loader implementation or source used to load the model (e.g., "transformers", "s3").
        quantization (str): Quantization/precision label applied to the model (e.g., "fp16", "int8", "none").
        latency (float): Observed load duration in seconds.
        success (bool): True if the load completed successfully; False for a failed attempt.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    status = "success" if success else "failure"
    _model_load_counter.labels(
        model=model,
        loader=loader,
        quantization=quantization,
        status=status,
    ).inc()

    if success:
        _model_load_latency.labels(
            model=model,
            loader=loader,
            quantization=quantization,
        ).observe(latency)


def set_models_loaded(count: int) -> None:
    """
    Set the gauge for the current number of loaded models.
    
    If the Prometheus client is not available, this function does nothing.
    
    Parameters:
        count (int): Total number of models currently loaded; updates the models-loaded gauge to this value.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    _models_loaded.set(count)


def record_gpu_memory(device: str, used_bytes: int, total_bytes: int) -> None:
    """
    Record GPU memory usage for a given device in the Prometheus metrics.
    
    Parameters:
        device (str): Identifier of the GPU device (label used in metrics).
        used_bytes (int): Number of bytes currently used on the device.
        total_bytes (int): Total number of bytes available on the device.
    """
    if not _prometheus_available:
        return

    _init_metrics()
    _gpu_memory_used.labels(device=device).set(used_bytes)
    _gpu_memory_total.labels(device=device).set(total_bytes)


# ============================================================================
# Context Managers and Decorators
# ============================================================================


@contextmanager
def track_request_latency(endpoint: str, method: str) -> Generator[None, None, None]:
    """
    Measure and emit request latency to Prometheus when available.
    
    Parameters:
        endpoint (str): The request endpoint label (e.g., path or route) used for the metric.
        method (str): The HTTP method label (e.g., "GET", "POST") used for the metric.
    
    Description:
        Acts as a context manager that records the elapsed time between entering and exiting the context and, if Prometheus metrics are enabled, observes that latency on the request latency histogram with the provided endpoint and method labels.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if _prometheus_available:
            _init_metrics()
            latency = time.perf_counter() - start
            _request_latency.labels(endpoint=endpoint, method=method).observe(latency)


F = TypeVar("F", bound=Callable[..., Any])


def track_generation(model_getter: Callable[[Any], str]) -> Callable[[F], F]:
    """
    Create a decorator that records generation metrics (token count and latency) for each invocation of the wrapped function.
    
    The provided `model_getter` is called with the wrapped function's positional and keyword arguments to determine the model name used for metrics. After the wrapped function executes, the decorator records the elapsed time and the number of tokens produced: it uses `result.token_count` if present, otherwise the length of `result.token_ids` if present, otherwise `0`. The wrapped function's return value is returned unchanged.
    
    Parameters:
        model_getter (Callable[[Any], str]): Function that accepts the wrapped function's `*args` and `**kwargs` and returns the model name to label metrics.
    
    Returns:
        Callable[[F], F]: A decorator that wraps a function to record generation latency and token count for the model returned by `model_getter`.
    """

    def decorator(func: F) -> F:
        """
        Wraps `func` to measure its execution latency, extract the model name and token count from the result, and record generation metrics.
        
        Parameters:
            func (callable): The function to wrap. The wrapper will call `func` with the same arguments.
        
        Returns:
            callable: A wrapper around `func` that, on each call, measures elapsed time, obtains the model name using the surrounding `model_getter`, determines tokens as `result.token_count` if present or `len(result.token_ids)` if present (0 if neither), calls `record_generation(model, tokens, latency)`, and returns `func`'s original result.
        """
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latency = time.perf_counter() - start

            model = model_getter(*args, **kwargs)
            tokens = getattr(result, "token_count", len(getattr(result, "token_ids", [])))
            record_generation(model, tokens, latency)

            return result

        return wrapper  # type: ignore[return-value]

    return decorator