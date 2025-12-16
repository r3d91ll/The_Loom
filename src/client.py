"""Client utilities for connecting to The Loom server.

Supports both HTTP and Unix socket connections.

Usage:
    # HTTP client
    client = LoomClient("http://localhost:8080")

    # Unix socket client
    client = LoomClient("unix:///tmp/loom.sock")

    # Generate with hidden states
    result = client.generate("meta-llama/Llama-3.1-8B", "Hello, world!")
    print(result["text"])
    print(result["hidden_states"])
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

import httpx


@dataclass
class StreamingToken:
    """A single token from streaming generation."""

    token: str
    token_id: int
    is_finished: bool = False
    finish_reason: str | None = None


@dataclass
class StreamingResult:
    """Final result from streaming generation."""

    text: str
    token_count: int
    token_ids: list[int]
    hidden_states: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class LoomClient:
    """Client for The Loom model server.

    Supports both HTTP and Unix socket connections.

    Args:
        base_url: Server URL. Use "http://host:port" for HTTP or
                  "unix:///path/to/socket" for Unix socket.
        timeout: Request timeout in seconds (default: 300 for model loading)
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client: httpx.Client | None = None
        self.socket_path: str | None = None

        # Parse URL to determine transport type
        parsed = urlparse(base_url)
        self.is_unix_socket = parsed.scheme == "unix"

        if self.is_unix_socket:
            # Extract socket path from URL
            self.socket_path = parsed.path
            self._http_base = "http://localhost"  # Dummy base for httpx
        else:
            self._http_base = base_url

    @property
    def client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            if self.is_unix_socket:
                # Create Unix socket transport
                transport = httpx.HTTPTransport(uds=self.socket_path)
                self._client = httpx.Client(
                    base_url=self._http_base,
                    transport=transport,
                    timeout=self.timeout,
                )
            else:
                self._client = httpx.Client(
                    base_url=self._http_base,
                    timeout=self.timeout,
                )
        return self._client

    def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> LoomClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ========================================================================
    # Health & Info
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """Check server health.

        Returns:
            Health status including GPU info and config
        """
        response = self.client.get("/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def list_models(self) -> dict[str, Any]:
        """List loaded models.

        Returns:
            Dict with loaded_models list and max_models
        """
        response = self.client.get("/models")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def list_loaders(self) -> dict[str, Any]:
        """List available loaders.

        Returns:
            Dict with loaders info and fallback order
        """
        response = self.client.get("/loaders")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def probe_loader(self, model_id: str) -> dict[str, Any]:
        """Probe which loader would handle a model.

        Args:
            model_id: Model identifier

        Returns:
            Loader selection info
        """
        response = self.client.get(f"/loaders/probe/{model_id}")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    # ========================================================================
    # Model Management
    # ========================================================================

    def load_model(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str = "auto",
        loader: str | None = None,
        quantization: str | None = None,
    ) -> dict[str, Any]:
        """Load a model into memory.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load on (e.g., "cuda:0")
            dtype: Data type (auto, float16, bfloat16, float32)
            loader: Force specific loader (auto-detect if None)
            quantization: Quantization mode (4bit, 8bit, gptq, awq)

        Returns:
            Model load info including hidden_size, num_layers, quantization
        """
        payload: dict[str, Any] = {
            "model": model_id,
            "dtype": dtype,
        }
        if device is not None:
            payload["device"] = device
        if loader is not None:
            payload["loader"] = loader
        if quantization is not None:
            payload["quantization"] = quantization

        response = self.client.post("/models/load", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def unload_model(self, model_id: str) -> dict[str, Any]:
        """Unload a model from memory.

        Args:
            model_id: Model identifier (use -- for / in path)

        Returns:
            Unload status
        """
        # Replace / with -- for URL
        safe_id = model_id.replace("/", "--")
        response = self.client.delete(f"/models/{safe_id}")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    # ========================================================================
    # Generation & Embedding
    # ========================================================================

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | None = None,
        hidden_state_format: str = "list",
        loader: str | None = None,
    ) -> dict[str, Any]:
        """Generate text with optional hidden state extraction.

        This is the core research endpoint - returns the geometric
        representation (hidden state) alongside the generated text.

        Args:
            model: Model ID
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            return_hidden_states: Whether to return hidden states
            hidden_state_layers: Which layers to return (-1 = last)
            hidden_state_format: Format for hidden states (list or base64)
            loader: Force specific loader

        Returns:
            Generation output with text, token_count, hidden_states, metadata
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_hidden_states": return_hidden_states,
            "hidden_state_format": hidden_state_format,
        }
        if hidden_state_layers is not None:
            payload["hidden_state_layers"] = hidden_state_layers
        if loader is not None:
            payload["loader"] = loader

        response = self.client.post("/generate", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def embed(
        self,
        model: str,
        text: str,
        pooling: str = "last_token",
        normalize: bool = False,
    ) -> dict[str, Any]:
        """Extract embedding for text.

        Args:
            model: Model ID
            text: Text to embed
            pooling: Pooling strategy (last_token, mean, first_token)
            normalize: Whether to L2 normalize the embedding

        Returns:
            Embedding output with embedding vector, shape, metadata
        """
        payload = {
            "model": model,
            "text": text,
            "pooling": pooling,
            "normalize": normalize,
        }

        response = self.client.post("/embed", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def analyze(
        self,
        model: str,
        text: str,
        pooling: str = "last_token",
    ) -> dict[str, Any]:
        """Extract embedding and compute diagnostic metrics.

        Returns hidden state analysis including D_eff estimation.

        Args:
            model: Model ID
            text: Text to analyze
            pooling: Pooling strategy

        Returns:
            Analysis with shape, stats, D_eff estimation, etc.
        """
        payload = {
            "model": model,
            "text": text,
            "pooling": pooling,
        }

        response = self.client.post("/analyze", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def generate_batch(
        self,
        model: str,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_hidden_states: bool = False,
        hidden_state_layers: list[int] | None = None,
        hidden_state_format: str = "list",
        loader: str | None = None,
    ) -> dict[str, Any]:
        """Generate text for multiple prompts in a batch.

        Efficiently processes all prompts with the same model.

        Args:
            model: Model ID
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            return_hidden_states: Whether to return hidden states
            hidden_state_layers: Which layers to return (-1 = last)
            hidden_state_format: Format for hidden states (list or base64)
            loader: Force specific loader

        Returns:
            Batch response with results array and totals
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompts": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_hidden_states": return_hidden_states,
            "hidden_state_format": hidden_state_format,
        }
        if hidden_state_layers is not None:
            payload["hidden_state_layers"] = hidden_state_layers
        if loader is not None:
            payload["loader"] = loader

        response = self.client.post("/generate/batch", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def embed_batch(
        self,
        model: str,
        texts: list[str],
        pooling: str = "last_token",
        normalize: bool = False,
    ) -> dict[str, Any]:
        """Extract embeddings for multiple texts in a batch.

        Efficiently processes all texts with the same model.

        Args:
            model: Model ID
            texts: List of texts to embed
            pooling: Pooling strategy (last_token, mean, first_token)
            normalize: Whether to L2 normalize the embeddings

        Returns:
            Batch response with embeddings array and totals
        """
        payload = {
            "model": model,
            "texts": texts,
            "pooling": pooling,
            "normalize": normalize,
        }

        response = self.client.post("/embed/batch", json=payload)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def generate_stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_hidden_states: bool = False,
        hidden_state_layers: list[int] | None = None,
        hidden_state_format: str = "list",
        loader: str | None = None,
    ) -> Iterator[StreamingToken | StreamingResult]:
        """Generate text with streaming token output.

        Uses Server-Sent Events for real-time token streaming.

        Args:
            model: Model ID
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            return_hidden_states: Whether to return hidden states in final event
            hidden_state_layers: Which layers to return (-1 = last)
            hidden_state_format: Format for hidden states (list or base64)
            loader: Force specific loader

        Yields:
            StreamingToken for each token, then StreamingResult at end
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_hidden_states": return_hidden_states,
            "hidden_state_format": hidden_state_format,
        }
        if hidden_state_layers is not None:
            payload["hidden_state_layers"] = hidden_state_layers
        if loader is not None:
            payload["loader"] = loader

        # Use streaming request
        with self.client.stream("POST", "/generate/stream", json=payload) as response:
            response.raise_for_status()

            event_type: str | None = None
            data_buffer: list[str] = []

            for line in response.iter_lines():
                line = line.strip()

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].strip())
                elif line == "" and event_type and data_buffer:
                    # End of event
                    data_str = "".join(data_buffer)
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        data_buffer = []
                        event_type = None
                        continue

                    if event_type == "token":
                        yield StreamingToken(
                            token=data.get("token", ""),
                            token_id=data.get("token_id", -1),
                            is_finished=data.get("is_finished", False),
                            finish_reason=data.get("finish_reason"),
                        )
                    elif event_type == "done":
                        yield StreamingResult(
                            text=data.get("text", ""),
                            token_count=data.get("token_count", 0),
                            token_ids=data.get("token_ids", []),
                            hidden_states=data.get("hidden_states"),
                            metadata=data.get("metadata"),
                        )
                    elif event_type == "error":
                        raise RuntimeError(data.get("error", "Unknown streaming error"))

                    data_buffer = []
                    event_type = None


# Convenience function for quick access
def connect(base_url: str = "http://localhost:8080", timeout: float = 300.0) -> LoomClient:
    """Create a client connection to The Loom server.

    Args:
        base_url: Server URL (http://... or unix://...)
        timeout: Request timeout in seconds

    Returns:
        LoomClient instance

    Example:
        >>> client = connect("unix:///tmp/loom.sock")
        >>> result = client.generate("llama3.1:8b", "Hello!")
    """
    return LoomClient(base_url, timeout)
