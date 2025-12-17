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
        """
        Initialize a LoomClient with the given base URL and request timeout.
        
        Parameters:
            base_url (str): Base address of the Loom server. Accepts HTTP URLs (e.g. "http://host:port")
                or a Unix socket URL using the "unix" scheme (e.g. "unix:///path/to/socket").
            timeout (float): Default request timeout in seconds.
        
        Behavior:
            - Determines whether the client will use an HTTP transport or a Unix domain socket by
              parsing `base_url`. If a Unix socket URL is provided, `socket_path` is set to the socket
              file path and an internal HTTP dummy base is used for the underlying HTTP client.
            - Stores `base_url`, `timeout`, and initializes internal client-related attributes.
        """
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
        """
        Lazily create and return a cached httpx.Client configured for the client's base URL and transport.
        
        Returns:
            httpx.Client: The cached HTTP client instance. For Unix socket base URLs the client uses a UDS transport; otherwise it uses a standard HTTP transport.
        """
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
        """
        Close the underlying HTTP client and clear the cached client reference.
        
        Closes the internal httpx.Client if present and sets the cached client to None; safe to call multiple times.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> LoomClient:
        """
        Enter a context manager and yield this LoomClient instance.
        
        Returns:
            LoomClient: The LoomClient instance.
        """
        return self

    def __exit__(self, *args: Any) -> None:
        """
        Close the client's underlying HTTP connection when exiting a context manager.
        
        Parameters:
            *args (Any): Standard context-manager exception information (exc_type, exc_value, traceback); these values are ignored by this method.
        
        Notes:
            This method does not suppress exceptions raised inside the context.
        """
        self.close()

    # ========================================================================
    # Health & Info
    # ========================================================================

    def health(self) -> dict[str, Any]:
        """
        Retrieve the server's health information.
        
        Returns:
            dict: Health information as returned by the server, typically including GPU details and server configuration.
        
        Raises:
            httpx.HTTPStatusError: If the server responds with a non-2xx status code.
        """
        response = self.client.get("/health")
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def list_models(self) -> dict[str, Any]:
        """
        Retrieve the server's list of currently loaded models and capacity information.
        
        Returns:
            dict: Dictionary with keys `loaded_models` (list) and `max_models` (int).
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
        """
        Determine which loader the server would select for the given model identifier.
        
        Parameters:
        	model_id (str): Model identifier to probe.
        
        Returns:
        	selection_info (dict[str, Any]): Server response describing loader selection and related metadata.
        """
        # Replace / with -- for URL safety
        safe_id = model_id.replace("/", "--")
        response = self.client.get(f"/loaders/probe/{safe_id}")
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
        """
        Unload the specified model from the server's memory.
        
        Parameters:
            model_id (str): Model identifier. Forward slashes in the identifier will be encoded as `--` for the request URL.
        
        Returns:
            dict[str, Any]: JSON response from the server describing the unload status.
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
        return_full_sequence: bool = False,
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
            return_full_sequence: Return hidden states for ALL tokens (manifold).
                Creates [num_tokens, hidden_size] tensor for geometric analysis.
            loader: Force specific loader

        Returns:
            Generation output with text, token_count, hidden_states,
            sequence_hidden_states (if return_full_sequence), metadata
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_hidden_states": return_hidden_states,
            "hidden_state_format": hidden_state_format,
            "return_full_sequence": return_full_sequence,
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
        """
        Compute an embedding vector for the given text using the specified model and pooling strategy.
        
        Parameters:
            pooling (str): Pooling strategy to reduce token-level representations into a single vector. Supported values: "last_token", "mean", "first_token".
            normalize (bool): If True, L2-normalize the resulting embedding vector.
        
        Returns:
            dict[str, Any]: Response containing the embedding vector (key "embedding"), its shape (key "shape"), and optional metadata (key "metadata").
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
        """
        Compute embeddings and diagnostic metrics for the given text using the specified model.
        
        Parameters:
            model (str): Model identifier to use for analysis.
            text (str): Input text to analyze.
            pooling (str): Pooling strategy for producing the embedding (e.g., "last_token").
        
        Returns:
            dict[str, Any]: Analysis results including embedding, tensor shapes, statistics, and an estimated effective dimensionality (`D_eff`), along with any additional diagnostic metadata.
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
        """
        Generate text for multiple prompts using the same model in a single request.
        
        Parameters:
            model (str): Model ID to use for all prompts.
            prompts (list[str]): Input prompts to generate text for.
            max_tokens (int): Maximum tokens to generate per prompt.
            temperature (float): Sampling temperature for generation.
            top_p (float): Nucleus sampling probability.
            return_hidden_states (bool): If true, include hidden states in each result.
            hidden_state_layers (list[int] | None): List of layer indices to return (use -1 for last layer).
            hidden_state_format (str): Format for hidden states, e.g., "list" or "base64".
            loader (str | None): Optional loader name to force for the request.
        
        Returns:
            dict[str, Any]: Response dictionary containing a `results` array with per-prompt outputs and any aggregate totals.
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
        """
        Extract embeddings for multiple texts using the specified model in a single batch.
        
        Parameters:
            model (str): Model identifier to use for embedding.
            texts (list[str]): Texts to embed; order of returned embeddings matches this list.
            pooling (str): Pooling strategy to produce a single vector per text. One of "last_token", "mean", or "first_token".
            normalize (bool): If true, L2-normalize each embedding vector.
        
        Returns:
            dict[str, Any]: Parsed JSON response containing the batch embeddings (typically under an "embeddings" key) and related totals/metadata.
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
        """
        Stream generation results from the server as token events followed by a final result.
        
        Parameters:
            model (str): Model identifier to use for generation.
            prompt (str): Input prompt text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature controlling randomness.
            top_p (float): Nucleus sampling probability threshold.
            return_hidden_states (bool): Include hidden states in the final result when True.
            hidden_state_layers (list[int] | None): Layers to include for hidden states (use -1 for last layer).
            hidden_state_format (str): Output format for hidden states, e.g. "list" or "base64".
            loader (str | None): If provided, force a specific loader to use.
        
        Returns:
            Iterator[StreamingToken | StreamingResult]: Yields a StreamingToken for each streamed token event, then a single StreamingResult when generation completes; raises RuntimeError if a streaming error event is received.
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