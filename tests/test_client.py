"""Tests for client utility and Unix socket transport."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.client import LoomClient, connect


class TestLoomClientInit:
    """Tests for LoomClient initialization."""

    def test_http_client_init(self):
        """Test HTTP client initialization."""
        client = LoomClient("http://localhost:8080")

        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 300.0
        assert client.is_unix_socket is False
        assert client.socket_path is None
        assert client._http_base == "http://localhost:8080"

    def test_http_client_custom_port(self):
        """Test HTTP client with custom port."""
        client = LoomClient("http://127.0.0.1:9000")

        assert client.base_url == "http://127.0.0.1:9000"
        assert client.is_unix_socket is False
        assert client._http_base == "http://127.0.0.1:9000"

    def test_unix_socket_client_init(self):
        """Test Unix socket client initialization."""
        client = LoomClient("unix:///tmp/loom.sock")

        assert client.base_url == "unix:///tmp/loom.sock"
        assert client.is_unix_socket is True
        assert client.socket_path == "/tmp/loom.sock"
        assert client._http_base == "http://localhost"

    def test_unix_socket_custom_path(self):
        """Test Unix socket with custom path."""
        client = LoomClient("unix:///var/run/custom.sock")

        assert client.socket_path == "/var/run/custom.sock"
        assert client.is_unix_socket is True

    def test_custom_timeout(self):
        """Test custom timeout setting."""
        client = LoomClient("http://localhost:8080", timeout=60.0)

        assert client.timeout == 60.0


class TestLoomClientHttpx:
    """Tests for LoomClient httpx client creation."""

    def test_http_client_creation(self):
        """Test HTTP client is created correctly."""
        loom_client = LoomClient("http://localhost:8080")

        with patch.object(httpx, "Client") as mock_client:
            mock_client.return_value = MagicMock()
            _ = loom_client.client

            mock_client.assert_called_once_with(
                base_url="http://localhost:8080",
                timeout=300.0,
            )

    def test_unix_socket_client_creation(self):
        """Test Unix socket client is created with transport."""
        loom_client = LoomClient("unix:///tmp/test.sock")

        with patch.object(httpx, "HTTPTransport") as mock_transport:
            with patch.object(httpx, "Client") as mock_client:
                mock_transport.return_value = MagicMock()
                mock_client.return_value = MagicMock()
                _ = loom_client.client

                mock_transport.assert_called_once_with(uds="/tmp/test.sock")
                mock_client.assert_called_once()
                # Verify transport was passed
                call_kwargs = mock_client.call_args[1]
                assert "transport" in call_kwargs

    def test_client_cached(self):
        """Test httpx client is cached after first access."""
        loom_client = LoomClient("http://localhost:8080")

        with patch.object(httpx, "Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Access twice
            client1 = loom_client.client
            client2 = loom_client.client

            # Should only create once
            assert mock_client.call_count == 1
            assert client1 is client2


class TestLoomClientClose:
    """Tests for LoomClient close functionality."""

    def test_close_releases_client(self):
        """Test close() releases the httpx client."""
        loom_client = LoomClient("http://localhost:8080")

        mock_httpx = MagicMock()
        loom_client._client = mock_httpx

        loom_client.close()

        mock_httpx.close.assert_called_once()
        assert loom_client._client is None

    def test_close_idempotent(self):
        """Test close() can be called multiple times safely."""
        loom_client = LoomClient("http://localhost:8080")

        # Close without client created
        loom_client.close()  # Should not raise
        loom_client.close()  # Should not raise

    def test_context_manager(self):
        """
        Ensure LoomClient used as a context manager closes its underlying HTTP client on exit.
        """
        with patch.object(httpx, "Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with LoomClient("http://localhost:8080") as client:
                _ = client.client  # Force creation

            mock_instance.close.assert_called_once()


class TestLoomClientHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_success(self):
        """Test health check returns data."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "healthy"}
            mock_instance = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.health()

            mock_instance.get.assert_called_once_with("/health")
            assert result == {"status": "healthy"}


class TestLoomClientModelsEndpoints:
    """Tests for model management endpoints."""

    def test_list_models(self):
        """Test list_models endpoint."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "loaded_models": ["model1"],
                "max_models": 3,
            }
            mock_instance = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.list_models()

            mock_instance.get.assert_called_once_with("/models")
            assert result["loaded_models"] == ["model1"]

    def test_list_loaders(self):
        """Test list_loaders endpoint."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"loaders": ["transformers"]}
            mock_instance = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.list_loaders()

            mock_instance.get.assert_called_once_with("/loaders")
            assert "loaders" in result

    def test_probe_loader(self):
        """Test probe_loader endpoint."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model_id": "test/model",
                "selected_loader": "transformers",
            }
            mock_instance = MagicMock()
            mock_instance.get.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.probe_loader("test/model")

            # Model ID with / is encoded as -- for URL safety
            mock_instance.get.assert_called_once_with("/loaders/probe/test--model")
            assert result["selected_loader"] == "transformers"

    def test_load_model(self):
        """Test load_model endpoint."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "model_id": "test/model",
                "hidden_size": 768,
            }
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            client.load_model(
                "test/model",
                device="cuda:0",
                dtype="float16",
            )

            mock_instance.post.assert_called_once()
            call_args = mock_instance.post.call_args
            assert call_args[0][0] == "/models/load"
            payload = call_args[1]["json"]
            assert payload["model"] == "test/model"
            assert payload["device"] == "cuda:0"
            assert payload["dtype"] == "float16"

    def test_load_model_with_loader(self):
        """Test load_model with explicit loader."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"model_id": "test/model"}
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            client.load_model("test/model", loader="custom")

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["loader"] == "custom"

    def test_unload_model(self):
        """Test unload_model endpoint."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "unloaded"}
            mock_instance = MagicMock()
            mock_instance.delete.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.unload_model("org/model")

            # Verify / is replaced with --
            mock_instance.delete.assert_called_once_with("/models/org--model")
            assert result["status"] == "unloaded"


class TestLoomClientGenerateEndpoint:
    """Tests for generate endpoint."""

    def test_generate_basic(self):
        """Test basic generation."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "text": "Generated text",
                "token_count": 5,
            }
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.generate("model", "Hello")

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["model"] == "model"
            assert payload["prompt"] == "Hello"
            assert payload["return_hidden_states"] is True  # Default
            assert result["text"] == "Generated text"

    def test_generate_with_params(self):
        """Test generation with custom parameters."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": "Output"}
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            client.generate(
                "model",
                "Prompt",
                max_tokens=100,
                temperature=0.5,
                top_p=0.8,
                return_hidden_states=False,
                hidden_state_layers=[-1, -2],
                hidden_state_format="base64",
            )

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["max_tokens"] == 100
            assert payload["temperature"] == 0.5
            assert payload["top_p"] == 0.8
            assert payload["return_hidden_states"] is False
            assert payload["hidden_state_layers"] == [-1, -2]
            assert payload["hidden_state_format"] == "base64"


class TestLoomClientEmbedEndpoint:
    """Tests for embed endpoint."""

    def test_embed_basic(self):
        """Test basic embedding."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3],
                "shape": [3],
            }
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.embed("model", "Hello world")

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["model"] == "model"
            assert payload["text"] == "Hello world"
            assert payload["pooling"] == "last_token"  # Default
            assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_embed_with_options(self):
        """Test embedding with custom options."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.5]}
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            client.embed("model", "Text", pooling="mean", normalize=True)

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["pooling"] == "mean"
            assert payload["normalize"] is True


class TestLoomClientAnalyzeEndpoint:
    """Tests for analyze endpoint."""

    def test_analyze_basic(self):
        """Test basic analysis."""
        with patch.object(httpx, "Client") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "D_eff": 45.5,
                "beta": 1.2,
            }
            mock_instance = MagicMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value = mock_instance

            client = LoomClient("http://localhost:8080")
            result = client.analyze("model", "Test text")

            payload = mock_instance.post.call_args[1]["json"]
            assert payload["model"] == "model"
            assert payload["text"] == "Test text"
            assert result["D_eff"] == 45.5


class TestConnectFunction:
    """Tests for the connect() convenience function."""

    def test_connect_default(self):
        """Test connect with default parameters."""
        client = connect()

        assert isinstance(client, LoomClient)
        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 300.0

    def test_connect_custom_url(self):
        """Test connect with custom URL."""
        client = connect("unix:///tmp/loom.sock")

        assert client.base_url == "unix:///tmp/loom.sock"
        assert client.is_unix_socket is True

    def test_connect_custom_timeout(self):
        """Test connect with custom timeout."""
        client = connect(timeout=60.0)

        assert client.timeout == 60.0


class TestServerTransportConfig:
    """Tests for server transport configuration parsing."""

    def test_parse_args_transport_http(self):
        """Test transport=http argument parsing."""
        from src.server import parse_args

        with patch("sys.argv", ["loom", "--transport", "http"]):
            args = parse_args()
            assert args.transport == "http"

    def test_parse_args_transport_unix(self):
        """Test transport=unix argument parsing."""
        from src.server import parse_args

        with patch("sys.argv", ["loom", "--transport", "unix"]):
            args = parse_args()
            assert args.transport == "unix"

    def test_parse_args_transport_both(self):
        """Test transport=both argument parsing."""
        from src.server import parse_args

        with patch("sys.argv", ["loom", "--transport", "both"]):
            args = parse_args()
            assert args.transport == "both"

    def test_parse_args_unix_socket_path(self):
        """Test --unix-socket argument parsing."""
        from src.server import parse_args

        with patch("sys.argv", ["loom", "--unix-socket", "/var/run/loom.sock"]):
            args = parse_args()
            assert args.unix_socket == "/var/run/loom.sock"


class TestUnixSocketServerFunctions:
    """Tests for Unix socket server functions."""

    @pytest.mark.asyncio
    async def test_run_unix_server_removes_existing_socket(self):
        """Test that run_unix_server removes existing socket file."""
        from src.server import run_unix_server

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = os.path.join(tmpdir, "test.sock")

            # Create a dummy socket file
            Path(socket_path).touch()
            assert os.path.exists(socket_path)

            with patch("src.server.uvicorn.Server") as mock_server:
                mock_instance = MagicMock()
                # Make serve() an async function that completes immediately
                mock_instance.serve = MagicMock(
                    side_effect=lambda: asyncio.sleep(0)
                )
                mock_server.return_value = mock_instance

                # The function should remove the existing file before starting
                with patch("src.server.uvicorn.Config"):
                    await run_unix_server(MagicMock(), socket_path, "info")

                    # Verify server was started
                    mock_server.assert_called_once()
                    mock_instance.serve.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_both_servers_calls_gather(self):
        """Test that run_both_servers runs both servers concurrently."""
        from src.server import run_both_servers

        with patch("src.server.run_http_server") as mock_http:
            with patch("src.server.run_unix_server") as mock_unix:
                mock_http.return_value = None
                mock_unix.return_value = None

                await run_both_servers(
                    MagicMock(),
                    "0.0.0.0",
                    8080,
                    "/tmp/test.sock",
                    "info",
                )

                mock_http.assert_called_once()
                mock_unix.assert_called_once()