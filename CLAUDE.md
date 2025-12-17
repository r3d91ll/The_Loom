# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**The Loom** is a Python-based model server that exposes hidden states for AI interpretability research. Part of the **Weaver ecosystem** for multi-agent orchestration and conveyance measurement.

**Core value proposition:** Access to the final hidden state (the "boundary object") before text generation - enabling conveyance measurement and interpretability research.

## Commands

```bash
# Install (Poetry)
poetry install                 # Install all dependencies

# Run server
poetry run loom                           # Default config (HTTP on port 8080)
poetry run loom --port 9000               # Custom HTTP port
poetry run loom --transport unix          # Unix socket only
poetry run loom --transport both          # HTTP and Unix socket
poetry run loom --unix-socket /tmp/my.sock  # Custom socket path
poetry run loom --preload model-id        # Preload model

# Run tests
poetry run pytest                             # All tests
poetry run pytest tests/test_extraction.py    # Specific file
poetry run pytest -m "not slow"               # Skip slow tests
poetry run pytest -v                          # Verbose

# Type checking and linting
poetry run mypy src --pretty
poetry run ruff format src tests
poetry run ruff check src tests
```

## Architecture

```
src/
├── server.py              # CLI entry point (`loom` command)
├── client.py              # Client utility (HTTP and Unix socket)
├── config.py              # Configuration (Pydantic-based)
├── loaders/
│   ├── base.py            # ModelLoader ABC, LoadedModel dataclass
│   ├── transformers_loader.py      # HuggingFace transformers (~80%)
│   ├── sentence_transformers_loader.py  # Embedding models (~15%)
│   ├── custom_loader.py   # Edge cases, research models (~5%)
│   └── registry.py        # LoaderRegistry with auto-detection
├── transport/
│   └── http.py            # FastAPI server, endpoints
├── extraction/
│   └── hidden_states.py   # D_eff, beta, geometric analysis
└── utils/
    ├── gpu.py             # GPU device management
    └── serialization.py   # Tensor to JSON conversion
```

## Key Design Decisions

1. **Multi-loader architecture** - Three loaders with auto-detection via LoaderRegistry
2. **TransformersLoader is primary** - Uses `output_hidden_states=True` for hidden state extraction
3. **SentenceTransformersLoader for embeddings** - Handles SBERT, BGE, E5, etc.
4. **CustomLoader for edge cases** - Registry-based custom model support
5. **ModelManager with LRU eviction** - Manages multiple loaded models with configurable max
6. **Pydantic for all validation** - Request/response models and configuration
7. **FastAPI for HTTP** - Async, automatic OpenAPI docs
8. **Hidden states as JSON** - Two formats: `list` (readable) and `base64` (efficient)

## API Patterns

```python
# Generate with hidden states (auto-detects loader)
POST /generate
{
  "model": "model-id",
  "prompt": "text",
  "return_hidden_states": true,
  "hidden_state_layers": [-1],  # -1 = last layer
  "loader": null  # or "transformers", "sentence_transformers", "custom"
}

# Extract embedding
POST /embed
{
  "model": "model-id",
  "text": "text",
  "pooling": "last_token"  # or "mean", "first_token"
}

# Probe loader selection (debugging)
GET /loaders/probe/{model_id}

# List available loaders
GET /loaders
```

## Multi-Loader Auto-Detection

The LoaderRegistry auto-selects loaders based on model ID patterns:

```python
# Pattern matching in order:
1. sentence-transformers/* → SentenceTransformersLoader
2. BAAI/bge-* → SentenceTransformersLoader
3. intfloat/e5-* → SentenceTransformersLoader
4. Everything else → TransformersLoader (fallback)
```

Override with `model_overrides` in config or `loader` parameter in requests.

## Testing Strategy

- **Unit tests** mock the model loader and GPU manager
- **Use TestClient from FastAPI** for endpoint testing
- **Avoid GPU-dependent tests** in CI (mark with `@pytest.mark.integration`)

## Configuration Priority

1. Environment variables (`LOOM_*`)
2. Config file (specified or auto-discovered from `~/.config/loom/`)
3. Default values

## Transport Options

The Loom supports multiple transport modes:

- **HTTP** (default): Standard HTTP server on configurable port
- **Unix socket**: Local IPC via Unix domain socket (faster for local calls)
- **Both**: Run HTTP and Unix socket simultaneously

```bash
# HTTP only (default)
poetry run loom --transport http --port 8080

# Unix socket only
poetry run loom --transport unix --unix-socket /tmp/loom.sock

# Both HTTP and Unix socket
poetry run loom --transport both
```

Environment variables:
- `LOOM_SERVER__TRANSPORT`: Override transport (http, unix, both)
- `LOOM_SERVER__HTTP_PORT`: Override HTTP port
- `LOOM_SERVER__UNIX_SOCKET`: Override socket path

## Python Client

The `LoomClient` class provides a convenient Python interface:

```python
from src.client import LoomClient, connect

# HTTP client
client = LoomClient("http://localhost:8080")

# Unix socket client (faster for local)
client = LoomClient("unix:///tmp/loom.sock")

# Or use convenience function
client = connect("unix:///tmp/loom.sock")

# Generate with hidden states
result = client.generate("meta-llama/Llama-3.1-8B", "Hello, world!")
print(result["text"])
print(result["hidden_states"])

# Extract embedding
embedding = client.embed("model-id", "Some text")

# Analyze (with D_eff metrics)
analysis = client.analyze("model-id", "Some text")

# Context manager for cleanup
with LoomClient("http://localhost:8080") as client:
    result = client.health()
```

## Integration Context

The Loom is called by WeaverCode (Go CLI) for conveyance measurement experiments. The hidden states enable geometric analysis of agent-to-agent communication.

```
WeaverCode (Go) ←──HTTP/Unix──→ The Loom (Python)
```
