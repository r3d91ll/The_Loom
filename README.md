# The Loom

**The Loom** is a model server that exposes hidden states for AI interpretability and conveyance measurement. Part of the **Weaver ecosystem** for multi-agent orchestration.

## Why The Loom?

Production servers (vLLM, Ollama, llama.cpp) optimize for throughput, not observability. They don't expose the final hidden state - the geometric representation of meaning before text generation.

```
Input → [Transformer Layers] → Final Hidden State → [lm_head] → Logits → Tokens
                                        ↑
                              THE LOOM EXPOSES THIS
                              (The "boundary object" for conveyance measurement)
```

A loom weaves threads into fabric. The Loom weaves hidden states into observable patterns for WeaverCode agents.

## Quick Start

```bash
# Install with Poetry
poetry install

# Start server
poetry run loom --port 8080

# Generate with hidden states
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "return_hidden_states": true
  }'
```

## API Endpoints

### POST /generate

Generate text with hidden state extraction.

**Request:**
```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, how are you?",
  "max_tokens": 256,
  "temperature": 0.7,
  "return_hidden_states": true,
  "hidden_state_layers": [-1]
}
```

**Response:**
```json
{
  "text": "I'm doing well, thank you!",
  "token_count": 8,
  "hidden_states": {
    "-1": {
      "data": [0.123, -0.456, ...],
      "shape": [4096],
      "dtype": "float32"
    }
  },
  "metadata": {
    "inference_time_ms": 234,
    "tokens_per_second": 34.2
  }
}
```

### POST /embed

Extract embedding for text.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "text": "Hello world",
  "pooling": "last_token",
  "normalize": true
}
```

### GET /health

Health check with GPU info.

### GET /models

List loaded models with details (loader type, device, etc.).

### POST /models/load

Preload a model into memory with optional loader specification.

```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "device": "cuda:0",
  "loader": "sentence_transformers"
}
```

### GET /loaders

List available loaders and fallback order.

### GET /loaders/probe/{model_id}

Probe which loader would handle a model without loading it.

```bash
curl http://localhost:8080/loaders/probe/meta-llama/Llama-3.1-8B-Instruct
```

## Multi-Loader Architecture

The Loom supports three loaders with automatic detection:

| Loader | Coverage | Best For |
|--------|----------|----------|
| `transformers` | ~80% | Standard HuggingFace models (LLaMA, Mistral, Qwen, etc.) |
| `sentence_transformers` | ~15% | Embedding models (SBERT, BGE, E5, etc.) |
| `custom` | ~5% | Edge cases, research models, custom architectures |

**Auto-detection** selects the best loader based on model ID patterns. Override with the `loader` parameter.

## Configuration

Create `~/.config/loom/config.yaml`:

```yaml
server:
  http_port: 8080

gpu:
  devices: [0, 1]
  default_device: 0
  memory_fraction: 0.9

models:
  max_loaded: 3
  default_dtype: auto

loaders:
  fallback_order:
    - transformers
    - sentence_transformers
    - custom

hidden_states:
  default_layers: [-1]
  precision: float32

# Per-model overrides
model_overrides:
  my-custom-model:
    loader: custom
    device: cuda:1
    dtype: float16
```

## Research Context: Conveyance Measurement

The Loom enables validation of the **Conveyance Hypothesis** - measuring information transfer effectiveness between AI agents through geometric analysis of hidden states.

**Key metrics:**
- **D_eff** - Effective dimensionality (semantic richness)
- **β** - Collapse indicator (information loss)
- **Geometric Alignment** - Similarity between agent representations

See [conveyance-hypothesis-v4.1.md](../conveyance-hypothesis-v4.1.md) for the theoretical framework.

## Development

```bash
# Install with Poetry (includes dev dependencies)
poetry install

# Run tests
poetry run pytest

# Type checking
poetry run mypy src

# Format
poetry run ruff format src tests
poetry run ruff check src tests
```

## The Weaver Ecosystem

```
┌────────────────────┐         ┌──────────────────────────┐
│   WeaverCode (Go)  │◄───────►│      The Loom (Python)   │
│                    │   HTTP  │                          │
│  - Agent mgmt      │         │  - Model loading         │
│  - Orchestration   │         │  - Hidden state extract  │
│  - Analysis        │         │  - GPU management        │
└────────────────────┘         └──────────────────────────┘
```

## License

Apache-2.0
