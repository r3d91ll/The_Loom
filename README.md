# The Loom

**Hidden state extraction for transformer models** - the capability that production inference servers don't provide.

## The Problem

Production inference servers (vLLM, Ollama, TGI, llama.cpp) are optimized for throughput. They don't expose the final hidden state - the geometric representation of meaning *before* text generation.

```
Input → [Transformer Layers] → Hidden State → [lm_head] → Logits → Tokens
                                    ↑
                          THE LOOM EXPOSES THIS
```

If you've been waiting for [vLLM #6165](https://github.com/vllm-project/vllm/issues/6165), [#12249](https://github.com/vllm-project/vllm/issues/12249), or similar issues to be resolved - The Loom is for you.

## Quick Start (Docker)

```bash
# Pull and run (requires nvidia-docker)
docker run -d --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  tbucy/loom:latest

# Generate with hidden states
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "What is the meaning of life?",
    "max_tokens": 50,
    "return_hidden_states": true
  }'
```

## Model Compatibility

Tested and working:

| Model Family | Example | Hidden Size | Status |
|-------------|---------|-------------|--------|
| **Llama** | `meta-llama/Llama-3.1-8B-Instruct` | 4096 | ✅ |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3` | 4096 | ✅ |
| **Qwen** | `Qwen/Qwen2.5-7B-Instruct` | 3584 | ✅ |
| **TinyLlama** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 | ✅ |
| **Embedding Models** | `BAAI/bge-small-en-v1.5` | 384 | ✅ |

Most HuggingFace transformer models should work. The Loom auto-detects the appropriate loader.

## API Reference

### POST /generate

Generate text with hidden state extraction.

**Request:**
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.3",
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
  "text": "I'm doing well, thank you for asking!",
  "token_count": 8,
  "hidden_states": {
    "-1": {
      "data": [0.123, -0.456, ...],
      "shape": [1, 4096],
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

Extract embeddings with configurable pooling.

```json
{
  "model": "BAAI/bge-small-en-v1.5",
  "text": "Hello world",
  "pooling": "mean",
  "normalize": true
}
```

### POST /generate/stream

Server-sent events streaming with optional hidden states at completion.

### POST /generate/batch & /embed/batch

Batch processing for multiple prompts/texts.

### GET /health

Health check with GPU info and loaded models.

### GET /models

List currently loaded models with details.

### POST /models/load

Preload a model into memory.

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "device": "cuda:0",
  "quantization": "4bit"
}
```

### DELETE /models/{model_id}

Unload a model from memory.

### GET /loaders

List available loaders and their fallback order.

### GET /loaders/probe/{model_id}

Check which loader would handle a model without loading it.

## Installation

### Docker (Recommended)

```bash
# Basic usage
docker run -d --gpus all -p 8080:8080 tbucy/loom:latest

# With persistent model cache (recommended)
docker run -d --gpus all -p 8080:8080 \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  tbucy/loom:latest

# With specific GPU
docker run -d --gpus '"device=0"' -p 8080:8080 \
  -e CUDA_VISIBLE_DEVICES=0 \
  tbucy/loom:latest

# With HuggingFace token for gated models
docker run -d --gpus all -p 8080:8080 \
  -e HF_TOKEN=your_token_here \
  -v ~/.cache/huggingface:/app/.cache/huggingface \
  tbucy/loom:latest
```

### Docker Compose

```yaml
services:
  loom:
    image: tbucy/loom:latest
    ports:
      - "8080:8080"
    environment:
      - LOOM_MAX_MODELS=2
      - LOOM_DEFAULT_DEVICE=cuda:0
    volumes:
      - ~/.cache/huggingface:/app/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### From Source

```bash
git clone https://github.com/r3d91ll/TheLoom.git
cd TheLoom/the-loom
poetry install
poetry run loom --port 8080
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOOM_PORT` | 8080 | HTTP port |
| `LOOM_HOST` | 0.0.0.0 | Bind address |
| `LOOM_MAX_MODELS` | 3 | Max models in memory (LRU eviction) |
| `LOOM_DEFAULT_DEVICE` | cuda:0 | Default GPU device |
| `LOOM_MEMORY_FRACTION` | 0.9 | GPU memory fraction to use |
| `HF_TOKEN` | - | HuggingFace token for gated models |

Or use a config file at `~/.config/loom/config.yaml`:

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

hidden_states:
  default_layers: [-1]
  precision: float32
```

## Multi-Loader Architecture

The Loom automatically selects the best loader for each model:

| Loader | Coverage | Models |
|--------|----------|--------|
| `transformers` | ~80% | LLaMA, Mistral, Qwen, Phi, GPT-2, etc. |
| `sentence_transformers` | ~15% | BGE, E5, SBERT, GTE, etc. |
| `custom` | ~5% | Research models, custom architectures |

Override auto-detection with the `loader` parameter in requests.

## Quantization Support

Load models with reduced memory using quantization:

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "quantization": "4bit"
}
```

Supported modes: `4bit`, `8bit`, `gptq`, `awq`

Requires `bitsandbytes` for 4bit/8bit (included in Docker image).

## Known Limitations

**This is a research tool, not a production inference server.**

- **Throughput**: Optimized for hidden state access, not maximum tokens/second
- **Concurrency**: Single-request processing (no batched inference across requests)
- **Memory**: Models loaded fully into GPU memory (no tensor parallelism)
- **Chat Templates**: Raw prompts only - apply chat templates client-side if needed

For production inference without hidden states, use vLLM, TGI, or similar.

## Use Cases

- **Interpretability Research**: Analyze internal representations during generation
- **Embedding Extraction**: Get hidden states instead of just final embeddings
- **Model Comparison**: Compare geometric representations across models
- **Alignment Research**: Study how representations change with prompts
- **Probing Classifiers**: Train classifiers on intermediate representations

## Python Client

```python
import httpx

client = httpx.Client(base_url="http://localhost:8080")

# Generate with hidden states
response = client.post("/generate", json={
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "Explain quantum computing:",
    "max_tokens": 100,
    "return_hidden_states": True,
})

result = response.json()
print(f"Text: {result['text']}")
print(f"Hidden state shape: {result['hidden_states']['-1']['shape']}")
```

## Development

```bash
# Install dev dependencies
poetry install

# Run tests
poetry run pytest

# Type checking
poetry run mypy src

# Linting
poetry run ruff check src tests
poetry run ruff format src tests
```

## Contributing

Contributions welcome! Areas where help is needed:

- Additional model family support
- Performance optimizations
- Documentation improvements
- Bug reports and fixes

## License

Apache-2.0

## Acknowledgments

Built with:
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
