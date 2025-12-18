# Contributing to The Loom

Thank you for your interest in contributing to The Loom! This project aims to provide hidden state extraction for transformer models, enabling ML research that production inference servers don't support.

## Ways to Contribute

### Reporting Issues

- **Bug reports**: Include steps to reproduce, expected vs actual behavior, and your environment (Python version, GPU, model used)
- **Feature requests**: Describe the use case and how it would benefit research workflows
- **Documentation improvements**: Typos, unclear explanations, missing examples

### Code Contributions

1. **Fork the repository** and create a feature branch
2. **Set up development environment**:
   ```bash
   cd the-loom
   pip install -e ".[dev]"
   ```
3. **Make your changes** following the code style below
4. **Run tests and checks**:
   ```bash
   pytest                    # All tests pass
   mypy src --pretty         # No type errors
   ruff format src tests     # Format code
   ruff check src tests      # No lint errors
   ```
5. **Submit a pull request** with a clear description of changes

### Code Style

- **Type hints**: All functions should have type annotations
- **Docstrings**: Use Google-style docstrings for public functions
- **Testing**: Add tests for new functionality
- **Commit messages**: Clear, concise descriptions of changes

### Areas Where Help is Needed

Check the [Roadmap](README.md#roadmap) section for planned features:

- [ ] Explicit device_map control for multi-GPU
- [ ] Tensor parallelism for large models
- [ ] Chat template support
- [ ] Concurrent request handling
- [ ] OpenAI-compatible API
- [ ] Attention weight extraction

### Adding New Model Loaders

If you need to support a model architecture that doesn't work with existing loaders:

1. Create a new loader in `src/loaders/`
2. Implement the `ModelLoader` abstract base class from `src/loaders/base.py`
3. Register patterns in `LoaderRegistry` (`src/loaders/registry.py`)
4. Add tests in `tests/test_loaders.py`
5. Document the supported models in README.md

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for integration tests)
- Poetry or pip

### Running Tests

```bash
# Unit tests (fast, no GPU required)
pytest -m "not integration"

# Integration tests (requires GPU, loads real models)
pytest tests/test_integration.py -v -s

# All tests
pytest
```

### Project Structure

```
the-loom/
├── src/
│   ├── loaders/          # Model loading (transformers, sentence_transformers, etc.)
│   ├── transport/        # HTTP API (FastAPI)
│   ├── extraction/       # Hidden state extraction and analysis
│   ├── utils/            # GPU management, serialization, metrics
│   └── server.py         # CLI entry point
├── tests/
│   ├── test_*.py         # Unit tests
│   └── test_integration.py  # GPU integration tests
└── examples/
    └── outputs/          # Sample API responses
```

## Questions?

Open an issue on GitHub - we're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
