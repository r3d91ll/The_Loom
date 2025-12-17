# The Loom - Hidden State Extraction Server
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Builder - Install dependencies
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml poetry.lock ./

# Export only production dependencies
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# =============================================================================
# Stage 2: Runtime - CUDA-enabled image
# =============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash loom
WORKDIR /app

# Copy requirements from builder
COPY --from=builder /app/requirements.txt .

# Install Python dependencies
# Using --extra-index-url for PyTorch CUDA wheels
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements.txt

# Copy application code
COPY --chown=loom:loom src/ ./src/
COPY --chown=loom:loom config/ ./config/
COPY --chown=loom:loom pyproject.toml README.md ./

# Install the package
RUN pip install --no-cache-dir .

# =============================================================================
# Configuration
# =============================================================================

# Environment variables for configuration
ENV LOOM_HOST=0.0.0.0
ENV LOOM_PORT=8080
ENV LOOM_LOG_LEVEL=INFO
ENV LOOM_MAX_MODELS=3
ENV LOOM_DEFAULT_DEVICE=cuda:0
ENV LOOM_MEMORY_FRACTION=0.9

# HuggingFace cache directory (mount as volume for persistence)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Create cache directory
RUN mkdir -p /app/.cache/huggingface && chown -R loom:loom /app/.cache

# Switch to non-root user
USER loom

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${LOOM_PORT}/health || exit 1

# Default command (uses LOOM_PORT env var)
CMD ["sh", "-c", "python -m src.server --host 0.0.0.0 --port ${LOOM_PORT}"]
