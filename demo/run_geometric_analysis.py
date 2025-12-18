#!/usr/bin/env python3
"""
The Loom - Geometric Analysis Demo

This demo showcases The Loom's core capability: extracting and analyzing
hidden states from transformer models for geometric/conveyance research.

What this demo does:
1. Loads two models on separate GPUs (cuda:0 and cuda:1)
2. Runs identical prompts through both models
3. Extracts hidden states from ALL layers
4. Computes geometric metrics (D_eff, beta) for each layer
5. Generates comparison visualizations
6. Saves everything for documentation/research

Usage:
    # Start The Loom server first:
    poetry run loom --port 8080

    # Then run this demo:
    poetry run python demo/run_geometric_analysis.py

    # Or with custom server URL:
    poetry run python demo/run_geometric_analysis.py --server http://localhost:8080

Output:
    demo/outputs/
    ├── model_a/
    │   ├── responses.json
    │   └── hidden_states_prompt_0.npz
    ├── model_b/
    │   └── ...
    └── analysis/
        ├── d_eff_by_layer.png
        ├── beta_by_layer.png
        ├── layer_geometry_pca.png
        └── metrics_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualizations will be skipped.")
    print("Install with: pip install matplotlib")

# Note: sklearn PCA could be used for more principled D_eff computation,
# but the current cumulative energy approach works well for this demo.


# =============================================================================
# Configuration
# =============================================================================

# Test prompts - diverse to see different geometric signatures
TEST_PROMPTS = [
    "Explain the concept of gravity in simple terms.",
    "Write a haiku about artificial intelligence.",
    "What are the main differences between Python and JavaScript?",
    "Describe a beautiful sunset over the ocean.",
    "How does photosynthesis work in plants?",
]

# Models to compare - using small models that fit on most GPUs
MODEL_CONFIGS = {
    "model_a": {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "device": "cuda:0",
        "display_name": "TinyLlama 1.1B",
    },
    "model_b": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "device": "cuda:1",
        "display_name": "Qwen2.5 0.5B",
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerMetrics:
    """Metrics for a single layer's hidden state."""
    layer_idx: int
    d_eff: float  # Effective dimensionality
    beta: float   # Collapse indicator
    mean_activation: float
    std_activation: float
    l2_norm: float


@dataclass
class ModelResults:
    """Results from running prompts through a model."""
    model_id: str
    display_name: str
    device: str
    num_layers: int
    hidden_size: int
    responses: list[dict[str, Any]] = field(default_factory=list)
    layer_metrics: list[list[LayerMetrics]] = field(default_factory=list)  # Per prompt, per layer
    load_time_seconds: float = 0.0


# =============================================================================
# Geometric Analysis Functions
# =============================================================================

def compute_d_eff(hidden_state: np.ndarray, variance_threshold: float = 0.90) -> float:
    """Compute effective dimensionality via cumulative energy distribution.

    D_eff represents the number of dimensions needed to capture
    variance_threshold of the total squared magnitude (energy) in the hidden state.

    This is a simplified approximation that doesn't require PCA - it sorts
    the squared values and finds how many are needed to reach the threshold.

    Args:
        hidden_state: 1D array of hidden state values
        variance_threshold: Fraction of energy to capture (default 90%)

    Returns:
        Effective dimensionality (float, clamped to [1, len(hidden_state)])
    """
    # Use squared values as proxy for "energy" distribution
    squared = hidden_state ** 2
    sorted_sq = np.sort(squared)[::-1]
    cumsum = np.cumsum(sorted_sq)
    total = cumsum[-1]

    if total == 0:
        return 1.0

    normalized = cumsum / total
    d_eff = np.searchsorted(normalized, variance_threshold) + 1

    return float(min(d_eff, len(hidden_state)))


def compute_beta(hidden_state: np.ndarray, reference_dim: int | None = None) -> float:
    """Compute collapse indicator (beta).

    Beta measures how much the representation has "collapsed" from its
    theoretical capacity. Lower is better (less collapse).

    β = reference_dim / D_eff

    Args:
        hidden_state: 1D array of hidden state values
        reference_dim: Expected dimensionality (default: len(hidden_state))

    Returns:
        Collapse indicator (>=1.0, lower is better)
    """
    if reference_dim is None:
        reference_dim = len(hidden_state)

    d_eff = compute_d_eff(hidden_state)

    if d_eff == 0:
        return float('inf')

    return reference_dim / d_eff


def analyze_hidden_state(hidden_state: np.ndarray, layer_idx: int) -> LayerMetrics:
    """Compute all metrics for a hidden state."""
    return LayerMetrics(
        layer_idx=layer_idx,
        d_eff=compute_d_eff(hidden_state),
        beta=compute_beta(hidden_state),
        mean_activation=float(np.mean(hidden_state)),
        std_activation=float(np.std(hidden_state)),
        l2_norm=float(np.linalg.norm(hidden_state)),
    )


# =============================================================================
# Server Communication
# =============================================================================

class LoomClient:
    """Simple client for The Loom server."""

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def health_check(self) -> dict[str, Any]:
        """Check server health."""
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def load_model(
        self,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "auto",
    ) -> dict[str, Any]:
        """Load a model onto specified device."""
        response = self.client.post(
            f"{self.base_url}/models/load",
            json={
                "model": model_id,
                "device": device,
                "dtype": dtype,
            },
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        return_hidden_states: bool = True,
        hidden_state_layers: list[int] | str = "all",
        use_chat_template: bool = False,  # False for all-layers extraction
    ) -> dict[str, Any]:
        """Generate text with hidden state extraction.

        Args:
            model_id: Model identifier
            prompt: Input prompt (will be wrapped as user message if use_chat_template=True)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_hidden_states: Whether to return hidden states
            hidden_state_layers: Which layers to extract ("all" or list of indices)
            use_chat_template: If True, use /v1/chat/completions with chat templates
        """
        if use_chat_template:
            # Use chat completions endpoint for proper chat template handling
            response = self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "return_hidden_states": return_hidden_states,
                },
            )
            response.raise_for_status()
            result = response.json()
            # Convert chat completion response to match generate format
            return {
                "text": result.get("text", ""),
                "token_count": result.get("usage", {}).get("completion_tokens", 0),
                "hidden_states": self._convert_chat_hidden_states(result, hidden_state_layers),
                "metadata": result.get("metadata", {}),
            }
        else:
            # Use raw generate endpoint
            response = self.client.post(
                f"{self.base_url}/generate",
                json={
                    "model": model_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "return_hidden_states": return_hidden_states,
                    "hidden_state_layers": hidden_state_layers,
                    "hidden_state_format": "list",
                },
            )
            response.raise_for_status()
            return response.json()

    def _convert_chat_hidden_states(
        self,
        result: dict[str, Any],
        hidden_state_layers: list[int] | str,
    ) -> dict[str, Any]:
        """Convert chat completion hidden state format to generate format."""
        hidden_state = result.get("hidden_state")
        if not hidden_state:
            return {}

        # Chat completions only return final layer, format it like generate
        layer_key = str(hidden_state.get("layer", -1))
        return {
            layer_key: {
                "data": hidden_state.get("final", []),
                "shape": hidden_state.get("shape", []),
                "dtype": hidden_state.get("dtype", "float32"),
            }
        }

    def list_models(self) -> dict[str, Any]:
        """List loaded models."""
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def unload_model(self, model_id: str) -> dict[str, Any]:
        """Unload a model."""
        # URL encode the model ID (replace / with --)
        encoded_id = model_id.replace("/", "--")
        response = self.client.delete(f"{self.base_url}/models/{encoded_id}")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the client."""
        self.client.close()


# =============================================================================
# Demo Runner
# =============================================================================

def run_model_analysis(
    client: LoomClient,
    config: dict[str, str],
    prompts: list[str],
    output_dir: Path,
) -> ModelResults:
    """Run analysis for a single model."""
    model_id = config["model_id"]
    device = config["device"]
    display_name = config["display_name"]

    print(f"\n{'='*60}")
    print(f"Analyzing: {display_name}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load model
    print(f"\n[1/3] Loading model on {device}...")
    start_time = time.time()
    load_result = client.load_model(model_id, device=device)
    load_time = time.time() - start_time

    num_layers = load_result["num_layers"]
    hidden_size = load_result["hidden_size"]

    print(f"      Loaded in {load_time:.1f}s")
    print(f"      Layers: {num_layers}, Hidden size: {hidden_size}")

    results = ModelResults(
        model_id=model_id,
        display_name=display_name,
        device=device,
        num_layers=num_layers,
        hidden_size=hidden_size,
        load_time_seconds=load_time,
    )

    # Run prompts
    print(f"\n[2/3] Running {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"      Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")

        gen_result = client.generate(
            model_id=model_id,
            prompt=prompt,
            max_tokens=64,
            temperature=0.7,
            return_hidden_states=True,
            hidden_state_layers="all",
        )

        # Save response
        response_data = {
            "prompt": prompt,
            "response": gen_result["text"],
            "token_count": gen_result["token_count"],
            "metadata": gen_result["metadata"],
        }
        results.responses.append(response_data)

        # Analyze hidden states for each layer
        prompt_layer_metrics = []
        hidden_states_data = {}

        if gen_result.get("hidden_states"):
            for layer_key, layer_data in gen_result["hidden_states"].items():
                layer_idx = int(layer_key)
                hidden_state = np.array(layer_data["data"], dtype=np.float32)

                # Flatten if needed (remove batch dimension)
                if len(hidden_state.shape) > 1:
                    hidden_state = hidden_state.flatten()

                # Compute metrics
                metrics = analyze_hidden_state(hidden_state, layer_idx)
                prompt_layer_metrics.append(metrics)

                # Store for saving
                hidden_states_data[layer_key] = hidden_state

        results.layer_metrics.append(prompt_layer_metrics)

        # Save hidden states as compressed numpy
        npz_path = output_dir / f"hidden_states_prompt_{i}.npz"
        np.savez_compressed(npz_path, **hidden_states_data)

        print(f"         Response: {gen_result['text'][:60]}...")
        print(f"         Tokens: {gen_result['token_count']}, Layers extracted: {len(hidden_states_data)}")

    # Save responses
    print(f"\n[3/3] Saving results to {output_dir}")
    responses_path = output_dir / "responses.json"
    with open(responses_path, "w") as f:
        json.dump(results.responses, f, indent=2)

    # Save metrics
    metrics_data = []
    for prompt_idx, prompt_metrics in enumerate(results.layer_metrics):
        for layer_metrics in prompt_metrics:
            metrics_data.append({
                "prompt_idx": prompt_idx,
                "layer_idx": layer_metrics.layer_idx,
                "d_eff": layer_metrics.d_eff,
                "beta": layer_metrics.beta,
                "mean_activation": layer_metrics.mean_activation,
                "std_activation": layer_metrics.std_activation,
                "l2_norm": layer_metrics.l2_norm,
            })

    metrics_path = output_dir / "layer_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"      Saved responses.json and layer_metrics.json")

    return results


def generate_visualizations(
    results_a: ModelResults,
    results_b: ModelResults,
    output_dir: Path,
) -> None:
    """Generate comparison visualizations."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not installed)")
        return

    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")

    # Aggregate metrics by layer (average across prompts)
    def aggregate_by_layer(results: ModelResults) -> dict[int, dict[str, float]]:
        layer_agg: dict[int, dict[str, list[float]]] = {}

        for prompt_metrics in results.layer_metrics:
            for lm in prompt_metrics:
                if lm.layer_idx not in layer_agg:
                    layer_agg[lm.layer_idx] = {"d_eff": [], "beta": [], "l2_norm": []}
                layer_agg[lm.layer_idx]["d_eff"].append(lm.d_eff)
                layer_agg[lm.layer_idx]["beta"].append(lm.beta)
                layer_agg[lm.layer_idx]["l2_norm"].append(lm.l2_norm)

        return {
            layer: {
                "d_eff": np.mean(vals["d_eff"]),
                "beta": np.mean(vals["beta"]),
                "l2_norm": np.mean(vals["l2_norm"]),
                "d_eff_std": np.std(vals["d_eff"]),
                "beta_std": np.std(vals["beta"]),
            }
            for layer, vals in layer_agg.items()
        }

    agg_a = aggregate_by_layer(results_a)
    agg_b = aggregate_by_layer(results_b)

    layers_a = sorted(agg_a.keys())
    layers_b = sorted(agg_b.keys())

    # Normalize layer indices to [0, 1] for comparison (models have different depths)
    def normalize_layers(layers: list[int]) -> np.ndarray:
        if len(layers) == 0:
            return np.array([])
        min_l, max_l = min(layers), max(layers)
        if max_l == min_l:
            return np.array([0.5])
        return np.array([(l - min_l) / (max_l - min_l) for l in layers])

    norm_layers_a = normalize_layers(layers_a)
    norm_layers_b = normalize_layers(layers_b)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("The Loom - Geometric Analysis Comparison", fontsize=14, fontweight='bold')

    # 1. D_eff by Layer
    ax1 = axes[0, 0]
    d_eff_a = [agg_a[l]["d_eff"] for l in layers_a]
    d_eff_b = [agg_b[l]["d_eff"] for l in layers_b]

    ax1.plot(norm_layers_a, d_eff_a, 'b-o', label=results_a.display_name, markersize=4)
    ax1.plot(norm_layers_b, d_eff_b, 'r-s', label=results_b.display_name, markersize=4)
    ax1.set_xlabel("Normalized Layer Depth (0=early, 1=final)")
    ax1.set_ylabel("Effective Dimensionality (D_eff)")
    ax1.set_title("D_eff Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Beta by Layer
    ax2 = axes[0, 1]
    beta_a = [agg_a[l]["beta"] for l in layers_a]
    beta_b = [agg_b[l]["beta"] for l in layers_b]

    ax2.plot(norm_layers_a, beta_a, 'b-o', label=results_a.display_name, markersize=4)
    ax2.plot(norm_layers_b, beta_b, 'r-s', label=results_b.display_name, markersize=4)
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Target threshold (β<2.0)')
    ax2.set_xlabel("Normalized Layer Depth")
    ax2.set_ylabel("Collapse Indicator (β)")
    ax2.set_title("β Across Layers (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. L2 Norm by Layer
    ax3 = axes[1, 0]
    l2_a = [agg_a[l]["l2_norm"] for l in layers_a]
    l2_b = [agg_b[l]["l2_norm"] for l in layers_b]

    ax3.plot(norm_layers_a, l2_a, 'b-o', label=results_a.display_name, markersize=4)
    ax3.plot(norm_layers_b, l2_b, 'r-s', label=results_b.display_name, markersize=4)
    ax3.set_xlabel("Normalized Layer Depth")
    ax3.set_ylabel("L2 Norm")
    ax3.set_title("Activation Magnitude Across Layers")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary Statistics Bar Chart
    ax4 = axes[1, 1]

    # Average D_eff and Beta for final layers (last 25%)
    final_layers_a = [l for l in layers_a if l >= min(layers_a) + 0.75 * (max(layers_a) - min(layers_a))]
    final_layers_b = [l for l in layers_b if l >= min(layers_b) + 0.75 * (max(layers_b) - min(layers_b))]

    if not final_layers_a:
        final_layers_a = layers_a[-1:]
    if not final_layers_b:
        final_layers_b = layers_b[-1:]

    avg_d_eff_a = np.mean([agg_a[l]["d_eff"] for l in final_layers_a])
    avg_d_eff_b = np.mean([agg_b[l]["d_eff"] for l in final_layers_b])
    avg_beta_a = np.mean([agg_a[l]["beta"] for l in final_layers_a])
    avg_beta_b = np.mean([agg_b[l]["beta"] for l in final_layers_b])

    x = np.arange(2)
    width = 0.35

    bars1 = ax4.bar(x - width/2, [avg_d_eff_a, avg_beta_a], width, label=results_a.display_name, color='blue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, [avg_d_eff_b, avg_beta_b], width, label=results_b.display_name, color='red', alpha=0.7)

    ax4.set_ylabel('Value')
    ax4.set_title('Final Layer Metrics (avg of last 25%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['D_eff', 'β'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / "geometric_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {fig_path}")

    plt.close()

    # Generate markdown summary
    generate_markdown_summary(results_a, results_b, agg_a, agg_b, output_dir)


def generate_markdown_summary(
    results_a: ModelResults,
    results_b: ModelResults,
    agg_a: dict[int, dict[str, float]],
    agg_b: dict[int, dict[str, float]],
    output_dir: Path,
) -> None:
    """Generate markdown summary of results."""

    layers_a = sorted(agg_a.keys())
    layers_b = sorted(agg_b.keys())

    # Calculate summary statistics
    final_layer_a = layers_a[-1] if layers_a else 0
    final_layer_b = layers_b[-1] if layers_b else 0

    final_d_eff_a = agg_a.get(final_layer_a, {}).get("d_eff", 0)
    final_d_eff_b = agg_b.get(final_layer_b, {}).get("d_eff", 0)
    final_beta_a = agg_a.get(final_layer_a, {}).get("beta", 0)
    final_beta_b = agg_b.get(final_layer_b, {}).get("beta", 0)

    avg_d_eff_a = np.mean([agg_a[l]["d_eff"] for l in layers_a]) if layers_a else 0
    avg_d_eff_b = np.mean([agg_b[l]["d_eff"] for l in layers_b]) if layers_b else 0
    avg_beta_a = np.mean([agg_a[l]["beta"] for l in layers_a]) if layers_a else 0
    avg_beta_b = np.mean([agg_b[l]["beta"] for l in layers_b]) if layers_b else 0

    md_content = f"""# The Loom - Geometric Analysis Results

## Overview

This analysis compares hidden state geometry between two transformer models,
demonstrating The Loom's capability to extract and analyze internal representations.

## Models Analyzed

| Property | {results_a.display_name} | {results_b.display_name} |
|----------|--------------------------|--------------------------|
| Model ID | `{results_a.model_id}` | `{results_b.model_id}` |
| Device | {results_a.device} | {results_b.device} |
| Layers | {results_a.num_layers} | {results_b.num_layers} |
| Hidden Size | {results_a.hidden_size} | {results_b.hidden_size} |
| Load Time | {results_a.load_time_seconds:.1f}s | {results_b.load_time_seconds:.1f}s |

## Geometric Metrics

### Final Layer Metrics

| Metric | {results_a.display_name} | {results_b.display_name} | Description |
|--------|--------------------------|--------------------------|-------------|
| D_eff | {final_d_eff_a:.1f} | {final_d_eff_b:.1f} | Effective dimensionality |
| β | {final_beta_a:.2f} | {final_beta_b:.2f} | Collapse indicator (lower is better) |

### Average Across All Layers

| Metric | {results_a.display_name} | {results_b.display_name} |
|--------|--------------------------|--------------------------|
| Avg D_eff | {avg_d_eff_a:.1f} | {avg_d_eff_b:.1f} |
| Avg β | {avg_beta_a:.2f} | {avg_beta_b:.2f} |

## Visualization

![Geometric Comparison](geometric_comparison.png)

## Interpretation

- **D_eff (Effective Dimensionality)**: Measures how many dimensions are actively used
  in the representation. Higher values indicate richer semantic encoding.

- **β (Collapse Indicator)**: Measures dimensional collapse. Values below 2.0 are
  generally healthy. High β suggests the model is compressing information too
  aggressively.

## Test Prompts Used

{chr(10).join(f'{i+1}. "{p}"' for i, p in enumerate(TEST_PROMPTS))}

## Files Generated

- `model_a/responses.json` - Generated text and metadata
- `model_a/hidden_states_prompt_*.npz` - Raw hidden states (numpy compressed)
- `model_a/layer_metrics.json` - Per-layer geometric metrics
- `model_b/` - Same structure for second model
- `analysis/geometric_comparison.png` - Visualization
- `analysis/metrics_summary.md` - This file

---

*Generated by The Loom - Hidden State Extraction for AI Research*
"""

    summary_path = output_dir / "metrics_summary.md"
    with open(summary_path, "w") as f:
        f.write(md_content)

    print(f"   Saved: {summary_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="The Loom - Geometric Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8080",
        help="The Loom server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--output-dir",
        default="demo/outputs",
        help="Output directory for results (default: demo/outputs)",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Run both models on cuda:0 (for single-GPU systems)",
    )

    args = parser.parse_args()

    # Setup output directories
    output_base = Path(args.output_dir)
    output_a = output_base / "model_a"
    output_b = output_base / "model_b"
    output_analysis = output_base / "analysis"

    output_a.mkdir(parents=True, exist_ok=True)
    output_b.mkdir(parents=True, exist_ok=True)
    output_analysis.mkdir(parents=True, exist_ok=True)

    # Adjust config for single GPU mode
    if args.single_gpu:
        MODEL_CONFIGS["model_b"]["device"] = "cuda:0"
        print("Running in single-GPU mode (both models on cuda:0)")

    print("\n" + "="*60)
    print("The Loom - Geometric Analysis Demo")
    print("="*60)
    print(f"\nServer: {args.server}")
    print(f"Output: {output_base}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Models: {len(MODEL_CONFIGS)}")

    # Create client
    client = LoomClient(args.server)

    try:
        # Health check
        print("\n[Health Check]")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   GPU Info: {health['gpu_info']}")

        # Run analysis for both models
        results_a = run_model_analysis(
            client,
            MODEL_CONFIGS["model_a"],
            TEST_PROMPTS,
            output_a,
        )

        results_b = run_model_analysis(
            client,
            MODEL_CONFIGS["model_b"],
            TEST_PROMPTS,
            output_b,
        )

        # Generate visualizations
        generate_visualizations(results_a, results_b, output_analysis)

        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print(f"\nResults saved to: {output_base}")
        print("\nFiles generated:")
        print(f"   {output_a}/responses.json")
        print(f"   {output_a}/layer_metrics.json")
        print(f"   {output_a}/hidden_states_prompt_*.npz")
        print(f"   {output_b}/ (same structure)")
        print(f"   {output_analysis}/geometric_comparison.png")
        print(f"   {output_analysis}/metrics_summary.md")

    except httpx.ConnectError:
        print(f"\nError: Could not connect to The Loom server at {args.server}")
        print("Make sure the server is running:")
        print("   poetry run loom --port 8080")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
