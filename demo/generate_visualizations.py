#!/usr/bin/env python3
"""
Generate visualizations from existing demo output data.

Reads layer_metrics.json files and creates comparison plots.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(output_dir: Path) -> dict:
    """Load layer metrics from JSON file and reorganize by layer."""
    metrics_file = output_dir / "layer_metrics.json"
    with open(metrics_file) as f:
        raw_data = json.load(f)

    # Reorganize: {layer_idx: {prompt_idx: metrics}}
    by_layer = defaultdict(dict)
    for entry in raw_data:
        layer = entry["layer_idx"]
        prompt = entry["prompt_idx"]
        by_layer[layer][prompt] = {
            "d_eff": entry["d_eff"],
            "beta": entry["beta"],
            "mean_activation": entry["mean_activation"],
            "std_activation": entry["std_activation"],
            "l2_norm": entry["l2_norm"],
        }

    return dict(by_layer)


def generate_deff_comparison(model_a_metrics: dict, model_b_metrics: dict, output_path: Path):
    """Generate D_eff by layer comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model A
    layers_a = sorted(model_a_metrics.keys())
    deff_means_a = [np.mean([p["d_eff"] for p in model_a_metrics[l].values()]) for l in layers_a]
    deff_stds_a = [np.std([p["d_eff"] for p in model_a_metrics[l].values()]) for l in layers_a]

    axes[0].errorbar(layers_a, deff_means_a, yerr=deff_stds_a, fmt='o-', capsize=3, color='#2196F3')
    axes[0].fill_between(layers_a,
                         np.array(deff_means_a) - np.array(deff_stds_a),
                         np.array(deff_means_a) + np.array(deff_stds_a),
                         alpha=0.2, color='#2196F3')
    axes[0].set_xlabel('Layer Index (negative = from output)', fontsize=12)
    axes[0].set_ylabel('Effective Dimensionality (D_eff)', fontsize=12)
    axes[0].set_title('TinyLlama 1.1B (22 layers)\nD_eff by Layer', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Quality threshold (≥20)')
    axes[0].legend()

    # Model B
    layers_b = sorted(model_b_metrics.keys())
    deff_means_b = [np.mean([p["d_eff"] for p in model_b_metrics[l].values()]) for l in layers_b]
    deff_stds_b = [np.std([p["d_eff"] for p in model_b_metrics[l].values()]) for l in layers_b]

    axes[1].errorbar(layers_b, deff_means_b, yerr=deff_stds_b, fmt='s-', capsize=3, color='#4CAF50')
    axes[1].fill_between(layers_b,
                         np.array(deff_means_b) - np.array(deff_stds_b),
                         np.array(deff_means_b) + np.array(deff_stds_b),
                         alpha=0.2, color='#4CAF50')
    axes[1].set_xlabel('Layer Index (negative = from output)', fontsize=12)
    axes[1].set_ylabel('Effective Dimensionality (D_eff)', fontsize=12)
    axes[1].set_title('Qwen2.5 0.5B (24 layers)\nD_eff by Layer', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Quality threshold (≥20)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / "deff_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'deff_by_layer.png'}")


def generate_beta_comparison(model_a_metrics: dict, model_b_metrics: dict, output_path: Path):
    """Generate beta (collapse indicator) by layer comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Model A
    layers_a = sorted(model_a_metrics.keys())
    beta_means_a = [np.mean([p["beta"] for p in model_a_metrics[l].values()]) for l in layers_a]
    beta_stds_a = [np.std([p["beta"] for p in model_a_metrics[l].values()]) for l in layers_a]

    axes[0].errorbar(layers_a, beta_means_a, yerr=beta_stds_a, fmt='o-', capsize=3, color='#FF5722')
    axes[0].fill_between(layers_a,
                         np.array(beta_means_a) - np.array(beta_stds_a),
                         np.array(beta_means_a) + np.array(beta_stds_a),
                         alpha=0.2, color='#FF5722')
    axes[0].set_xlabel('Layer Index (negative = from output)', fontsize=12)
    axes[0].set_ylabel('Collapse Indicator (β)', fontsize=12)
    axes[0].set_title('TinyLlama 1.1B\nβ by Layer (lower = better)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Warning threshold (β < 2.0)')
    axes[0].legend()

    # Model B
    layers_b = sorted(model_b_metrics.keys())
    beta_means_b = [np.mean([p["beta"] for p in model_b_metrics[l].values()]) for l in layers_b]
    beta_stds_b = [np.std([p["beta"] for p in model_b_metrics[l].values()]) for l in layers_b]

    axes[1].errorbar(layers_b, beta_means_b, yerr=beta_stds_b, fmt='s-', capsize=3, color='#9C27B0')
    axes[1].fill_between(layers_b,
                         np.array(beta_means_b) - np.array(beta_stds_b),
                         np.array(beta_means_b) + np.array(beta_stds_b),
                         alpha=0.2, color='#9C27B0')
    axes[1].set_xlabel('Layer Index (negative = from output)', fontsize=12)
    axes[1].set_ylabel('Collapse Indicator (β)', fontsize=12)
    axes[1].set_title('Qwen2.5 0.5B\nβ by Layer (lower = better)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Warning threshold (β < 2.0)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / "beta_by_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'beta_by_layer.png'}")


def generate_activation_stats(model_a_metrics: dict, model_b_metrics: dict, output_path: Path):
    """Generate activation statistics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Model A - Mean activation
    layers_a = sorted(model_a_metrics.keys())
    mean_acts_a = [np.mean([p["mean_activation"] for p in model_a_metrics[l].values()]) for l in layers_a]
    std_acts_a = [np.mean([p["std_activation"] for p in model_a_metrics[l].values()]) for l in layers_a]

    axes[0, 0].plot(layers_a, mean_acts_a, 'o-', color='#2196F3', label='TinyLlama')
    axes[0, 0].set_xlabel('Layer Index')
    axes[0, 0].set_ylabel('Mean Activation')
    axes[0, 0].set_title('TinyLlama: Mean Activation by Layer')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(layers_a, std_acts_a, 'o-', color='#2196F3', label='TinyLlama')
    axes[0, 1].set_xlabel('Layer Index')
    axes[0, 1].set_ylabel('Std Activation')
    axes[0, 1].set_title('TinyLlama: Activation Std Dev by Layer')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Model B
    layers_b = sorted(model_b_metrics.keys())
    mean_acts_b = [np.mean([p["mean_activation"] for p in model_b_metrics[l].values()]) for l in layers_b]
    std_acts_b = [np.mean([p["std_activation"] for p in model_b_metrics[l].values()]) for l in layers_b]

    axes[1, 0].plot(layers_b, mean_acts_b, 's-', color='#4CAF50', label='Qwen2.5')
    axes[1, 0].set_xlabel('Layer Index')
    axes[1, 0].set_ylabel('Mean Activation')
    axes[1, 0].set_title('Qwen2.5: Mean Activation by Layer')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(layers_b, std_acts_b, 's-', color='#4CAF50', label='Qwen2.5')
    axes[1, 1].set_xlabel('Layer Index')
    axes[1, 1].set_ylabel('Std Activation')
    axes[1, 1].set_title('Qwen2.5: Activation Std Dev by Layer')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path / "activation_stats.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'activation_stats.png'}")


def generate_combined_overlay(model_a_metrics: dict, model_b_metrics: dict, output_path: Path):
    """Generate combined overlay plot for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # D_eff comparison (normalized by layer position)
    layers_a = sorted(model_a_metrics.keys())
    layers_b = sorted(model_b_metrics.keys())

    # Normalize to 0-1 range for comparison (layers are negative, so map -N...-1 to 0...1)
    norm_layers_a = (np.array(layers_a) - min(layers_a)) / (max(layers_a) - min(layers_a)) if len(layers_a) > 1 else [0.5]
    norm_layers_b = (np.array(layers_b) - min(layers_b)) / (max(layers_b) - min(layers_b)) if len(layers_b) > 1 else [0.5]

    deff_means_a = [np.mean([p["d_eff"] for p in model_a_metrics[l].values()]) for l in layers_a]
    deff_means_b = [np.mean([p["d_eff"] for p in model_b_metrics[l].values()]) for l in layers_b]

    axes[0].plot(norm_layers_a, deff_means_a, 'o-', color='#2196F3', label='TinyLlama 1.1B (2048d)', markersize=6)
    axes[0].plot(norm_layers_b, deff_means_b, 's-', color='#4CAF50', label='Qwen2.5 0.5B (896d)', markersize=6)
    axes[0].set_xlabel('Normalized Layer Position (0=first, 1=output)', fontsize=12)
    axes[0].set_ylabel('Effective Dimensionality (D_eff)', fontsize=12)
    axes[0].set_title('D_eff Comparison\n(Normalized by Network Depth)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Quality threshold')
    axes[0].legend()

    # Beta comparison
    beta_means_a = [np.mean([p["beta"] for p in model_a_metrics[l].values()]) for l in layers_a]
    beta_means_b = [np.mean([p["beta"] for p in model_b_metrics[l].values()]) for l in layers_b]

    axes[1].plot(norm_layers_a, beta_means_a, 'o-', color='#FF5722', label='TinyLlama 1.1B', markersize=6)
    axes[1].plot(norm_layers_b, beta_means_b, 's-', color='#9C27B0', label='Qwen2.5 0.5B', markersize=6)
    axes[1].set_xlabel('Normalized Layer Position (0=first, 1=output)', fontsize=12)
    axes[1].set_ylabel('Collapse Indicator (β)', fontsize=12)
    axes[1].set_title('β Comparison\n(Normalized by Network Depth)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Warning threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path / 'model_comparison.png'}")


def generate_summary_table(model_a_metrics: dict, model_b_metrics: dict,
                          model_a_name: str, model_b_name: str, output_path: Path):
    """Generate markdown summary table."""

    # Compute aggregates for Model A
    all_deff_a = []
    all_beta_a = []
    for layer_data in model_a_metrics.values():
        for prompt_data in layer_data.values():
            all_deff_a.append(prompt_data["d_eff"])
            all_beta_a.append(prompt_data["beta"])

    # Compute aggregates for Model B
    all_deff_b = []
    all_beta_b = []
    for layer_data in model_b_metrics.values():
        for prompt_data in layer_data.values():
            all_deff_b.append(prompt_data["d_eff"])
            all_beta_b.append(prompt_data["beta"])

    # Find peak D_eff layers
    layers_a = sorted(model_a_metrics.keys())
    layers_b = sorted(model_b_metrics.keys())

    deff_by_layer_a = {l: np.mean([p["d_eff"] for p in model_a_metrics[l].values()]) for l in layers_a}
    deff_by_layer_b = {l: np.mean([p["d_eff"] for p in model_b_metrics[l].values()]) for l in layers_b}

    peak_layer_a = max(deff_by_layer_a, key=deff_by_layer_a.get)
    peak_layer_b = max(deff_by_layer_b, key=deff_by_layer_b.get)

    summary = f"""# Geometric Analysis Summary

## Models Compared

| Model | Hidden Size | Layers | GPU |
|-------|-------------|--------|-----|
| {model_a_name} | 2048 | 22 | cuda:0 |
| {model_b_name} | 896 | 24 | cuda:1 |

## Key Metrics

| Metric | {model_a_name} | {model_b_name} |
|--------|--------------|---------------|
| Mean D_eff | {np.mean(all_deff_a):.2f} | {np.mean(all_deff_b):.2f} |
| D_eff Std | {np.std(all_deff_a):.2f} | {np.std(all_deff_b):.2f} |
| Min D_eff | {np.min(all_deff_a):.2f} | {np.min(all_deff_b):.2f} |
| Max D_eff | {np.max(all_deff_a):.2f} | {np.max(all_deff_b):.2f} |
| Peak D_eff Layer | {peak_layer_a} | {peak_layer_b} |
| Mean β | {np.mean(all_beta_a):.3f} | {np.mean(all_beta_b):.3f} |
| β Std | {np.std(all_beta_a):.3f} | {np.std(all_beta_b):.3f} |

## Interpretation

### D_eff (Effective Dimensionality)
- **Higher is better** (more semantic richness)
- Quality threshold: ≥20 for meaningful representations
- {model_a_name}: {"PASSES" if np.mean(all_deff_a) >= 20 else "BELOW"} threshold (mean={np.mean(all_deff_a):.2f})
- {model_b_name}: {"PASSES" if np.mean(all_deff_b) >= 20 else "BELOW"} threshold (mean={np.mean(all_deff_b):.2f})

### β (Collapse Indicator)
- **Lower is better** (less dimensional collapse)
- Warning threshold: < 2.0
- {model_a_name}: {"OK" if np.mean(all_beta_a) < 2.0 else "WARNING - above threshold"} (mean={np.mean(all_beta_a):.3f})
- {model_b_name}: {"OK" if np.mean(all_beta_b) < 2.0 else "WARNING - above threshold"} (mean={np.mean(all_beta_b):.3f})

## Visualizations

- `deff_by_layer.png` - D_eff across all layers for both models
- `beta_by_layer.png` - β (collapse indicator) across layers
- `activation_stats.png` - Activation mean and std by layer
- `model_comparison.png` - Normalized overlay comparison

## Test Prompts Used

1. "Explain the concept of gravity in simple terms."
2. "Write a haiku about artificial intelligence."
3. "What are the main differences between Python and JavaScript?"
4. "Describe a beautiful sunset over the ocean."
5. "How does photosynthesis work in plants?"

---
*Generated by The Loom Geometric Analysis Demo*
"""

    with open(output_path / "metrics_summary.md", "w") as f:
        f.write(summary)
    print(f"  Saved: {output_path / 'metrics_summary.md'}")


def main():
    """Generate all visualizations from existing demo data."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "outputs"

    print("\n" + "="*60)
    print("The Loom - Generating Visualizations")
    print("="*60 + "\n")

    # Load metrics
    print("Loading metrics...")
    model_a_metrics = load_metrics(output_dir / "model_a")
    model_b_metrics = load_metrics(output_dir / "model_b")
    print(f"  Model A: {len(model_a_metrics)} layers")
    print(f"  Model B: {len(model_b_metrics)} layers")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_deff_comparison(model_a_metrics, model_b_metrics, output_dir)
    generate_beta_comparison(model_a_metrics, model_b_metrics, output_dir)
    generate_activation_stats(model_a_metrics, model_b_metrics, output_dir)
    generate_combined_overlay(model_a_metrics, model_b_metrics, output_dir)

    # Generate summary
    print("\nGenerating summary...")
    generate_summary_table(
        model_a_metrics, model_b_metrics,
        "TinyLlama 1.1B", "Qwen2.5 0.5B",
        output_dir
    )

    print("\n" + "="*60)
    print("Visualization generation complete!")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("*.md")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
