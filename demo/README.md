# The Loom - Geometric Analysis Demo

This demo showcases The Loom's core capability: extracting and analyzing hidden states from transformer models for geometric/conveyance research.

## What This Demo Does

1. **Loads two models on separate GPUs** - Tests multi-GPU capability
2. **Runs identical prompts through both** - Enables fair comparison
3. **Extracts hidden states from ALL layers** - Full geometric profile
4. **Computes geometric metrics** - D_eff, beta, activation statistics
5. **Generates comparison visualizations** - Publication-ready plots
6. **Saves everything** - For further research or documentation

## Prerequisites

```bash
# The Loom must be installed
cd the-loom
pip install -e ".[dev]"

# Optional: matplotlib and scikit-learn for visualizations
pip install matplotlib scikit-learn
```

## Usage

### Step 1: Start The Loom Server

```bash
# In one terminal
poetry run loom --port 8080
```

### Step 2: Run the Demo

```bash
# In another terminal
poetry run python demo/run_geometric_analysis.py

# For single-GPU systems
poetry run python demo/run_geometric_analysis.py --single-gpu

# Custom server URL
poetry run python demo/run_geometric_analysis.py --server http://localhost:9000
```

## Output Structure

```
demo/outputs/
├── model_a/
│   ├── responses.json          # Generated text and metadata
│   ├── layer_metrics.json      # D_eff, beta per layer
│   └── hidden_states_prompt_*.npz  # Raw hidden states
├── model_b/
│   └── ... (same structure)
└── analysis/
    ├── geometric_comparison.png    # Visualization
    └── metrics_summary.md          # Results table
```

## Models Used

By default, the demo uses small models that fit on most GPUs:

| Model | GPU | Size |
|-------|-----|------|
| TinyLlama 1.1B Chat | cuda:0 | ~2.2GB VRAM |
| Qwen2.5 0.5B Instruct | cuda:1 | ~1GB VRAM |

> **Note on Response Quality:** Chat models (TinyLlama, Qwen) may produce short or empty text responses when given raw prompts without proper chat template formatting. This is expected behavior - the demo uses raw prompts to enable all-layers hidden state extraction. The geometric analysis remains valid regardless of text output length since we analyze the hidden state representations, not the generated text.

## Understanding the Metrics

### D_eff (Effective Dimensionality)
Measures how many dimensions are actively used in the representation.
- Higher values = richer semantic encoding
- Typical range: 20-200 for final layers

### β (Beta / Collapse Indicator)
Measures dimensional collapse (hidden_size / D_eff).
- Target: < 2.0 (lower is better)
- High β suggests over-compression

### Layer Progression
- Early layers: Lower D_eff (basic features)
- Middle layers: Peak D_eff (complex representations)
- Final layers: May decrease slightly (task-specific compression)

## Sample Visualization

![Geometric Comparison](outputs/analysis/geometric_comparison.png)

## Customization

Edit `run_geometric_analysis.py` to change:

```python
# Test prompts
TEST_PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
]

# Models to compare
MODEL_CONFIGS = {
    "model_a": {
        "model_id": "your/model-id",
        "device": "cuda:0",
        "display_name": "Your Model",
    },
    ...
}
```

## Using the Raw Data

Load the hidden states for your own analysis:

```python
import numpy as np

# Load hidden states for a specific prompt
data = np.load("demo/outputs/model_a/hidden_states_prompt_0.npz")

# Access specific layers (negative indices)
final_layer = data["-1"]  # Shape: [hidden_size]
layer_10 = data["-10"]

# List all layers
print(data.files)  # ['-1', '-2', '-3', ...]
```

## Troubleshooting

### "Could not connect to server"
Make sure The Loom is running:
```bash
poetry run loom --port 8080
```

### "CUDA out of memory"
Use single-GPU mode or smaller models:
```bash
poetry run python demo/run_geometric_analysis.py --single-gpu
```

### "matplotlib not installed"
The demo will still run and save data, but skip visualizations:
```bash
pip install matplotlib scikit-learn
```
