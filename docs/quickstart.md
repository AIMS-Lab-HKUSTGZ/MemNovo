# Quick Start Guide

This guide demonstrates how to use MemNovo for enhanced de novo peptide sequencing.

## Basic Usage

### 1. Load a Pre-trained Model with MemNovo

```python
from memnovo import MemNovoModel

# Load InstaNovo with MemNovo enhancement
model = MemNovoModel.from_pretrained(
    base_model="instanovo",
    config="configs/memnovo.yaml"
)
```

### 2. Run Inference

```python
# Single spectrum inference
predictions = model.predict("path/to/spectra.mgf")

# With custom batch size
predictions = model.predict(
    "path/to/spectra.mgf",
    batch_size=32
)
```

### 3. Evaluate Results

```python
from evaluation import Evaluator

evaluator = Evaluator()
metrics = evaluator.evaluate(predictions, targets)

print(f"AA Precision: {metrics['aa_precision']:.4f}")
print(f"Peptide Precision: {metrics['pep_precision']:.4f}")
```

## Using the MemNovoManager Directly

For more control, use the manager interface:

```python
from memnovo import MemNovoManager
import torch

# Initialize manager with custom parameters
manager = MemNovoManager({
    'residual_scale': 0.005,  # Alpha parameter
    'apply_to_last_n_layers': 1,  # Which decoder layers to enhance
    'confidence_threshold': None  # No gating
})

# Register hooks on your model
manager.register(model.decoder)

# Run inference as normal
with torch.no_grad():
    output = model(spectra, precursors)

# Remove hooks when done
manager.unregister()
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| residual_scale | 0.005 | Injection strength (α) |
| apply_to_last_n_layers | 1 | Number of decoder layers |
| confidence_threshold | None | Gating threshold (None = always apply) |

## Command Line Interface

```bash
# Run inference with default settings
python scripts/run_inference.py \
    --spectra data/spectra.mgf \
    --output results/predictions.csv \
    --config configs/memnovo.yaml

# Compare with baseline
python scripts/run_inference.py \
    --spectra data/spectra.mgf \
    --output results/baseline.csv \
    --config configs/baseline_instanovo.yaml
```

## Next Steps

- See [API Reference](api.md) for detailed documentation
- See [Experiments](experiments.md) for reproducing paper results
- See [Methods](methods.md) for technical details
