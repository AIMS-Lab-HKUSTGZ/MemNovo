# Quick Start Guide

This guide demonstrates how to use MemNovo for enhanced de novo peptide sequencing.

## Basic Usage

### 1. Load a Pre-trained Model with MemNovo

```python
from memnovo.models import MemNovoModel

# Load InstaNovo with MemNovo enhancement
model = MemNovoModel(
    model_name="instanovo",
    checkpoint_path="../weights/instanovo-v1.1.0.ckpt",
    config_path="configs/memnovo_instanovo.yaml",
    device="cuda",
)
```

### 2. Run Inference

```python
# Single-file inference
predictions = model.predict("path/to/spectra.mgf")
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
    --input path/to/spectra.mgf \
    --output predictions.jsonl \
    --config configs/memnovo_instanovo.yaml

# Compare with baseline
python scripts/run_inference.py \
    --input path/to/spectra.mgf \
    --output baseline.jsonl \
    --config configs/baseline_instanovo.yaml
```

## Next Steps

- See [API Reference](api.md) for detailed documentation
- See [Experiments](experiments.md) for reproducing paper results
- See [Methods](methods.md) for technical details
- See [PrimeNovo](primenovo.md) for the third-backbone extension
