# Reproducing Experiments

This guide covers reproducing the experiments from the paper.

## Dataset: Nine Species Benchmark

The Nine Species dataset contains ~2.8 million spectra from 9 taxonomically diverse organisms.

### Downloading the Dataset

```bash
# Follow instructions in the script
bash scripts/download_data.sh
```

Or manually download from MassIVE:
- Dataset ID: MSV000081142
- URL: https://massive.ucsd.edu/

### Dataset Structure

```
data/
├── nine_species/
│   ├── bacillus/
│   ├── clambacteria/
│   ├── honeybee/
│   ├── human/
│   ├── mmazei/
│   ├── mouse/
│   ├── rice/
│   ├── tomato/
│   └── yeast/
```

## Experiment 1: Sensitivity Scaling

Measure decoder sensitivity to each modality.

### Running the Experiment

```bash
python -m sensitivity_scaling.experiment \
    --model instanovo \
    --checkpoint models/instanovo.ckpt \
    --data data/nine_species/human/ \
    --output results/sensitivity/ \
    --scale-factors 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0
```

### Generating Plots

```python
from sensitivity_scaling.visualize import plot_sensitivity_curves

# Load results
spectrum_results = load_json("results/sensitivity/spectrum_scaling.json")
peptide_results = load_json("results/sensitivity/peptide_scaling.json")

# Generate comparison plot
plot_sensitivity_curves(
    spectrum_results,
    peptide_results,
    output_path="figures/sensitivity_comparison.pdf"
)
```

### Expected Results

| Model | Spectrum Sensitivity | Peptide Sensitivity | Ratio |
|-------|---------------------|---------------------|-------|
| Casanovo | 0.032 | 0.493 | 15.4x |
| InstaNovo | 0.089 | 0.267 | 3.0x |

## Experiment 2: MemNovo Evaluation

Compare baseline vs MemNovo-enhanced models.

### Running Full Benchmark

```bash
# Full nine species evaluation
bash scripts/run_nine_species.sh
```

Or run individual species:

```bash
# Single species evaluation
python scripts/run_inference.py \
    --spectra data/nine_species/human/*.mgf \
    --output results/human_memnovo.csv \
    --config configs/memnovo.yaml

# Baseline comparison
python scripts/run_inference.py \
    --spectra data/nine_species/human/*.mgf \
    --output results/human_baseline.csv \
    --config configs/baseline_instanovo.yaml
```

### Comparing Results

```python
from evaluation import Evaluator

evaluator = Evaluator()

# Load predictions
memnovo_preds = load_predictions("results/human_memnovo.csv")
baseline_preds = load_predictions("results/human_baseline.csv")
targets = load_targets("data/nine_species/human/")

# Evaluate
memnovo_metrics = evaluator.evaluate(memnovo_preds, targets)
baseline_metrics = evaluator.evaluate(baseline_preds, targets)

print("Baseline AA Precision:", baseline_metrics['aa_precision'])
print("MemNovo AA Precision:", memnovo_metrics['aa_precision'])
print("Improvement:", memnovo_metrics['aa_precision'] - baseline_metrics['aa_precision'])
```

### Expected Results (InstaNovo + MemNovo)

| Species | Baseline | MemNovo | Delta |
|---------|----------|---------|-------|
| Human | 0.421 | 0.457 | +3.6% |
| Mouse | 0.398 | 0.432 | +3.4% |
| Yeast | 0.445 | 0.489 | +4.4% |
| Rice | 0.367 | 0.401 | +3.4% |
| Honeybee | 0.412 | 0.451 | +3.9% |
| Tomato | 0.389 | 0.423 | +3.4% |
| Bacillus | 0.456 | 0.498 | +4.2% |
| C. Bacteria | 0.423 | 0.461 | +3.8% |
| M. Mazei | 0.401 | 0.438 | +3.7% |
| **Average** | **0.412** | **0.450** | **+3.8%** |

## Experiment 3: Ablation Studies

Test different MemNovo configurations:

```bash
# Default configuration
python scripts/run_inference.py \
    --config configs/memnovo.yaml \
    --output results/memnovo.csv

# Custom parameters
python scripts/run_inference.py \
    --config configs/memnovo.yaml \
    --output results/custom.csv
```

## Experiment 4: Length-Stratified Analysis

Evaluate performance across peptide length bins.

```python
from evaluation import Evaluator

evaluator = Evaluator()

metrics_by_length = evaluator.evaluate_by_length(
    predictions,
    targets,
    length_bins=[(7, 10), (11, 15), (16, 20), (21, 30)]
)

for bin_name, metrics in metrics_by_length.items():
    print(f"{bin_name}: AA Precision = {metrics['aa_precision']:.4f}")
```

## Computational Requirements

| Experiment | GPU Memory | Time (9 species) |
|------------|-----------|------------------|
| Baseline Inference | 8 GB | ~4 hours |
| MemNovo Inference | 8 GB | ~4.5 hours |
| Sensitivity Scaling | 16 GB | ~8 hours |

All experiments tested on NVIDIA A100 40GB.
