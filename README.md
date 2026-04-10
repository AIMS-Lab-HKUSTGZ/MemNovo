# MemNovo

**A Plug-and-Play Framework for Mitigating Sensitivity Imbalance in De Novo Peptide Sequencing**

## Installation

### Prerequisites

- Python 3.12 or higher
- CUDA 11.8+ (for GPU acceleration)

### Step 1: Clone Repository

```bash
git clone https://github.com/SmallBlueWolf/MemNovo.git
cd MemNovo
```

This repository vendors the base-model source trees used by the current
MemNovo wrappers:

- `external/casanovo`
- `external/instanovo`
- `external/primenovo`

### Step 2: Create Conda Environment

```bash
conda env create -f environment.yml
conda activate memnovo
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Step 3: Install MemNovo Package

```bash
pip install -e .
```

### Step 4: Download Pre-trained Models

```bash
bash scripts/download_models.sh
```

This downloads:
- InstaNovo v1.1.0 checkpoint
- Casanovo v5.0.0 checkpoint

PrimeNovo uses a separate checkpoint (`model_massive.ckpt`) that is not
auto-downloaded by the current script. See
[docs/installation.md](docs/installation.md) for the expected path.

### Step 5: Download Datasets (Optional)

For reproducing paper experiments:

```bash
bash scripts/download_data.sh
```

This downloads the Nine Species benchmark dataset (~10 GB).

## Quick Start

### Basic Inference

```python
from memnovo import MemNovoModel
from memnovo.utils import load_spectra

# Load model with MemNovo enhancement
model = MemNovoModel.from_pretrained(
    "instanovo",
    config="configs/memnovo.yaml"
)

# Load spectra
spectra = load_spectra("examples/sample_spectra.mgf")

# Run inference
predictions = model.predict(spectra)

for spectrum_id, peptide_seq in predictions.items():
    print(f"{spectrum_id}: {peptide_seq}")
```

### Command-Line Inference

```bash
python scripts/run_inference.py \
    --config configs/memnovo.yaml \
    --input examples/sample_spectra.mgf \
    --output results/predictions.csv
```

### Running with Baseline (No MemNovo)

```bash
python scripts/run_inference.py \
    --config configs/baseline_instanovo.yaml \
    --input examples/sample_spectra.mgf \
    --output results/baseline_predictions.csv
```

## Reproducing Paper Results

### Sensitivity Scaling Experiments (Figure 2)

Quantify the sensitivity imbalance in baseline models:

```bash
bash scripts/run_sensitivity.sh instanovo
bash scripts/run_sensitivity.sh casanovo
```

This uses the local workspace datasets referenced in the paper appendix:

- `../dataset/hc_pt/test.parquet` for InstaNovo
- `../dataset/novobench/test.parquet` for Casanovo

The scaling range is model-specific:

- Casanovo: `[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]`
- InstaNovo: `[0.990, ..., 1.010]`

Results are saved under `results/sensitivity/<model>/`.

Visualize results:

```bash
python sensitivity_scaling/visualize.py \
    --input results/sensitivity/instanovo/instanovo_sensitivity_results.json \
    --output assets/instanovo_sensitivity_curves.pdf
```

### Nine Species Benchmark (Table 2)

Run comprehensive evaluation on all 9 species:

```bash
bash scripts/run_nine_species.sh
```

This evaluates both baseline and MemNovo on:
- Bacillus-subtilis
- Saccharomyces-cerevisiae
- Methanosarcina-mazei
- Apis-mellifera
- Solanum-lycopersicum
- Vigna-mungo
- H.-sapiens
- Mus-musculus
- Candidatus-endoloripes

Results saved to `results/nine_species/comparison.csv`.

### Case Studies

Analyze specific examples where MemNovo corrects baseline errors:

```bash
python case_studies/analyze_cases.py \
    --baseline results/baseline/ \
    --memnovo results/memnovo/ \
    --output results/case_studies/
```

## Configuration

### MemNovo Configuration (Recommended)

The default configuration:

```yaml
model:
  name: instanovo
  checkpoint: ../weights/instanovo-v1.1.0.ckpt

memnovo:
  enabled: true
  residual_scale: 0.005        # 0.5% residual weight
  apply_to_last_n_layers: 1    # Final layer only
  confidence_threshold: null   # Always-on injection
```

### Key Parameters

- `residual_scale`: Strength of spectral memory injection (α). Range: [0.001, 0.020]
  - 0.005 (default): Best balance of improvement and stability
  - Lower: More conservative, safer but smaller gains
  - Higher: Stronger correction but risk of disruption

- `apply_to_last_n_layers`: Inject into the last `k` decoder layers
  - 1 (default): Final layer only, most stable
  - 2: Last 2 layers, stronger intervention

- `confidence_threshold`: Optional confidence gate
  - null (default): Always apply
  - numeric value: Only inject on low-confidence states

## Project Structure

```
MemNovo/
├── memnovo/                  # Core implementation
│   ├── hooks.py             # PyTorch hook management
│   ├── layers.py            # Cross-attention layers
│   ├── manager.py           # MemNovo manager
│   ├── models.py            # Model interfaces
│   └── utils.py             # Utility functions
│
├── sensitivity_scaling/      # Feature scaling experiments
│   ├── experiment.py        # Experiment runner
│   ├── analyze.py           # Analysis functions
│   └── visualize.py         # Result visualization
│
├── evaluation/              # Evaluation framework
│   ├── data_handler.py      # Data loading
│   ├── evaluator.py         # Unified evaluator
│   └── metrics.py           # Metric computations
│
├── configs/                 # YAML configurations
│   ├── memnovo.yaml         # Default MemNovo config
│   └── baseline_*.yaml      # Baseline configs
│
├── scripts/                 # Executable scripts
│   ├── run_inference.py     # Main inference
│   ├── run_nine_species.sh  # Nine species evaluation
│   └── download_*.sh        # Download helpers
│
├── external/                # Vendored base-model source trees
│   ├── casanovo/
│   ├── instanovo/
│   └── primenovo/
│
└── docs/                    # Documentation
    ├── installation.md      # Detailed installation
    ├── quickstart.md        # Tutorial
    ├── api.md               # API reference
    ├── methods.md           # Technical methods
    ├── experiments.md       # Reproduction guide
    └── primenovo.md         # PrimeNovo extension notes
```

## API Reference

### Core Classes

#### MemNovoModel

Main interface for inference with MemNovo enhancement.

```python
from memnovo import MemNovoModel

model = MemNovoModel.from_pretrained(
    model_name="instanovo",  # or "casanovo"
    config="configs/memnovo.yaml",
    device="cuda"
)

predictions = model.predict(
    spectra,                 # List of spectra or path to MGF file
    beam_size=5,             # Beam search width
    batch_size=64            # Batch size for inference
)
```

#### SensitivityScaler

Diagnostic tool for measuring modal imbalance.

```python
from sensitivity_scaling import SensitivityScaler

scaler = SensitivityScaler(model, dataset)

results = scaler.run_experiment(
    modality="spectrum",     # or "peptide" or "both"
    scale_factors=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    metrics=["aa_precision", "peptide_recall"]
)

sensitivity_ratio = scaler.compute_sensitivity_ratio()
print(f"Sensitivity ratio: {sensitivity_ratio:.2f}x")
```

## Acknowledgments

This work builds upon:
- [InstaNovo](https://github.com/instadeepai/InstaNovo) - Fast transformer-based de novo sequencing
- [Casanovo](https://github.com/Noble-Lab/casanovo) - Pioneering transformer encoder-decoder for peptides

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
