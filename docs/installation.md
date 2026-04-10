# Installation Guide

## Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## Quick Installation

### Using pip

```bash
git clone https://github.com/SmallBlueWolf/MemNovo.git
cd MemNovo
pip install -e .
```

### Using conda

```bash
git clone https://github.com/SmallBlueWolf/MemNovo.git
cd MemNovo
conda env create -f environment.yml
conda activate memnovo
pip install -e .
```

## Installing Dependencies

### Base Model Dependencies

The repository currently vendors the source trees used by the wrappers under:

- `external/casanovo`
- `external/instanovo`
- `external/primenovo`

In the default local setup you do not need to install these separately before
running the MemNovo scripts.

### Downloading Model Checkpoints

```bash
# Download InstaNovo + Casanovo checkpoints
bash scripts/download_models.sh
```

Expected checkpoint locations:

- `weights/instanovo-v1.1.0.ckpt`
- `weights/casanovo_v5_0_0.ckpt`
- `weights/model_massive.ckpt` for PrimeNovo

At present the helper script downloads the first two automatically. The
PrimeNovo checkpoint should be placed manually at `weights/model_massive.ckpt`.

## Verifying Installation

```python
import memnovo
from memnovo import MemNovoManager

# Check version
print(memnovo.__version__)

# Verify layer imports
from memnovo.layers import CrossAttentionRetrieval
layer = CrossAttentionRetrieval(dim_model=512)
print("Installation successful!")
```

## GPU Support

MemNovo automatically detects and uses available GPUs. To verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues

1. **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system CUDA.

2. **Memory errors**: Reduce batch size in configuration files.

3. **Import errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

### Getting Help

Open an issue on GitHub with:
- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Full error traceback
