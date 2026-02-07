# External Dependencies

This directory contains git submodules for base model implementations.

## Submodules

### Casanovo

De novo mass spectrometry peptide sequencing with a transformer model.

- Repository: https://github.com/Noble-Lab/casanovo
- Paper: Yilmaz et al., Nature Communications 2024

```bash
# Initialize submodule
git submodule update --init external/casanovo
```

### InstaNovo

De novo peptide sequencing with transformer architecture.

- Repository: https://github.com/instadeepai/instanovo
- Paper: Eloff et al., Nature Machine Intelligence 2025

```bash
# Initialize submodule
git submodule update --init external/instanovo
```

## Manual Installation

If you prefer not to use submodules, install base models directly:

```bash
# Via pip
pip install casanovo
pip install instanovo

# Or from source
git clone https://github.com/Noble-Lab/casanovo.git
cd casanovo && pip install -e .

git clone https://github.com/instadeepai/instanovo.git
cd instanovo && pip install -e .
```
