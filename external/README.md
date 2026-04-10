# External Dependencies

This directory contains the vendored base-model source trees used by the
current MemNovo wrappers.

## Included backbones

### Casanovo

- upstream repository: https://github.com/Noble-Lab/casanovo
- role here:
  - official encoder/decoder implementation for the Casanovo backend
  - reused directly by the MemNovo wrappers

### InstaNovo

- upstream repository: https://github.com/instadeepai/InstaNovo
- role here:
  - official transformer implementation for the InstaNovo backend
  - reused directly by the MemNovo wrappers and official-predictor experiments

### PrimeNovo

- role here:
  - vendored source tree for the PrimeNovo backbone used in the third-backbone
    MemNovo extension experiments

## Notes

- These directories are committed as regular source folders in this repository.
- They are not expected to be initialized via `git submodule update`.
- MemNovo itself remains training-free and plug-and-play with respect to the
  pretrained checkpoints; vendoring these code snapshots is purely for
  reproducible local execution.
