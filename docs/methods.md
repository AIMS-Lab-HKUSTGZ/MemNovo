# Technical Methods

This document describes the technical details of MemNovo.

## Problem: Sensitivity Imbalance

Transformer-based de novo sequencing models suffer from **sensitivity imbalance**: decoders over-rely on peptide linguistic priors while under-utilizing spectral evidence.

### Sensitivity Scaling Framework

We quantify this through controlled scaling experiments:

```
output_scaled = Decoder(α · spectrum_features, peptide_features)
```

Where α ∈ [0, 2] modulates feature magnitude while preserving semantics.

### Key Findings

| Model | Peptide Sensitivity | Spectrum Sensitivity | Ratio |
|-------|---------------------|---------------------|-------|
| Casanovo | High | Low | 15.4x |
| InstaNovo | Moderate | Moderate | 3.0x |

**Observation**: Peptide features dominate decoder attention, causing missed spectral evidence.

## Solution: MemNovo

MemNovo addresses sensitivity imbalance through **conservative spectral memory injection**.

### Core Mechanism

At each target decoder layer:

```
H' = H + α · pool(Attention(H, M))
```

Where:
- `H` = decoder hidden states [B, L, D]
- `M` = spectral memory features [B, N, D]
- `α` = residual scale (default: 0.005)
- `pool` = mean pooling over spectral positions

### Cross-Attention Retrieval

Unlike standard cross-attention, we use **projection-free attention**:

```python
# Standard (with projections)
Q = W_q @ H
K = W_k @ M
V = W_v @ M
attention = softmax(Q @ K.T / sqrt(d)) @ V

# MemNovo (projection-free)
attention = softmax(H @ M.T / sqrt(d)) @ M
```

Benefits:
1. No additional parameters to train
2. Preserves original feature semantics
3. Training-free enhancement

### Implementation via Hooks

MemNovo uses PyTorch forward hooks for non-invasive integration:

```python
def enhancement_hook(module, input, output):
    hidden = output[0]  # Decoder hidden states
    enhanced = hidden + alpha * retrieve_spectral(hidden, memory)
    return (enhanced,) + output[1:]
```

This allows enhancement without modifying model architecture.

## Default Configuration

```yaml
residual_scale: 0.005
target_layers: [-1]  # Last layer only
use_gating: false
```

Rationale: Minimal intervention at the final decision layer provides best stability.

## Theoretical Analysis

### Why Conservative Injection Works

1. **Preserves learned representations**: Small α ensures original model behavior is largely maintained.

2. **Targeted intervention**: Last-layer enhancement affects final logits without disrupting intermediate processing.

3. **Additive refinement**: Spectral information adds missing evidence rather than overwriting.

### Sensitivity Ratio After Enhancement

With MemNovo on InstaNovo:
- Original ratio: 3.0x
- Enhanced ratio: ~1.5x

The more balanced ratio indicates better utilization of both modalities.

## Comparison with Alternatives

| Approach | Training Required | Parameters Added | Improvement |
|----------|------------------|------------------|-------------|
| Retraining | Yes | 0 | Variable |
| Adapter layers | Yes | Many | Moderate |
| MemNovo | No | 0 | +4-5% |

MemNovo achieves improvements without any training or additional parameters.
