# PrimeNovo Extension

This note summarizes how `MemNovo` was applied to `PrimeNovo`, what worked,
and what did not.

## Architecture Positioning

`PrimeNovo` is not a fully non-Transformer model. Internally it still uses
transformer-style encoder/decoder blocks:

- [`external/primenovo/denovo/model.py`](../external/primenovo/denovo/model.py)
  - `self.encoder = SpectrumEncoder(...)`
  - `self.decoder = PeptideDecoder(...)`

However, it differs substantially from `Casanovo` and `InstaNovo` at inference
time:

- it uses a `CTC-style / non-autoregressive` decoding path
- the main prediction route is:
  - `encoder(spectra) -> decoder(None, precursors, memory, mask) -> CTC beam decoder`

This matters because the original online MemNovo mechanism was motivated by
progressive modality imbalance during autoregressive decoding.

## How MemNovo Was Applied

We ended up with two PrimeNovo adaptation lines:

1. `paper-final online MemNovo injection`
2. `training-free offline beam reranking`

The first keeps the method closest to the paper. The second is what produced
the strongest gains on PrimeNovo.

### Backend loading

PrimeNovo support is wired through:

- [`memnovo/backends.py`](../memnovo/backends.py)
  - `load_primenovo_backend(...)`

This loader:

- reads the vendored PrimeNovo config from
  [`external/primenovo/config.yaml`](../external/primenovo/config.yaml)
- restores the checkpoint expected at `weights/model_massive.ckpt`
- keeps PrimeNovo hyperparameters frozen unless explicitly overridden

### Online paper-final injection

The online path uses the same hook framework as the other backbones:

- [`memnovo/hooks.py`](../memnovo/hooks.py)
- [`memnovo/models.py`](../memnovo/models.py)

For PrimeNovo specifically:

- [`memnovo/models.py`](../memnovo/models.py)
  - `_predict_primenovo(...)`
- [`memnovo/hooks.py`](../memnovo/hooks.py)
  - wraps `encoder.forward(...)` to capture spectral memory
  - registers a last-layer decoder hook
  - computes retrieval attention over encoder memory
  - injects the pooled memory via:
    - `hidden + alpha * pooled`

The paper-faithful PrimeNovo config is:

- [`configs/memnovo_primenovo.yaml`](../configs/memnovo_primenovo.yaml)

Key settings:

- `residual_scale: 0.005`
- `apply_to_last_n_layers: 1`
- `confidence_threshold: null`
- `beam_size: 5`
- `batch_size: 64`
- `fp16: false`

This preserves the intended constraints:

- `training-free`
- `plug-and-play`
- inference-time only

### Offline beam reranking

The more effective line keeps the backbone frozen and only reranks beam
candidates after PrimeNovo decoding.

Core scripts:

- [`scripts/replay_rerank_probe.py`](../scripts/replay_rerank_probe.py)
- [`scripts/sweep_hybrid_rerank.py`](../scripts/sweep_hybrid_rerank.py)
- [`scripts/sweep_hybrid_fastmatch.py`](../scripts/sweep_hybrid_fastmatch.py)

Mechanically:

1. export PrimeNovo top-`k` beam candidates with `save_beams=true`
2. keep the original backbone frozen
3. compute a spectrum-aware score for beam candidates
4. rerank only uncertain / locally ambiguous cases

This is still:

- `training-free`
- `plug-and-play`

but it is no longer the same as the paper-final online residual injection.

## Main Experimental Findings

### 1. Full nine-species online paper-faithful MemNovo is essentially neutral

Using the paper-faithful online setting:

- `beam=5`
- `batch=64`
- `fp32`
- final-layer injection
- `alpha=0.005`
- no gating

the full nine-species PrimeNovo result was effectively flat:

- baseline species-average peptide recall: `0.6881073369553304`
- MemNovo species-average peptide recall: `0.688103673074468`
- delta: `-3.66e-06`

Interpretation:

- the paper-final online mechanism does not automatically transfer to PrimeNovo
- the likely reason is that PrimeNovo’s non-AR / CTC decoding dynamics differ
  from the failure mode targeted by the original online method

### 2. Fixed weighted 4898 subset shows real beam-rerank headroom

On the fixed weighted `4898` subset:

- top-1 exact: `3668 / 4898 = 74.89%`
- top-5 exact: `3907 / 4898 = 79.77%`
- top-5 mass-match: `3827 / 4898 = 78.13%`

Best hybrid rerank on this subset:

- baseline peptide recall: `0.748877`
- best peptide recall: `0.757044`
- relative gain: `+1.091%`
- matched peptides: `3668 -> 3708`

Best fixed-subset config:

- `alpha=0.75`
- `confidence_threshold=0.71`
- `spec_gap_threshold=0.0`
- `decoder_margin_threshold=0.2`
- `ion_mode=both`

### 3. Hard-species 5k confirmations are stronger

#### Mus-musculus 5k

- baseline peptide recall: `0.5548`
- best peptide recall: `0.5776`
- relative gain: `+4.110%`
- matched peptides: `2774 -> 2888`

Best config:

- `alpha=0.75`
- `spec_gap_threshold=0.1`
- `decoder_margin_threshold=0.7`
- `confidence_threshold=0.8`
- `ion_mode=both`

#### Candidatus-endoloripes 5k

- baseline peptide recall: `0.5264`
- best peptide recall: `0.5418`
- relative gain: `+2.926%`
- matched peptides: `2632 -> 2709`

Best config:

- `alpha=0.75`
- `spec_gap_threshold=0.0`
- `decoder_margin_threshold=0.7`
- `confidence_threshold=0.75`
- `ion_mode=both`

#### Apis-mellifera 5k

- baseline peptide recall: `0.6576`
- best peptide recall: `0.6712`
- relative gain: `+2.068%`
- matched peptides: `3288 -> 3356`

Best config:

- `alpha=0.75`
- `spec_gap_threshold=0.0`
- `decoder_margin_threshold=0.7`
- `confidence_threshold=0.71`
- `ion_mode=both`

### 4. Full-species confirmations show the gains are not tiny-subset artifacts

#### Mus-musculus full

- baseline peptide recall: `0.560745`
- best peptide recall: `0.579735`
- relative gain: `+3.386%`
- matched peptides: `14322 -> 14807`

Replayed config:

- `alpha=0.75`
- `spec_gap_threshold=0.1`
- `decoder_margin_threshold=0.7`
- `confidence_threshold=0.8`
- `ion_mode=both`

#### Candidatus-endoloripes full

- baseline peptide recall: `0.530927`
- best peptide recall: `0.544586`
- relative gain: `+2.573%`
- matched peptides: `43690 -> 44814`

Replayed config:

- `alpha=0.75`
- `spec_gap_threshold=0.0`
- `decoder_margin_threshold=0.7`
- `confidence_threshold=0.75`
- `ion_mode=both`

## Case-study Interpretation

The strongest PrimeNovo case studies indicate:

- there is real beam-ordering headroom on hard species
- the reranker can recover a meaningful fraction of beam-fixable errors
- the main remaining failure mode is still generation failure rather than beam
  ordering alone

So the most accurate summary is:

- `paper-final online MemNovo` is essentially neutral on PrimeNovo
- `training-free offline hybrid reranking` is clearly positive on PrimeNovo
- the best confirmed PrimeNovo gains are:
  - `Mus 5k: +4.110%`
  - `Mus full: +3.386%`
  - `Candidatus 5k: +2.926%`
  - `Candidatus full: +2.573%`
  - `Apis 5k: +2.068%`

## Recommended Positioning

PrimeNovo is useful as a third-backbone extension, but the effective adaptation
is architecture-dependent.

- If the claim is specifically about the paper-final online final-layer
  injection, PrimeNovo is mostly neutral.
- If the claim is about the broader training-free plug-and-play spectral
  rebalancing idea, PrimeNovo is clearly positive once beam candidates are
  exposed and reranked.
