# API Reference

## Core Module: `memnovo`

### MemNovoManager

Main class for managing MemNovo enhancement.

```python
class MemNovoManager:
    """
    Manages MemNovo hook registration and inference enhancement.

    Parameters
    ----------
    config : dict
        Configuration dictionary with:
        - residual_scale : float, default=0.005
            Scaling factor (alpha) for residual injection.
        - apply_to_last_n_layers : int, default=1
            Number of decoder layers to enhance.
        - confidence_threshold : float or None, default=None
            Confidence threshold for gating (None = always apply).

    Examples
    --------
    >>> config = {'residual_scale': 0.005, 'apply_to_last_n_layers': 1}
    >>> manager = MemNovoManager(config)
    >>> manager.register(model.decoder)
    >>> output = model(spectra)
    >>> manager.unregister()
    """

    def register(self, model: nn.Module) -> None:
        """Register forward hooks on decoder layers."""

    def unregister(self) -> None:
        """Remove all registered hooks."""

    def set_spectral_memory(self, memory: torch.Tensor) -> None:
        """Set spectral memory for current batch."""

    def reset(self) -> None:
        """Clear stored spectral memory."""
```

### CrossAttentionRetrieval

Core attention layer for spectral feature retrieval.

```python
class CrossAttentionRetrieval(nn.Module):
    """
    Cross-attention layer for retrieving spectral information.

    Uses scaled dot-product attention without learned projections.

    Parameters
    ----------
    dim_model : int
        Model dimension (must match decoder hidden size).
    residual_scale : float, default=0.005
        Scaling factor for residual connection.
    use_softmax : bool, default=True
        Whether to apply softmax to attention weights.

    Examples
    --------
    >>> layer = CrossAttentionRetrieval(dim_model=512)
    >>> enhanced = layer(hidden, spectral_memory)
    """

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Apply cross-attention retrieval.

        Parameters
        ----------
        hidden : Tensor[B, L, D]
            Decoder hidden states.
        memory : Tensor[B, N, D]
            Spectral memory features.
        mask : Tensor[B, N], optional
            Attention mask for spectral positions.

        Returns
        -------
        Tensor[B, L, D]
            Enhanced hidden states.
        """
```

### SpectrumEnhancer

Alternative enhancement layer with multiple modes.

```python
class SpectrumEnhancer(nn.Module):
    """
    Spectrum-based hidden state enhancer.

    Parameters
    ----------
    dim_model : int
        Model dimension.
    mode : str, default='additive'
        Enhancement mode: 'additive', 'gated', or 'residual'.
    scale : float, default=0.005
        Enhancement scale factor.
    """
```

## Sensitivity Scaling Module

### SensitivityScaler

Diagnostic tool for measuring modality sensitivity.

```python
class SensitivityScaler:
    """
    Applies scaling factors to measure decoder sensitivity.

    Parameters
    ----------
    model : nn.Module
        Base de novo sequencing model.
    modality : str
        Which modality to scale: 'spectrum' or 'peptide'.

    Examples
    --------
    >>> scaler = SensitivityScaler(model, modality='spectrum')
    >>> results = scaler.run_experiment(
    ...     dataloader,
    ...     scale_factors=[0.0, 0.5, 1.0, 1.5, 2.0]
    ... )
    """

    def run_experiment(
        self,
        dataloader: DataLoader,
        scale_factors: list[float]
    ) -> dict:
        """Run sensitivity scaling experiment."""
```

### Analysis Functions

```python
def compute_sensitivity_ratio(
    spectrum_results: dict,
    peptide_results: dict,
    metric: str = 'aa_precision'
) -> float:
    """
    Compute sensitivity ratio (peptide/spectrum sensitivity).

    Returns
    -------
    float
        Ratio indicating relative sensitivity. >1 means peptide-dominant.
    """

def compute_elasticity(
    results: dict,
    metric: str = 'aa_precision'
) -> float:
    """
    Compute elasticity (rate of change) at scale=1.0.
    """
```

## Evaluation Module

### Evaluator

```python
class Evaluator:
    """
    Evaluation utilities for peptide predictions.

    Examples
    --------
    >>> evaluator = Evaluator()
    >>> metrics = evaluator.evaluate(predictions, targets)
    """

    def evaluate(
        self,
        predictions: list[str],
        targets: list[str]
    ) -> dict:
        """
        Compute all metrics.

        Returns
        -------
        dict
            Contains: aa_precision, aa_recall, pep_precision, pep_recall
        """

    def evaluate_by_length(
        self,
        predictions: list[str],
        targets: list[str],
        length_bins: list[tuple] = [(7, 10), (11, 15), (16, 20), (21, 30)]
    ) -> dict:
        """Evaluate metrics stratified by sequence length."""

    def evaluate_by_species(
        self,
        predictions: list[str],
        targets: list[str],
        species: list[str]
    ) -> dict:
        """Evaluate metrics stratified by species."""
```

### Metric Functions

```python
def aa_precision(predicted: str, target: str) -> float:
    """Amino acid level precision."""

def aa_recall(predicted: str, target: str) -> float:
    """Amino acid level recall."""

def peptide_precision(predicted: str, target: str) -> float:
    """Peptide level precision (exact match)."""

def normalize_sequence(seq: str) -> str:
    """Normalize sequence (uppercase, I/L equivalence, remove mods)."""

def aggregate_metrics(
    predictions: list[str],
    targets: list[str]
) -> dict:
    """Compute aggregated metrics over multiple samples."""
```

## Utility Functions

```python
def load_config(path: str) -> dict:
    """Load YAML configuration file."""

def load_spectra(path: str) -> list[dict]:
    """Load spectra from MGF file."""

def save_predictions(
    predictions: list[str],
    path: str,
    format: str = 'csv'
) -> None:
    """Save predictions to file."""
```
