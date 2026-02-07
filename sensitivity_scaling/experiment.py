"""
Sensitivity Scaling Experiment Runner

Implements the Sensitivity Scaling Framework for diagnosing modal
imbalance in transformer-based de novo peptide sequencing models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SensitivityScaler:
    """
    Sensitivity Scaling Framework for modal imbalance diagnosis.

    Applies scaling factors to spectrum and peptide features independently
    to measure model sensitivity to each modality.

    Args:
        model: De novo sequencing model (Casanovo or InstaNovo)
        device: Computation device
    """

    DEFAULT_SCALE_FACTORS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Detect model type
        self.model_type = self._detect_model_type()

        # Registered hooks for scaling
        self.hooks = []
        self.current_scale = 1.0
        self.scaling_modality = None

        logger.info(f"SensitivityScaler initialized for {self.model_type}")

    def _detect_model_type(self) -> str:
        """Detect whether model is Casanovo or InstaNovo."""
        if hasattr(self.model, 'residue_set'):
            return 'instanovo'
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'transformer_decoder'):
            return 'casanovo'
        else:
            return 'unknown'

    def run_experiment(
        self,
        dataloader,
        scale_factors: Optional[List[float]] = None,
        modality: str = 'spectrum',
        evaluator = None,
    ) -> Dict[str, Any]:
        """
        Run sensitivity scaling experiment.

        Args:
            dataloader: Data loader with spectra
            scale_factors: List of scaling factors (default: [0.1, 0.2, ..., 10.0])
            modality: Which modality to scale ('spectrum' or 'peptide')
            evaluator: Evaluator for computing metrics

        Returns:
            Dictionary with results for each scale factor
        """
        if scale_factors is None:
            scale_factors = self.DEFAULT_SCALE_FACTORS

        logger.info(f"Running sensitivity experiment for {modality} modality")
        logger.info(f"Scale factors: {scale_factors}")

        results = []

        for scale in tqdm(scale_factors, desc=f"Scaling {modality}"):
            try:
                # Apply scaling
                self._register_scaling_hooks(modality, scale)

                # Run inference
                predictions = self._run_inference(dataloader)

                # Evaluate
                if evaluator:
                    metrics = evaluator.evaluate(predictions)
                else:
                    metrics = self._compute_basic_metrics(predictions)

                metrics['scale_factor'] = scale
                metrics['modality'] = modality
                results.append(metrics)

                logger.info(
                    f"Scale={scale:.1f}: AA_prec={metrics.get('aa_precision', 0):.4f}, "
                    f"Pep_prec={metrics.get('pep_precision', 0):.4f}"
                )

            except Exception as e:
                logger.error(f"Error at scale {scale}: {e}")
                results.append({
                    'scale_factor': scale,
                    'modality': modality,
                    'error': str(e),
                })

            finally:
                self._remove_scaling_hooks()

        return {
            'modality': modality,
            'results': results,
            'scale_factors': scale_factors,
        }

    def _register_scaling_hooks(self, modality: str, scale: float) -> None:
        """Register hooks for feature scaling."""
        self.current_scale = scale
        self.scaling_modality = modality

        if modality == 'spectrum':
            self._register_spectrum_scaling_hooks(scale)
        elif modality == 'peptide':
            self._register_peptide_scaling_hooks(scale)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _register_spectrum_scaling_hooks(self, scale: float) -> None:
        """Register hooks to scale spectrum (encoder) features."""
        def scale_hook(module, input_tuple, output):
            if isinstance(output, torch.Tensor):
                return output * scale
            elif isinstance(output, tuple):
                return tuple(
                    o * scale if isinstance(o, torch.Tensor) else o
                    for o in output
                )
            return output

        # Hook on encoder output
        if hasattr(self.model, 'encoder'):
            hook = self.model.encoder.register_forward_hook(scale_hook)
            self.hooks.append(hook)

    def _register_peptide_scaling_hooks(self, scale: float) -> None:
        """Register hooks to scale peptide (decoder input) features."""
        def scale_hook(module, input_tuple, output):
            if isinstance(output, torch.Tensor):
                return output * scale
            return output

        # Hook on embedding layer
        if hasattr(self.model, 'aa_embed'):
            hook = self.model.aa_embed.register_forward_hook(scale_hook)
            self.hooks.append(hook)
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'aa_embed'):
            hook = self.model.decoder.aa_embed.register_forward_hook(scale_hook)
            self.hooks.append(hook)

    def _remove_scaling_hooks(self) -> None:
        """Remove all registered scaling hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.current_scale = 1.0
        self.scaling_modality = None

    def _run_inference(self, dataloader) -> List[Dict[str, Any]]:
        """Run inference on dataloader."""
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = self._to_device(batch)

                # Run model
                try:
                    output = self.model(batch)
                    predictions.extend(self._process_output(output))
                except Exception as e:
                    logger.debug(f"Batch inference error: {e}")
                    continue

        return predictions

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _process_output(self, output) -> List[Dict[str, Any]]:
        """Process model output to predictions."""
        # This is model-specific - implement based on actual output format
        return []

    def _compute_basic_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute basic evaluation metrics."""
        if not predictions:
            return {'aa_precision': 0.0, 'pep_precision': 0.0}

        # Placeholder - actual implementation depends on prediction format
        return {
            'aa_precision': 0.0,
            'pep_precision': 0.0,
            'n_samples': len(predictions),
        }

    def compute_sensitivity(
        self,
        results: Dict[str, Any],
        baseline_scale: float = 1.0,
    ) -> float:
        """
        Compute sensitivity metric for a modality.

        Sensitivity = E[|Perf(α) - Perf(1.0)| / Perf(1.0)]

        Args:
            results: Results from run_experiment
            baseline_scale: Reference scale factor (default: 1.0)

        Returns:
            Sensitivity value
        """
        result_list = results.get('results', [])

        # Find baseline performance
        baseline_perf = None
        for r in result_list:
            if r.get('scale_factor') == baseline_scale:
                baseline_perf = r.get('aa_precision', 0)
                break

        if baseline_perf is None or baseline_perf == 0:
            return 0.0

        # Compute sensitivity
        deviations = []
        for r in result_list:
            if r.get('scale_factor') != baseline_scale:
                perf = r.get('aa_precision', 0)
                deviation = abs(perf - baseline_perf) / baseline_perf
                deviations.append(deviation)

        return np.mean(deviations) if deviations else 0.0


def run_sensitivity_experiment(
    model_path: str,
    data_path: str,
    output_dir: str,
    model_type: str = 'instanovo',
    scale_factors: Optional[List[float]] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Run complete sensitivity scaling experiment.

    Args:
        model_path: Path to model checkpoint
        data_path: Path to test data (MGF file)
        output_dir: Directory to save results
        model_type: 'instanovo' or 'casanovo'
        scale_factors: List of scaling factors
        device: Computation device

    Returns:
        Complete experiment results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running sensitivity experiment")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data: {data_path}")

    # Load model (placeholder - implement based on actual loading)
    # model = load_model(model_type, model_path)

    # Create scaler
    # scaler = SensitivityScaler(model, device)

    # Run experiments for both modalities
    results = {
        'model_path': model_path,
        'data_path': data_path,
        'model_type': model_type,
        'scale_factors': scale_factors or SensitivityScaler.DEFAULT_SCALE_FACTORS,
        'spectrum_results': [],
        'peptide_results': [],
    }

    # Save results
    results_path = output_dir / f"{model_type}_sensitivity_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    return results
