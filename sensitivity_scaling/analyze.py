"""
Sensitivity Analysis Functions

Utilities for analyzing sensitivity scaling experiment results.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import json
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)


def compute_sensitivity_ratio(
    spectrum_results: List[Dict[str, Any]],
    peptide_results: List[Dict[str, Any]],
    metric: str = 'aa_precision',
    baseline_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute sensitivity ratio between peptide and spectrum modalities.

    Sensitivity Ratio = Δ_peptide / Δ_spectrum

    A ratio > 1 indicates heavier reliance on peptide priors than
    spectral evidence.

    Args:
        spectrum_results: Results from spectrum scaling experiment
        peptide_results: Results from peptide scaling experiment
        metric: Metric to use for sensitivity calculation
        baseline_scale: Reference scale factor

    Returns:
        Dictionary with sensitivity metrics
    """
    spectrum_sensitivity = _compute_modality_sensitivity(
        spectrum_results, metric, baseline_scale
    )
    peptide_sensitivity = _compute_modality_sensitivity(
        peptide_results, metric, baseline_scale
    )

    ratio = peptide_sensitivity / max(spectrum_sensitivity, 1e-6)

    return {
        'spectrum_sensitivity': spectrum_sensitivity,
        'peptide_sensitivity': peptide_sensitivity,
        'sensitivity_ratio': ratio,
        'metric': metric,
        'interpretation': _interpret_ratio(ratio),
    }


def _compute_modality_sensitivity(
    results: List[Dict[str, Any]],
    metric: str,
    baseline_scale: float,
) -> float:
    """Compute sensitivity for a single modality."""
    # Find baseline performance
    baseline_perf = None
    for r in results:
        if r.get('scale_factor') == baseline_scale:
            baseline_perf = r.get(metric, 0)
            break

    if baseline_perf is None or baseline_perf == 0:
        return 0.0

    # Compute average deviation from baseline
    deviations = []
    for r in results:
        scale = r.get('scale_factor')
        if scale != baseline_scale and 'error' not in r:
            perf = r.get(metric, 0)
            deviation = abs(perf - baseline_perf) / baseline_perf
            deviations.append(deviation)

    return np.mean(deviations) if deviations else 0.0


def _interpret_ratio(ratio: float) -> str:
    """Provide interpretation of sensitivity ratio."""
    if ratio > 10:
        return "Severe peptide dominance - model heavily over-relies on linguistic priors"
    elif ratio > 5:
        return "Strong peptide dominance - significant spectral under-utilization"
    elif ratio > 2:
        return "Moderate peptide dominance - some spectral under-utilization"
    elif ratio > 1:
        return "Slight peptide dominance - mild imbalance"
    elif ratio > 0.5:
        return "Balanced - roughly equal sensitivity to both modalities"
    else:
        return "Spectrum dominant - unusual, may indicate encoder issues"


def analyze_results(
    results_path: str,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze sensitivity scaling results from a JSON file.

    Args:
        results_path: Path to results JSON file
        output_path: Optional path to save analysis

    Returns:
        Analysis dictionary
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    spectrum_results = results.get('spectrum_results', [])
    peptide_results = results.get('peptide_results', [])

    # Compute sensitivities for different metrics
    analysis = {
        'model_type': results.get('model_type', 'unknown'),
        'n_scale_factors': len(results.get('scale_factors', [])),
        'aa_precision': compute_sensitivity_ratio(
            spectrum_results, peptide_results, 'aa_precision'
        ),
        'pep_precision': compute_sensitivity_ratio(
            spectrum_results, peptide_results, 'pep_precision'
        ),
    }

    # Compute asymmetry (upscale vs downscale)
    if spectrum_results:
        analysis['spectrum_asymmetry'] = _compute_asymmetry(spectrum_results)
    if peptide_results:
        analysis['peptide_asymmetry'] = _compute_asymmetry(peptide_results)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {output_path}")

    return analysis


def _compute_asymmetry(
    results: List[Dict[str, Any]],
    metric: str = 'aa_precision',
) -> Dict[str, float]:
    """
    Compute asymmetry between upscaling and downscaling effects.

    Positive asymmetry: downscaling hurts less than upscaling helps
    Negative asymmetry: upscaling hurts more than downscaling helps
    """
    # Find baseline
    baseline_perf = None
    for r in results:
        if r.get('scale_factor') == 1.0:
            baseline_perf = r.get(metric, 0)
            break

    if baseline_perf is None or baseline_perf == 0:
        return {'asymmetry': 0.0, 'downscale_effect': 0.0, 'upscale_effect': 0.0}

    # Compute average effects
    downscale_effects = []
    upscale_effects = []

    for r in results:
        scale = r.get('scale_factor')
        if scale is None or 'error' in r:
            continue

        perf = r.get(metric, 0)
        change = (perf - baseline_perf) / baseline_perf

        if scale < 1.0:
            downscale_effects.append(change)
        elif scale > 1.0:
            upscale_effects.append(change)

    avg_downscale = np.mean(downscale_effects) if downscale_effects else 0.0
    avg_upscale = np.mean(upscale_effects) if upscale_effects else 0.0

    return {
        'asymmetry': avg_downscale - avg_upscale,
        'downscale_effect': avg_downscale,
        'upscale_effect': avg_upscale,
    }


def summarize_experiments(
    results_dir: str,
    models: List[str] = None,
) -> Dict[str, Any]:
    """
    Summarize sensitivity experiments across multiple models.

    Args:
        results_dir: Directory containing result JSON files
        models: List of model names to include

    Returns:
        Summary dictionary
    """
    results_dir = Path(results_dir)
    summary = {'models': {}}

    for results_file in results_dir.glob("*_sensitivity_results.json"):
        model_name = results_file.stem.replace('_sensitivity_results', '')

        if models and model_name not in models:
            continue

        try:
            analysis = analyze_results(str(results_file))
            summary['models'][model_name] = analysis
        except Exception as e:
            logger.warning(f"Error analyzing {results_file}: {e}")

    # Compute overall statistics
    ratios = [
        m['aa_precision']['sensitivity_ratio']
        for m in summary['models'].values()
        if 'aa_precision' in m
    ]

    if ratios:
        summary['overall'] = {
            'mean_ratio': np.mean(ratios),
            'max_ratio': max(ratios),
            'min_ratio': min(ratios),
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sensitivity scaling results")
    parser.add_argument("--input", required=True, help="Sensitivity results JSON file")
    parser.add_argument("--output", default=None, help="Optional analysis JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze_results(args.input, args.output)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
