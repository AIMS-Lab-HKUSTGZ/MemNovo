"""
Sensitivity Scaling Visualization

Functions for visualizing sensitivity scaling experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-paper')
COLORS = {
    'spectrum': '#1f77b4',      # Blue
    'peptide': '#ff7f0e',       # Orange
    'baseline': '#2ca02c',      # Green
    'highlight': '#d62728',     # Red
}


def plot_sensitivity_curves(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    metric: str = 'aa_precision',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot sensitivity curves for spectrum and peptide modalities.

    Args:
        results: Results dictionary from sensitivity experiment
        output_path: Optional path to save figure
        metric: Metric to plot
        title: Optional custom title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract data
    spectrum_results = results.get('spectrum_results', [])
    peptide_results = results.get('peptide_results', [])

    # Plot spectrum sensitivity
    if spectrum_results:
        scales = [r['scale_factor'] for r in spectrum_results if 'error' not in r]
        values = [r.get(metric, 0) for r in spectrum_results if 'error' not in r]

        ax.plot(
            scales, values,
            'o-',
            color=COLORS['spectrum'],
            linewidth=2,
            markersize=8,
            label='Spectrum',
        )

    # Plot peptide sensitivity
    if peptide_results:
        scales = [r['scale_factor'] for r in peptide_results if 'error' not in r]
        values = [r.get(metric, 0) for r in peptide_results if 'error' not in r]

        ax.plot(
            scales, values,
            's-',
            color=COLORS['peptide'],
            linewidth=2,
            markersize=8,
            label='Peptide',
        )

    # Add baseline reference
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Formatting
    ax.set_xlabel('Scale Factor', fontsize=12)
    ax.set_ylabel(_format_metric_name(metric), fontsize=12)
    ax.set_xscale('log')

    if title:
        ax.set_title(title, fontsize=14)
    else:
        model_type = results.get('model_type', 'Model')
        ax.set_title(f'{model_type.capitalize()} Sensitivity Analysis', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_comparison(
    results_list: List[Dict[str, Any]],
    model_names: List[str],
    output_path: Optional[str] = None,
    metric: str = 'aa_precision',
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot sensitivity comparison across multiple models.

    Args:
        results_list: List of results dictionaries
        model_names: List of model names
        output_path: Optional path to save figure
        metric: Metric to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    for idx, (results, name) in enumerate(zip(results_list, model_names)):
        ax = axes[idx]

        # Plot spectrum
        spectrum_results = results.get('spectrum_results', [])
        if spectrum_results:
            scales = [r['scale_factor'] for r in spectrum_results if 'error' not in r]
            values = [r.get(metric, 0) for r in spectrum_results if 'error' not in r]
            ax.plot(scales, values, 'o-', color=COLORS['spectrum'], label='Spectrum', linewidth=2)

        # Plot peptide
        peptide_results = results.get('peptide_results', [])
        if peptide_results:
            scales = [r['scale_factor'] for r in peptide_results if 'error' not in r]
            values = [r.get(metric, 0) for r in peptide_results if 'error' not in r]
            ax.plot(scales, values, 's-', color=COLORS['peptide'], label='Peptide', linewidth=2)

        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xlabel('Scale Factor', fontsize=11)
        ax.set_xscale('log')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)

    axes[0].set_ylabel(_format_metric_name(metric), fontsize=11)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    return fig


def plot_sensitivity_ratio_bar(
    ratios: Dict[str, float],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """
    Plot sensitivity ratios as a bar chart.

    Args:
        ratios: Dictionary mapping model names to sensitivity ratios
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = list(ratios.keys())
    values = list(ratios.values())

    # Color bars based on ratio
    colors = [
        COLORS['highlight'] if v > 5 else
        COLORS['peptide'] if v > 2 else
        COLORS['baseline']
        for v in values
    ]

    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.1f}x',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )

    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Balanced')
    ax.axhline(y=5.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Severe imbalance')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Sensitivity Ratio (Peptide / Spectrum)', fontsize=12)
    ax.set_title('Modal Sensitivity Imbalance', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

    return fig


def _format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    mapping = {
        'aa_precision': 'AA Precision',
        'aa_recall': 'AA Recall',
        'pep_precision': 'Peptide Precision',
        'pep_recall': 'Peptide Recall',
    }
    return mapping.get(metric, metric)


def generate_report(
    results_dir: str,
    output_dir: str,
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Generate complete sensitivity analysis report with figures.

    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save report and figures
        model_names: Optional list of models to include
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find result files
    result_files = list(results_dir.glob("*_sensitivity_results.json"))

    if model_names:
        result_files = [
            f for f in result_files
            if any(m in f.stem for m in model_names)
        ]

    logger.info(f"Generating report for {len(result_files)} models")

    # Load results
    results_list = []
    names = []
    ratios = {}

    for result_file in result_files:
        with open(result_file) as f:
            results = json.load(f)

        name = results.get('model_type', result_file.stem)
        results_list.append(results)
        names.append(name)

        # Compute sensitivity ratio
        from .analyze import compute_sensitivity_ratio
        ratio_data = compute_sensitivity_ratio(
            results.get('spectrum_results', []),
            results.get('peptide_results', []),
        )
        ratios[name] = ratio_data['sensitivity_ratio']

    # Generate figures
    if results_list:
        # Individual sensitivity curves
        for results, name in zip(results_list, names):
            plot_sensitivity_curves(
                results,
                output_path=output_dir / f"{name}_sensitivity_curves.pdf",
            )

        # Comparison plot
        if len(results_list) > 1:
            plot_comparison(
                results_list, names,
                output_path=output_dir / "sensitivity_comparison.pdf",
            )

        # Ratio bar chart
        if ratios:
            plot_sensitivity_ratio_bar(
                ratios,
                output_path=output_dir / "sensitivity_ratios.pdf",
            )

    logger.info(f"Report generated in {output_dir}")
