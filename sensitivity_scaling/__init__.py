"""
Sensitivity Scaling Framework

Diagnostic tool for quantifying modal imbalance in de novo peptide
sequencing models. Applies continuous scaling factors to feature
magnitudes to measure model sensitivity to each input source.
"""

from .experiment import SensitivityScaler, run_sensitivity_experiment
from .analyze import compute_sensitivity_ratio, analyze_results
from .visualize import plot_sensitivity_curves, plot_comparison

__all__ = [
    'SensitivityScaler',
    'run_sensitivity_experiment',
    'compute_sensitivity_ratio',
    'analyze_results',
    'plot_sensitivity_curves',
    'plot_comparison',
]
