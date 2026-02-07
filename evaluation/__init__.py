"""
Evaluation Framework

Provides unified evaluation utilities for de novo peptide sequencing
models, compatible with both Casanovo and InstaNovo metrics.
"""

from .evaluator import Evaluator, compute_metrics
from .data_handler import DataHandler, load_mgf_file
from .metrics import (
    aa_precision,
    aa_recall,
    peptide_precision,
    peptide_recall,
    compute_all_metrics,
)

__all__ = [
    'Evaluator',
    'compute_metrics',
    'DataHandler',
    'load_mgf_file',
    'aa_precision',
    'aa_recall',
    'peptide_precision',
    'peptide_recall',
    'compute_all_metrics',
]
