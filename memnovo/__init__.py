"""
MemNovo: A Plug-and-Play Framework for Mitigating Sensitivity Imbalance
in De Novo Peptide Sequencing

This package provides:
1. MemNovo enhancement mechanism for transformer-based de novo sequencing models
2. Sensitivity Scaling Framework for diagnosing modal imbalance
3. Unified evaluation utilities
"""

__version__ = "0.1.0"

from .manager import MemNovoManager
from .layers import CrossAttentionRetrieval
from .hooks import HookManager
from .models import MemNovoModel

__all__ = [
    "MemNovoManager",
    "CrossAttentionRetrieval",
    "HookManager",
    "MemNovoModel",
    "__version__",
]
