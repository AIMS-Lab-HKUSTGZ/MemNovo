"""
MemNovo Manager

Central management class for MemNovo enhancement.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .hooks import HookManager
from .layers import CrossAttentionRetrieval

logger = logging.getLogger(__name__)


class MemNovoManager:
    """
    MemNovo Manager - Central controller for memory-enhanced decoding.

    Args:
        config: Configuration dictionary with MemNovo parameters
    """

    # Default configuration
    DEFAULT_CONFIG = {
        'residual_scale': 0.005,
        'apply_to_last_n_layers': 1,
        'confidence_threshold': None,
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize MemNovo manager with configuration."""
        self.config = config
        self.enabled = config.get('enabled', True)

        # Get parameters with defaults
        self.residual_scale = config.get('residual_scale', self.DEFAULT_CONFIG['residual_scale'])
        self.apply_to_last_n_layers = config.get('apply_to_last_n_layers', self.DEFAULT_CONFIG['apply_to_last_n_layers'])
        self.confidence_threshold = config.get('confidence_threshold', self.DEFAULT_CONFIG['confidence_threshold'])

        # Initialize hook manager
        hook_config = {
            'enabled': self.enabled,
            'residual_scale': self.residual_scale,
            'apply_to_last_n_layers': self.apply_to_last_n_layers,
            'confidence_threshold': self.confidence_threshold,
        }
        self.hook_manager = HookManager(hook_config)

        # Model reference
        self.model = None

        logger.info(
            f"MemNovoManager initialized: "
            f"scale={self.residual_scale}, layers={self.apply_to_last_n_layers}, "
            f"confidence_gate={self.confidence_threshold}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MemNovoManager':
        """
        Create manager from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Configured MemNovoManager instance
        """
        return cls(config)

    @classmethod
    def default(cls, **kwargs) -> 'MemNovoManager':
        """
        Create manager with default configuration.

        Args:
            **kwargs: Configuration overrides

        Returns:
            Configured MemNovoManager instance
        """
        config = {'enabled': True}
        config.update(cls.DEFAULT_CONFIG)
        config.update(kwargs)
        return cls(config)

    def register(self, model: nn.Module) -> None:
        """
        Register MemNovo hooks on the model.

        Args:
            model: De novo sequencing model (Casanovo or InstaNovo)
        """
        if not self.enabled:
            logger.info("MemNovo disabled, skipping registration")
            return

        self.model = model
        self.hook_manager.register_hooks(model)

    def unregister(self) -> None:
        """Remove all MemNovo hooks from the model."""
        self.hook_manager.remove_hooks()
        self.model = None

    def set_spectral_memory(
        self,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Manually set spectral memory for injection.

        Args:
            memory: Encoder output [batch, n_peaks, dim]
            mask: Optional padding mask [batch, n_peaks]
        """
        self.hook_manager.set_spectral_memory(memory, mask)

    def reset(self) -> None:
        """Reset runtime state for new batch."""
        self.hook_manager.reset_state()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.hook_manager.get_stats()

    @property
    def is_enabled(self) -> bool:
        """Check if MemNovo is enabled."""
        return self.enabled

    @property
    def is_registered(self) -> bool:
        """Check if hooks are registered."""
        return len(self.hook_manager.hooks) > 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unregister()
        return False

    def __repr__(self) -> str:
        return (
            f"MemNovoManager("
            f"scale={self.residual_scale}, "
            f"layers={self.apply_to_last_n_layers}, "
            f"enabled={self.enabled})"
        )


def create_memnovo_manager(**config) -> MemNovoManager:
    """
    Factory function for creating MemNovo manager.

    Args:
        **config: Configuration parameters

    Returns:
        Configured MemNovoManager
    """
    return MemNovoManager.default(**config)
