"""
MemNovo Hook Management

Provides PyTorch hook registration and management for injecting MemNovo
into pre-trained transformer decoders without modifying model code.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable
import logging
import math

logger = logging.getLogger(__name__)


class HookManager:
    """
    Manages PyTorch forward hooks for MemNovo injection.

    This class handles:
    1. Hook registration on specific decoder layers
    2. Spectral memory capture and storage
    3. Dynamic batch size handling for beam search
    4. Hook lifecycle management

    Args:
        config: MemNovo configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.residual_scale = config.get('residual_scale', 0.005)
        self.apply_to_last_n_layers = config.get('apply_to_last_n_layers', 1)
        self.confidence_threshold = config.get('confidence_threshold', None)

        # Runtime state
        self.model = None
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.spectral_memory: Optional[torch.Tensor] = None
        self.spectral_mask: Optional[torch.Tensor] = None
        self.original_batch_size: int = 0

        # Statistics
        self.stats = {
            'total_calls': 0,
            'effective_calls': 0,
            'skipped_by_confidence': 0,
        }

        logger.info(
            f"HookManager initialized: enabled={self.enabled}, "
            f"scale={self.residual_scale}, layers={self.apply_to_last_n_layers}"
        )

    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks on decoder layers.

        Args:
            model: The de novo sequencing model (Casanovo or InstaNovo)
        """
        if not self.enabled:
            logger.info("MemNovo disabled, skipping hook registration")
            return

        self.model = model

        # Capture spectral encoding from model.init
        self._wrap_init_method(model)

        # Find and hook decoder layers
        decoder_layers = self._find_decoder_layers(model)
        if not decoder_layers:
            logger.warning("No decoder layers found, cannot register hooks")
            return

        num_layers = len(decoder_layers)
        start_layer = max(0, num_layers - self.apply_to_last_n_layers)

        logger.info(
            f"Registering hooks on decoder layers {start_layer}-{num_layers-1} "
            f"(last {self.apply_to_last_n_layers} of {num_layers} layers)"
        )

        for layer_idx in range(start_layer, num_layers):
            layer = decoder_layers[layer_idx]
            hook = layer.register_forward_hook(
                self._create_injection_hook(layer_idx, num_layers)
            )
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} hooks successfully")

    def _find_decoder_layers(self, model: nn.Module) -> Optional[nn.ModuleList]:
        """Find decoder layers in the model."""
        # Try common patterns
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            return model.decoder.layers

        if hasattr(model, 'decoder') and hasattr(model.decoder, 'transformer_decoder'):
            if hasattr(model.decoder.transformer_decoder, 'layers'):
                return model.decoder.transformer_decoder.layers

        # Search recursively
        for name, module in model.named_modules():
            if 'decoder' in name.lower() and hasattr(module, 'layers'):
                if isinstance(module.layers, nn.ModuleList):
                    return module.layers

        return None

    def _wrap_init_method(self, model: nn.Module) -> None:
        """Wrap model.init to capture spectral encoding."""
        if not hasattr(model, 'init'):
            logger.warning("Model has no 'init' method, cannot capture spectral memory")
            return

        original_init = model.init
        manager = self

        def wrapped_init(*args, **kwargs):
            result = original_init(*args, **kwargs)

            try:
                # Extract spectral encoding from init result
                if isinstance(result, tuple) and len(result) >= 1:
                    first = result[0]
                    if isinstance(first, tuple) and len(first) >= 1:
                        encoding = first[0]
                    else:
                        encoding = first

                    if isinstance(encoding, torch.Tensor) and encoding.dim() == 3:
                        manager.spectral_memory = encoding.clone()
                        manager.original_batch_size = encoding.shape[0]
                        logger.debug(f"Captured spectral memory: {encoding.shape}")
            except Exception as e:
                logger.debug(f"Failed to capture spectral memory: {e}")

            return result

        model.init = wrapped_init
        logger.info("Wrapped model.init for spectral memory capture")

    def _create_injection_hook(
        self,
        layer_idx: int,
        total_layers: int,
    ) -> Callable:
        """Create a forward hook for spectral injection."""

        def hook_fn(module, input_tuple, output):
            self.stats['total_calls'] += 1

            # Skip if disabled or no memory
            if not self.enabled or self.spectral_memory is None:
                return output

            # Ensure output is a tensor
            if not isinstance(output, torch.Tensor) or output.dim() != 3:
                return output

            try:
                # Detect tensor ordering (batch-first vs seq-first)
                if output.shape[0] < output.shape[1]:
                    # Likely seq-first: (seq, batch, dim)
                    output = output.transpose(0, 1)
                    transposed = True
                else:
                    transposed = False

                batch_size = output.shape[0]

                # Get matched spectral memory
                memory = self._get_matched_memory(batch_size)
                if memory is None:
                    return output.transpose(0, 1) if transposed else output

                # Apply spectral injection
                enhanced = self._apply_injection(output, memory)

                if enhanced is not None:
                    self.stats['effective_calls'] += 1
                    if transposed:
                        enhanced = enhanced.transpose(0, 1)
                    return enhanced

            except Exception as e:
                logger.debug(f"Injection error at layer {layer_idx}: {e}")

            return output

        return hook_fn

    def _get_matched_memory(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get spectral memory matched to current batch size."""
        if self.spectral_memory is None:
            return None

        mem_batch = self.spectral_memory.shape[0]

        if mem_batch == batch_size:
            return self.spectral_memory

        # Beam search expansion
        if batch_size % self.original_batch_size == 0:
            beam_size = batch_size // self.original_batch_size
            return self.spectral_memory.repeat_interleave(beam_size, dim=0)

        # Last batch (smaller)
        if batch_size < self.original_batch_size:
            return self.spectral_memory[:batch_size]

        return None

    def _apply_injection(
        self,
        hidden_state: torch.Tensor,
        spectral_memory: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Apply spectral memory injection."""
        try:
            with torch.no_grad():
                # Compute attention
                scores = torch.matmul(
                    hidden_state,
                    spectral_memory.transpose(-2, -1)
                ) / math.sqrt(hidden_state.shape[-1])

                attention = torch.softmax(scores, dim=-1)

                # Pool spectral features
                pooled = torch.matmul(attention, spectral_memory)

                # Residual injection
                enhanced = hidden_state + self.residual_scale * pooled

            return enhanced

        except Exception as e:
            logger.debug(f"Injection computation error: {e}")
            return None

    def set_spectral_memory(
        self,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Manually set spectral memory."""
        self.spectral_memory = memory
        self.spectral_mask = mask
        self.original_batch_size = memory.shape[0]

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.spectral_memory = None
        self.spectral_mask = None
        logger.info("Removed all MemNovo hooks")

    def reset_state(self) -> None:
        """Reset runtime state."""
        self.spectral_memory = None
        self.spectral_mask = None
        self.original_batch_size = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics."""
        effectiveness = (
            self.stats['effective_calls'] / max(self.stats['total_calls'], 1)
        )
        return {
            **self.stats,
            'effectiveness_rate': effectiveness,
            'residual_scale': self.residual_scale,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
