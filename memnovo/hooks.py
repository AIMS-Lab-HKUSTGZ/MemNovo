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
        self.memory_token_trim_left = int(config.get('memory_token_trim_left', 0) or 0)
        self.use_softmax = bool(config.get('use_softmax', True))

        # Runtime state
        self.model = None
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.spectral_memory: Optional[torch.Tensor] = None
        self.spectral_mask: Optional[torch.Tensor] = None
        self.original_batch_size: int = 0
        self._original_init = None
        self._original_encoder_forward = None

        # Statistics
        self.stats = {
            'total_calls': 0,
            'effective_calls': 0,
            'skipped_by_confidence': 0,
            'use_softmax': self.use_softmax,
        }

        logger.info(
            f"HookManager initialized: enabled={self.enabled}, "
            f"scale={self.residual_scale}, layers={self.apply_to_last_n_layers}, "
            f"trim_left={self.memory_token_trim_left}, "
            f"use_softmax={self.use_softmax}"
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
        self._setup_memory_capture(model)

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
            hook_fn = self._create_injection_hook(layer_idx, num_layers)
            try:
                hook = layer.register_forward_hook(hook_fn, with_kwargs=True)
            except TypeError:
                hook = layer.register_forward_hook(hook_fn)
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

    def _setup_memory_capture(self, model: nn.Module) -> None:
        """Set up runtime capture of encoder outputs as spectral memory."""
        if self._wrap_init_method(model):
            return
        if self._wrap_encoder_forward(model):
            return
        logger.warning("Unable to wrap model.init or encoder.forward for spectral memory capture")

    def _wrap_init_method(self, model: nn.Module) -> bool:
        """Wrap model.init to capture spectral encoding."""
        if not hasattr(model, 'init'):
            return False

        if self._original_init is not None:
            return True

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
                        encoding, _ = manager._trim_memory_and_mask(encoding, None)
                        manager.spectral_memory = encoding
                        manager.original_batch_size = encoding.shape[0]
                        logger.debug(f"Captured spectral memory: {encoding.shape}")
            except Exception as e:
                logger.debug(f"Failed to capture spectral memory: {e}")

            return result

        self._original_init = original_init
        model.init = wrapped_init
        logger.info("Wrapped model.init for spectral memory capture")
        return True

    def _wrap_encoder_forward(self, model: nn.Module) -> bool:
        """Wrap encoder.forward to capture spectral memory for models without model.init."""
        encoder = getattr(model, 'encoder', None)
        if encoder is None or not hasattr(encoder, 'forward'):
            return False

        if self._original_encoder_forward is not None:
            return True

        original_forward = encoder.forward
        manager = self

        def wrapped_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)

            try:
                if isinstance(result, tuple) and len(result) >= 1:
                    memory = result[0]
                    mask = result[1] if len(result) >= 2 else None
                    if isinstance(memory, torch.Tensor) and memory.dim() == 3:
                        memory, mask = manager._trim_memory_and_mask(memory, mask if isinstance(mask, torch.Tensor) else None)
                        manager.spectral_memory = memory
                        manager.original_batch_size = memory.shape[0]
                        manager.spectral_mask = mask if isinstance(mask, torch.Tensor) else None
                        logger.debug(f"Captured encoder spectral memory: {memory.shape}")
            except Exception as e:
                logger.debug(f"Failed to capture encoder spectral memory: {e}")

            return result

        self._original_encoder_forward = original_forward
        encoder.forward = wrapped_forward
        logger.info("Wrapped encoder.forward for spectral memory capture")
        return True

    def _create_injection_hook(
        self,
        layer_idx: int,
        total_layers: int,
    ) -> Callable:
        """Create a forward hook for spectral injection."""

        def hook_fn(module, input_tuple, kwargs_or_output, maybe_output=None):
            if maybe_output is None:
                kwargs = {}
                output = kwargs_or_output
            else:
                kwargs = kwargs_or_output if isinstance(kwargs_or_output, dict) else {}
                output = maybe_output
            self.stats['total_calls'] += 1

            # Skip if disabled or no memory
            if not self.enabled:
                return output

            if isinstance(output, tuple):
                if not output or not isinstance(output[0], torch.Tensor) or output[0].dim() != 3:
                    return output
                output_tensor = output[0]
                output_is_tuple = True
            elif isinstance(output, torch.Tensor) and output.dim() == 3:
                output_tensor = output
                output_is_tuple = False
            else:
                return output

            try:
                # Prefer the actual decoder memory passed to this layer.
                memory, mask = self._extract_runtime_memory(input_tuple, kwargs)
                transposed = False
                if memory is not None:
                    mem_batch = memory.shape[0]
                    if output_tensor.shape[0] == mem_batch:
                        transposed = False
                    elif output_tensor.shape[1] == mem_batch:
                        output_tensor = output_tensor.transpose(0, 1)
                        transposed = True
                elif output_tensor.shape[0] < output_tensor.shape[1]:
                    # Fallback heuristic for models where live decoder memory is not exposed.
                    output_tensor = output_tensor.transpose(0, 1)
                    transposed = True

                batch_size = output_tensor.shape[0]

                if memory is None:
                    memory = self._get_matched_memory(batch_size)
                    mask = self._get_matched_mask(batch_size)
                if memory is None:
                    return output

                # Apply spectral injection
                enhanced = self._apply_injection(output_tensor, memory, mask)

                if enhanced is not None:
                    self.stats['effective_calls'] += 1
                    if transposed:
                        enhanced = enhanced.transpose(0, 1)
                    if output_is_tuple:
                        return (enhanced, *output[1:])
                    return enhanced

            except Exception as e:
                logger.debug(f"Injection error at layer {layer_idx}: {e}")

            return output

        return hook_fn

    def _extract_runtime_memory(
        self,
        input_tuple,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract decoder memory and mask from the layer's live inputs when available."""
        kwargs = kwargs or {}
        runtime_memory = None
        runtime_mask = None

        if len(input_tuple) >= 2 and isinstance(input_tuple[1], torch.Tensor):
            candidate = input_tuple[1]
            if candidate.dim() == 3:
                runtime_memory = candidate

        if len(input_tuple) >= 6 and isinstance(input_tuple[5], torch.Tensor):
            runtime_mask = input_tuple[5]

        kw_memory = kwargs.get("memory")
        if isinstance(kw_memory, torch.Tensor) and kw_memory.dim() == 3:
            runtime_memory = kw_memory

        kw_mask = kwargs.get("memory_key_padding_mask")
        if isinstance(kw_mask, torch.Tensor):
            runtime_mask = kw_mask

        if runtime_memory is not None:
            return self._trim_memory_and_mask(runtime_memory, runtime_mask)

        return None, None

    def _trim_memory_and_mask(
        self,
        memory: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if memory is None:
            return None, mask

        trim_left = self.memory_token_trim_left
        if trim_left <= 0:
            return memory, mask

        if memory.dim() != 3 or memory.shape[1] <= trim_left:
            return None, None

        trimmed_memory = memory[:, trim_left:, :]
        trimmed_mask = mask[:, trim_left:] if isinstance(mask, torch.Tensor) and mask.dim() == 2 and mask.shape[1] >= trim_left else mask
        return trimmed_memory, trimmed_mask

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

    def _get_matched_mask(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get spectral mask matched to current batch size."""
        if self.spectral_mask is None:
            return None

        mask_batch = self.spectral_mask.shape[0]

        if mask_batch == batch_size:
            return self.spectral_mask

        if self.original_batch_size and batch_size % self.original_batch_size == 0:
            beam_size = batch_size // self.original_batch_size
            return self.spectral_mask.repeat_interleave(beam_size, dim=0)

        if batch_size < self.original_batch_size:
            return self.spectral_mask[:batch_size]

        return None

    def _apply_injection(
        self,
        hidden_state: torch.Tensor,
        spectral_memory: torch.Tensor,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Apply spectral memory injection."""
        try:
            with torch.no_grad():
                # Compute attention
                scores = torch.matmul(
                    hidden_state,
                    spectral_memory.transpose(-2, -1)
                ) / math.sqrt(hidden_state.shape[-1])

                if spectral_mask is not None:
                    if spectral_mask.dtype == torch.bool:
                        invalid_mask = spectral_mask
                    else:
                        invalid_mask = spectral_mask <= 0
                    scores = scores.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))

                if self.use_softmax:
                    attention = torch.softmax(scores, dim=-1)
                    attention = torch.nan_to_num(attention, nan=0.0)
                else:
                    attention = torch.relu(scores)
                    if spectral_mask is not None:
                        attention = attention.masked_fill(invalid_mask.unsqueeze(1), 0.0)

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
        memory, mask = self._trim_memory_and_mask(memory, mask)
        if memory is None:
            self.spectral_memory = None
            self.spectral_mask = None
            self.original_batch_size = 0
            return
        self.spectral_memory = memory
        self.spectral_mask = mask
        self.original_batch_size = memory.shape[0]

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if self.model is not None and self._original_init is not None:
            self.model.init = self._original_init
            self._original_init = None
        if self.model is not None and self._original_encoder_forward is not None and hasattr(self.model, 'encoder'):
            self.model.encoder.forward = self._original_encoder_forward
            self._original_encoder_forward = None
        self.spectral_memory = None
        self.spectral_mask = None
        self.original_batch_size = 0
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
