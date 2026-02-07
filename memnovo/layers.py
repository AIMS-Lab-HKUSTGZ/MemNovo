"""
MemNovo Cross-Attention Layers

Implements the core memory retracing mechanism for spectral memory injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class CrossAttentionRetrieval(nn.Module):
    """
    Cross-attention based spectral memory retrieval.

    Implements projection-free memory retrieval via scaled dot-product attention:
        A = softmax(H @ M^T / sqrt(d))
        S_pool = A @ M
        H_enhanced = H + α * S_pool

    Key design choices:
    - No learned projections (W_Q, W_K, W_V)
    - Softmax normalization for attention weights
    - Conservative residual injection (α << 1)

    Args:
        dim_model: Model hidden dimension
        residual_scale: Scaling factor for residual injection (default: 0.005)
        use_softmax: Use softmax (True) or ReLU (False) for attention weights
    """

    def __init__(
        self,
        dim_model: int,
        residual_scale: float = 0.005,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.residual_scale = residual_scale
        self.use_softmax = use_softmax
        self.scale = 1.0 / math.sqrt(dim_model)

        logger.info(
            f"CrossAttentionRetrieval initialized: "
            f"dim={dim_model}, scale={residual_scale}, softmax={use_softmax}"
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        spectral_memory: torch.Tensor,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply spectral memory retrieval and injection.

        Args:
            hidden_state: Decoder hidden state [batch, seq_len, dim]
            spectral_memory: Encoder spectral features [batch, n_peaks, dim]
            spectral_mask: Optional mask [batch, n_peaks], 1=valid, 0=padding

        Returns:
            Enhanced hidden state [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = hidden_state.shape
        _, n_peaks, mem_dim = spectral_memory.shape

        # Validate dimensions
        if dim != self.dim_model or mem_dim != self.dim_model:
            logger.warning(
                f"Dimension mismatch: expected {self.dim_model}, "
                f"got hidden={dim}, memory={mem_dim}"
            )
            return hidden_state

        # Handle batch size mismatch (beam search expansion)
        if hidden_state.shape[0] != spectral_memory.shape[0]:
            return hidden_state

        try:
            with torch.no_grad():
                # Compute attention scores
                # [batch, seq_len, dim] @ [batch, dim, n_peaks] -> [batch, seq_len, n_peaks]
                scores = torch.matmul(hidden_state, spectral_memory.transpose(-2, -1))
                scores = scores * self.scale

                # Apply mask if provided
                if spectral_mask is not None:
                    # Expand mask: [batch, n_peaks] -> [batch, 1, n_peaks]
                    mask = spectral_mask.unsqueeze(1)
                    scores = scores.masked_fill(mask == 0, float('-inf'))

                # Compute attention weights
                if self.use_softmax:
                    attention = F.softmax(scores, dim=-1)
                    # Handle all-masked case
                    attention = torch.nan_to_num(attention, nan=0.0)
                else:
                    # ReLU activation (alternative)
                    attention = F.relu(scores)
                    if spectral_mask is not None:
                        attention = attention * mask

                # Pool spectral features
                # [batch, seq_len, n_peaks] @ [batch, n_peaks, dim] -> [batch, seq_len, dim]
                pooled = torch.matmul(attention, spectral_memory)

                # Residual injection
                enhanced = hidden_state + self.residual_scale * pooled

            return enhanced

        except Exception as e:
            logger.warning(f"CrossAttentionRetrieval error: {e}")
            return hidden_state


class SpectrumEnhancer(nn.Module):
    """
    Enhanced spectrum injection with configurable strategies.

    Supports multiple injection modes:
    - 'additive': H' = H + α * S_pool
    - 'gated': H' = H + α * σ(gate) * S_pool
    - 'residual': H' = (1-α) * H + α * S_pool

    Args:
        dim_model: Model hidden dimension
        residual_scale: Injection strength
        mode: Injection mode ('additive', 'gated', 'residual')
    """

    def __init__(
        self,
        dim_model: int,
        residual_scale: float = 0.005,
        mode: str = 'additive',
    ):
        super().__init__()
        self.dim_model = dim_model
        self.residual_scale = residual_scale
        self.mode = mode

        self.cross_attn = CrossAttentionRetrieval(
            dim_model=dim_model,
            residual_scale=1.0,  # We handle scaling here
            use_softmax=True,
        )

        if mode == 'gated':
            self.gate = nn.Linear(dim_model, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        spectral_memory: torch.Tensor,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply spectrum enhancement."""
        # Get pooled spectral features (without injection)
        with torch.no_grad():
            scores = torch.matmul(
                hidden_state,
                spectral_memory.transpose(-2, -1)
            ) / math.sqrt(self.dim_model)

            if spectral_mask is not None:
                mask = spectral_mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attention = F.softmax(scores, dim=-1)
            attention = torch.nan_to_num(attention, nan=0.0)
            pooled = torch.matmul(attention, spectral_memory)

        # Apply injection based on mode
        if self.mode == 'additive':
            enhanced = hidden_state + self.residual_scale * pooled

        elif self.mode == 'gated':
            gate = torch.sigmoid(self.gate(hidden_state))
            enhanced = hidden_state + self.residual_scale * gate * pooled

        elif self.mode == 'residual':
            enhanced = (1 - self.residual_scale) * hidden_state + self.residual_scale * pooled

        else:
            enhanced = hidden_state + self.residual_scale * pooled

        return enhanced
