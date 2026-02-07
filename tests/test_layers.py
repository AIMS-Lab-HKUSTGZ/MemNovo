"""
Tests for MemNovo layers module.
"""

import pytest
import torch
import torch.nn as nn

from memnovo.layers import CrossAttentionRetrieval, SpectrumEnhancer


class TestCrossAttentionRetrieval:
    """Tests for CrossAttentionRetrieval layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = CrossAttentionRetrieval(dim_model=512, residual_scale=0.005)
        assert layer.dim_model == 512
        assert layer.residual_scale == 0.005
        assert layer.use_softmax is True

    def test_forward_shape(self):
        """Test output shape matches input."""
        layer = CrossAttentionRetrieval(dim_model=256)

        batch_size = 4
        seq_len = 10
        n_peaks = 50
        dim = 256

        hidden = torch.randn(batch_size, seq_len, dim)
        memory = torch.randn(batch_size, n_peaks, dim)

        output = layer(hidden, memory)

        assert output.shape == hidden.shape

    def test_forward_with_mask(self):
        """Test forward pass with spectral mask."""
        layer = CrossAttentionRetrieval(dim_model=128)

        batch_size = 2
        seq_len = 5
        n_peaks = 20
        dim = 128

        hidden = torch.randn(batch_size, seq_len, dim)
        memory = torch.randn(batch_size, n_peaks, dim)
        mask = torch.ones(batch_size, n_peaks)
        mask[:, 10:] = 0  # Mask second half

        output = layer(hidden, memory, mask)

        assert output.shape == hidden.shape

    def test_residual_scale_effect(self):
        """Test that residual scale affects output magnitude."""
        dim = 64
        layer_small = CrossAttentionRetrieval(dim_model=dim, residual_scale=0.001)
        layer_large = CrossAttentionRetrieval(dim_model=dim, residual_scale=0.1)

        hidden = torch.randn(2, 5, dim)
        memory = torch.randn(2, 10, dim)

        output_small = layer_small(hidden, memory)
        output_large = layer_large(hidden, memory)

        # Larger scale should produce larger deviation from original
        diff_small = (output_small - hidden).abs().mean()
        diff_large = (output_large - hidden).abs().mean()

        assert diff_large > diff_small

    def test_batch_mismatch_passthrough(self):
        """Test that batch size mismatch returns original input."""
        layer = CrossAttentionRetrieval(dim_model=64)

        hidden = torch.randn(4, 5, 64)
        memory = torch.randn(2, 10, 64)  # Different batch size

        output = layer(hidden, memory)

        assert torch.allclose(output, hidden)


class TestSpectrumEnhancer:
    """Tests for SpectrumEnhancer layer."""

    def test_additive_mode(self):
        """Test additive injection mode."""
        layer = SpectrumEnhancer(dim_model=128, mode='additive')

        hidden = torch.randn(2, 5, 128)
        memory = torch.randn(2, 10, 128)

        output = layer(hidden, memory)
        assert output.shape == hidden.shape

    def test_gated_mode(self):
        """Test gated injection mode."""
        layer = SpectrumEnhancer(dim_model=128, mode='gated')

        hidden = torch.randn(2, 5, 128)
        memory = torch.randn(2, 10, 128)

        output = layer(hidden, memory)
        assert output.shape == hidden.shape

    def test_residual_mode(self):
        """Test residual injection mode."""
        layer = SpectrumEnhancer(dim_model=128, mode='residual')

        hidden = torch.randn(2, 5, 128)
        memory = torch.randn(2, 10, 128)

        output = layer(hidden, memory)
        assert output.shape == hidden.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
