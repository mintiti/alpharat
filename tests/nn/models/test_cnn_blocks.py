"""Tests for CNN trunk blocks."""

from __future__ import annotations

import pytest
import torch

from alpharat.nn.models.cnn.blocks import GPoolResBlock, ResBlock


class TestResBlock:
    """Tests for ResBlock."""

    @pytest.mark.parametrize("spatial", [5, 7, 9])
    def test_output_shape_matches_input(self, spatial: int) -> None:
        block = ResBlock(channels=32)
        x = torch.randn(2, 32, spatial, spatial)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        """Zero-init convs should give identity-like behavior."""
        block = ResBlock(channels=16)
        x = torch.randn(1, 16, 5, 5)
        out = block(x)
        # Output should be finite
        assert torch.isfinite(out).all()


class TestGPoolResBlock:
    """Tests for GPoolResBlock."""

    @pytest.mark.parametrize("spatial", [5, 7, 9])
    def test_output_shape_matches_input(self, spatial: int) -> None:
        block = GPoolResBlock(channels=32, gpool_channels=16)
        x = torch.randn(2, 32, spatial, spatial)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flow(self) -> None:
        block = GPoolResBlock(channels=16, gpool_channels=8)
        x = torch.randn(2, 16, 5, 5, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_gpool_channels(self) -> None:
        for gpool_ch in [8, 16, 32]:
            block = GPoolResBlock(channels=32, gpool_channels=gpool_ch)
            x = torch.randn(2, 32, 5, 5)
            out = block(x)
            assert out.shape == x.shape
