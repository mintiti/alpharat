"""Tests for CNN head modules."""

from __future__ import annotations

import torch

from alpharat.nn.models.cnn.heads import MLPPolicyHead, PointValueHead, PooledValueHead


class TestMLPPolicyHead:
    """Tests for MLPPolicyHead."""

    def test_output_shape(self) -> None:
        head = MLPPolicyHead(hidden_dim=64, num_actions=5)
        h_i = torch.randn(4, 32)
        agg = torch.randn(4, 32)
        out = head(h_i, agg)
        assert out.shape == (4, 5)

    def test_different_hidden_dims(self) -> None:
        head = MLPPolicyHead(hidden_dim=128, num_actions=5)
        h_i = torch.randn(2, 64)
        agg = torch.randn(2, 64)
        out = head(h_i, agg)
        assert out.shape == (2, 5)


class TestPointValueHead:
    """Tests for PointValueHead."""

    def test_output_shape(self) -> None:
        head = PointValueHead(hidden_dim=64)
        h_i = torch.randn(4, 32)
        agg = torch.randn(4, 32)
        out = head(h_i, agg)
        assert out.shape == (4,)

    def test_non_negative_output(self) -> None:
        """Softplus ensures non-negative outputs."""
        head = PointValueHead(hidden_dim=64)
        h_i = torch.randn(10, 32)
        agg = torch.randn(10, 32)
        out = head(h_i, agg)
        assert (out >= 0).all()

    def test_no_needs_spatial(self) -> None:
        head = PointValueHead(hidden_dim=64)
        assert not getattr(head, "needs_spatial", False)


class TestPooledValueHead:
    """Tests for PooledValueHead."""

    def test_output_shape(self) -> None:
        head = PooledValueHead(hidden_dim=64, hidden_channels=32)
        h_i = torch.randn(4, 32)
        agg = torch.randn(4, 32)
        spatial = torch.randn(4, 32, 5, 5)
        out = head(h_i, agg, spatial)
        assert out.shape == (4,)

    def test_non_negative_output(self) -> None:
        head = PooledValueHead(hidden_dim=64, hidden_channels=32)
        h_i = torch.randn(10, 32)
        agg = torch.randn(10, 32)
        spatial = torch.randn(10, 32, 7, 7)
        out = head(h_i, agg, spatial)
        assert (out >= 0).all()

    def test_needs_spatial_flag(self) -> None:
        head = PooledValueHead(hidden_dim=64, hidden_channels=32)
        assert head.needs_spatial is True

    def test_gradient_flow(self) -> None:
        head = PooledValueHead(hidden_dim=64, hidden_channels=32)
        spatial = torch.randn(2, 32, 5, 5, requires_grad=True)
        h_i = torch.randn(2, 32)
        agg = torch.randn(2, 32)
        out = head(h_i, agg, spatial)
        out.sum().backward()
        assert spatial.grad is not None
