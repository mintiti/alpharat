"""Shared loss functions for neural network training."""

from __future__ import annotations

from alpharat.nn.losses.ownership import compute_ownership_loss

__all__ = [
    "compute_ownership_loss",
]
