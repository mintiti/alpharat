"""Shared loss functions for neural network training."""

from __future__ import annotations

from alpharat.nn.losses.constant_sum import constant_sum_loss
from alpharat.nn.losses.nash_consistency import nash_consistency_loss
from alpharat.nn.losses.ownership import compute_ownership_loss
from alpharat.nn.losses.sparse_payout import sparse_payout_loss

__all__ = [
    "sparse_payout_loss",
    "nash_consistency_loss",
    "constant_sum_loss",
    "compute_ownership_loss",
]
