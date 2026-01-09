"""MLP architecture for PyRat."""

from __future__ import annotations

from alpharat.nn.architectures.mlp.config import MLPModelConfig, MLPOptimConfig
from alpharat.nn.architectures.mlp.loss import compute_mlp_losses
from alpharat.nn.models.mlp import PyRatMLP

__all__ = [
    "PyRatMLP",
    "MLPModelConfig",
    "MLPOptimConfig",
    "compute_mlp_losses",
]
