"""Neural network utilities for observation encoding and target extraction."""

from alpharat.nn.extraction import from_game_arrays, from_pyrat_game
from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    MetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
    explained_variance,
    payout_correlation,
    policy_entropy,
    target_entropy,
    top_k_accuracy,
)
from alpharat.nn.streaming import StreamingDataset
from alpharat.nn.targets import build_targets
from alpharat.nn.types import ObservationInput, TargetBundle

__all__ = [
    "GPUDataset",
    "MetricsAccumulator",
    "ObservationInput",
    "StreamingDataset",
    "TargetBundle",
    "build_targets",
    "compute_payout_metrics",
    "compute_policy_metrics",
    "compute_value_metrics",
    "explained_variance",
    "from_game_arrays",
    "from_pyrat_game",
    "payout_correlation",
    "policy_entropy",
    "target_entropy",
    "top_k_accuracy",
]
