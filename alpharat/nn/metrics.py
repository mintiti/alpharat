"""Training metrics for policy and value learning diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Mapping


def top_k_accuracy(logits: Tensor, target: Tensor, k: int = 1) -> Tensor:
    """Compute top-k accuracy for policy predictions.

    Checks if the target's argmax is in the top-k predictions.

    Args:
        logits: Raw logits from model, shape (batch, num_actions).
        target: Target probabilities, shape (batch, num_actions).
        k: Number of top predictions to consider.

    Returns:
        Scalar tensor with mean accuracy across batch.
    """
    target_argmax = target.argmax(dim=-1)  # (batch,)
    _, top_k_indices = logits.topk(k, dim=-1)  # (batch, k)
    matches = (top_k_indices == target_argmax.unsqueeze(-1)).any(dim=-1)  # (batch,)
    return matches.float().mean()


def policy_entropy(logits: Tensor) -> Tensor:
    """Compute entropy of predicted policy distribution.

    Args:
        logits: Raw logits from model, shape (batch, num_actions).

    Returns:
        Scalar tensor with mean entropy across batch.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (batch,)
    return entropy.mean()


def target_entropy(target: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute entropy of target probability distribution.

    Args:
        target: Target probabilities, shape (batch, num_actions).
        eps: Small constant to avoid log(0).

    Returns:
        Scalar tensor with mean entropy across batch.
    """
    # Clamp to avoid log(0)
    target_clamped = target.clamp(min=eps)
    entropy = -(target * target_clamped.log()).sum(dim=-1)  # (batch,)
    return entropy.mean()


def explained_variance(pred: Tensor, target: Tensor) -> Tensor:
    """Compute explained variance for value predictions.

    EV = max(-1, 1 - Var(target - pred) / Var(target))

    Returns 1.0 for perfect predictions, 0.0 for predicting the mean,
    and clamped to -1.0 for predictions worse than the mean.

    Args:
        pred: Predicted values, shape (batch, ...).
        target: Target values, shape (batch, ...).

    Returns:
        Scalar tensor with explained variance in [-1, 1].
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    target_var = target_flat.var()
    if target_var < 1e-8:
        # Target has no variance — EV is undefined, return 0
        return torch.tensor(0.0, device=pred.device)

    residual_var = (target_flat - pred_flat).var()
    ev = 1.0 - residual_var / target_var
    return torch.clamp(ev, min=-1.0)


def value_correlation(pred: Tensor, target: Tensor) -> Tensor:
    """Compute Pearson correlation between predicted and target values.

    Args:
        pred: Predicted values, shape (batch,).
        target: Target values, shape (batch,).

    Returns:
        Scalar tensor with correlation coefficient.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()

    numerator = (pred_centered * target_centered).sum()
    denominator = (pred_centered.pow(2).sum() * target_centered.pow(2).sum()).sqrt()

    if denominator < 1e-8:
        # No variance — correlation undefined, return 0
        return torch.tensor(0.0, device=pred.device)

    return numerator / denominator


def compute_policy_metrics(logits: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Compute all policy metrics for a batch.

    Args:
        logits: Raw logits from model, shape (batch, 5).
        target: Target probabilities (Nash equilibrium), shape (batch, 5).

    Returns:
        Dict with top1_accuracy, top2_accuracy, entropy_pred, entropy_target as tensors.
    """
    return {
        "top1_accuracy": top_k_accuracy(logits, target, k=1),
        "top2_accuracy": top_k_accuracy(logits, target, k=2),
        "entropy_pred": policy_entropy(logits),
        "entropy_target": target_entropy(target),
    }


def compute_value_metrics(
    pred_v1: Tensor,
    pred_v2: Tensor,
    target_v1: Tensor,
    target_v2: Tensor,
) -> dict[str, Tensor]:
    """Compute metrics for scalar value predictions.

    Args:
        pred_v1: Predicted P1 values, shape (batch,).
        pred_v2: Predicted P2 values, shape (batch,).
        target_v1: Target P1 values, shape (batch,) or (batch, 1).
        target_v2: Target P2 values, shape (batch,) or (batch, 1).

    Returns:
        Dict with explained_variance and correlation for each player's value as tensors.
    """
    target_p1 = target_v1.squeeze()
    target_p2 = target_v2.squeeze()

    return {
        "p1_explained_variance": explained_variance(pred_v1, target_p1),
        "p1_correlation": value_correlation(pred_v1, target_p1),
        "p2_explained_variance": explained_variance(pred_v2, target_p2),
        "p2_correlation": value_correlation(pred_v2, target_p2),
    }


class MetricsAccumulator:
    """Accumulates batch metrics for epoch-level reporting.

    Tracks weighted sums and counts to compute proper averages when
    batch sizes vary (e.g., last batch may be smaller).

    Usage:
        acc = MetricsAccumulator()
        for batch in loader:
            metrics = compute_policy_metrics(logits, target)
            acc.update(metrics, batch_size=len(batch))
        epoch_metrics = acc.compute()
    """

    def __init__(self) -> None:
        """Initialize empty accumulator."""
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: Mapping[str, float | Tensor], batch_size: int = 1) -> None:
        """Add batch metrics weighted by batch size.

        Args:
            metrics: Mapping of metric name to scalar value (float or 0-dim Tensor).
            batch_size: Number of samples in this batch (for weighted average).
        """
        for key, value in metrics.items():
            if key not in self._sums:
                self._sums[key] = 0.0
                self._counts[key] = 0
            # Handle both float and Tensor values
            v = value.item() if isinstance(value, Tensor) else value
            self._sums[key] += v * batch_size
            self._counts[key] += batch_size

    def compute(self) -> dict[str, float]:
        """Compute weighted averages for all accumulated metrics.

        Returns:
            Dict of metric name to epoch-level average.
        """
        return {key: self._sums[key] / self._counts[key] for key in self._sums}

    def reset(self) -> None:
        """Clear all accumulated values for next epoch."""
        self._sums.clear()
        self._counts.clear()


class GPUMetricsAccumulator:
    """Accumulates metrics as GPU tensors, syncs only at compute().

    Unlike MetricsAccumulator which requires .item() calls per batch (forcing
    GPU-CPU sync), this class keeps tensors on GPU and only transfers to CPU
    once at epoch end. This can dramatically improve GPU utilization.

    Usage:
        acc = GPUMetricsAccumulator(device)
        for batch in loader:
            out = compute_losses(...)
            acc.update({
                "loss": out["loss"],  # No .item() - stays on GPU
                "loss_p1": out["loss_p1"],
            }, batch_size=len(batch))
        epoch_metrics = acc.compute()  # Single sync point
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize empty accumulator.

        Args:
            device: GPU device for tensor operations.
        """
        self._device = device
        self._tensors: dict[str, list[Tensor]] = {}
        self._weights: dict[str, list[int]] = {}

    def update(self, metrics: dict[str, Tensor], batch_size: int) -> None:
        """Store detached tensors without syncing to CPU.

        Args:
            metrics: Dict of metric name to scalar tensor (0-dim).
            batch_size: Number of samples in this batch (for weighted average).
        """
        for key, value in metrics.items():
            if key not in self._tensors:
                self._tensors[key] = []
                self._weights[key] = []
            self._tensors[key].append(value.detach())
            self._weights[key].append(batch_size)

    def compute(self) -> dict[str, float]:
        """Compute weighted averages and transfer to CPU.

        This is the single sync point - all accumulated tensors are
        processed on GPU first, then results transferred to CPU.

        Returns:
            Dict of metric name to epoch-level average (as float).
        """
        results: dict[str, float] = {}
        for key in self._tensors:
            stacked = torch.stack(self._tensors[key])
            weights = torch.tensor(self._weights[key], device=self._device, dtype=stacked.dtype)
            weighted_avg = (stacked * weights).sum() / weights.sum()
            results[key] = weighted_avg.item()
        return results

    def reset(self) -> None:
        """Clear all accumulated values for next epoch."""
        self._tensors.clear()
        self._weights.clear()
