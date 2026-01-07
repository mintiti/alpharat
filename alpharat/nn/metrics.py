"""Training metrics for policy and value learning diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


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


def payout_correlation(pred: Tensor, target: Tensor) -> Tensor:
    """Compute Pearson correlation between predicted and target payout matrices.

    Args:
        pred: Predicted payout matrices, shape (batch, 5, 5).
        target: Target payout matrices, shape (batch, 5, 5).

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


def compute_policy_metrics(logits: Tensor, target: Tensor) -> dict[str, float]:
    """Compute all policy metrics for a batch.

    Args:
        logits: Raw logits from model, shape (batch, 5).
        target: Target probabilities (Nash equilibrium), shape (batch, 5).

    Returns:
        Dict with top1_accuracy, top2_accuracy, entropy_pred, entropy_target.
    """
    return {
        "top1_accuracy": top_k_accuracy(logits, target, k=1).item(),
        "top2_accuracy": top_k_accuracy(logits, target, k=2).item(),
        "entropy_pred": policy_entropy(logits).item(),
        "entropy_target": target_entropy(target).item(),
    }


def compute_payout_metrics(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Compute payout matrix metrics for a batch, per player.

    Args:
        pred: Predicted payout matrices, shape (batch, 2, 5, 5).
        target: Target payout matrices, shape (batch, 2, 5, 5).

    Returns:
        Dict with explained_variance and correlation for each player.
    """
    return {
        "p1_explained_variance": explained_variance(pred[:, 0], target[:, 0]).item(),
        "p1_correlation": payout_correlation(pred[:, 0], target[:, 0]).item(),
        "p2_explained_variance": explained_variance(pred[:, 1], target[:, 1]).item(),
        "p2_correlation": payout_correlation(pred[:, 1], target[:, 1]).item(),
    }


def compute_value_metrics(
    pred_payout: Tensor,
    action_p1: Tensor,
    action_p2: Tensor,
    p1_value: Tensor,
    p2_value: Tensor,
) -> dict[str, float]:
    """Compute metrics for value predictions at played action pair.

    Compares pred_payout[:, player, a1, a2] to actual game outcomes.

    Args:
        pred_payout: Predicted payout matrices, shape (batch, 2, 5, 5).
        action_p1: P1 action indices, shape (batch,) or (batch, 1).
        action_p2: P2 action indices, shape (batch,) or (batch, 1).
        p1_value: P1's actual remaining score, shape (batch,) or (batch, 1).
        p2_value: P2's actual remaining score, shape (batch,) or (batch, 1).

    Returns:
        Dict with explained_variance and correlation for each player's value.
    """
    batch_size = pred_payout.shape[0]
    batch_idx = torch.arange(batch_size, device=pred_payout.device)

    a1 = action_p1.squeeze().long()
    a2 = action_p2.squeeze().long()

    pred_p1 = pred_payout[batch_idx, 0, a1, a2]
    pred_p2 = pred_payout[batch_idx, 1, a1, a2]
    target_p1 = p1_value.squeeze()
    target_p2 = p2_value.squeeze()

    return {
        "p1_explained_variance": explained_variance(pred_p1, target_p1).item(),
        "p1_correlation": payout_correlation(pred_p1, target_p1).item(),
        "p2_explained_variance": explained_variance(pred_p2, target_p2).item(),
        "p2_correlation": payout_correlation(pred_p2, target_p2).item(),
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

    def update(self, metrics: dict[str, float], batch_size: int = 1) -> None:
        """Add batch metrics weighted by batch size.

        Args:
            metrics: Dict of metric name to scalar value.
            batch_size: Number of samples in this batch (for weighted average).
        """
        for key, value in metrics.items():
            if key not in self._sums:
                self._sums[key] = 0.0
                self._counts[key] = 0
            self._sums[key] += value * batch_size
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
