"""Loss function for LocalValueMLP architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alpharat.nn.architectures.local_value.config import LocalValueOptimConfig  # noqa: TC001
from alpharat.nn.losses import compute_ownership_loss, sparse_payout_loss
from alpharat.nn.training.keys import BatchKey, LossKey, ModelOutput


def compute_local_value_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LocalValueOptimConfig,
) -> dict[str, torch.Tensor]:
    """Compute losses for LocalValueMLP.

    Args:
        model_output: Dict from model.forward() with logits, payout, ownership.
        batch: Training batch with targets.
        config: OptimConfig with loss weights.

    Returns:
        Dict with LossKey keys containing individual and total losses.
    """
    logits_p1 = model_output[ModelOutput.LOGITS_P1]
    logits_p2 = model_output[ModelOutput.LOGITS_P2]
    pred_payout = model_output[ModelOutput.PAYOUT]
    ownership_logits = model_output[ModelOutput.OWNERSHIP_LOGITS]

    # Policy losses
    loss_p1 = F.cross_entropy(logits_p1, batch[BatchKey.POLICY_P1])
    loss_p2 = F.cross_entropy(logits_p2, batch[BatchKey.POLICY_P2])

    # Sparse payout loss
    loss_value = sparse_payout_loss(
        pred_payout,
        batch[BatchKey.ACTION_P1],
        batch[BatchKey.ACTION_P2],
        batch[BatchKey.P1_VALUE],
        batch[BatchKey.P2_VALUE],
    )

    # Ownership loss (auxiliary task)
    loss_ownership = compute_ownership_loss(ownership_logits, batch[BatchKey.CHEESE_OUTCOMES])

    loss = (
        config.policy_weight * (loss_p1 + loss_p2)
        + config.value_weight * loss_value
        + config.ownership_weight * loss_ownership
    )

    return {
        LossKey.TOTAL: loss,
        LossKey.POLICY_P1: loss_p1,
        LossKey.POLICY_P2: loss_p2,
        LossKey.VALUE: loss_value,
        LossKey.OWNERSHIP: loss_ownership,
    }
