"""Loss function for SymmetricMLP architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alpharat.nn.architectures.symmetric.config import SymmetricOptimConfig  # noqa: TC001
from alpharat.nn.losses import (
    constant_sum_loss,
    nash_consistency_loss,
    sparse_payout_loss,
)
from alpharat.nn.training.keys import BatchKey, LossKey, ModelOutput


def compute_symmetric_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: SymmetricOptimConfig,
) -> dict[str, torch.Tensor]:
    """Compute losses for SymmetricMLP.

    Args:
        model_output: Dict from model.forward() with logits and payout.
        batch: Training batch with targets.
        config: OptimConfig with loss weights.

    Returns:
        Dict with LossKey keys containing individual and total losses.
    """
    logits_p1 = model_output[ModelOutput.LOGITS_P1]
    logits_p2 = model_output[ModelOutput.LOGITS_P2]
    pred_payout = model_output[ModelOutput.PAYOUT]

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

    # Nash consistency loss (optional)
    if config.nash_weight > 0:
        if config.nash_mode == "predicted":
            pi1 = F.softmax(logits_p1, dim=-1)
            pi2 = F.softmax(logits_p2, dim=-1)
        else:
            pi1 = batch[BatchKey.POLICY_P1]
            pi2 = batch[BatchKey.POLICY_P2]
        loss_nash, loss_indiff, loss_dev = nash_consistency_loss(pred_payout, pi1, pi2)
    else:
        zero = torch.tensor(0.0, device=pred_payout.device)
        loss_nash, loss_indiff, loss_dev = zero, zero, zero

    # Constant-sum regularization (optional)
    if config.constant_sum_weight > 0:
        loss_csum = constant_sum_loss(
            pred_payout, batch[BatchKey.P1_VALUE], batch[BatchKey.P2_VALUE]
        )
    else:
        loss_csum = torch.tensor(0.0, device=pred_payout.device)

    loss = (
        config.policy_weight * (loss_p1 + loss_p2)
        + config.value_weight * loss_value
        + config.nash_weight * loss_nash
        + config.constant_sum_weight * loss_csum
    )

    return {
        LossKey.TOTAL: loss,
        LossKey.POLICY_P1: loss_p1,
        LossKey.POLICY_P2: loss_p2,
        LossKey.VALUE: loss_value,
        LossKey.NASH: loss_nash,
        LossKey.NASH_INDIFF: loss_indiff,
        LossKey.NASH_DEV: loss_dev,
        LossKey.CONSTANT_SUM: loss_csum,
    }
