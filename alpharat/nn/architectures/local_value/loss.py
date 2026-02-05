"""Loss function for LocalValueMLP architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from alpharat.nn.losses import compute_ownership_loss
from alpharat.nn.training.keys import BatchKey, LossKey, ModelOutput

if TYPE_CHECKING:
    from alpharat.nn.architectures.local_value.config import LocalValueOptimConfig


def compute_local_value_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LocalValueOptimConfig,
) -> dict[str, torch.Tensor]:
    """Compute losses for LocalValueMLP.

    Args:
        model_output: Dict from model.forward() with LOGITS_P1, LOGITS_P2,
            VALUE_P1, VALUE_P2, OWNERSHIP_LOGITS.
        batch: Training batch with policy_p1, policy_p2, p1_value, p2_value,
            cheese_outcomes.
        config: LocalValueOptimConfig with loss weights.

    Returns:
        Dict with LossKey.TOTAL and individual loss components.
    """
    logits_p1 = model_output[ModelOutput.LOGITS_P1]
    logits_p2 = model_output[ModelOutput.LOGITS_P2]
    pred_v1 = model_output[ModelOutput.VALUE_P1]
    pred_v2 = model_output[ModelOutput.VALUE_P2]
    ownership_logits = model_output[ModelOutput.OWNERSHIP_LOGITS]

    # Policy losses (cross-entropy with soft targets)
    loss_p1 = F.cross_entropy(logits_p1, batch[BatchKey.POLICY_P1])
    loss_p2 = F.cross_entropy(logits_p2, batch[BatchKey.POLICY_P2])

    # Value losses (MSE on scalar values)
    target_v1 = batch[BatchKey.VALUE_P1].squeeze(-1)
    target_v2 = batch[BatchKey.VALUE_P2].squeeze(-1)
    loss_v1 = F.mse_loss(pred_v1, target_v1)
    loss_v2 = F.mse_loss(pred_v2, target_v2)
    loss_value = 0.5 * (loss_v1 + loss_v2)

    # Ownership loss (auxiliary task)
    loss_ownership = compute_ownership_loss(ownership_logits, batch[BatchKey.CHEESE_OUTCOMES])

    # Combine losses
    loss = (
        config.policy_weight * (loss_p1 + loss_p2)
        + config.value_weight * loss_value
        + config.ownership_weight * loss_ownership
    )

    result: dict[str, torch.Tensor] = {
        LossKey.POLICY_P1: loss_p1,
        LossKey.POLICY_P2: loss_p2,
        LossKey.VALUE: loss_value,
        LossKey.VALUE_P1: loss_v1,
        LossKey.VALUE_P2: loss_v2,
        LossKey.OWNERSHIP: loss_ownership,
        LossKey.TOTAL: loss,
    }

    return result
