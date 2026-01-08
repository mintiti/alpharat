"""Loss computation for MLP architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from alpharat.nn.losses import constant_sum_loss, nash_consistency_loss, sparse_payout_loss
from alpharat.nn.training.keys import LossKey, ModelOutput

if TYPE_CHECKING:
    from alpharat.nn.architectures.mlp.config import MLPOptimConfig


def compute_mlp_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: MLPOptimConfig,
) -> dict[str, torch.Tensor]:
    """Compute losses for MLP model.

    Args:
        model_output: Dict from model.forward() with LOGITS_P1, LOGITS_P2, PAYOUT.
        batch: Training batch with policy_p1, policy_p2, action_p1, action_p2,
            p1_value, p2_value.
        config: MLPOptimConfig with loss weights.

    Returns:
        Dict with LossKey.TOTAL and individual loss components.
    """
    logits_p1 = model_output[ModelOutput.LOGITS_P1]
    logits_p2 = model_output[ModelOutput.LOGITS_P2]
    pred_payout = model_output[ModelOutput.PAYOUT]

    # Policy losses (cross-entropy with soft targets)
    loss_p1 = F.cross_entropy(logits_p1, batch["policy_p1"])
    loss_p2 = F.cross_entropy(logits_p2, batch["policy_p2"])

    # Payout loss (sparse - only at played action pair)
    loss_value = sparse_payout_loss(
        pred_payout,
        batch["action_p1"],
        batch["action_p2"],
        batch["p1_value"],
        batch["p2_value"],
    )

    # Combine base losses
    loss = config.policy_weight * (loss_p1 + loss_p2) + config.value_weight * loss_value

    result: dict[str, torch.Tensor] = {
        LossKey.POLICY_P1: loss_p1,
        LossKey.POLICY_P2: loss_p2,
        LossKey.VALUE: loss_value,
    }

    # Optional Nash consistency loss
    loss_nash = torch.tensor(0.0, device=loss.device)
    loss_indiff = torch.tensor(0.0, device=loss.device)
    loss_dev = torch.tensor(0.0, device=loss.device)

    if config.nash_weight > 0:
        if config.nash_mode == "target":
            pi1 = batch["policy_p1"]
            pi2 = batch["policy_p2"]
        else:  # predicted
            pi1 = F.softmax(logits_p1, dim=-1)
            pi2 = F.softmax(logits_p2, dim=-1)

        loss_nash, loss_indiff, loss_dev = nash_consistency_loss(pred_payout, pi1, pi2)
        loss = loss + config.nash_weight * loss_nash

    result[LossKey.NASH] = loss_nash
    result[LossKey.NASH_INDIFF] = loss_indiff
    result[LossKey.NASH_DEV] = loss_dev

    # Optional constant-sum regularization
    loss_constant_sum = torch.tensor(0.0, device=loss.device)

    if config.constant_sum_weight > 0:
        loss_constant_sum = constant_sum_loss(
            pred_payout,
            batch["p1_value"],
            batch["p2_value"],
        )
        loss = loss + config.constant_sum_weight * loss_constant_sum

    result[LossKey.CONSTANT_SUM] = loss_constant_sum

    # Total loss for backward
    result[LossKey.TOTAL] = loss

    return result
