"""Constant-sum regularization loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def constant_sum_loss(
    pred_payout: torch.Tensor,
    p1_value: torch.Tensor,
    p2_value: torch.Tensor,
) -> torch.Tensor:
    """Regularize payout matrix toward constant-sum (approximately zero-sum game).

    PyRat is approximately constant-sum: total cheese collected is bounded by
    remaining cheese. This loss encourages all action pairs to sum to approximately
    the same value (the total cheese collected in the actual game).

    Args:
        pred_payout: Predicted bimatrix, shape (batch, 2, 5, 5).
        p1_value: P1's actual cheese gained, shape (batch,) or (batch, 1).
        p2_value: P2's actual cheese gained, shape (batch,) or (batch, 1).

    Returns:
        MSE loss between sum of predicted payouts and total collected.
    """
    # Sum of both players' predicted payouts for each action pair
    sum_payout = pred_payout[:, 0] + pred_payout[:, 1]  # [batch, 5, 5]

    # Target: total cheese collected in the actual game
    total_collected = p1_value.squeeze(-1) + p2_value.squeeze(-1)  # [batch]

    # MSE between predicted sum and actual total (broadcast across all action pairs)
    return F.mse_loss(sum_payout, total_collected.view(-1, 1, 1).expand_as(sum_payout))
