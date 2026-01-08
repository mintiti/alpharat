"""Sparse payout loss - supervise payout matrix at played action pairs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_payout_loss(
    pred_payout: torch.Tensor,
    action_p1: torch.Tensor,
    action_p2: torch.Tensor,
    p1_value: torch.Tensor,
    p2_value: torch.Tensor,
) -> torch.Tensor:
    """MSE loss at played action pair using actual game outcomes.

    Supervises the NN's payout prediction with ground truth outcomes:
    - p1_value: how much P1 actually scored from this position
    - p2_value: how much P2 actually scored from this position

    Args:
        pred_payout: Predicted bimatrix, shape (batch, 2, 5, 5).
        action_p1: P1's action, shape (batch, 1).
        action_p2: P2's action, shape (batch, 1).
        p1_value: P1's actual remaining score, shape (batch,) or (batch, 1).
        p2_value: P2's actual remaining score, shape (batch,) or (batch, 1).

    Returns:
        Average MSE loss over both players' values at played action.
    """
    n = pred_payout.shape[0]
    batch_idx = torch.arange(n, device=pred_payout.device)
    a1 = action_p1.squeeze(-1).long()
    a2 = action_p2.squeeze(-1).long()

    # Extract predicted values at played action pair for both players
    pred_p1 = pred_payout[batch_idx, 0, a1, a2]
    pred_p2 = pred_payout[batch_idx, 1, a1, a2]

    # Targets are actual game outcomes
    target_p1 = p1_value.squeeze(-1)
    target_p2 = p2_value.squeeze(-1)

    # Average loss over both players
    return 0.5 * (F.mse_loss(pred_p1, target_p1) + F.mse_loss(pred_p2, target_p2))
