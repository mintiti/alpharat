"""Nash consistency loss - enforce game-theoretic consistency."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nash_consistency_loss(
    pred_payout: torch.Tensor,
    pi1: torch.Tensor,
    pi2: torch.Tensor,
    support_threshold: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce game-theoretic consistency between payout matrix and policies.

    Forces the predicted payout matrix to be a valid game whose Nash equilibrium
    is the given policy. Two components:

    1. Indifference: Actions in support (π > threshold) must have equal expected utility.
    2. No profitable deviation: Actions outside support must not be better
       than the equilibrium value.

    Can be used with either:
    - Target policies (from MCTS): enforces payout consistency with MCTS Nash
    - Predicted policies (from NN): enforces self-consistency between NN heads

    Args:
        pred_payout: Predicted bimatrix, shape (batch, 2, 5, 5).
            pred_payout[:, 0] is P1's payoff, pred_payout[:, 1] is P2's payoff.
        pi1: P1's policy, shape (batch, 5). Either target or predicted.
        pi2: P2's policy, shape (batch, 5). Either target or predicted.
        support_threshold: Actions with π > threshold are considered in support.
            Default 1e-3 corresponds to ~1 MCTS visit out of 1000 simulations.

    Returns:
        Tuple of (total_loss, indifference_loss, deviation_loss).
    """
    # P1's expected payoff per action against P2's strategy
    # pred_payout[:, 0] is [batch, 5, 5], pi2 is [batch, 5]
    # exp1[i] = sum_j P1[i,j] * pi2[j]
    exp1 = torch.einsum("bij,bj->bi", pred_payout[:, 0], pi2)  # [batch, 5]

    # P2's expected payoff per action against P1's strategy
    # pred_payout[:, 1] is [batch, 5, 5] where [i,j] = P2's payoff when P1 plays i, P2 plays j
    # exp2[j] = sum_i P2[i,j] * pi1[i]
    exp2 = torch.einsum("bij,bi->bj", pred_payout[:, 1], pi1)  # [batch, 5]

    # Equilibrium values
    val1 = (pi1 * exp1).sum(dim=-1, keepdim=True)  # [batch, 1]
    val2 = (pi2 * exp2).sum(dim=-1, keepdim=True)  # [batch, 1]

    # Support masks
    support1 = pi1 > support_threshold  # [batch, 5]
    support2 = pi2 > support_threshold  # [batch, 5]

    # Indifference loss: actions in support should have equal expected payoff
    indiff1 = (support1.float() * (exp1 - val1) ** 2).mean()
    indiff2 = (support2.float() * (exp2 - val2) ** 2).mean()
    indifference_loss = indiff1 + indiff2

    # No profitable deviation: actions outside support shouldn't be better than V
    outside1 = ~support1
    outside2 = ~support2
    dev1 = (outside1.float() * F.relu(exp1 - val1) ** 2).mean()
    dev2 = (outside2.float() * F.relu(exp2 - val2) ** 2).mean()
    deviation_loss = dev1 + dev2

    total_loss = indifference_loss + deviation_loss

    return total_loss, indifference_loss, deviation_loss
