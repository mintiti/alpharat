"""Training target extraction from game data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alpharat.nn.types import TargetBundle

if TYPE_CHECKING:
    from alpharat.data.types import GameData, PositionData

# Sentinel value for cells with no active cheese at this position.
# Used to mask ownership loss: only cells with cheese_outcomes >= 0 contribute.
CHEESE_INACTIVE = -1


def build_targets(game: GameData, position: PositionData) -> TargetBundle:
    """Build training targets for a single position.

    Policy targets come from the Nash equilibrium computed by MCTS
    at this position (stored in position.policy_p1/p2).

    Value target is the remaining score differential from this position
    to game end. This is the expected future reward sum with gamma=1:
        V(s) = final_diff - current_diff
             = (final_p1 - final_p2) - (current_p1 - current_p2)

    A value of +2.0 means P1 will gain 2 more cheese than P2 from
    this position onwards.

    Cheese outcomes are masked to only include cheese still on the board
    at this position. Cells without active cheese get CHEESE_INACTIVE (-1).
    At training time, derive the loss mask as: cheese_outcomes >= 0.

    Args:
        game: Game-level data containing final scores and cheese outcomes.
        position: Position-level data containing Nash policies and current scores.

    Returns:
        TargetBundle with policy, value, payout_matrix, action, and cheese targets.
    """
    final_diff = game.final_p1_score - game.final_p2_score
    current_diff = position.p1_score - position.p2_score
    remaining_diff = final_diff - current_diff

    assert game.cheese_outcomes is not None

    # Build position-level cheese outcomes:
    # - Start with -1 (inactive) for all cells
    # - Copy actual outcomes only for cheese still on the board
    # This way, the loss mask is simply: cheese_outcomes >= 0
    cheese_outcomes = np.full((game.height, game.width), CHEESE_INACTIVE, dtype=np.int8)
    for x, y in position.cheese_positions:
        cheese_outcomes[y, x] = game.cheese_outcomes[y, x]

    return TargetBundle(
        policy_p1=position.policy_p1.astype(np.float32),
        policy_p2=position.policy_p2.astype(np.float32),
        value=remaining_diff,
        payout_matrix=position.payout_matrix.astype(np.float32),
        action_p1=position.action_p1,
        action_p2=position.action_p2,
        cheese_outcomes=cheese_outcomes,
    )
