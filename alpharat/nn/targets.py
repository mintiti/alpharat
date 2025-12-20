"""Training target extraction from game data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alpharat.nn.types import TargetBundle

if TYPE_CHECKING:
    from alpharat.data.types import GameData, PositionData


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

    Args:
        game: Game-level data containing final scores.
        position: Position-level data containing Nash policies and current scores.

    Returns:
        TargetBundle with policy and value targets.
    """
    final_diff = game.final_p1_score - game.final_p2_score
    current_diff = position.p1_score - position.p2_score
    remaining_diff = final_diff - current_diff

    return TargetBundle(
        policy_p1=position.policy_p1.astype(np.float32),
        policy_p2=position.policy_p2.astype(np.float32),
        value=remaining_diff,
    )
