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

    Value target is the final score differential from the game
    outcome (final_p1_score - final_p2_score). This is the same
    for all positions in a game - the model learns to predict
    the game's eventual outcome from any position.

    Args:
        game: Game-level data containing final scores.
        position: Position-level data containing Nash policies.

    Returns:
        TargetBundle with policy and value targets.
    """
    return TargetBundle(
        policy_p1=position.policy_p1.astype(np.float32),
        policy_p2=position.policy_p2.astype(np.float32),
        value=game.final_p1_score - game.final_p2_score,
    )
