"""Data types for neural network inputs and targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: TC002 - needed at runtime for dataclass fields


@dataclass
class ObservationInput:
    """Source-agnostic input for observation building.

    Intermediate representation holding everything needed to build an
    observation tensor. Can be constructed from stored game data (training)
    or live PyRat game (inference).

    Attributes:
        maze: Adjacency costs, shape (H, W, 4). Values: -1=wall, 1=normal, >1=mud.
        p1_pos: Player 1 position as (x, y).
        p2_pos: Player 2 position as (x, y).
        cheese_mask: Boolean mask of cheese locations, shape (H, W).
        p1_score: Player 1 current score.
        p2_score: Player 2 current score.
        turn: Current turn number.
        max_turns: Maximum turns in the game.
        p1_mud: Turns remaining in mud for player 1.
        p2_mud: Turns remaining in mud for player 2.
        width: Maze width.
        height: Maze height.
    """

    maze: np.ndarray  # int8 (H, W, 4)
    p1_pos: tuple[int, int]
    p2_pos: tuple[int, int]
    cheese_mask: np.ndarray  # bool (H, W)
    p1_score: float
    p2_score: float
    turn: int
    max_turns: int
    p1_mud: int
    p2_mud: int
    width: int
    height: int


@dataclass
class TargetBundle:
    """Training targets for a single position.

    Attributes:
        policy_p1: Nash policy for player 1, shape (5,). Sums to 1.
        policy_p2: Nash policy for player 2, shape (5,). Sums to 1.
        p1_value: P1's remaining score (final_p1_score - current_p1_score).
            Ground truth outcome for P1 from this position.
        p2_value: P2's remaining score (final_p2_score - current_p2_score).
            Ground truth outcome for P2 from this position.
        payout_matrix: MCTS refined bimatrix, shape (2, 5, 5).
            [0] = P1's payoffs, [1] = P2's payoffs.
        action_p1: Action taken by player 1 (0-4).
        action_p2: Action taken by player 2 (0-4).
        cheese_outcomes: Per-cell ownership outcomes, shape (H, W).
            Position-level targets for the local value head.

            Values:
                -1 = No active cheese (cell never had cheese, or cheese
                     was already collected before this position). Skip
                     in loss computation.
                 0 = P1_WIN: P1 will collect this cheese alone
                 1 = SIMULTANEOUS: Both players collect at same time
                 2 = UNCOLLECTED: Nobody collects before game end
                 3 = P2_WIN: P2 will collect this cheese alone

            At training time, derive the loss mask as: cheese_outcomes >= 0
    """

    policy_p1: np.ndarray  # float32 (5,)
    policy_p2: np.ndarray  # float32 (5,)
    p1_value: float  # P1's actual remaining score
    p2_value: float  # P2's actual remaining score
    payout_matrix: np.ndarray  # float32 (2, 5, 5)
    action_p1: int
    action_p2: int
    cheese_outcomes: np.ndarray  # int8 (H, W), -1 sentinel for inactive cells
