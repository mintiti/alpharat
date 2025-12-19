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
        value: Score differential target (final_p1_score - final_p2_score).
    """

    policy_p1: np.ndarray  # float32 (5,)
    policy_p2: np.ndarray  # float32 (5,)
    value: float
