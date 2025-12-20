"""Data types for game recording and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PositionData:
    """Data captured at a single game position.

    Stores game state and MCTS outputs for one turn. Cheese positions
    are stored as a list and converted to a mask at save time.
    """

    p1_pos: tuple[int, int]
    p2_pos: tuple[int, int]
    p1_score: float
    p2_score: float
    p1_mud: int
    p2_mud: int
    cheese_positions: list[tuple[int, int]]
    turn: int
    payout_matrix: np.ndarray  # (5, 5)
    visit_counts: np.ndarray  # (5, 5)
    prior_p1: np.ndarray  # (5,)
    prior_p2: np.ndarray  # (5,)
    policy_p1: np.ndarray  # (5,)
    policy_p2: np.ndarray  # (5,)
    action_p1: int  # action taken by player 1 (0-4)
    action_p2: int  # action taken by player 2 (0-4)


@dataclass
class GameData:
    """Complete data for a single game.

    Game-level data is captured once at start (maze, initial cheese).
    Position-level data accumulates turn by turn.
    Final result and scores are set at game end.
    """

    maze: np.ndarray  # int8[H, W, 4]
    initial_cheese: np.ndarray  # bool[H, W]
    max_turns: int
    width: int
    height: int
    positions: list[PositionData] = field(default_factory=list)
    result: int = 0  # 1=P1 win, 2=P2 win, 0=draw
    final_p1_score: float = 0.0
    final_p2_score: float = 0.0
