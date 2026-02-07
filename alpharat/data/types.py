"""Data types for game recording and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class GameFileKey(StrEnum):
    """Keys for game/bundle npz files.

    Used by GameRecorder (write) and loader (read) to ensure
    consistent key naming in the serialization format.
    """

    # Game-level
    MAZE = "maze"
    INITIAL_CHEESE = "initial_cheese"
    CHEESE_OUTCOMES = "cheese_outcomes"
    MAX_TURNS = "max_turns"
    RESULT = "result"
    FINAL_P1_SCORE = "final_p1_score"
    FINAL_P2_SCORE = "final_p2_score"
    NUM_POSITIONS = "num_positions"

    # Bundle metadata
    GAME_LENGTHS = "game_lengths"

    # Position-level
    P1_POS = "p1_pos"
    P2_POS = "p2_pos"
    P1_SCORE = "p1_score"
    P2_SCORE = "p2_score"
    P1_MUD = "p1_mud"
    P2_MUD = "p2_mud"
    CHEESE_MASK = "cheese_mask"
    TURN = "turn"
    VALUE_P1 = "value_p1"
    VALUE_P2 = "value_p2"
    VISIT_COUNTS_P1 = "visit_counts_p1"
    VISIT_COUNTS_P2 = "visit_counts_p2"
    PRIOR_P1 = "prior_p1"
    PRIOR_P2 = "prior_p2"
    POLICY_P1 = "policy_p1"
    POLICY_P2 = "policy_p2"
    ACTION_P1 = "action_p1"
    ACTION_P2 = "action_p2"


class CheeseOutcome(IntEnum):
    """Per-cheese outcome from P1's perspective.

    Each cheese in the game has exactly one outcome:
    - P1_WIN: P1 collected it alone
    - SIMULTANEOUS: Both players collected it at the same time (0.5 points each)
    - UNCOLLECTED: Game ended before anyone collected it
    - P2_WIN: P2 collected it alone (loss for P1)
    """

    P1_WIN = 0
    SIMULTANEOUS = 1
    UNCOLLECTED = 2
    P2_WIN = 3


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
    value_p1: float  # MCTS root value estimate for P1
    value_p2: float  # MCTS root value estimate for P2
    visit_counts_p1: np.ndarray  # (5,) marginal visit counts for P1
    visit_counts_p2: np.ndarray  # (5,) marginal visit counts for P2
    prior_p1: np.ndarray  # (5,)
    prior_p2: np.ndarray  # (5,)
    policy_p1: np.ndarray  # (5,) visit-proportional policy
    policy_p2: np.ndarray  # (5,) visit-proportional policy
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
    cheese_outcomes: np.ndarray | None = None  # int8[H, W], CheeseOutcome values
