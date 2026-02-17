"""PyRat game helper types.

This module provides convenience types for working with the PyRat game engine.
The main game interface is now `PyRat` from `pyrat_engine.core`.

Types:
    - MoveUndo: Wrapper for undo information with convenient property access
    - GameResult: Named tuple for step() results (optional convenience type)
"""

from dataclasses import dataclass
from typing import NamedTuple

from pyrat_engine.core import MoveUndo as _RustMoveUndo
from pyrat_engine.core.types import Coordinates

__all__ = ["GameResult", "MoveUndo"]


@dataclass(frozen=True)
class MoveUndo:
    """Information needed to undo a move in the game.

    This class stores all state information required to reverse a move,
    enabling the game engine to support move undo/redo functionality.
    This is particularly useful for implementing game tree search algorithms
    and analyzing different game strategies.

    Example:
        >>> from pyrat_engine import PyRat, Direction
        >>> game = PyRat(width=15, height=15)
        >>> # Make a move and store undo information
        >>> undo_info = game.make_move(Direction.RIGHT, Direction.LEFT)
        >>> # Undo the move
        >>> game.unmake_move(undo_info)
    """

    _undo: _RustMoveUndo  # Internal rust undo data

    @property
    def p1_position(self) -> Coordinates:
        """Player 1's position before the move."""
        return self._undo.p1_pos

    @property
    def p2_position(self) -> Coordinates:
        """Player 2's position before the move."""
        return self._undo.p2_pos

    @property
    def scores(self) -> tuple[float, float]:
        """Scores before the move."""
        return (self._undo.p1_score, self._undo.p2_score)

    @property
    def collected_cheese(self) -> list[Coordinates]:
        """Cheese collected during this move."""
        return list(self._undo.collected_cheese)

    @property
    def turn(self) -> int:
        """Turn number before the move."""
        return self._undo.turn

    def __repr__(self) -> str:
        return (
            f"MoveUndo(turn={self.turn}, "
            f"p1_pos={self.p1_position}, "
            f"p2_pos={self.p2_position}, "
            f"scores={self.scores})"
        )


class GameResult(NamedTuple):
    """Result of a game step.

    Contains information about the outcome of a single game step.
    This is an optional convenience type - you can also use the raw
    tuple returned by PyRat.step() directly.

    Attributes:
        game_over: True if the game has ended
        collected_cheese: List of positions where cheese was collected this turn
        p1_score: Player 1's current score
        p2_score: Player 2's current score

    Example:
        >>> from pyrat_engine import PyRat, Direction
        >>> from pyrat_engine.game import GameResult
        >>> game = PyRat(width=15, height=15)
        >>> game_over, collected = game.step(Direction.RIGHT, Direction.LEFT)
        >>> # Or wrap in GameResult for named access:
        >>> result = GameResult(game_over, collected, game.player1_score, game.player2_score)
    """

    game_over: bool
    collected_cheese: list[Coordinates]
    p1_score: float
    p2_score: float
