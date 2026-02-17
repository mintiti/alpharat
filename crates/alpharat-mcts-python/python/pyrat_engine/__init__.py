"""PyRat Engine - High-performance PyRat game engine with Python bindings

This package provides a fast implementation of the PyRat game engine written in Rust
with Python bindings. It features both a raw game engine interface and a
PettingZoo-compatible environment.

Key Features:
    - High-performance Rust core engine
    - PettingZoo Parallel environment interface
    - Support for move undo/redo
    - Symmetric and asymmetric maze generation
    - Customizable game parameters

Example:
    >>> from pyrat_engine import PyRat, Direction
    >>> game = PyRat(width=15, height=15)
    >>> # Make moves
    >>> game_over, collected = game.step(Direction.RIGHT, Direction.LEFT)
    >>> # Check game state
    >>> print(f"Player 1 score: {game.player1_score}")
"""

from pyrat_engine.core import MoveUndo, PyRat
from pyrat_engine.core.types import (
    Coordinates,
    Direction,
    Mud,
    Wall,
)
from pyrat_engine.game import GameResult

__version__ = "0.1.0"
__all__ = [
    "Coordinates",
    "Direction",
    "GameResult",
    "MoveUndo",
    "Mud",
    "PyRat",
    "Wall",
]
