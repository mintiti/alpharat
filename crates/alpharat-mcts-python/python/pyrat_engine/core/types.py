"""Core types for the PyRat game engine.

This module provides the fundamental types used throughout the engine:
- Coordinates: Position on the game board (from Rust)
- Direction: Movement directions (Python IntEnum)
- Wall: Barriers between cells (from Rust)
- Mud: Passages that slow movement (from Rust)
"""

from enum import IntEnum

# Import the compiled module directly
import pyrat_engine._core as _impl

# Re-export Rust type classes
Coordinates = _impl.types.Coordinates
Wall = _impl.types.Wall
Mud = _impl.types.Mud


class Direction(IntEnum):
    """Movement directions in the game.

    This is an IntEnum with the following values:
    - UP = 0: Move up (increase y)
    - RIGHT = 1: Move right (increase x)
    - DOWN = 2: Move down (decrease y)
    - LEFT = 3: Move left (decrease x)
    - STAY = 4: Don't move

    As an IntEnum, Direction members are also integers:
    - Direction.UP == 0 is True
    - isinstance(Direction.UP, int) is True

    Example:
        >>> Direction.UP
        <Direction.UP: 0>
        >>> Direction.UP.name
        'UP'
        >>> Direction.UP.value
        0
        >>> Direction["UP"]
        <Direction.UP: 0>
        >>> Direction(0)
        <Direction.UP: 0>
        >>> list(Direction)
        [<Direction.UP: 0>, <Direction.RIGHT: 1>, ...]
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


__all__ = [
    "Coordinates",
    "Direction",
    "Mud",
    "Wall",
]
