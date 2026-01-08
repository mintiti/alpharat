"""Maze array building utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from pyrat_engine.core.types import Coordinates, Direction

# Delta tuple -> Direction mapping (Y-up coordinate system: UP increases Y)
_DELTA_TO_DIRECTION: dict[tuple[int, int], Direction] = {
    (0, 1): Direction.UP,
    (1, 0): Direction.RIGHT,
    (0, -1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
}


def build_maze_array(game: Any, width: int, height: int) -> np.ndarray:
    """Build maze adjacency array from game's wall/mud data.

    Queries the PyRat game instance for walls and mud, building a dense
    adjacency representation suitable for neural network observation encoding.

    Args:
        game: PyRat game instance with wall_entries() and mud_entries() methods.
        width: Maze width.
        height: Maze height.

    Returns:
        int8 array of shape (H, W, 4) where maze[y, x, d] is:
        -1 if there's a wall in direction d from (x, y)
        1 if traversable in 1 turn (normal)
        >1 if traversable in N turns (mud)

        Directions indexed by Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
    """
    # Default: all interior cells can move in all directions (cost 1)
    maze = np.ones((height, width, 4), dtype=np.int8)

    # Mark edge boundaries as walls (-1)
    # Y-up coordinate system: y=0 is bottom, y=height-1 is top
    maze[:, 0, Direction.LEFT] = -1
    maze[:, width - 1, Direction.RIGHT] = -1
    maze[0, :, Direction.DOWN] = -1  # Bottom edge: can't go DOWN
    maze[height - 1, :, Direction.UP] = -1  # Top edge: can't go UP

    # Mark walls from game.wall_entries()
    for wall in game.wall_entries():
        d1 = _coords_to_direction(wall.pos1, wall.pos2)
        d2 = _opposite_direction(d1)
        maze[wall.pos1.y, wall.pos1.x, d1] = -1
        maze[wall.pos2.y, wall.pos2.x, d2] = -1

    # Mark mud costs from game.mud_entries()
    for mud in game.mud_entries():
        d1 = _coords_to_direction(mud.pos1, mud.pos2)
        d2 = _opposite_direction(d1)
        maze[mud.pos1.y, mud.pos1.x, d1] = mud.value
        maze[mud.pos2.y, mud.pos2.x, d2] = mud.value

    return maze


def _coords_to_direction(from_pos: Coordinates, to_pos: Coordinates) -> Direction:
    """Determine direction from from_pos to adjacent to_pos.

    Uses Coordinates subtraction to get the delta tuple.

    Args:
        from_pos: Starting position.
        to_pos: Adjacent target position.

    Returns:
        Direction enum value.

    Raises:
        ValueError: If positions are not adjacent.
    """
    delta = to_pos - from_pos
    direction = _DELTA_TO_DIRECTION.get(delta)
    if direction is None:
        raise ValueError(f"Non-adjacent positions: {from_pos} -> {to_pos}")
    return direction


def _opposite_direction(d: Direction) -> Direction:
    """Get the opposite direction.

    Args:
        d: Direction enum value.

    Returns:
        Opposite direction.
    """
    return Direction((d + 2) % 4)
