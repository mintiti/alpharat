"""ObservationInput extraction from game data and live games."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alpharat.nn.types import ObservationInput

if TYPE_CHECKING:
    from alpharat.data.types import GameData, PositionData


def from_game_arrays(game: GameData, position: PositionData) -> ObservationInput:
    """Extract ObservationInput from stored game data.

    Combines game-level data (maze, dimensions, max_turns) with
    position-level data (positions, scores, cheese, mud, turn).

    Args:
        game: Game-level data containing maze and configuration.
        position: Position-level data for a single turn.

    Returns:
        ObservationInput ready for observation building.
    """
    # Build cheese mask from position list
    cheese_mask = _cheese_positions_to_mask(position.cheese_positions, game.height, game.width)

    return ObservationInput(
        maze=game.maze,
        p1_pos=position.p1_pos,
        p2_pos=position.p2_pos,
        cheese_mask=cheese_mask,
        p1_score=position.p1_score,
        p2_score=position.p2_score,
        turn=position.turn,
        max_turns=game.max_turns,
        p1_mud=position.p1_mud,
        p2_mud=position.p2_mud,
        width=game.width,
        height=game.height,
    )


def from_pyrat_game(game: Any, maze: np.ndarray, max_turns: int) -> ObservationInput:
    """Extract ObservationInput from a live PyRat game instance.

    Used during inference/self-play. The maze array should be pre-built
    once at game start using build_maze_array() since it doesn't change.

    Args:
        game: Live PyRat game instance.
        maze: Pre-built maze array from build_maze_array(game, width, height).
        max_turns: Maximum turns for the game (typically game.max_turns).

    Returns:
        ObservationInput ready for observation building.
    """
    height, width = maze.shape[:2]

    # Extract positions from Coordinates objects
    p1_pos = (game.player1_position.x, game.player1_position.y)
    p2_pos = (game.player2_position.x, game.player2_position.y)

    # Build cheese mask from Coordinates list
    cheese_positions = [(c.x, c.y) for c in game.cheese_positions()]
    cheese_mask = _cheese_positions_to_mask(cheese_positions, height, width)

    return ObservationInput(
        maze=maze,
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        cheese_mask=cheese_mask,
        p1_score=float(game.player1_score),
        p2_score=float(game.player2_score),
        turn=int(game.turn),
        max_turns=max_turns,
        p1_mud=int(game.player1_mud_turns),
        p2_mud=int(game.player2_mud_turns),
        width=width,
        height=height,
    )


def _cheese_positions_to_mask(
    cheese_positions: list[tuple[int, int]], height: int, width: int
) -> np.ndarray:
    """Convert cheese position list to boolean mask.

    Args:
        cheese_positions: List of (x, y) cheese positions.
        height: Maze height.
        width: Maze width.

    Returns:
        Boolean array of shape (height, width) with True at cheese locations.
    """
    mask = np.zeros((height, width), dtype=bool)
    for x, y in cheese_positions:
        mask[y, x] = True
    return mask
