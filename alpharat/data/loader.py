"""Game data loading from npz files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from alpharat.data.types import GameData, PositionData

if TYPE_CHECKING:
    from collections.abc import Iterator


def is_bundle_file(path: Path | str) -> bool:
    """Check if an npz file is a bundle (vs single game).

    Args:
        path: Path to the npz file.

    Returns:
        True if the file contains a 'game_lengths' array (bundle format).
    """
    path = Path(path)
    with np.load(path) as data:
        return "game_lengths" in data.files


def load_game_data(path: Path | str) -> GameData:
    """Load game data from an npz file.

    Reconstructs the GameData and PositionData dataclasses from the
    serialized numpy arrays.

    Args:
        path: Path to the npz file.

    Returns:
        GameData with all positions reconstructed.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If required arrays are missing.
    """
    path = Path(path)
    data = np.load(path)

    # Extract game-level data
    maze = data["maze"]
    height, width = maze.shape[:2]

    initial_cheese = data["initial_cheese"]
    cheese_outcomes = data["cheese_outcomes"].copy()
    max_turns = int(data["max_turns"])
    result = int(data["result"])
    final_p1_score = float(data["final_p1_score"])
    final_p2_score = float(data["final_p2_score"])
    num_positions = int(data["num_positions"])

    # Reconstruct positions
    positions: list[PositionData] = []
    for i in range(num_positions):
        position = PositionData(
            p1_pos=(int(data["p1_pos"][i, 0]), int(data["p1_pos"][i, 1])),
            p2_pos=(int(data["p2_pos"][i, 0]), int(data["p2_pos"][i, 1])),
            p1_score=float(data["p1_score"][i]),
            p2_score=float(data["p2_score"][i]),
            p1_mud=int(data["p1_mud"][i]),
            p2_mud=int(data["p2_mud"][i]),
            cheese_positions=_mask_to_cheese(data["cheese_mask"][i]),
            turn=int(data["turn"][i]),
            value_p1=float(data["value_p1"][i]),
            value_p2=float(data["value_p2"][i]),
            visit_counts_p1=data["visit_counts_p1"][i].copy(),
            visit_counts_p2=data["visit_counts_p2"][i].copy(),
            prior_p1=data["prior_p1"][i].copy(),
            prior_p2=data["prior_p2"][i].copy(),
            policy_p1=data["policy_p1"][i].copy(),
            policy_p2=data["policy_p2"][i].copy(),
            action_p1=int(data["action_p1"][i]),
            action_p2=int(data["action_p2"][i]),
        )
        positions.append(position)

    return GameData(
        maze=maze.copy(),
        initial_cheese=initial_cheese.copy(),
        max_turns=max_turns,
        width=width,
        height=height,
        positions=positions,
        result=result,
        final_p1_score=final_p1_score,
        final_p2_score=final_p2_score,
        cheese_outcomes=cheese_outcomes,
    )


def _mask_to_cheese(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert bool[H, W] mask to list of (x, y) positions.

    Args:
        mask: Boolean array of shape (height, width).

    Returns:
        List of (x, y) tuples where mask is True.
    """
    ys, xs = np.where(mask)
    return [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]


def load_game_bundle(path: Path | str) -> list[GameData]:
    """Load all games from a bundle npz file.

    Args:
        path: Path to the bundle npz file.

    Returns:
        List of GameData objects, one per game in the bundle.

    Raises:
        KeyError: If required arrays are missing.
        ValueError: If the file is not a bundle format.
    """
    return list(iter_games_from_bundle(path))


def iter_games_from_bundle(path: Path | str) -> Iterator[GameData]:
    """Iterate over games in a bundle npz file.

    Memory-efficient: yields one GameData at a time, though the entire
    npz is loaded into memory (numpy limitation).

    Args:
        path: Path to the bundle npz file.

    Yields:
        GameData objects, one per game in the bundle.

    Raises:
        KeyError: If required arrays are missing.
        ValueError: If the file is not a bundle format.
    """
    path = Path(path)
    data = np.load(path)

    if "game_lengths" not in data.files:
        raise ValueError(f"Not a bundle file (no game_lengths): {path}")

    game_lengths = data["game_lengths"]
    num_games = len(game_lengths)

    # Compute position offsets for slicing
    pos_offsets = np.zeros(num_games + 1, dtype=np.int64)
    pos_offsets[1:] = np.cumsum(game_lengths)

    # Extract arrays (all loaded, but we slice per game)
    maze_all = data["maze"]
    initial_cheese_all = data["initial_cheese"]
    cheese_outcomes_all = data["cheese_outcomes"]
    max_turns_all = data["max_turns"]
    result_all = data["result"]
    final_p1_score_all = data["final_p1_score"]
    final_p2_score_all = data["final_p2_score"]

    p1_pos_all = data["p1_pos"]
    p2_pos_all = data["p2_pos"]
    p1_score_all = data["p1_score"]
    p2_score_all = data["p2_score"]
    p1_mud_all = data["p1_mud"]
    p2_mud_all = data["p2_mud"]
    cheese_mask_all = data["cheese_mask"]
    turn_all = data["turn"]
    value_p1_all = data["value_p1"]
    value_p2_all = data["value_p2"]
    visit_counts_p1_all = data["visit_counts_p1"]
    visit_counts_p2_all = data["visit_counts_p2"]
    prior_p1_all = data["prior_p1"]
    prior_p2_all = data["prior_p2"]
    policy_p1_all = data["policy_p1"]
    policy_p2_all = data["policy_p2"]
    action_p1_all = data["action_p1"]
    action_p2_all = data["action_p2"]

    for gi in range(num_games):
        start = pos_offsets[gi]
        end = pos_offsets[gi + 1]

        maze = maze_all[gi]
        height, width = maze.shape[:2]

        # Reconstruct positions for this game
        positions: list[PositionData] = []
        for pi in range(start, end):
            position = PositionData(
                p1_pos=(int(p1_pos_all[pi, 0]), int(p1_pos_all[pi, 1])),
                p2_pos=(int(p2_pos_all[pi, 0]), int(p2_pos_all[pi, 1])),
                p1_score=float(p1_score_all[pi]),
                p2_score=float(p2_score_all[pi]),
                p1_mud=int(p1_mud_all[pi]),
                p2_mud=int(p2_mud_all[pi]),
                cheese_positions=_mask_to_cheese(cheese_mask_all[pi]),
                turn=int(turn_all[pi]),
                value_p1=float(value_p1_all[pi]),
                value_p2=float(value_p2_all[pi]),
                visit_counts_p1=visit_counts_p1_all[pi].copy(),
                visit_counts_p2=visit_counts_p2_all[pi].copy(),
                prior_p1=prior_p1_all[pi].copy(),
                prior_p2=prior_p2_all[pi].copy(),
                policy_p1=policy_p1_all[pi].copy(),
                policy_p2=policy_p2_all[pi].copy(),
                action_p1=int(action_p1_all[pi]),
                action_p2=int(action_p2_all[pi]),
            )
            positions.append(position)

        yield GameData(
            maze=maze.copy(),
            initial_cheese=initial_cheese_all[gi].copy(),
            max_turns=int(max_turns_all[gi]),
            width=width,
            height=height,
            positions=positions,
            result=int(result_all[gi]),
            final_p1_score=float(final_p1_score_all[gi]),
            final_p2_score=float(final_p2_score_all[gi]),
            cheese_outcomes=cheese_outcomes_all[gi].copy(),
        )
