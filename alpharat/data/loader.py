"""Game data loading from npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alpharat.data.types import GameData, PositionData


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
            payout_matrix=data["payout_matrix"][i].copy(),
            visit_counts=data["visit_counts"][i].copy(),
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
