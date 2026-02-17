"""Batched predict_fn adapter for Rust MCTS.

The Rust MCTS calls predict_fn(list[PyRat]) -> 4-tuple of numpy arrays.
This module adapts existing single-game NN inference into batched form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from alpharat.data.maze import build_maze_array
from alpharat.nn.extraction import from_pyrat_game

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def make_batched_predict_fn(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Callable[[list[Any]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create a batched predict_fn for the Rust MCTS binding.

    Loads a model from checkpoint and returns a callable that takes a list of
    PyRat game states and returns batched numpy arrays.

    Args:
        checkpoint_path: Path to the NN checkpoint.
        device: Device for inference ("cpu", "cuda", "mps").

    Returns:
        Callable with signature:
            (games: list[PyRat]) -> (policy_p1[N,5], policy_p2[N,5],
                                     value_p1[N], value_p2[N])
        All arrays are float32.
    """
    import torch

    from alpharat.config.checkpoint import load_model_from_checkpoint
    from alpharat.nn.training_utils import select_device

    model, builder, width, height = load_model_from_checkpoint(
        checkpoint_path, device=device, compile_model=True
    )
    resolved_device = select_device(device)

    def predict_fn(
        games: list[Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Batched NN evaluation over a list of PyRat game states."""
        # Build observation for each game state.
        # Maze is rebuilt per game because leaf states may differ in cheese/topology.
        # For same-maze games this is redundant but cheap.
        observations = []
        for game in games:
            maze = build_maze_array(game, width, height)
            obs_input = from_pyrat_game(game, maze, game.max_turns)
            obs = builder.build(obs_input)
            observations.append(obs)

        # Stack into batch tensor
        batch = np.stack(observations)
        batch_tensor = torch.from_numpy(batch).to(resolved_device)

        with torch.inference_mode():
            result = model.predict(batch_tensor)  # type: ignore[operator]
            policy_p1 = result["policy_p1"].cpu().numpy().astype(np.float32)
            policy_p2 = result["policy_p2"].cpu().numpy().astype(np.float32)
            value_p1 = result["pred_value_p1"].reshape(-1).cpu().numpy().astype(np.float32)
            value_p2 = result["pred_value_p2"].reshape(-1).cpu().numpy().astype(np.float32)

        return policy_p1, policy_p2, value_p1, value_p2

    return predict_fn
