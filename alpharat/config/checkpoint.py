"""Checkpoint loading utilities using ModelConfig.build_model().

This module provides a centralized way to load models from checkpoints without
manual dispatch based on model type. The checkpoint contains the full model_config,
which is parsed back into a ModelConfig using Pydantic's discriminated union,
then build_model() constructs the correct architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from pyrat_engine.core.game import PyRat
    from torch import nn

    from alpharat.nn.builders import ObservationBuilder
    from alpharat.nn.training.protocols import TrainableModel


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
    compile_model: bool = True,
) -> tuple[nn.Module, ObservationBuilder, int, int]:
    """Load a model from checkpoint using ModelConfig.build_model().

    This function:
    1. Loads the checkpoint
    2. Parses config["model"] as a ModelConfig using discriminated union
    3. Calls model_config.build_model() to construct the correct architecture
    4. Loads the state dict

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load model onto ("cpu", "cuda", "mps").
        compile_model: Whether to torch.compile the model (CUDA only).

    Returns:
        Tuple of (model, observation_builder, width, height).

    Raises:
        ValueError: If checkpoint doesn't contain required config.
    """
    from pydantic import TypeAdapter

    from alpharat.nn.config import ModelConfig

    checkpoint_path = Path(checkpoint_path)
    torch_device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)

    # Extract dimensions
    width = ckpt.get("width")
    height = ckpt.get("height")
    if width is None or height is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing width/height. "
            "Checkpoints must include width and height."
        )

    # Get config
    config = ckpt.get("config", {})
    model_config_dict = config.get("model", {})

    if not model_config_dict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing config.model. "
            "Checkpoints must include the full model config."
        )

    if "architecture" not in model_config_dict:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing config.model.architecture. "
            "Checkpoints must include the architecture discriminator."
        )

    # Parse model config using discriminated union
    # Pydantic auto-dispatches based on 'architecture' field
    model_config: ModelConfig = TypeAdapter(ModelConfig).validate_python(model_config_dict)

    # Set data dimensions before building
    model_config.set_data_dimensions(width, height)

    # Build model (TrainableModel is a protocol; actual model is nn.Module)
    model: nn.Module = model_config.build_model()  # type: ignore[assignment]
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(torch_device)
    model.eval()

    # Compile for faster inference (CUDA only - MPS has issues)
    if compile_model and torch_device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

    # Build observation builder
    builder = model_config.build_observation_builder(width, height)

    return model, builder, width, height


def make_predict_fn(
    model: TrainableModel,
    builder: ObservationBuilder,
    simulator: PyRat,
    width: int,
    height: int,
    device: str,
) -> Callable[[Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create predict_fn closure for MCTS that reads from simulator.

    The closure captures the simulator reference. The tree mutates the simulator
    during search, and the predict_fn reads its current state.

    Args:
        model: Loaded NN model (must have .predict() method).
        builder: Observation builder for the model architecture.
        simulator: Game simulator (will be mutated by tree).
        width: Maze width (must match model training dimensions).
        height: Maze height (must match model training dimensions).
        device: Device for inference ("cpu", "cuda", "mps").

    Returns:
        Function that returns (policy_p1, policy_p2, payout) predictions.
    """
    from alpharat.data.maze import build_maze_array
    from alpharat.nn.extraction import from_pyrat_game

    maze = build_maze_array(simulator, width, height)
    max_turns = simulator.max_turns

    def predict_fn(_observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run NN inference on current simulator state."""
        obs_input = from_pyrat_game(simulator, maze, max_turns)
        obs = builder.build(obs_input)

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.inference_mode():
            result = model.predict(obs_tensor)
            # Dict interface - all models now return dicts
            policy_p1 = result["policy_p1"].squeeze(0).cpu().numpy()
            policy_p2 = result["policy_p2"].squeeze(0).cpu().numpy()
            payout = result["payout"].squeeze(0).cpu().numpy()

        return policy_p1, policy_p2, payout

    return predict_fn
