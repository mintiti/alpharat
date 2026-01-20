"""Checkpoint loading utilities using ModelConfig.build_model().

This module provides a centralized way to load models from checkpoints without
manual dispatch based on model type. The checkpoint contains the full model_config,
which is parsed back into a ModelConfig using Pydantic's discriminated union,
then build_model() constructs the correct architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from alpharat.nn.builders import ObservationBuilder


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
    model_config = TypeAdapter(ModelConfig).validate_python(model_config_dict)

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
