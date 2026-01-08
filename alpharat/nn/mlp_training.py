"""Mini-batch training for PyRat MLP neural network.

Backward-compatible wrapper around the generic training loop.
Uses MLPModelConfig and MLPOptimConfig under the hood.

Usage:
    from alpharat.nn.training import TrainConfig, run_training

    config = TrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    run_training(config, epochs=100, output_dir=Path("runs"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from alpharat.nn.architectures.mlp.config import MLPModelConfig, MLPOptimConfig
from alpharat.nn.training.config import DataConfig
from alpharat.nn.training.loop import run_training as _run_training

logger = logging.getLogger(__name__)


# --- Config Models (backward-compatible) ---


class ModelConfig(BaseModel):
    """Model architecture parameters."""

    hidden_dim: int = 256
    dropout: float = 0.0


class OptimConfig(BaseModel):
    """Optimization parameters."""

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    nash_weight: float = 0.0
    nash_mode: Literal["target", "predicted"] = "target"
    constant_sum_weight: float = 0.0
    p_augment: float = 0.5
    batch_size: int = 4096


class TrainConfig(BaseModel):
    """Full training configuration for MLP.

    Either `model` or `resume_from` must be provided:
    - model only: fresh start with specified architecture
    - resume_from only: architecture loaded from checkpoint
    - both: validate they match, then resume
    """

    model: ModelConfig | None = None
    optim: OptimConfig = OptimConfig()
    data: DataConfig
    resume_from: str | None = None
    seed: int = 42


def run_training(
    config: TrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    use_amp: bool | None = None,
) -> Path:
    """Run MLP training loop.

    Wrapper around the generic training loop for backward compatibility.

    Args:
        config: Training configuration (model, optimizer, data).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (for tensorboard).
        use_amp: Enable AMP. None auto-detects.

    Returns:
        Path to best model checkpoint.
    """
    import torch

    from alpharat.nn.gpu_dataset import GPUDataset
    from alpharat.nn.training_utils import select_device

    # Resolve model config from checkpoint if resuming
    model_config = config.model
    if config.resume_from is not None:
        torch_device = select_device(device)
        checkpoint = torch.load(config.resume_from, map_location=torch_device, weights_only=False)
        checkpoint_model = ModelConfig(**checkpoint["config"]["model"])

        if model_config is not None:
            if model_config != checkpoint_model:
                raise ValueError(
                    f"Model config mismatch.\n"
                    f"  Config: {model_config}\n"
                    f"  Checkpoint: {checkpoint_model}"
                )
        else:
            model_config = checkpoint_model
            logger.info(f"Using model config from checkpoint: {model_config}")

    if model_config is None:
        raise ValueError("Either 'model' or 'resume_from' must be provided in config")

    # Load data to get obs_dim (needed for model config)
    train_dir = Path(config.data.train_dir)
    torch_device = select_device(device)
    train_dataset = GPUDataset(train_dir, torch_device)
    width, height = train_dataset.width, train_dataset.height
    obs_dim = width * height * 7 + 6

    # Convert to new architecture configs
    mlp_model_config = MLPModelConfig(
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout,
        p_augment=config.optim.p_augment,
        obs_dim=obs_dim,
    )

    mlp_optim_config = MLPOptimConfig(
        lr=config.optim.lr,
        policy_weight=config.optim.policy_weight,
        value_weight=config.optim.value_weight,
        nash_weight=config.optim.nash_weight,
        nash_mode=config.optim.nash_mode,
        constant_sum_weight=config.optim.constant_sum_weight,
        batch_size=config.optim.batch_size,
    )

    data_config = DataConfig(
        train_dir=config.data.train_dir,
        val_dir=config.data.val_dir,
    )

    # Delegate to generic training loop
    return _run_training(
        model_config=mlp_model_config,
        optim_config=mlp_optim_config,
        data_config=data_config,
        epochs=epochs,
        checkpoint_every=checkpoint_every,
        metrics_every=metrics_every,
        device=device,
        output_dir=output_dir,
        run_name=run_name,
        use_amp=use_amp,
        seed=config.seed,
        resume_from=config.resume_from,
    )
