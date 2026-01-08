"""Training for LocalValueMLP with ownership loss.

Backward-compatible wrapper around the generic training loop.
Uses LocalValueModelConfig and LocalValueOptimConfig under the hood.

Usage:
    from alpharat.nn.local_value_training import LocalValueTrainConfig, run_local_value_training

    config = LocalValueTrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    run_local_value_training(config, epochs=100, output_dir=Path("runs"))
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from alpharat.nn.architectures.local_value.config import (
    LocalValueModelConfig,
    LocalValueOptimConfig,
)
from alpharat.nn.training.config import DataConfig
from alpharat.nn.training.loop import run_training as _run_training

logger = logging.getLogger(__name__)


# --- Config Models (backward-compatible) ---


class LocalValueModelConfigOld(BaseModel):
    """Model architecture parameters for LocalValueMLP."""

    hidden_dim: int = 256
    dropout: float = 0.0


class LocalValueOptimConfigOld(BaseModel):
    """Optimization parameters for local value training."""

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    ownership_weight: float = 1.0
    p_augment: float = 0.5
    batch_size: int = 4096


class LocalValueDataConfig(BaseModel):
    """Data paths."""

    train_dir: str
    val_dir: str


class LocalValueTrainConfig(BaseModel):
    """Full training configuration for LocalValueMLP.

    Either `model` or `resume_from` must be provided:
    - model only: fresh start with specified architecture
    - resume_from only: architecture loaded from checkpoint
    - both: validate they match, then resume
    """

    model: LocalValueModelConfigOld | None = None
    optim: LocalValueOptimConfigOld = LocalValueOptimConfigOld()
    data: LocalValueDataConfig
    resume_from: str | None = None
    seed: int = 42


def run_local_value_training(
    config: LocalValueTrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    use_amp: bool | None = None,
) -> Path:
    """Run LocalValueMLP training loop.

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
        checkpoint_model = LocalValueModelConfigOld(**checkpoint["config"]["model"])

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

    # Load data to get dimensions (needed for model config)
    train_dir = Path(config.data.train_dir)
    torch_device = select_device(device)
    train_dataset = GPUDataset(train_dir, torch_device)
    width, height = train_dataset.width, train_dataset.height
    obs_dim = width * height * 7 + 6

    # Convert to new architecture configs
    local_value_model_config = LocalValueModelConfig(
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout,
        p_augment=config.optim.p_augment,
        obs_dim=obs_dim,
        width=width,
        height=height,
    )

    local_value_optim_config = LocalValueOptimConfig(
        lr=config.optim.lr,
        policy_weight=config.optim.policy_weight,
        value_weight=config.optim.value_weight,
        ownership_weight=config.optim.ownership_weight,
        batch_size=config.optim.batch_size,
    )

    data_config = DataConfig(
        train_dir=config.data.train_dir,
        val_dir=config.data.val_dir,
    )

    # Delegate to generic training loop
    return _run_training(
        model_config=local_value_model_config,
        optim_config=local_value_optim_config,
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
