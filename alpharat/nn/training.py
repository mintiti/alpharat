"""Mini-batch training for PyRat neural network.

Two loss variants:
- "mcts": Full payout matrix supervision (MSE on all 25 entries from MCTS estimates)
- "dqn": Sparse supervision (MSE only on played action pair vs actual outcome)

Usage:
    from alpharat.nn.training import TrainConfig, run_training

    config = TrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    run_training(config, epochs=100, output_dir=Path("runs"))
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from alpharat.nn.augmentation import swap_player_perspective_batch
from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    MetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.models import PyRatMLP

logger = logging.getLogger(__name__)


# --- Config Models ---


class ModelConfig(BaseModel):
    """Model architecture parameters."""

    hidden_dim: int = 256
    dropout: float = 0.0


class OptimConfig(BaseModel):
    """Optimization parameters."""

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    loss_variant: Literal["mcts", "dqn"] = "dqn"
    p_augment: float = 0.5
    batch_size: int = 4096


class DataConfig(BaseModel):
    """Data paths."""

    train_dir: str
    val_dir: str


class TrainConfig(BaseModel):
    """Full training configuration.

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


# --- Loss Functions ---


def sparse_payout_loss(
    pred_payout: torch.Tensor,
    action_p1: torch.Tensor,
    action_p2: torch.Tensor,
    p1_value: torch.Tensor,
    p2_value: torch.Tensor,
) -> torch.Tensor:
    """MSE loss at played action pair using actual game outcomes.

    Supervises the NN's payout prediction with ground truth outcomes:
    - p1_value: how much P1 actually scored from this position
    - p2_value: how much P2 actually scored from this position

    Args:
        pred_payout: Predicted bimatrix, shape (batch, 2, 5, 5).
        action_p1: P1's action, shape (batch, 1).
        action_p2: P2's action, shape (batch, 1).
        p1_value: P1's actual remaining score, shape (batch,) or (batch, 1).
        p2_value: P2's actual remaining score, shape (batch,) or (batch, 1).

    Returns:
        Average MSE loss over both players' values at played action.
    """
    n = pred_payout.shape[0]
    batch_idx = torch.arange(n, device=pred_payout.device)
    a1 = action_p1.squeeze(-1).long()
    a2 = action_p2.squeeze(-1).long()

    # Extract predicted values at played action pair for both players
    pred_p1 = pred_payout[batch_idx, 0, a1, a2]
    pred_p2 = pred_payout[batch_idx, 1, a1, a2]

    # Targets are actual game outcomes
    target_p1 = p1_value.squeeze(-1)
    target_p2 = p2_value.squeeze(-1)

    # Average loss over both players
    return 0.5 * (F.mse_loss(pred_p1, target_p1) + F.mse_loss(pred_p2, target_p2))


def compute_losses(
    model: PyRatMLP,
    data: dict[str, torch.Tensor],
    loss_variant: Literal["mcts", "dqn"],
    policy_weight: float,
    value_weight: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Forward pass and loss computation.

    Returns:
        (loss, loss_p1, loss_p2, loss_value, logits_p1, logits_p2, pred_payout)
    """
    logits_p1, logits_p2, pred_payout = model(data["observation"])

    loss_p1 = F.cross_entropy(logits_p1, data["policy_p1"])
    loss_p2 = F.cross_entropy(logits_p2, data["policy_p2"])

    if loss_variant == "mcts":
        # Full matrix MSE - still uses MCTS estimates (deprecated)
        loss_value = F.mse_loss(pred_payout, data["payout_matrix"])
    else:
        # Sparse DQN: supervise with actual game outcomes
        loss_value = sparse_payout_loss(
            pred_payout, data["action_p1"], data["action_p2"], data["p1_value"], data["p2_value"]
        )

    loss = policy_weight * (loss_p1 + loss_p2) + value_weight * loss_value

    return loss, loss_p1, loss_p2, loss_value, logits_p1, logits_p2, pred_payout


def compute_detailed_metrics(
    data: dict[str, torch.Tensor],
    logits_p1: torch.Tensor,
    logits_p2: torch.Tensor,
    pred_payout: torch.Tensor,
) -> dict[str, float]:
    """Compute diagnostic metrics (expensive on full dataset)."""
    p1_metrics = compute_policy_metrics(logits_p1.detach(), data["policy_p1"])
    p2_metrics = compute_policy_metrics(logits_p2.detach(), data["policy_p2"])
    payout_metrics = compute_payout_metrics(pred_payout.detach(), data["payout_matrix"])
    value_metrics = compute_value_metrics(
        pred_payout.detach(),
        data["action_p1"],
        data["action_p2"],
        data["p1_value"],
        data["p2_value"],
    )

    metrics: dict[str, float] = {}
    metrics.update({f"p1/{k}": v for k, v in p1_metrics.items()})
    metrics.update({f"p2/{k}": v for k, v in p2_metrics.items()})
    metrics.update({f"payout/{k}": v for k, v in payout_metrics.items()})
    metrics.update({f"value/{k}": v for k, v in value_metrics.items()})

    return metrics


# --- Main Training Function ---


def _resolve_model_config(
    config: TrainConfig,
    torch_device: torch.device,
) -> tuple[ModelConfig, dict | None]:
    """Resolve model config from config.model or checkpoint.

    Returns:
        Tuple of (effective model config, checkpoint dict or None).

    Raises:
        ValueError: If neither model nor resume_from is provided, or if they conflict.
    """
    checkpoint = None

    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location=torch_device, weights_only=False)
        checkpoint_model = ModelConfig(**checkpoint["config"]["model"])

        if config.model is not None:
            # Both provided: validate they match
            if config.model != checkpoint_model:
                raise ValueError(
                    f"Model config mismatch.\n"
                    f"  Config: {config.model}\n"
                    f"  Checkpoint: {checkpoint_model}"
                )
            model_config = config.model
        else:
            # Only resume_from: use checkpoint's config
            model_config = checkpoint_model
            logger.info(f"Using model config from checkpoint: {model_config}")
    elif config.model is not None:
        # Only model: fresh start
        model_config = config.model
    else:
        raise ValueError("Either 'model' or 'resume_from' must be provided in config")

    return model_config, checkpoint


def run_training(
    config: TrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
) -> Path:
    """Run mini-batch training loop.

    Args:
        config: Training configuration (model, optimizer, data).
            Must have either `model` (fresh start) or `resume_from` (continue training).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs (losses always computed).
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (for tensorboard). Auto-generated if None.

    Returns:
        Path to best model checkpoint.
    """
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"

    # Set seed
    torch.manual_seed(config.seed)

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)
    logger.info(f"Using device: {torch_device}")

    # Resolve model config (from config.model or checkpoint)
    model_config, checkpoint = _resolve_model_config(config, torch_device)

    # Create output directory for this run
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"Run directory: {run_dir}")

    # Load all data to GPU
    train_dir = Path(config.data.train_dir)
    val_dir = Path(config.data.val_dir)

    logger.info(f"Loading training data from {train_dir}")
    train_dataset = GPUDataset(train_dir, torch_device)
    train = train_dataset._data

    logger.info(f"Loading validation data from {val_dir}")
    val_dataset = GPUDataset(val_dir, torch_device)
    val = val_dataset._data

    width, height = train_dataset.width, train_dataset.height
    obs_dim = width * height * 7 + 6
    logger.info(f"obs_dim={obs_dim}, train={len(train_dataset)}, val={len(val_dataset)}")

    # Create model
    model = PyRatMLP(
        obs_dim=obs_dim,
        hidden_dim=model_config.hidden_dim,
        dropout=model_config.dropout,
    ).to(torch_device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.optim.lr)

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_loss = float("inf")

    if checkpoint is not None:
        logger.info(f"Resuming from {config.resume_from}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    best_model_path = run_dir / "best_model.pt"
    optim_cfg = config.optim
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    batch_size = optim_cfg.batch_size

    for epoch in range(start_epoch, start_epoch + epochs):
        compute_detailed = epoch == 1 or epoch % metrics_every == 0

        # === Train ===
        model.train()

        # Shuffle training data
        train_indices = torch.randperm(n_train, device=torch_device)

        # Accumulate metrics over mini-batches
        train_acc = MetricsAccumulator()

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_idx = train_indices[start_idx:end_idx]
            curr_batch_size = len(batch_idx)

            # Extract batch
            batch = {
                "observation": train["observation"][batch_idx],
                "policy_p1": train["policy_p1"][batch_idx],
                "policy_p2": train["policy_p2"][batch_idx],
                "value": train["value"][batch_idx],
                "payout_matrix": train["payout_matrix"][batch_idx],
                "action_p1": train["action_p1"][batch_idx],
                "action_p2": train["action_p2"][batch_idx],
            }

            # Per-batch augmentation
            aug_mask = torch.rand(curr_batch_size, device=torch_device) < optim_cfg.p_augment
            if aug_mask.any():
                swap_player_perspective_batch(batch, aug_mask, width, height)

            optimizer.zero_grad()

            loss, loss_p1, loss_p2, loss_value, logits_p1, logits_p2, pred_payout = compute_losses(
                model,
                batch,
                optim_cfg.loss_variant,
                optim_cfg.policy_weight,
                optim_cfg.value_weight,
            )

            loss.backward()
            optimizer.step()

            # Accumulate loss metrics
            train_acc.update(
                {
                    "loss_total": loss.item(),
                    "loss_policy_p1": loss_p1.item(),
                    "loss_policy_p2": loss_p2.item(),
                    "loss_value": loss_value.item(),
                },
                batch_size=curr_batch_size,
            )

            # Accumulate detailed metrics (expensive, only when needed)
            if compute_detailed:
                with torch.no_grad():
                    train_acc.update(
                        compute_detailed_metrics(batch, logits_p1, logits_p2, pred_payout),
                        batch_size=curr_batch_size,
                    )

        train_metrics = train_acc.compute()

        # === Validate ===
        model.eval()
        val_acc = MetricsAccumulator()

        with torch.no_grad():
            for start_idx in range(0, n_val, batch_size):
                end_idx = min(start_idx + batch_size, n_val)
                curr_batch_size = end_idx - start_idx

                val_batch = {
                    "observation": val["observation"][start_idx:end_idx],
                    "policy_p1": val["policy_p1"][start_idx:end_idx],
                    "policy_p2": val["policy_p2"][start_idx:end_idx],
                    "payout_matrix": val["payout_matrix"][start_idx:end_idx],
                    "action_p1": val["action_p1"][start_idx:end_idx],
                    "action_p2": val["action_p2"][start_idx:end_idx],
                    "value": val["value"][start_idx:end_idx],
                }

                (
                    vl,
                    vl_p1,
                    vl_p2,
                    vl_value,
                    vl_logits_p1,
                    vl_logits_p2,
                    vl_pred_payout,
                ) = compute_losses(
                    model,
                    val_batch,
                    optim_cfg.loss_variant,
                    optim_cfg.policy_weight,
                    optim_cfg.value_weight,
                )

                val_acc.update(
                    {
                        "loss_total": vl.item(),
                        "loss_policy_p1": vl_p1.item(),
                        "loss_policy_p2": vl_p2.item(),
                        "loss_value": vl_value.item(),
                    },
                    batch_size=curr_batch_size,
                )

                if compute_detailed:
                    val_acc.update(
                        compute_detailed_metrics(
                            val_batch, vl_logits_p1, vl_logits_p2, vl_pred_payout
                        ),
                        batch_size=curr_batch_size,
                    )

            val_metrics = val_acc.compute()

        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log to console
        logger.info(
            f"Epoch {epoch} - "
            f"Train: {train_metrics['loss_total']:.4f} "
            f"(p1={train_metrics['loss_policy_p1']:.4f}, "
            f"p2={train_metrics['loss_policy_p2']:.4f}, "
            f"val={train_metrics['loss_value']:.4f}) | "
            f"Val: {val_metrics['loss_total']:.4f}"
        )

        # Build effective config for saving (always include resolved model config)
        effective_config = config.model_dump()
        effective_config["model"] = model_config.model_dump()

        # Save best model
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "best_val_loss": best_val_loss,
                    "config": effective_config,
                    "width": width,
                    "height": height,
                },
                best_model_path,
            )
            logger.info(f"  New best model (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss_total"],
                    "best_val_loss": best_val_loss,
                    "config": effective_config,
                    "width": width,
                    "height": height,
                },
                run_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return best_model_path
