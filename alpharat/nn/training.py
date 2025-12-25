"""Full-batch training for PyRat neural network.

Single forward/backward pass on entire dataset per epoch. Faster than mini-batch
for datasets that fit in GPU memory.

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
    loss_variant: Literal["mcts", "dqn"] = "mcts"
    p_augment: float = 0.5


class DataConfig(BaseModel):
    """Data paths."""

    train_dir: str
    val_dir: str


class TrainConfig(BaseModel):
    """Full training configuration."""

    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig
    seed: int = 42


# --- Loss Functions ---


def sparse_payout_loss(
    pred_payout: torch.Tensor,
    action_p1: torch.Tensor,
    action_p2: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    """MSE loss only on the played action pair."""
    n = pred_payout.shape[0]
    batch_idx = torch.arange(n, device=pred_payout.device)
    a1 = action_p1.squeeze(-1).long()
    a2 = action_p2.squeeze(-1).long()
    pred_value = pred_payout[batch_idx, a1, a2]
    return F.mse_loss(pred_value, target_value.squeeze(-1))


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
        loss_value = F.mse_loss(pred_payout, data["payout_matrix"])
    else:
        loss_value = sparse_payout_loss(
            pred_payout, data["action_p1"], data["action_p2"], data["value"]
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
        pred_payout.detach(), data["action_p1"], data["action_p2"], data["value"]
    )

    metrics: dict[str, float] = {}
    metrics.update({f"p1/{k}": v for k, v in p1_metrics.items()})
    metrics.update({f"p2/{k}": v for k, v in p2_metrics.items()})
    metrics.update({f"payout/{k}": v for k, v in payout_metrics.items()})
    metrics.update({f"value/{k}": v for k, v in value_metrics.items()})

    return metrics


# --- Main Training Function ---


def run_training(
    config: TrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    resume: Path | None = None,
) -> Path:
    """Run full-batch training loop.

    Args:
        config: Training configuration (model, optimizer, data).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs (losses always computed).
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (for tensorboard). Auto-generated if None.
        resume: Path to checkpoint to resume from.

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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard" / run_name
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"Tensorboard run: {run_name}")

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
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    ).to(torch_device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.optim.lr)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float("inf")

    if resume is not None:
        logger.info(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=torch_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from epoch {start_epoch}")

    # Training loop
    best_model_path = output_dir / "best_model.pt"
    optim = config.optim

    for epoch in range(start_epoch, start_epoch + epochs):
        # === Train ===
        model.train()

        # In-place augmentation
        mask = torch.rand(len(train_dataset), device=torch_device) < optim.p_augment
        if mask.any():
            swap_player_perspective_batch(train, mask, width, height)

        optimizer.zero_grad()

        loss, loss_p1, loss_p2, loss_value, logits_p1, logits_p2, pred_payout = compute_losses(
            model, train, optim.loss_variant, optim.policy_weight, optim.value_weight
        )

        loss.backward()
        optimizer.step()

        # Train metrics
        train_metrics = {
            "loss_total": loss.item(),
            "loss_policy_p1": loss_p1.item(),
            "loss_policy_p2": loss_p2.item(),
            "loss_value": loss_value.item(),
        }
        compute_detailed = epoch == 1 or epoch % metrics_every == 0
        if compute_detailed:
            train_metrics.update(compute_detailed_metrics(train, logits_p1, logits_p2, pred_payout))

        # === Validate ===
        model.eval()
        with torch.no_grad():
            (
                val_loss,
                val_loss_p1,
                val_loss_p2,
                val_loss_value,
                val_logits_p1,
                val_logits_p2,
                val_pred_payout,
            ) = compute_losses(
                model, val, optim.loss_variant, optim.policy_weight, optim.value_weight
            )

            val_metrics = {
                "loss_total": val_loss.item(),
                "loss_policy_p1": val_loss_p1.item(),
                "loss_policy_p2": val_loss_p2.item(),
                "loss_value": val_loss_value.item(),
            }
            if compute_detailed:
                val_metrics.update(
                    compute_detailed_metrics(val, val_logits_p1, val_logits_p2, val_pred_payout)
                )

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
                    "config": config.model_dump(),
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
                    "config": config.model_dump(),
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return best_model_path
