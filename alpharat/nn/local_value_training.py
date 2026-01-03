"""Training for LocalValueMLP with ownership loss.

Similar to training.py but uses:
- LocalValueMLP instead of PyRatMLP
- Ownership loss (masked cross-entropy) instead of payout matrix loss
- Value derived from ownership predictions, not a separate head

Loss components:
- policy_loss: cross_entropy(logits, target_policy) for P1 and P2
- ownership_loss: cross_entropy(ownership_logits, cheese_outcomes) masked to active cheese

The ownership loss uses the sentinel-based cheese_outcomes:
- -1 = inactive (no cheese at this position) -> skip in loss
- 0-3 = outcome class -> include in loss
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
from alpharat.nn.models import LocalValueMLP

logger = logging.getLogger(__name__)


# --- Config Models ---


class LocalValueModelConfig(BaseModel):
    """Model architecture parameters for LocalValueMLP."""

    hidden_dim: int = 256
    dropout: float = 0.0


class LocalValueOptimConfig(BaseModel):
    """Optimization parameters for local value training."""

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    ownership_weight: float = 1.0
    loss_variant: Literal["mcts", "dqn"] = "mcts"
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

    model: LocalValueModelConfig | None = None
    optim: LocalValueOptimConfig = LocalValueOptimConfig()
    data: LocalValueDataConfig
    resume_from: str | None = None
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


def compute_ownership_loss(
    ownership_logits: torch.Tensor,
    cheese_outcomes: torch.Tensor,
) -> torch.Tensor:
    """Compute masked cross-entropy loss for ownership prediction.

    Only computes loss on cells with active cheese (cheese_outcomes >= 0).
    Cells with -1 (inactive) are excluded from the loss.

    Args:
        ownership_logits: Per-cell logits, shape (B, H, W, 4).
        cheese_outcomes: Target outcomes, shape (B, H, W).
            Values: -1=inactive (skip), 0-3=outcome class.

    Returns:
        Scalar loss tensor (mean over active cheese cells in batch).
    """
    b, h, w, c = ownership_logits.shape

    # Flatten spatial dimensions
    logits_flat = ownership_logits.view(b * h * w, c)  # (B*H*W, 4)
    targets_flat = cheese_outcomes.view(b * h * w).long()  # (B*H*W,)

    # Create mask for active cheese cells (outcomes >= 0)
    mask_flat = targets_flat >= 0  # (B*H*W,) bool

    if not mask_flat.any():
        # No active cheese in batch (shouldn't happen in practice)
        return torch.tensor(0.0, device=ownership_logits.device)

    # Compute per-element cross-entropy
    # Note: targets with -1 will cause issues with cross_entropy, but we mask them out
    # To be safe, clamp targets to valid range before computing
    targets_clamped = targets_flat.clamp(min=0)
    ce_all = F.cross_entropy(logits_flat, targets_clamped, reduction="none")  # (B*H*W,)

    # Average only over active cheese cells
    loss = (ce_all * mask_flat.float()).sum() / mask_flat.float().sum()

    return loss


def compute_local_value_losses(
    model: LocalValueMLP,
    data: dict[str, torch.Tensor],
    loss_variant: Literal["mcts", "dqn"],
    policy_weight: float,
    value_weight: float,
    ownership_weight: float,
) -> dict[str, torch.Tensor]:
    """Forward pass and loss computation for LocalValueMLP.

    Args:
        model: LocalValueMLP model.
        data: Batch dict with observation, policy_p1, policy_p2, payout_matrix,
            action_p1, action_p2, value, cheese_outcomes.
        loss_variant: "mcts" for full matrix MSE, "dqn" for sparse (played action only).
        policy_weight: Weight for policy loss.
        value_weight: Weight for value/payout loss.
        ownership_weight: Weight for ownership loss.

    Returns:
        Dict with all losses and model outputs:
            - loss: Total weighted loss
            - loss_p1, loss_p2: Policy losses
            - loss_value: Payout matrix loss (MSE, full or sparse)
            - loss_ownership: Ownership loss
            - logits_p1, logits_p2: Policy logits
            - pred_payout: Predicted payout matrix
            - ownership_logits: Ownership logits
    """
    # Derive cheese mask from outcomes for ownership value computation
    cheese_mask = data["cheese_outcomes"] >= 0  # (B, H, W) bool

    logits_p1, logits_p2, pred_payout, ownership_logits, _ = model(
        data["observation"], cheese_mask=cheese_mask
    )

    # Policy losses
    loss_p1 = F.cross_entropy(logits_p1, data["policy_p1"])
    loss_p2 = F.cross_entropy(logits_p2, data["policy_p2"])

    # Value loss (payout matrix)
    if loss_variant == "mcts":
        loss_value = F.mse_loss(pred_payout, data["payout_matrix"])
    else:
        loss_value = sparse_payout_loss(
            pred_payout, data["action_p1"], data["action_p2"], data["value"]
        )

    # Ownership loss (auxiliary task)
    loss_ownership = compute_ownership_loss(ownership_logits, data["cheese_outcomes"])

    # Total loss
    loss = (
        policy_weight * (loss_p1 + loss_p2)
        + value_weight * loss_value
        + ownership_weight * loss_ownership
    )

    return {
        "loss": loss,
        "loss_p1": loss_p1,
        "loss_p2": loss_p2,
        "loss_value": loss_value,
        "loss_ownership": loss_ownership,
        "logits_p1": logits_p1,
        "logits_p2": logits_p2,
        "pred_payout": pred_payout,
        "ownership_logits": ownership_logits,
    }


def compute_ownership_metrics(
    ownership_logits: torch.Tensor,
    cheese_outcomes: torch.Tensor,
) -> dict[str, float]:
    """Compute diagnostic metrics for ownership prediction.

    Args:
        ownership_logits: Per-cell logits, shape (B, H, W, 4).
        cheese_outcomes: Target outcomes, shape (B, H, W).

    Returns:
        Dict with accuracy, per-class accuracy, etc.
    """
    b, h, w, c = ownership_logits.shape

    # Flatten
    logits_flat = ownership_logits.view(b * h * w, c)
    targets_flat = cheese_outcomes.view(b * h * w)

    # Mask for active cells
    mask = targets_flat >= 0
    if not mask.any():
        return {"accuracy": 0.0}

    # Predictions
    preds = logits_flat[mask].argmax(dim=-1)
    targets = targets_flat[mask]

    # Overall accuracy
    accuracy = (preds == targets).float().mean().item()

    # Per-class accuracy (if enough samples)
    metrics: dict[str, float] = {"accuracy": accuracy}
    class_names = ["p1_win", "simultaneous", "uncollected", "p2_win"]
    for c, name in enumerate(class_names):
        class_mask = targets == c
        if class_mask.any():
            class_acc = (preds[class_mask] == c).float().mean().item()
            metrics[f"acc_{name}"] = class_acc

    return metrics


# --- Main Training Function ---


def _resolve_model_config(
    config: LocalValueTrainConfig,
    torch_device: torch.device,
) -> tuple[LocalValueModelConfig, dict | None]:
    """Resolve model config from config.model or checkpoint.

    Returns:
        Tuple of (effective model config, checkpoint dict or None).

    Raises:
        ValueError: If neither model nor resume_from is provided, or if they conflict.
    """
    checkpoint = None

    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location=torch_device, weights_only=False)
        checkpoint_model = LocalValueModelConfig(**checkpoint["config"]["model"])

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


def run_local_value_training(
    config: LocalValueTrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
) -> Path:
    """Run training loop for LocalValueMLP.

    Args:
        config: Training configuration (model, optimizer, data).
            Must have either `model` (fresh start) or `resume_from` (continue training).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (for tensorboard). Auto-generated if None.

    Returns:
        Path to best model checkpoint.
    """
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"local_value_{timestamp}"

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
    model = LocalValueMLP(
        obs_dim=obs_dim,
        width=width,
        height=height,
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
                "cheese_outcomes": train["cheese_outcomes"][batch_idx],
            }

            # Per-batch augmentation
            aug_mask = torch.rand(curr_batch_size, device=torch_device) < optim_cfg.p_augment
            if aug_mask.any():
                swap_player_perspective_batch(batch, aug_mask, width, height)

            optimizer.zero_grad()

            out = compute_local_value_losses(
                model,
                batch,
                optim_cfg.loss_variant,
                optim_cfg.policy_weight,
                optim_cfg.value_weight,
                optim_cfg.ownership_weight,
            )

            out["loss"].backward()
            optimizer.step()

            # Accumulate loss metrics
            train_acc.update(
                {
                    "loss_total": out["loss"].item(),
                    "loss_policy_p1": out["loss_p1"].item(),
                    "loss_policy_p2": out["loss_p2"].item(),
                    "loss_value": out["loss_value"].item(),
                    "loss_ownership": out["loss_ownership"].item(),
                },
                batch_size=curr_batch_size,
            )

            # Accumulate detailed metrics (expensive, only when needed)
            if compute_detailed:
                with torch.no_grad():
                    p1_metrics = compute_policy_metrics(
                        out["logits_p1"].detach(), batch["policy_p1"]
                    )
                    p2_metrics = compute_policy_metrics(
                        out["logits_p2"].detach(), batch["policy_p2"]
                    )
                    payout_mets = compute_payout_metrics(
                        out["pred_payout"].detach(), batch["payout_matrix"]
                    )
                    value_mets = compute_value_metrics(
                        out["pred_payout"].detach(),
                        batch["action_p1"],
                        batch["action_p2"],
                        batch["value"],
                    )
                    own_metrics = compute_ownership_metrics(
                        out["ownership_logits"].detach(), batch["cheese_outcomes"]
                    )

                    detailed = {}
                    detailed.update({f"p1/{k}": v for k, v in p1_metrics.items()})
                    detailed.update({f"p2/{k}": v for k, v in p2_metrics.items()})
                    detailed.update({f"payout/{k}": v for k, v in payout_mets.items()})
                    detailed.update({f"value/{k}": v for k, v in value_mets.items()})
                    detailed.update({f"ownership/{k}": v for k, v in own_metrics.items()})
                    train_acc.update(detailed, batch_size=curr_batch_size)

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
                    "cheese_outcomes": val["cheese_outcomes"][start_idx:end_idx],
                }

                vl_out = compute_local_value_losses(
                    model,
                    val_batch,
                    optim_cfg.loss_variant,
                    optim_cfg.policy_weight,
                    optim_cfg.value_weight,
                    optim_cfg.ownership_weight,
                )

                val_acc.update(
                    {
                        "loss_total": vl_out["loss"].item(),
                        "loss_policy_p1": vl_out["loss_p1"].item(),
                        "loss_policy_p2": vl_out["loss_p2"].item(),
                        "loss_value": vl_out["loss_value"].item(),
                        "loss_ownership": vl_out["loss_ownership"].item(),
                    },
                    batch_size=curr_batch_size,
                )

                if compute_detailed:
                    p1_metrics = compute_policy_metrics(vl_out["logits_p1"], val_batch["policy_p1"])
                    p2_metrics = compute_policy_metrics(vl_out["logits_p2"], val_batch["policy_p2"])
                    payout_mets = compute_payout_metrics(
                        vl_out["pred_payout"], val_batch["payout_matrix"]
                    )
                    value_mets = compute_value_metrics(
                        vl_out["pred_payout"],
                        val_batch["action_p1"],
                        val_batch["action_p2"],
                        val_batch["value"],
                    )
                    own_metrics = compute_ownership_metrics(
                        vl_out["ownership_logits"], val_batch["cheese_outcomes"]
                    )

                    detailed = {}
                    detailed.update({f"p1/{k}": v for k, v in p1_metrics.items()})
                    detailed.update({f"p2/{k}": v for k, v in p2_metrics.items()})
                    detailed.update({f"payout/{k}": v for k, v in payout_mets.items()})
                    detailed.update({f"value/{k}": v for k, v in value_mets.items()})
                    detailed.update({f"ownership/{k}": v for k, v in own_metrics.items()})
                    val_acc.update(detailed, batch_size=curr_batch_size)

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
            f"val={train_metrics['loss_value']:.4f}, "
            f"own={train_metrics['loss_ownership']:.4f}) | "
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
