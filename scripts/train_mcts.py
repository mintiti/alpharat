#!/usr/bin/env python3
"""MCTS-style training with full payout matrix supervision.

Losses:
- Policy: Cross-entropy with soft Nash targets: -sum(target * log_pred)
- Payout: MSE on full 5x5 payout matrix

Usage:
    uv run python scripts/train_mcts.py --train-dir data/train --val-dir data/val

Design Decision: Full Matrix vs DQN-Style Learning
---------------------------------------------------
We supervise the entire 5x5 payout matrix from MCTS output, not just the
action pair that was actually played.

Alternative (DQN-style): Only update payout[a1, a2] (and equivalent action
pairs due to walls/mud) based on actual game outcome, leaving other entries
untouched. More "honest" — only learns from actions actually tried — but
needs far more samples and has higher variance.

Current choice (full matrix): Every sample gives gradient signal on all 25
entries. Higher sample efficiency, but we're learning MCTS's estimates, not
ground truth outcomes. If MCTS is wrong about unexplored action pairs, the
NN inherits those mistakes.

This is a deliberate trade-off. Revisit if:
- We have abundant data (DQN-style becomes viable)
- NN predictions diverge from actual outcomes in evaluation
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    MetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.models import PyRatMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _train_step_impl(
    model: PyRatMLP,
    obs: torch.Tensor,
    target_p1: torch.Tensor,
    target_p2: torch.Tensor,
    target_payout: torch.Tensor,
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
    """Compiled forward + loss computation.

    Returns:
        Tuple of (loss, loss_p1, loss_p2, loss_payout, logits_p1, logits_p2, pred_payout).
    """
    logits_p1, logits_p2, pred_payout = model(obs)
    loss_p1 = F.cross_entropy(logits_p1, target_p1)
    loss_p2 = F.cross_entropy(logits_p2, target_p2)
    loss_payout = F.mse_loss(pred_payout, target_payout)
    loss = policy_weight * (loss_p1 + loss_p2) + value_weight * loss_payout
    return loss, loss_p1, loss_p2, loss_payout, logits_p1, logits_p2, pred_payout


# torch.compile causes NaN on MPS after ~300 batches, disable for now
# TODO: re-enable when MPS support improves
train_step = _train_step_impl


def train_epoch(
    model: PyRatMLP,
    dataset: GPUDataset,
    optimizer: AdamW,
    batch_size: int,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Returns:
        Dict with losses and diagnostic metrics.
    """
    model.train()
    acc = MetricsAccumulator()

    for batch in dataset.epoch_iter(batch_size, augment=True):
        obs = batch["observation"]
        target_p1 = batch["policy_p1"]
        target_p2 = batch["policy_p2"]
        target_payout = batch["payout_matrix"]
        action_p1 = batch["action_p1"]
        action_p2 = batch["action_p2"]
        target_value = batch["value"]
        n = obs.size(0)

        optimizer.zero_grad()

        loss, loss_p1, loss_p2, loss_payout, logits_p1, logits_p2, pred_payout = train_step(
            model,
            obs,
            target_p1,
            target_p2,
            target_payout,
            policy_weight,
            value_weight,
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses (outside compiled region)
        acc.update(
            {
                "loss_total": loss.item(),
                "loss_policy_p1": loss_p1.item(),
                "loss_policy_p2": loss_p2.item(),
                "loss_payout": loss_payout.item(),
            },
            n,
        )

        # Diagnostic metrics
        p1_metrics = compute_policy_metrics(logits_p1.detach(), target_p1)
        p2_metrics = compute_policy_metrics(logits_p2.detach(), target_p2)
        payout_metrics = compute_payout_metrics(pred_payout.detach(), target_payout)
        value_metrics = compute_value_metrics(
            pred_payout.detach(), action_p1, action_p2, target_value
        )

        acc.update(
            {
                **{f"p1/{k}": v for k, v in p1_metrics.items()},
                **{f"p2/{k}": v for k, v in p2_metrics.items()},
                **{f"payout/{k}": v for k, v in payout_metrics.items()},
                **{f"value/{k}": v for k, v in value_metrics.items()},
            },
            n,
        )

    return acc.compute()


@torch.no_grad()
def validate(
    model: PyRatMLP,
    dataset: GPUDataset,
    batch_size: int,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> dict[str, float]:
    """Validate model.

    Note: No augmentation or shuffling during validation for consistent metrics.

    Returns:
        Dict with losses and diagnostic metrics.
    """
    model.eval()
    acc = MetricsAccumulator()

    for batch in dataset.epoch_iter(batch_size, augment=False, shuffle=False):
        obs = batch["observation"]
        target_p1 = batch["policy_p1"]
        target_p2 = batch["policy_p2"]
        target_payout = batch["payout_matrix"]
        action_p1 = batch["action_p1"]
        action_p2 = batch["action_p2"]
        target_value = batch["value"]
        n = obs.size(0)

        logits_p1, logits_p2, pred_payout = model(obs)

        loss_p1 = F.cross_entropy(logits_p1, target_p1)
        loss_p2 = F.cross_entropy(logits_p2, target_p2)
        loss_payout = F.mse_loss(pred_payout, target_payout)

        loss = policy_weight * (loss_p1 + loss_p2) + value_weight * loss_payout

        # Accumulate losses
        acc.update(
            {
                "loss_total": loss.item(),
                "loss_policy_p1": loss_p1.item(),
                "loss_policy_p2": loss_p2.item(),
                "loss_payout": loss_payout.item(),
            },
            n,
        )

        # Diagnostic metrics
        p1_metrics = compute_policy_metrics(logits_p1, target_p1)
        p2_metrics = compute_policy_metrics(logits_p2, target_p2)
        payout_metrics = compute_payout_metrics(pred_payout, target_payout)
        value_metrics = compute_value_metrics(pred_payout, action_p1, action_p2, target_value)

        acc.update(
            {
                **{f"p1/{k}": v for k, v in p1_metrics.items()},
                **{f"p2/{k}": v for k, v in p2_metrics.items()},
                **{f"payout/{k}": v for k, v in payout_metrics.items()},
                **{f"value/{k}": v for k, v in value_metrics.items()},
            },
            n,
        )

    return acc.compute()


def main() -> None:
    """Main training loop."""
    parser = argparse.ArgumentParser(description="MCTS-style training")
    parser.add_argument("--train-dir", type=Path, required=True, help="Training data directory")
    parser.add_argument("--val-dir", type=Path, required=True, help="Validation data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output dir")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Value loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for tensorboard")
    args = parser.parse_args()

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mcts_{timestamp}"

    # Set seed
    torch.manual_seed(args.seed)

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard" / args.run_name
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"Tensorboard run: {args.run_name}")

    # Load datasets to GPU
    logger.info(f"Loading training data from {args.train_dir} to {device}")
    train_data = GPUDataset(args.train_dir, device)

    logger.info(f"Loading validation data from {args.val_dir} to {device}")
    val_data = GPUDataset(args.val_dir, device)

    # Get dimensions from manifest
    width = train_data.width
    height = train_data.height
    obs_dim = width * height * 7 + 4
    logger.info(f"Observation dimension: {obs_dim}")
    logger.info(f"Training positions: {len(train_data)}")
    logger.info(f"Validation positions: {len(val_data)}")

    # Create model
    model = PyRatMLP(obs_dim=obs_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model,
            train_data,
            optimizer,
            args.batch_size,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
        )

        # Validate
        val_metrics = validate(
            model,
            val_data,
            args.batch_size,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
        )

        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log to console
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train: {train_metrics['loss_total']:.4f} "
            f"(p1={train_metrics['loss_policy_p1']:.4f}, "
            f"p2={train_metrics['loss_policy_p2']:.4f}, "
            f"payout={train_metrics['loss_payout']:.4f}) | "
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
                    "args": vars(args),
                },
                args.output_dir / "best_model.pt",
            )
            logger.info(f"  New best model saved (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint
        if epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss_total"],
                    "args": vars(args),
                },
                args.output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
