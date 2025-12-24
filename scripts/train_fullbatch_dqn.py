#!/usr/bin/env python3
"""Full-batch DQN training - forward/backward on entire dataset each epoch.

DQN variant: sparse payout loss (only played action pair, not full 5x5 matrix).
No batching, no DataLoader. Simpler and faster for small datasets that fit in GPU memory.

Usage:
    uv run python scripts/train_fullbatch_dqn.py --train-dir data/train --val-dir data/val
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

from alpharat.nn.augmentation import swap_player_perspective_batch
from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.models import PyRatMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


def compute_loss_metrics(
    loss_p1: torch.Tensor,
    loss_p2: torch.Tensor,
    loss_value: torch.Tensor,
    loss_total: torch.Tensor,
) -> dict[str, float]:
    """Compute loss-only metrics (fast)."""
    return {
        "loss_total": loss_total.item(),
        "loss_policy_p1": loss_p1.item(),
        "loss_policy_p2": loss_p2.item(),
        "loss_value": loss_value.item(),
    }


def compute_detailed_metrics(
    logits_p1: torch.Tensor,
    logits_p2: torch.Tensor,
    pred_payout: torch.Tensor,
    target_p1: torch.Tensor,
    target_p2: torch.Tensor,
    target_payout: torch.Tensor,
    action_p1: torch.Tensor,
    action_p2: torch.Tensor,
    target_value: torch.Tensor,
) -> dict[str, float]:
    """Compute diagnostic metrics (expensive on full batch)."""
    p1_metrics = compute_policy_metrics(logits_p1.detach(), target_p1)
    p2_metrics = compute_policy_metrics(logits_p2.detach(), target_p2)
    payout_metrics = compute_payout_metrics(pred_payout.detach(), target_payout)
    value_metrics = compute_value_metrics(pred_payout.detach(), action_p1, action_p2, target_value)

    metrics: dict[str, float] = {}
    metrics.update({f"p1/{k}": v for k, v in p1_metrics.items()})
    metrics.update({f"p2/{k}": v for k, v in p2_metrics.items()})
    metrics.update({f"payout/{k}": v for k, v in payout_metrics.items()})
    metrics.update({f"value/{k}": v for k, v in value_metrics.items()})

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-batch DQN training")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--p-swap", type=float, default=0.5, help="Augmentation swap probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--metrics-every", type=int, default=10, help="Compute detailed metrics every N epochs"
    )
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f"fullbatch_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    torch.manual_seed(args.seed)

    # Device
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

    # Output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard" / args.run_name))
    logger.info(f"Tensorboard run: {args.run_name}")

    # Load all data to GPU
    logger.info(f"Loading training data from {args.train_dir}")
    train_data = GPUDataset(args.train_dir, device)
    train = train_data._data
    width, height = train_data.width, train_data.height

    logger.info(f"Loading validation data from {args.val_dir}")
    val_data = GPUDataset(args.val_dir, device)
    val = val_data._data

    obs_dim = width * height * 7 + 6
    logger.info(f"obs_dim={obs_dim}, train={len(train_data)}, val={len(val_data)}")

    # Model
    model = PyRatMLP(obs_dim=obs_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # === Train ===
        model.train()

        # Augment (in-place on train data)
        mask = torch.rand(len(train_data), device=device) < args.p_swap
        if mask.any():
            swap_player_perspective_batch(train, mask, width, height)

        optimizer.zero_grad()

        logits_p1, logits_p2, pred_payout = model(train["observation"])

        loss_p1 = F.cross_entropy(logits_p1, train["policy_p1"])
        loss_p2 = F.cross_entropy(logits_p2, train["policy_p2"])
        loss_value = sparse_payout_loss(
            pred_payout, train["action_p1"], train["action_p2"], train["value"]
        )
        train_loss = args.policy_weight * (loss_p1 + loss_p2) + args.value_weight * loss_value

        train_loss.backward()
        optimizer.step()

        # Train metrics (losses always, detailed metrics periodically)
        train_metrics = compute_loss_metrics(loss_p1, loss_p2, loss_value, train_loss)
        compute_detailed = epoch == 1 or epoch % args.metrics_every == 0

        if compute_detailed:
            train_metrics.update(
                compute_detailed_metrics(
                    logits_p1,
                    logits_p2,
                    pred_payout,
                    train["policy_p1"],
                    train["policy_p2"],
                    train["payout_matrix"],
                    train["action_p1"],
                    train["action_p2"],
                    train["value"],
                )
            )

        # === Validate ===
        model.eval()
        with torch.no_grad():
            val_logits_p1, val_logits_p2, val_pred_payout = model(val["observation"])

            val_loss_p1 = F.cross_entropy(val_logits_p1, val["policy_p1"])
            val_loss_p2 = F.cross_entropy(val_logits_p2, val["policy_p2"])
            val_loss_value = sparse_payout_loss(
                val_pred_payout, val["action_p1"], val["action_p2"], val["value"]
            )
            val_loss = (
                args.policy_weight * (val_loss_p1 + val_loss_p2)
                + args.value_weight * val_loss_value
            )

            val_metrics = compute_loss_metrics(val_loss_p1, val_loss_p2, val_loss_value, val_loss)

            if compute_detailed:
                val_metrics.update(
                    compute_detailed_metrics(
                        val_logits_p1,
                        val_logits_p2,
                        val_pred_payout,
                        val["policy_p1"],
                        val["policy_p2"],
                        val["payout_matrix"],
                        val["action_p1"],
                        val["action_p2"],
                        val["value"],
                    )
                )

        # Log to tensorboard
        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        # Log to console
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train: {train_metrics['loss_total']:.4f} "
            f"(p1={train_metrics['loss_policy_p1']:.4f}, "
            f"p2={train_metrics['loss_policy_p2']:.4f}, "
            f"val={train_metrics['loss_value']:.4f}) | "
            f"Val: {val_metrics['loss_total']:.4f}"
        )

        # Save best
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
            logger.info(f"  New best (val_loss={best_val_loss:.4f})")

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
    logger.info(f"Done. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
