"""Generic training loop for all model architectures.

This module provides a single training function that works with any architecture
by using dependency injection through the ModelConfig interface.

Usage:
    from alpharat.nn.training.loop import TrainConfig, run_training
    from alpharat.nn.architectures.mlp.config import MLPModelConfig, MLPOptimConfig

    config = TrainConfig(
        model=MLPModelConfig(hidden_dim=256),
        optim=MLPOptimConfig(lr=1e-3),
        data=DataConfig(train_dir="data/train", val_dir="data/val"),
    )
    run_training(config, epochs=100)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    GPUMetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.training.keys import LossKey
from alpharat.nn.training_utils import (
    backward_with_amp,
    select_device,
    setup_amp,
    setup_cuda_optimizations,
)

if TYPE_CHECKING:
    from torch import nn

    from alpharat.nn.training.config import BaseModelConfig, BaseOptimConfig, DataConfig
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction

logger = logging.getLogger(__name__)


def _compute_detailed_metrics(
    all_outputs: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute detailed metrics from accumulated batch outputs.

    Args:
        all_outputs: List of dicts with logits, targets, and payouts per batch.

    Returns:
        Dict of metric name -> value.
    """
    # Concatenate all batch outputs
    all_logits_p1 = torch.cat([d["logits_p1"] for d in all_outputs])
    all_logits_p2 = torch.cat([d["logits_p2"] for d in all_outputs])
    all_pred_payout = torch.cat([d["pred_payout"] for d in all_outputs])
    all_policy_p1 = torch.cat([d["policy_p1"] for d in all_outputs])
    all_policy_p2 = torch.cat([d["policy_p2"] for d in all_outputs])
    all_payout_matrix = torch.cat([d["payout_matrix"] for d in all_outputs])
    all_action_p1 = torch.cat([d["action_p1"] for d in all_outputs])
    all_action_p2 = torch.cat([d["action_p2"] for d in all_outputs])
    all_p1_value = torch.cat([d["p1_value"] for d in all_outputs])
    all_p2_value = torch.cat([d["p2_value"] for d in all_outputs])

    # Compute metrics on full epoch data
    p1_metrics = compute_policy_metrics(all_logits_p1, all_policy_p1)
    p2_metrics = compute_policy_metrics(all_logits_p2, all_policy_p2)
    payout_metrics = compute_payout_metrics(all_pred_payout, all_payout_matrix)
    value_metrics = compute_value_metrics(
        all_pred_payout, all_action_p1, all_action_p2, all_p1_value, all_p2_value
    )

    # Convert tensors to floats
    metrics: dict[str, float] = {}
    for k, v in p1_metrics.items():
        metrics[f"p1/{k}"] = v.item()
    for k, v in p2_metrics.items():
        metrics[f"p2/{k}"] = v.item()
    for k, v in payout_metrics.items():
        metrics[f"payout/{k}"] = v.item()
    for k, v in value_metrics.items():
        metrics[f"value/{k}"] = v.item()

    return metrics


def run_training(
    model_config: BaseModelConfig,
    optim_config: BaseOptimConfig,
    data_config: DataConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    use_amp: bool | None = None,
    seed: int = 42,
    resume_from: str | None = None,
) -> Path:
    """Run training loop for any model architecture.

    This is the generic training loop that works with any architecture through
    dependency injection. The model_config provides:
        - build_model(): Create the model instance
        - build_loss_fn(): Get the loss function
        - build_augmentation(): Get the augmentation strategy

    Args:
        model_config: Architecture-specific model configuration.
        optim_config: Optimization parameters (lr, weights, batch_size).
        data_config: Training and validation data paths.
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run. Auto-generated if None.
        use_amp: Enable AMP. None auto-detects, True/False forces on/off.
        seed: Random seed for reproducibility.
        resume_from: Path to checkpoint to resume from.

    Returns:
        Path to best model checkpoint.
    """
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"train_{timestamp}"

    # Set seed
    torch.manual_seed(seed)

    # Device and AMP setup
    torch_device = select_device(device)
    logger.info(f"Using device: {torch_device}")

    amp = setup_amp(torch_device, use_amp)
    amp_enabled = amp.enabled
    amp_dtype = amp.dtype
    scaler = amp.scaler

    # Create output directory
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"Run directory: {run_dir}")

    # Load data
    train_dir = Path(data_config.train_dir)
    val_dir = Path(data_config.val_dir)

    logger.info(f"Loading training data from {train_dir}")
    train_dataset = GPUDataset(train_dir, torch_device)
    train = train_dataset._data

    logger.info(f"Loading validation data from {val_dir}")
    val_dataset = GPUDataset(val_dir, torch_device)
    val = val_dataset._data

    width, height = train_dataset.width, train_dataset.height
    logger.info(
        f"width={width}, height={height}, train={len(train_dataset)}, val={len(val_dataset)}"
    )

    # Build model from config (models are nn.Module subclasses)
    model: nn.Module = model_config.build_model()  # type: ignore[assignment]
    model = model.to(torch_device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build loss function and augmentation from config
    loss_fn: LossFunction = model_config.build_loss_fn()
    augmentation: AugmentationStrategy = model_config.build_augmentation()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=optim_config.lr)

    # Resume from checkpoint if provided
    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint = None

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=torch_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resuming from {resume_from}, epoch {start_epoch}")

    # CUDA optimizations (TF32 + torch.compile)
    unwrapped_model = model
    model = setup_cuda_optimizations(model, torch_device)

    # Training loop
    best_model_path = run_dir / "best_model.pt"
    batch_size = optim_config.batch_size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    for epoch in range(start_epoch, start_epoch + epochs):
        compute_detailed = epoch == 1 or epoch % metrics_every == 0

        # === Train ===
        model.train()

        # Shuffle training data
        train_indices = torch.randperm(n_train, device=torch_device)

        # Accumulate metrics (GPU-resident, no per-batch sync)
        train_acc = GPUMetricsAccumulator(torch_device)
        train_detailed_outputs: list[dict[str, torch.Tensor]] = []

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_idx = train_indices[start_idx:end_idx]
            curr_batch_size = len(batch_idx)

            # Extract batch
            batch = {
                "observation": train["observation"][batch_idx],
                "policy_p1": train["policy_p1"][batch_idx],
                "policy_p2": train["policy_p2"][batch_idx],
                "p1_value": train["p1_value"][batch_idx],
                "p2_value": train["p2_value"][batch_idx],
                "payout_matrix": train["payout_matrix"][batch_idx],
                "action_p1": train["action_p1"][batch_idx],
                "action_p2": train["action_p2"][batch_idx],
            }

            # Include cheese_outcomes if available (for LocalValueMLP)
            if "cheese_outcomes" in train:
                batch["cheese_outcomes"] = train["cheese_outcomes"][batch_idx]

            # Apply augmentation
            if augmentation.needs_augmentation:
                batch = augmentation(batch, width, height)

            optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Pass cheese_mask if available (for LocalValueMLP ownership value)
                model_kwargs = {}
                if "cheese_outcomes" in batch:
                    model_kwargs["cheese_mask"] = batch["cheese_outcomes"] >= 0

                model_output = model(batch["observation"], **model_kwargs)
                losses = loss_fn(model_output, batch, optim_config)

            # Backward pass
            backward_with_amp(losses[LossKey.TOTAL], optimizer, scaler)

            # Accumulate loss metrics (no .item() - stays on GPU)
            loss_metrics = {k: v for k, v in losses.items() if k.startswith("loss")}
            train_acc.update(loss_metrics, batch_size=curr_batch_size)

            # Accumulate outputs for detailed metrics
            if compute_detailed:
                from alpharat.nn.training.keys import ModelOutput

                train_detailed_outputs.append(
                    {
                        "logits_p1": model_output[ModelOutput.LOGITS_P1].detach().clone(),
                        "logits_p2": model_output[ModelOutput.LOGITS_P2].detach().clone(),
                        "pred_payout": model_output[ModelOutput.PAYOUT].detach().clone(),
                        "policy_p1": batch["policy_p1"],
                        "policy_p2": batch["policy_p2"],
                        "payout_matrix": batch["payout_matrix"],
                        "action_p1": batch["action_p1"],
                        "action_p2": batch["action_p2"],
                        "p1_value": batch["p1_value"],
                        "p2_value": batch["p2_value"],
                    }
                )

        # Compute epoch metrics (single sync point)
        train_metrics = train_acc.compute()

        # Compute detailed metrics at epoch end
        if compute_detailed and train_detailed_outputs:
            with torch.no_grad():
                detailed = _compute_detailed_metrics(train_detailed_outputs)
                train_metrics.update(detailed)
            del train_detailed_outputs

        # === Validate ===
        model.eval()
        val_acc = GPUMetricsAccumulator(torch_device)
        val_detailed_outputs: list[dict[str, torch.Tensor]] = []

        with torch.no_grad():
            for start_idx in range(0, n_val, batch_size):
                end_idx = min(start_idx + batch_size, n_val)
                curr_batch_size = end_idx - start_idx

                val_batch = {
                    "observation": val["observation"][start_idx:end_idx],
                    "policy_p1": val["policy_p1"][start_idx:end_idx],
                    "policy_p2": val["policy_p2"][start_idx:end_idx],
                    "p1_value": val["p1_value"][start_idx:end_idx],
                    "p2_value": val["p2_value"][start_idx:end_idx],
                    "payout_matrix": val["payout_matrix"][start_idx:end_idx],
                    "action_p1": val["action_p1"][start_idx:end_idx],
                    "action_p2": val["action_p2"][start_idx:end_idx],
                }

                if "cheese_outcomes" in val:
                    val_batch["cheese_outcomes"] = val["cheese_outcomes"][start_idx:end_idx]

                with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Pass cheese_mask if available (for LocalValueMLP ownership value)
                    model_kwargs = {}
                    if "cheese_outcomes" in val_batch:
                        model_kwargs["cheese_mask"] = val_batch["cheese_outcomes"] >= 0

                    model_output = model(val_batch["observation"], **model_kwargs)
                    losses = loss_fn(model_output, val_batch, optim_config)

                loss_metrics = {k: v for k, v in losses.items() if k.startswith("loss")}
                val_acc.update(loss_metrics, batch_size=curr_batch_size)

                if compute_detailed:
                    from alpharat.nn.training.keys import ModelOutput

                    val_detailed_outputs.append(
                        {
                            "logits_p1": model_output[ModelOutput.LOGITS_P1].clone(),
                            "logits_p2": model_output[ModelOutput.LOGITS_P2].clone(),
                            "pred_payout": model_output[ModelOutput.PAYOUT].clone(),
                            "policy_p1": val_batch["policy_p1"],
                            "policy_p2": val_batch["policy_p2"],
                            "payout_matrix": val_batch["payout_matrix"],
                            "action_p1": val_batch["action_p1"],
                            "action_p2": val_batch["action_p2"],
                            "p1_value": val_batch["p1_value"],
                            "p2_value": val_batch["p2_value"],
                        }
                    )

            val_metrics = val_acc.compute()

            if compute_detailed and val_detailed_outputs:
                detailed = _compute_detailed_metrics(val_detailed_outputs)
                val_metrics.update(detailed)
                del val_detailed_outputs

        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log to console
        total_key = str(LossKey.TOTAL)
        p1_key = str(LossKey.POLICY_P1)
        p2_key = str(LossKey.POLICY_P2)
        value_key = str(LossKey.VALUE)

        logger.info(
            f"Epoch {epoch} - "
            f"Train: {train_metrics.get(total_key, 0):.4f} "
            f"(p1={train_metrics.get(p1_key, 0):.4f}, "
            f"p2={train_metrics.get(p2_key, 0):.4f}, "
            f"val={train_metrics.get(value_key, 0):.4f}) | "
            f"Val: {val_metrics.get(total_key, 0):.4f}"
        )

        # Build config for saving
        effective_config = {
            "model": model_config.model_dump(),
            "optim": optim_config.model_dump(),
            "data": data_config.model_dump(),
        }

        # Save best model
        if val_metrics.get(total_key, float("inf")) < best_val_loss:
            best_val_loss = val_metrics.get(total_key, float("inf"))
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
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
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics.get(total_key, 0),
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
