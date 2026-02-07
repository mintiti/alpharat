"""Generic training loop for all model architectures.

This module provides a single training function that works with any architecture
by using dependency injection through the ModelConfig interface.

Usage:
    import yaml
    from alpharat.nn.config import TrainConfig
    from alpharat.nn.training import run_training

    config = TrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    run_training(config, epochs=100)

The architecture is determined by the 'architecture' field in the YAML config,
and Pydantic automatically dispatches to the correct config class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from alpharat.nn.metrics import (
    GPUMetricsAccumulator,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.training.keys import BatchKey, LossKey, ModelOutput
from alpharat.nn.training_utils import (
    backward_with_amp,
    select_device,
    setup_amp,
    setup_cuda_optimizations,
)

if TYPE_CHECKING:
    from torch import nn

    from alpharat.nn.config import TrainConfig
    from alpharat.nn.training.protocols import AugmentationStrategy, LossFunction

logger = logging.getLogger(__name__)


def _compute_detailed_metrics(
    all_outputs: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute detailed metrics from accumulated batch outputs.

    Args:
        all_outputs: List of dicts with logits, targets, and values per batch.

    Returns:
        Dict of metric name -> value.
    """
    # Concatenate all batch outputs
    all_logits_p1 = torch.cat([d[ModelOutput.LOGITS_P1] for d in all_outputs])
    all_logits_p2 = torch.cat([d[ModelOutput.LOGITS_P2] for d in all_outputs])
    all_pred_v1 = torch.cat([d[ModelOutput.VALUE_P1] for d in all_outputs])
    all_pred_v2 = torch.cat([d[ModelOutput.VALUE_P2] for d in all_outputs])
    all_policy_p1 = torch.cat([d[BatchKey.POLICY_P1] for d in all_outputs])
    all_policy_p2 = torch.cat([d[BatchKey.POLICY_P2] for d in all_outputs])
    all_p1_value = torch.cat([d[BatchKey.VALUE_P1] for d in all_outputs])
    all_p2_value = torch.cat([d[BatchKey.VALUE_P2] for d in all_outputs])

    # Compute metrics on full epoch data
    p1_metrics = compute_policy_metrics(all_logits_p1, all_policy_p1)
    p2_metrics = compute_policy_metrics(all_logits_p2, all_policy_p2)
    value_metrics = compute_value_metrics(all_pred_v1, all_pred_v2, all_p1_value, all_p2_value)

    # Convert tensors to floats
    metrics: dict[str, float] = {}
    for k, v in p1_metrics.items():
        metrics[f"p1/{k}"] = v.item()
    for k, v in p2_metrics.items():
        metrics[f"p2/{k}"] = v.item()
    for k, v in value_metrics.items():
        metrics[f"value/{k}"] = v.item()

    return metrics


def run_training(
    config: TrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str,
    use_amp: bool | None = None,
    checkpoints_subdir: str = "",
) -> Path:
    """Run training loop for any model architecture.

    This is the generic training loop that works with any architecture through
    dependency injection. The model_config provides:
        - build_model(): Create the model instance
        - build_loss_fn(): Get the loss function
        - build_augmentation(): Get the augmentation strategy

    Args:
        config: Full training configuration (model, optim, data, seed, resume_from).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs.
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (required, from config.name).
        use_amp: Enable AMP. None auto-detects, True/False forces on/off.
        checkpoints_subdir: Subdirectory for checkpoints within run_dir.
            Set to "checkpoints" when using ExperimentManager.

    Returns:
        Path to best model checkpoint.
    """
    # Unpack config
    model_config = config.model
    optim_config = config.optim
    data_config = config.data
    game_config = config.game
    seed = config.seed
    resume_from = config.resume_from

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

    from alpharat.nn.gpu_dataset import GPUDataset

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

    # Inject data dimensions into model config before building
    model_config.set_data_dimensions(width, height)

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
    # Checkpoints directory (subdir of run_dir if specified)
    checkpoint_dir = run_dir / checkpoints_subdir if checkpoints_subdir else run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pt"
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

            # Extract batch (typed as dict[str, ...] since BatchKey is StrEnum)
            batch: dict[str, torch.Tensor] = {
                BatchKey.OBSERVATION: train[BatchKey.OBSERVATION][batch_idx],
                BatchKey.POLICY_P1: train[BatchKey.POLICY_P1][batch_idx],
                BatchKey.POLICY_P2: train[BatchKey.POLICY_P2][batch_idx],
                BatchKey.VALUE_P1: train[BatchKey.VALUE_P1][batch_idx],
                BatchKey.VALUE_P2: train[BatchKey.VALUE_P2][batch_idx],
                BatchKey.ACTION_P1: train[BatchKey.ACTION_P1][batch_idx],
                BatchKey.ACTION_P2: train[BatchKey.ACTION_P2][batch_idx],
            }

            # Include cheese_outcomes if available (for LocalValueMLP)
            if BatchKey.CHEESE_OUTCOMES in train:
                batch[BatchKey.CHEESE_OUTCOMES] = train[BatchKey.CHEESE_OUTCOMES][batch_idx]

            # Apply augmentation
            if augmentation.needs_augmentation:
                batch = augmentation(batch, width, height)

            optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Pass cheese_mask if available (for LocalValueMLP ownership value)
                model_kwargs = {}
                if BatchKey.CHEESE_OUTCOMES in batch:
                    model_kwargs["cheese_mask"] = batch[BatchKey.CHEESE_OUTCOMES] >= 0

                model_output = model(batch[BatchKey.OBSERVATION], **model_kwargs)
                losses = loss_fn(model_output, batch, optim_config)

            # Backward pass
            backward_with_amp(losses[LossKey.TOTAL], optimizer, scaler)

            # Accumulate loss metrics (no .item() - stays on GPU)
            loss_metrics = {k: v for k, v in losses.items() if k.startswith("loss")}
            train_acc.update(loss_metrics, batch_size=curr_batch_size)

            # Accumulate outputs for detailed metrics
            if compute_detailed:
                train_detailed_outputs.append(
                    {
                        ModelOutput.LOGITS_P1: model_output[ModelOutput.LOGITS_P1].detach().clone(),
                        ModelOutput.LOGITS_P2: model_output[ModelOutput.LOGITS_P2].detach().clone(),
                        ModelOutput.VALUE_P1: model_output[ModelOutput.VALUE_P1].detach().clone(),
                        ModelOutput.VALUE_P2: model_output[ModelOutput.VALUE_P2].detach().clone(),
                        BatchKey.POLICY_P1: batch[BatchKey.POLICY_P1],
                        BatchKey.POLICY_P2: batch[BatchKey.POLICY_P2],
                        BatchKey.VALUE_P1: batch[BatchKey.VALUE_P1],
                        BatchKey.VALUE_P2: batch[BatchKey.VALUE_P2],
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

                val_batch: dict[str, torch.Tensor] = {
                    BatchKey.OBSERVATION: val[BatchKey.OBSERVATION][start_idx:end_idx],
                    BatchKey.POLICY_P1: val[BatchKey.POLICY_P1][start_idx:end_idx],
                    BatchKey.POLICY_P2: val[BatchKey.POLICY_P2][start_idx:end_idx],
                    BatchKey.VALUE_P1: val[BatchKey.VALUE_P1][start_idx:end_idx],
                    BatchKey.VALUE_P2: val[BatchKey.VALUE_P2][start_idx:end_idx],
                    BatchKey.ACTION_P1: val[BatchKey.ACTION_P1][start_idx:end_idx],
                    BatchKey.ACTION_P2: val[BatchKey.ACTION_P2][start_idx:end_idx],
                }

                if BatchKey.CHEESE_OUTCOMES in val:
                    cheese_key = BatchKey.CHEESE_OUTCOMES
                    val_batch[cheese_key] = val[cheese_key][start_idx:end_idx]

                with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                    # Pass cheese_mask if available (for LocalValueMLP ownership value)
                    model_kwargs = {}
                    if BatchKey.CHEESE_OUTCOMES in val_batch:
                        model_kwargs["cheese_mask"] = val_batch[BatchKey.CHEESE_OUTCOMES] >= 0

                    model_output = model(val_batch[BatchKey.OBSERVATION], **model_kwargs)
                    losses = loss_fn(model_output, val_batch, optim_config)

                loss_metrics = {k: v for k, v in losses.items() if k.startswith("loss")}
                val_acc.update(loss_metrics, batch_size=curr_batch_size)

                if compute_detailed:
                    val_detailed_outputs.append(
                        {
                            ModelOutput.LOGITS_P1: model_output[ModelOutput.LOGITS_P1].clone(),
                            ModelOutput.LOGITS_P2: model_output[ModelOutput.LOGITS_P2].clone(),
                            ModelOutput.VALUE_P1: model_output[ModelOutput.VALUE_P1].clone(),
                            ModelOutput.VALUE_P2: model_output[ModelOutput.VALUE_P2].clone(),
                            BatchKey.POLICY_P1: val_batch[BatchKey.POLICY_P1],
                            BatchKey.POLICY_P2: val_batch[BatchKey.POLICY_P2],
                            BatchKey.VALUE_P1: val_batch[BatchKey.VALUE_P1],
                            BatchKey.VALUE_P2: val_batch[BatchKey.VALUE_P2],
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
            "game": game_config.model_dump() if game_config else None,
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
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return best_model_path
