"""Training for SymmetricMLP with structural P1/P2 symmetry.

Similar to training.py but uses SymmetricMLP which has structural symmetry,
so no player swap augmentation is needed.

Usage:
    from alpharat.nn.symmetric_training import SymmetricTrainConfig, run_symmetric_training

    config = SymmetricTrainConfig.model_validate(yaml.safe_load(config_path.read_text()))
    run_symmetric_training(config, epochs=100, output_dir=Path("runs"))
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

from alpharat.nn.gpu_dataset import GPUDataset
from alpharat.nn.metrics import (
    GPUMetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    compute_value_metrics,
)
from alpharat.nn.models import SymmetricMLP
from alpharat.nn.training import constant_sum_loss, nash_consistency_loss

logger = logging.getLogger(__name__)


# --- Config Models ---


class SymmetricModelConfig(BaseModel):
    """Model architecture parameters for SymmetricMLP."""

    hidden_dim: int = 256
    dropout: float = 0.0


class SymmetricOptimConfig(BaseModel):
    """Optimization parameters for symmetric training.

    No p_augment field - structural symmetry makes augmentation unnecessary.
    """

    lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    nash_weight: float = 0.0  # Nash consistency loss weight (0 = disabled)
    nash_mode: Literal["target", "predicted"] = "target"
    constant_sum_weight: float = 0.0  # Constant-sum regularization weight (0 = disabled)
    batch_size: int = 4096


class SymmetricDataConfig(BaseModel):
    """Data paths."""

    train_dir: str
    val_dir: str


class SymmetricTrainConfig(BaseModel):
    """Full training configuration for SymmetricMLP.

    Either `model` or `resume_from` must be provided:
    - model only: fresh start with specified architecture
    - resume_from only: architecture loaded from checkpoint
    - both: validate they match, then resume
    """

    model: SymmetricModelConfig | None = None
    optim: SymmetricOptimConfig = SymmetricOptimConfig()
    data: SymmetricDataConfig
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
    model: SymmetricMLP,
    data: dict[str, torch.Tensor],
    policy_weight: float,
    value_weight: float,
    nash_weight: float = 0.0,
    nash_mode: Literal["target", "predicted"] = "target",
    constant_sum_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Forward pass and loss computation.

    Returns:
        Dict with keys: loss, loss_p1, loss_p2, loss_value, loss_nash, loss_indiff,
        loss_dev, loss_constant_sum, logits_p1, logits_p2, pred_payout.
    """
    logits_p1, logits_p2, pred_payout = model(data["observation"])

    loss_p1 = F.cross_entropy(logits_p1, data["policy_p1"])
    loss_p2 = F.cross_entropy(logits_p2, data["policy_p2"])

    # Sparse loss: supervise with actual game outcomes at played action pair
    loss_value = sparse_payout_loss(
        pred_payout,
        data["action_p1"],
        data["action_p2"],
        data["p1_value"],
        data["p2_value"],
    )

    # Nash consistency loss (game-theoretic constraint)
    if nash_weight > 0:
        if nash_mode == "predicted":
            # Use NN's own predicted policies — self-consistency regularization
            pi1 = F.softmax(logits_p1, dim=-1)
            pi2 = F.softmax(logits_p2, dim=-1)
        else:
            # Use target policies from MCTS
            pi1 = data["policy_p1"]
            pi2 = data["policy_p2"]
        loss_nash, loss_indiff, loss_dev = nash_consistency_loss(pred_payout, pi1, pi2)
    else:
        # Return zeros when disabled (for consistent logging)
        zero = torch.tensor(0.0, device=pred_payout.device)
        loss_nash, loss_indiff, loss_dev = zero, zero, zero

    # Constant-sum regularization (encourages P1 + P2 ≈ total collected)
    if constant_sum_weight > 0:
        loss_csum = constant_sum_loss(pred_payout, data["p1_value"], data["p2_value"])
    else:
        loss_csum = torch.tensor(0.0, device=pred_payout.device)

    loss = (
        policy_weight * (loss_p1 + loss_p2)
        + value_weight * loss_value
        + nash_weight * loss_nash
        + constant_sum_weight * loss_csum
    )

    return {
        "loss": loss,
        "loss_p1": loss_p1,
        "loss_p2": loss_p2,
        "loss_value": loss_value,
        "loss_nash": loss_nash,
        "loss_indiff": loss_indiff,
        "loss_dev": loss_dev,
        "loss_constant_sum": loss_csum,
        "logits_p1": logits_p1,
        "logits_p2": logits_p2,
        "pred_payout": pred_payout,
    }


# --- Main Training Function ---


def _resolve_model_config(
    config: SymmetricTrainConfig,
    torch_device: torch.device,
) -> tuple[SymmetricModelConfig, dict | None]:
    """Resolve model config from config.model or checkpoint.

    Returns:
        Tuple of (effective model config, checkpoint dict or None).

    Raises:
        ValueError: If neither model nor resume_from is provided, or if they conflict.
    """
    checkpoint = None

    if config.resume_from is not None:
        checkpoint = torch.load(config.resume_from, map_location=torch_device, weights_only=False)
        checkpoint_model = SymmetricModelConfig(**checkpoint["config"]["model"])

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


def _should_use_amp(device: torch.device) -> bool:
    """Check if AMP should be enabled based on device capabilities.

    Returns True for:
    - CUDA with compute capability >= 7.0 (Volta+ has Tensor Cores)
    - MPS (Apple Silicon has good float16 support)

    Returns False for:
    - CUDA with compute capability < 7.0 (modest benefit not worth complexity)
    - CPU (no benefit)
    """
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 7
    elif device.type == "mps":
        return True
    return False


def run_symmetric_training(
    config: SymmetricTrainConfig,
    *,
    epochs: int = 100,
    checkpoint_every: int = 10,
    metrics_every: int = 10,
    device: str = "auto",
    output_dir: Path = Path("checkpoints"),
    run_name: str | None = None,
    use_amp: bool | None = None,
) -> Path:
    """Run training loop for SymmetricMLP.

    Args:
        config: Training configuration (model, optimizer, data).
            Must have either `model` (fresh start) or `resume_from` (continue training).
        epochs: Number of epochs to train.
        checkpoint_every: Save checkpoint every N epochs.
        metrics_every: Compute detailed metrics every N epochs (losses always computed).
        device: Device to use ("auto", "cpu", "cuda", "mps").
        output_dir: Directory for checkpoints and logs.
        run_name: Name for this run (for tensorboard). Auto-generated if None.
        use_amp: Enable automatic mixed precision. None (default) auto-detects based
            on GPU capability. True forces AMP on, False forces it off.

    Returns:
        Path to best model checkpoint.
    """
    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"symmetric_{timestamp}"

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

    # AMP setup (auto-detect if not specified)
    amp_enabled = _should_use_amp(torch_device) if use_amp is None else use_amp
    amp_dtype = torch.float16 if amp_enabled else None

    # GradScaler only for CUDA (MPS doesn't support it)
    scaler = (
        torch.amp.GradScaler("cuda") if (amp_enabled and torch_device.type == "cuda") else None
    )

    if amp_enabled:
        capability = ""
        if torch_device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(torch_device)
            capability = f" (SM {major}.{minor})"
        scaler_str = "yes" if scaler else "no"
        logger.info(f"AMP enabled{capability}: dtype={amp_dtype}, scaler={scaler_str}")

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
    logger.info(
        f"width={width}, height={height}, train={len(train_dataset)}, val={len(val_dataset)}"
    )

    # Create model
    model = SymmetricMLP(
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

    # CUDA optimizations
    # Keep reference to original model for saving (torch.compile wraps it)
    unwrapped_model = model

    if torch_device.type == "cuda":
        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, A100, etc.)
        torch.set_float32_matmul_precision("high")
        logger.info("Enabled TensorFloat32 matmul precision")

        # Compile model for faster training
        # Options: "default" (balanced), "max-autotune" (slower start, faster run)
        model = torch.compile(model, mode="default")
        logger.info("Model compiled with torch.compile(mode='default')")

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

        # Accumulate metrics over mini-batches (GPU-resident, no per-batch sync)
        train_acc = GPUMetricsAccumulator(torch_device)

        # For detailed metrics: accumulate outputs, compute once at epoch end
        train_detailed_outputs: list[dict[str, torch.Tensor]] = []

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            batch_idx = train_indices[start_idx:end_idx]
            curr_batch_size = len(batch_idx)

            # Extract batch (no augmentation needed - structural symmetry)
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

            optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = compute_losses(
                    model,
                    batch,
                    optim_cfg.policy_weight,
                    optim_cfg.value_weight,
                    optim_cfg.nash_weight,
                    optim_cfg.nash_mode,
                    optim_cfg.constant_sum_weight,
                )

            # Backward pass with optional gradient scaling (CUDA only)
            if scaler is not None:
                scaler.scale(out["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out["loss"].backward()
                optimizer.step()

            # Accumulate loss metrics (no .item() - stays on GPU)
            train_acc.update(
                {
                    "loss_total": out["loss"],
                    "loss_policy_p1": out["loss_p1"],
                    "loss_policy_p2": out["loss_p2"],
                    "loss_value": out["loss_value"],
                    "loss_nash": out["loss_nash"],
                    "loss_indiff": out["loss_indiff"],
                    "loss_dev": out["loss_dev"],
                    "loss_constant_sum": out["loss_constant_sum"],
                },
                batch_size=curr_batch_size,
            )

            # Accumulate outputs for detailed metrics (computed at epoch end)
            # Note: .clone() needed because CUDA graphs reuse memory buffers
            if compute_detailed:
                train_detailed_outputs.append({
                    "logits_p1": out["logits_p1"].detach().clone(),
                    "logits_p2": out["logits_p2"].detach().clone(),
                    "pred_payout": out["pred_payout"].detach().clone(),
                    "policy_p1": batch["policy_p1"],
                    "policy_p2": batch["policy_p2"],
                    "payout_matrix": batch["payout_matrix"],
                    "action_p1": batch["action_p1"],
                    "action_p2": batch["action_p2"],
                    "p1_value": batch["p1_value"],
                    "p2_value": batch["p2_value"],
                })

        # Compute epoch metrics (single sync point)
        train_metrics = train_acc.compute()

        # Compute detailed metrics at epoch end (if needed)
        if compute_detailed and train_detailed_outputs:
            with torch.no_grad():
                # Concatenate all batch outputs
                all_logits_p1 = torch.cat([d["logits_p1"] for d in train_detailed_outputs])
                all_logits_p2 = torch.cat([d["logits_p2"] for d in train_detailed_outputs])
                all_pred_payout = torch.cat([d["pred_payout"] for d in train_detailed_outputs])
                all_policy_p1 = torch.cat([d["policy_p1"] for d in train_detailed_outputs])
                all_policy_p2 = torch.cat([d["policy_p2"] for d in train_detailed_outputs])
                all_payout_matrix = torch.cat([d["payout_matrix"] for d in train_detailed_outputs])
                all_action_p1 = torch.cat([d["action_p1"] for d in train_detailed_outputs])
                all_action_p2 = torch.cat([d["action_p2"] for d in train_detailed_outputs])
                all_p1_value = torch.cat([d["p1_value"] for d in train_detailed_outputs])
                all_p2_value = torch.cat([d["p2_value"] for d in train_detailed_outputs])

                # Compute metrics on full epoch data
                p1_metrics = compute_policy_metrics(all_logits_p1, all_policy_p1)
                p2_metrics = compute_policy_metrics(all_logits_p2, all_policy_p2)
                payout_metrics = compute_payout_metrics(all_pred_payout, all_payout_matrix)
                value_metrics = compute_value_metrics(
                    all_pred_payout, all_action_p1, all_action_p2, all_p1_value, all_p2_value
                )

                # Add to train_metrics (convert tensors to floats)
                for k, v in p1_metrics.items():
                    train_metrics[f"p1/{k}"] = v.item()
                for k, v in p2_metrics.items():
                    train_metrics[f"p2/{k}"] = v.item()
                for k, v in payout_metrics.items():
                    train_metrics[f"payout/{k}"] = v.item()
                for k, v in value_metrics.items():
                    train_metrics[f"value/{k}"] = v.item()

            # Free memory
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

                # Validation forward pass with AMP autocast
                with torch.autocast(torch_device.type, dtype=amp_dtype, enabled=amp_enabled):
                    vl_out = compute_losses(
                        model,
                        val_batch,
                        optim_cfg.policy_weight,
                        optim_cfg.value_weight,
                        optim_cfg.nash_weight,
                        optim_cfg.nash_mode,
                        optim_cfg.constant_sum_weight,
                    )

                val_acc.update(
                    {
                        "loss_total": vl_out["loss"],
                        "loss_policy_p1": vl_out["loss_p1"],
                        "loss_policy_p2": vl_out["loss_p2"],
                        "loss_value": vl_out["loss_value"],
                        "loss_nash": vl_out["loss_nash"],
                        "loss_indiff": vl_out["loss_indiff"],
                        "loss_dev": vl_out["loss_dev"],
                        "loss_constant_sum": vl_out["loss_constant_sum"],
                    },
                    batch_size=curr_batch_size,
                )

                if compute_detailed:
                    val_detailed_outputs.append({
                        "logits_p1": vl_out["logits_p1"].clone(),
                        "logits_p2": vl_out["logits_p2"].clone(),
                        "pred_payout": vl_out["pred_payout"].clone(),
                        "policy_p1": val_batch["policy_p1"],
                        "policy_p2": val_batch["policy_p2"],
                        "payout_matrix": val_batch["payout_matrix"],
                        "action_p1": val_batch["action_p1"],
                        "action_p2": val_batch["action_p2"],
                        "p1_value": val_batch["p1_value"],
                        "p2_value": val_batch["p2_value"],
                    })

            # Compute epoch metrics (single sync point)
            val_metrics = val_acc.compute()

            # Compute detailed metrics at epoch end (if needed)
            if compute_detailed and val_detailed_outputs:
                # Concatenate all batch outputs
                all_logits_p1 = torch.cat([d["logits_p1"] for d in val_detailed_outputs])
                all_logits_p2 = torch.cat([d["logits_p2"] for d in val_detailed_outputs])
                all_pred_payout = torch.cat([d["pred_payout"] for d in val_detailed_outputs])
                all_policy_p1 = torch.cat([d["policy_p1"] for d in val_detailed_outputs])
                all_policy_p2 = torch.cat([d["policy_p2"] for d in val_detailed_outputs])
                all_payout_matrix = torch.cat([d["payout_matrix"] for d in val_detailed_outputs])
                all_action_p1 = torch.cat([d["action_p1"] for d in val_detailed_outputs])
                all_action_p2 = torch.cat([d["action_p2"] for d in val_detailed_outputs])
                all_p1_value = torch.cat([d["p1_value"] for d in val_detailed_outputs])
                all_p2_value = torch.cat([d["p2_value"] for d in val_detailed_outputs])

                # Compute metrics on full epoch data
                p1_metrics = compute_policy_metrics(all_logits_p1, all_policy_p1)
                p2_metrics = compute_policy_metrics(all_logits_p2, all_policy_p2)
                payout_metrics = compute_payout_metrics(all_pred_payout, all_payout_matrix)
                value_metrics = compute_value_metrics(
                    all_pred_payout, all_action_p1, all_action_p2, all_p1_value, all_p2_value
                )

                # Add to val_metrics (convert tensors to floats)
                for k, v in p1_metrics.items():
                    val_metrics[f"p1/{k}"] = v.item()
                for k, v in p2_metrics.items():
                    val_metrics[f"p2/{k}"] = v.item()
                for k, v in payout_metrics.items():
                    val_metrics[f"payout/{k}"] = v.item()
                for k, v in value_metrics.items():
                    val_metrics[f"value/{k}"] = v.item()

                del val_detailed_outputs

        # Log to tensorboard
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        # Log to console
        nash_str = ""
        if optim_cfg.nash_weight > 0:
            nash_str = f", nash={train_metrics['loss_nash']:.4f}"
        logger.info(
            f"Epoch {epoch} - "
            f"Train: {train_metrics['loss_total']:.4f} "
            f"(p1={train_metrics['loss_policy_p1']:.4f}, "
            f"p2={train_metrics['loss_policy_p2']:.4f}, "
            f"val={train_metrics['loss_value']:.4f}{nash_str}) | "
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
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "best_val_loss": best_val_loss,
                    "config": effective_config,
                    "model_type": "symmetric",  # For loading detection
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
                    "val_loss": val_metrics["loss_total"],
                    "best_val_loss": best_val_loss,
                    "config": effective_config,
                    "model_type": "symmetric",
                    "width": width,
                    "height": height,
                },
                run_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return best_model_path
