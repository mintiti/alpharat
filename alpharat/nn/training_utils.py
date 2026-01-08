"""Shared utilities for training modules.

This module extracts common patterns used across training.py, symmetric_training.py,
and local_value_training.py to ensure consistent behavior and reduce duplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def select_device(device: str) -> torch.device:
    """Select compute device with auto-detection.

    Args:
        device: Device specifier ("auto", "cpu", "cuda", "mps").

    Returns:
        torch.device for the selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def should_use_amp(device: torch.device) -> bool:
    """Check if AMP should be enabled based on device capabilities.

    Returns True for:
    - CUDA with compute capability >= 7.0 (Volta+ has Tensor Cores)
    - MPS (Apple Silicon has good float16 support)

    Returns False for:
    - CUDA with compute capability < 7.0 (modest benefit not worth complexity)
    - CPU (no benefit)

    Args:
        device: The torch device being used.

    Returns:
        Whether AMP is recommended for this device.
    """
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 7
    elif device.type == "mps":
        return True
    return False


@dataclass
class AMPContext:
    """Context for Automatic Mixed Precision training.

    Attributes:
        enabled: Whether AMP is enabled.
        dtype: The dtype to use for autocast (float16 or None).
        scaler: GradScaler for CUDA (None for MPS/CPU).
    """

    enabled: bool
    dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None


def setup_amp(device: torch.device, use_amp: bool | None = None) -> AMPContext:
    """Configure Automatic Mixed Precision for a device.

    Args:
        device: The torch device being used.
        use_amp: Override for AMP. None auto-detects, True forces on, False forces off.

    Returns:
        AMPContext with enabled flag, dtype, and optional scaler.
    """
    enabled = should_use_amp(device) if use_amp is None else use_amp
    dtype = torch.float16 if enabled else None

    # GradScaler only for CUDA (MPS doesn't support it)
    scaler = torch.amp.GradScaler("cuda") if (enabled and device.type == "cuda") else None

    if enabled:
        capability = ""
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            capability = f" (SM {major}.{minor})"
        scaler_str = "yes" if scaler else "no"
        logger.info(f"AMP enabled{capability}: dtype={dtype}, scaler={scaler_str}")

    return AMPContext(enabled=enabled, dtype=dtype, scaler=scaler)


def setup_cuda_optimizations(model: nn.Module, device: torch.device) -> nn.Module:
    """Apply CUDA-specific optimizations: TF32 matmul and torch.compile.

    Args:
        model: The model to optimize.
        device: The torch device.

    Returns:
        The (possibly compiled) model. Always keep a reference to the original
        model for saving checkpoints, as torch.compile wraps the model.
    """
    if device.type != "cuda":
        return model

    # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, A100, etc.)
    torch.set_float32_matmul_precision("high")
    logger.info("Enabled TensorFloat32 matmul precision")

    # Compile model for faster training
    # torch.compile returns a callable that acts like the model
    compiled: nn.Module = torch.compile(model, mode="default")  # type: ignore[assignment]
    logger.info("Model compiled with torch.compile(mode='default')")

    return compiled


def backward_with_amp(
    loss: Tensor,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
) -> None:
    """Perform backward pass with optional gradient scaling.

    Args:
        loss: The loss tensor to backpropagate.
        optimizer: The optimizer to step.
        scaler: GradScaler for CUDA AMP (None for MPS/CPU/no-AMP).
    """
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
