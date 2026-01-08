"""Ownership loss for LocalValueMLP - per-cheese ownership prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
