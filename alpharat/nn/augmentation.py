"""Data augmentation for training."""

from __future__ import annotations

import numpy as np  # noqa: TC002 - needed at runtime
import torch

from alpharat.nn.builders.flat import FlatObsLayout
from alpharat.nn.training.keys import BatchKey


def swap_player_perspective(
    observation: np.ndarray,
    policy_p1: np.ndarray,
    policy_p2: np.ndarray,
    value_p1: np.ndarray,
    value_p2: np.ndarray,
    action_p1: np.ndarray,
    action_p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Swap P1/P2 perspective in all tensors.

    Transforms the game state to view it from P2's perspective instead of P1's.
    This is an involution: applying it twice returns the original.

    Args:
        observation: Flat observation tensor from FlatObservationBuilder.
        policy_p1: P1 policy, shape (5,).
        policy_p2: P2 policy, shape (5,).
        value_p1: P1's remaining score, shape (1,).
        value_p2: P2's remaining score, shape (1,).
        action_p1: P1 action, shape (1,).
        action_p2: P2 action, shape (1,).
        width: Maze width.
        height: Maze height.

    Returns:
        Tuple of transformed tensors in same order as inputs.
    """
    lo = FlatObsLayout(width, height)
    s = lo.scalars_start

    # Transform observation
    new_obs = observation.copy()

    # Swap p1_pos and p2_pos segments
    new_obs[lo.p1_pos] = observation[lo.p2_pos]
    new_obs[lo.p2_pos] = observation[lo.p1_pos]

    # Negate score_diff
    new_obs[s + lo.SCORE_DIFF] = -observation[s + lo.SCORE_DIFF]

    # Swap p1_mud and p2_mud
    new_obs[s + lo.P1_MUD] = observation[s + lo.P2_MUD]
    new_obs[s + lo.P2_MUD] = observation[s + lo.P1_MUD]

    # Swap p1_score and p2_score
    new_obs[s + lo.P1_SCORE] = observation[s + lo.P2_SCORE]
    new_obs[s + lo.P2_SCORE] = observation[s + lo.P1_SCORE]

    # Transform targets
    new_policy_p1 = policy_p2.copy()
    new_policy_p2 = policy_p1.copy()

    # Swap values: new P1's value = old P2's value
    new_value_p1 = value_p2.copy()
    new_value_p2 = value_p1.copy()

    new_action_p1 = action_p2.copy()
    new_action_p2 = action_p1.copy()

    return (
        new_obs,
        new_policy_p1,
        new_policy_p2,
        new_value_p1,
        new_value_p2,
        new_action_p1,
        new_action_p2,
    )


@torch.compile
def swap_player_perspective_batch(
    batch: dict[str, torch.Tensor],
    mask: torch.Tensor,
    width: int,
    height: int,
) -> dict[str, torch.Tensor]:
    """Swap P1/P2 perspective for masked samples in a batch.

    Uses torch.where to avoid masked indexing, reducing GPU kernel launches.
    Compiled with torch.compile for kernel fusion (may have issues on MPS).

    Transforms applied:
        - observation: swap p1_pos/p2_pos, negate score_diff, swap mud/scores
        - policy_p1/p2: swap
        - action_p1/p2: swap
        - value_p1/value_p2: swap
        - cheese_outcomes: swap P1_WIN(0) <-> P2_WIN(3), keep others unchanged

    Args:
        batch: Dict with keys observation, policy_p1, policy_p2, value_p1, value_p2,
            action_p1, action_p2, cheese_outcomes. Shapes are (N, ...).
        mask: Boolean tensor of shape (N,) indicating which samples to augment.
        width: Maze width.
        height: Maze height.

    Returns:
        New batch dict with masked samples transformed.
    """
    if not mask.any():
        return batch

    # Expand mask for broadcasting with different tensor shapes
    # mask: (N,) -> mask_2d: (N, 1) -> broadcasts with (N, D) tensors
    # mask: (N,) -> mask_3d: (N, 1, 1) -> broadcasts with (N, 5, 5) or (N, H, W) tensors
    mask_2d = mask.unsqueeze(-1)
    mask_3d = mask_2d.unsqueeze(-1)

    lo = FlatObsLayout(width, height)
    s = lo.scalars_start

    # === Observation ===
    # Build fully-swapped observation, then select with torch.where
    obs = batch[BatchKey.OBSERVATION]
    swapped_obs = obs.clone()

    # Swap p1_pos and p2_pos (copy from original to avoid read-after-write issues)
    swapped_obs[:, lo.p1_pos] = obs[:, lo.p2_pos]
    swapped_obs[:, lo.p2_pos] = obs[:, lo.p1_pos]

    # Negate score_diff
    swapped_obs[:, s + lo.SCORE_DIFF] = -obs[:, s + lo.SCORE_DIFF]

    # Swap mud counters
    swapped_obs[:, s + lo.P1_MUD] = obs[:, s + lo.P2_MUD]
    swapped_obs[:, s + lo.P2_MUD] = obs[:, s + lo.P1_MUD]

    # Swap scores
    swapped_obs[:, s + lo.P1_SCORE] = obs[:, s + lo.P2_SCORE]
    swapped_obs[:, s + lo.P2_SCORE] = obs[:, s + lo.P1_SCORE]

    batch[BatchKey.OBSERVATION] = torch.where(mask_2d, swapped_obs, obs)

    # === Policies ===
    # Swap p1 and p2 policies based on mask
    p1 = batch[BatchKey.POLICY_P1]
    p2 = batch[BatchKey.POLICY_P2]
    batch[BatchKey.POLICY_P1] = torch.where(mask_2d, p2, p1)
    batch[BatchKey.POLICY_P2] = torch.where(mask_2d, p1, p2)

    # === Actions ===
    # Swap p1 and p2 actions based on mask
    a1 = batch[BatchKey.ACTION_P1]
    a2 = batch[BatchKey.ACTION_P2]
    batch[BatchKey.ACTION_P1] = torch.where(mask_2d, a2, a1)
    batch[BatchKey.ACTION_P2] = torch.where(mask_2d, a1, a2)

    # === Values ===
    # Swap value_p1 and value_p2 for swapped samples
    v1 = batch[BatchKey.VALUE_P1]
    v2 = batch[BatchKey.VALUE_P2]
    batch[BatchKey.VALUE_P1] = torch.where(mask_2d, v2, v1)
    batch[BatchKey.VALUE_P2] = torch.where(mask_2d, v1, v2)

    # === Cheese outcomes ===
    # Swap P1_WIN (0) <-> P2_WIN (3), keep SIMULTANEOUS (1), UNCOLLECTED (2), and -1 unchanged
    # This is an involution: swap(swap(x)) = x
    if BatchKey.CHEESE_OUTCOMES in batch:
        outcomes = batch[BatchKey.CHEESE_OUTCOMES]  # (N, H, W), int8 with values -1, 0, 1, 2, 3
        # Build swapped version: 0->3, 3->0, others unchanged
        swapped_outcomes = outcomes.clone()
        swapped_outcomes = torch.where(
            outcomes == 0, torch.tensor(3, device=outcomes.device), swapped_outcomes
        )
        swapped_outcomes = torch.where(
            outcomes == 3, torch.tensor(0, device=outcomes.device), swapped_outcomes
        )
        batch[BatchKey.CHEESE_OUTCOMES] = torch.where(mask_3d, swapped_outcomes, outcomes)

    return batch


class BatchAugmentation:
    """GPU batch-level augmentation pipeline.

    Applies augmentations to entire batches of data on GPU, much faster than
    per-sample augmentation in the dataset.

    Example:
        augment = BatchAugmentation(width=5, height=5, p_swap=0.5)

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = augment(batch)
            # ... forward pass
    """

    def __init__(self, width: int, height: int, *, p_swap: float = 0.5) -> None:
        """Initialize batch augmentation.

        Args:
            width: Maze width.
            height: Maze height.
            p_swap: Probability of applying player swap to each sample.
        """
        self.width = width
        self.height = height
        self.p_swap = p_swap

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply augmentations to a batch.

        Args:
            batch: Dict with observation, policy_p1, policy_p2, value_p1,
                value_p2, action_p1, action_p2 tensors.

        Returns:
            Augmented batch (modified in-place where possible).
        """
        n = batch[BatchKey.OBSERVATION].shape[0]
        device = batch[BatchKey.OBSERVATION].device

        # Generate augmentation mask for entire batch at once
        mask = torch.rand(n, device=device) < self.p_swap

        if mask.any():
            batch = swap_player_perspective_batch(batch, mask, self.width, self.height)

        return batch


# --- AugmentationStrategy implementations ---


class PlayerSwapStrategy:
    """Augmentation strategy that randomly swaps P1/P2 perspective.

    Implements the AugmentationStrategy protocol for use with the generic trainer.
    """

    def __init__(self, p_swap: float = 0.5) -> None:
        """Initialize player swap strategy.

        Args:
            p_swap: Probability of swapping each sample.
        """
        self.p_swap = p_swap

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor]:
        """Apply player swap augmentation."""
        n = batch[BatchKey.OBSERVATION].shape[0]
        device = batch[BatchKey.OBSERVATION].device
        mask = torch.rand(n, device=device) < self.p_swap

        if mask.any():
            batch = swap_player_perspective_batch(batch, mask, width, height)

        return batch

    @property
    def needs_augmentation(self) -> bool:
        """Player swap is a real augmentation."""
        return self.p_swap > 0


class NoAugmentation:
    """No-op augmentation strategy.

    Used for models with structural symmetry (e.g., SymmetricMLP) that don't
    need player swap augmentation.
    """

    def __call__(
        self,
        batch: dict[str, torch.Tensor],
        width: int,
        height: int,
    ) -> dict[str, torch.Tensor]:
        """Return batch unchanged."""
        return batch

    @property
    def needs_augmentation(self) -> bool:
        """No augmentation needed."""
        return False
