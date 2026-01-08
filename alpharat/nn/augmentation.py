"""Data augmentation for training."""

from __future__ import annotations

import numpy as np  # noqa: TC002 - needed at runtime
import torch


def swap_player_perspective(
    observation: np.ndarray,
    policy_p1: np.ndarray,
    policy_p2: np.ndarray,
    p1_value: np.ndarray,
    p2_value: np.ndarray,
    payout_matrix: np.ndarray,
    action_p1: np.ndarray,
    action_p2: np.ndarray,
    width: int,
    height: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Swap P1/P2 perspective in all tensors.

    Transforms the game state to view it from P2's perspective instead of P1's.
    This is an involution: applying it twice returns the original.

    Observation layout (FlatObservationBuilder):
        [maze (H×W×4)] [p1_pos (H×W)] [p2_pos (H×W)] [cheese (H×W)]
        [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]

    Args:
        observation: Flat observation tensor from FlatObservationBuilder.
        policy_p1: P1 policy, shape (5,).
        policy_p2: P2 policy, shape (5,).
        p1_value: P1's remaining score, shape (1,).
        p2_value: P2's remaining score, shape (1,).
        payout_matrix: Payout matrix, shape (2, 5, 5).
        action_p1: P1 action, shape (1,).
        action_p2: P2 action, shape (1,).
        width: Maze width.
        height: Maze height.

    Returns:
        Tuple of transformed tensors in same order as inputs.
    """
    spatial = width * height

    # Compute offsets
    maze_end = spatial * 4
    p1_pos_start = maze_end
    p1_pos_end = maze_end + spatial
    p2_pos_start = p1_pos_end
    p2_pos_end = p2_pos_start + spatial
    # cheese is from p2_pos_end to p2_pos_end + spatial (preserved)
    scalars_start = spatial * 7  # After maze + p1_pos + p2_pos + cheese

    # Transform observation
    new_obs = observation.copy()

    # Swap p1_pos and p2_pos segments
    new_obs[p1_pos_start:p1_pos_end] = observation[p2_pos_start:p2_pos_end]
    new_obs[p2_pos_start:p2_pos_end] = observation[p1_pos_start:p1_pos_end]

    # Negate score_diff (index 0 of scalars)
    new_obs[scalars_start] = -observation[scalars_start]

    # Swap p1_mud and p2_mud (indices 2 and 3 of scalars)
    new_obs[scalars_start + 2] = observation[scalars_start + 3]
    new_obs[scalars_start + 3] = observation[scalars_start + 2]

    # Swap p1_score and p2_score (indices 4 and 5 of scalars)
    new_obs[scalars_start + 4] = observation[scalars_start + 5]
    new_obs[scalars_start + 5] = observation[scalars_start + 4]

    # Transform targets
    new_policy_p1 = policy_p2.copy()
    new_policy_p2 = policy_p1.copy()

    # Swap values: new P1's value = old P2's value
    new_p1_value = p2_value.copy()
    new_p2_value = p1_value.copy()

    # Bimatrix: swap player indices and transpose
    # new_payout[0] = payout[1].T (new P1's payoffs = old P2's, transposed)
    # new_payout[1] = payout[0].T (new P2's payoffs = old P1's, transposed)
    new_payout = np.empty_like(payout_matrix)
    new_payout[0] = payout_matrix[1].T
    new_payout[1] = payout_matrix[0].T

    new_action_p1 = action_p2.copy()
    new_action_p2 = action_p1.copy()

    return (
        new_obs,
        new_policy_p1,
        new_policy_p2,
        new_p1_value,
        new_p2_value,
        new_payout,
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
        - p1_value/p2_value: swap
        - payout_matrix: swap player indices and transpose
        - cheese_outcomes: swap P1_WIN(0) <-> P2_WIN(3), keep others unchanged

    Args:
        batch: Dict with keys observation, policy_p1, policy_p2, p1_value, p2_value,
            payout_matrix, action_p1, action_p2, cheese_outcomes. Shapes are (N, ...).
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

    spatial = width * height

    # Compute offsets into flat observation
    p1_pos_start = spatial * 4
    p1_pos_end = spatial * 5
    p2_pos_start = spatial * 5
    p2_pos_end = spatial * 6
    scalars_start = spatial * 7

    # === Observation ===
    # Build fully-swapped observation, then select with torch.where
    obs = batch["observation"]
    swapped_obs = obs.clone()

    # Swap p1_pos and p2_pos (copy from original to avoid read-after-write issues)
    swapped_obs[:, p1_pos_start:p1_pos_end] = obs[:, p2_pos_start:p2_pos_end]
    swapped_obs[:, p2_pos_start:p2_pos_end] = obs[:, p1_pos_start:p1_pos_end]

    # Negate score_diff
    swapped_obs[:, scalars_start] = -obs[:, scalars_start]

    # Swap mud counters (indices 2, 3 of scalars)
    swapped_obs[:, scalars_start + 2] = obs[:, scalars_start + 3]
    swapped_obs[:, scalars_start + 3] = obs[:, scalars_start + 2]

    # Swap scores (indices 4, 5 of scalars)
    swapped_obs[:, scalars_start + 4] = obs[:, scalars_start + 5]
    swapped_obs[:, scalars_start + 5] = obs[:, scalars_start + 4]

    batch["observation"] = torch.where(mask_2d, swapped_obs, obs)

    # === Policies ===
    # Swap p1 and p2 policies based on mask
    p1 = batch["policy_p1"]
    p2 = batch["policy_p2"]
    batch["policy_p1"] = torch.where(mask_2d, p2, p1)
    batch["policy_p2"] = torch.where(mask_2d, p1, p2)

    # === Actions ===
    # Swap p1 and p2 actions based on mask
    a1 = batch["action_p1"]
    a2 = batch["action_p2"]
    batch["action_p1"] = torch.where(mask_2d, a2, a1)
    batch["action_p2"] = torch.where(mask_2d, a1, a2)

    # === Values ===
    # Swap p1_value and p2_value for swapped samples
    v1 = batch["p1_value"]
    v2 = batch["p2_value"]
    batch["p1_value"] = torch.where(mask_2d, v2, v1)
    batch["p2_value"] = torch.where(mask_2d, v1, v2)

    # === Payout matrix ===
    # Bimatrix: swap player indices and transpose for swapped samples
    # new_payout[:, 0] = payout[:, 1].T, new_payout[:, 1] = payout[:, 0].T
    payout = batch["payout_matrix"]  # (N, 2, 5, 5)
    swapped_payout = payout.flip(dims=[1]).transpose(-1, -2)
    mask_4d = mask_2d.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1, 1)
    batch["payout_matrix"] = torch.where(mask_4d, swapped_payout, payout)

    # === Cheese outcomes ===
    # Swap P1_WIN (0) <-> P2_WIN (3), keep SIMULTANEOUS (1), UNCOLLECTED (2), and -1 unchanged
    # This is an involution: swap(swap(x)) = x
    if "cheese_outcomes" in batch:
        outcomes = batch["cheese_outcomes"]  # (N, H, W), int8 with values -1, 0, 1, 2, 3
        # Build swapped version: 0->3, 3->0, others unchanged
        swapped_outcomes = outcomes.clone()
        swapped_outcomes = torch.where(
            outcomes == 0, torch.tensor(3, device=outcomes.device), swapped_outcomes
        )
        swapped_outcomes = torch.where(
            outcomes == 3, torch.tensor(0, device=outcomes.device), swapped_outcomes
        )
        batch["cheese_outcomes"] = torch.where(mask_3d, swapped_outcomes, outcomes)

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
            batch: Dict with observation, policy_p1, policy_p2, value,
                payout_matrix, action_p1, action_p2 tensors.

        Returns:
            Augmented batch (modified in-place where possible).
        """
        n = batch["observation"].shape[0]
        device = batch["observation"].device

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
        n = batch["observation"].shape[0]
        device = batch["observation"].device
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
