"""GPU-resident dataset for fast training on small datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from alpharat.nn.augmentation import swap_player_perspective_batch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alpharat.data.sharding import TrainingSetManifest


class GPUDataset:
    """GPU-resident dataset with epoch-level augmentation.

    Loads all data to GPU at initialization. Each epoch:
    1. Applies augmentation to entire dataset (in-place)
    2. Shuffles indices on GPU
    3. Yields pre-sliced batches

    For small datasets (~1GB or less) this eliminates all CPU-GPU transfers
    during training.

    Example:
        dataset = GPUDataset(train_dir, device)

        for epoch in range(epochs):
            for batch in dataset.epoch_iter(batch_size=256):
                # batch already on GPU, already augmented
                logits = model(batch["observation"])
                ...
    """

    def __init__(self, data_dir: Path | str, device: torch.device) -> None:
        """Load all shards to GPU.

        Args:
            data_dir: Path to training set with manifest.json and shards.
            device: Target device (cuda, mps, cpu).
        """
        from alpharat.data.sharding import load_training_set_manifest

        self._data_dir = Path(data_dir)
        self._manifest = load_training_set_manifest(self._data_dir)
        self._device = device

        # Load all shards
        shard_paths = [
            self._data_dir / f"shard_{i:04d}.npz" for i in range(self._manifest.shard_count)
        ]

        arrays: dict[str, list[np.ndarray]] = {
            "observation": [],
            "policy_p1": [],
            "policy_p2": [],
            "value": [],
            "payout_matrix": [],
            "action_p1": [],
            "action_p2": [],
        }

        for path in shard_paths:
            with np.load(path) as data:
                arrays["observation"].append(np.array(data["observations"]))
                arrays["policy_p1"].append(np.array(data["policy_p1"]))
                arrays["policy_p2"].append(np.array(data["policy_p2"]))
                arrays["value"].append(np.array(data["value"]))
                arrays["payout_matrix"].append(np.array(data["payout_matrix"]))
                arrays["action_p1"].append(np.array(data["action_p1"]))
                arrays["action_p2"].append(np.array(data["action_p2"]))

        # Concatenate and move to GPU
        self._data: dict[str, torch.Tensor] = {}
        for key, arrs in arrays.items():
            concat = np.concatenate(arrs, axis=0)
            self._data[key] = torch.from_numpy(concat).to(device)

        # Ensure value and actions have shape (N, 1) for consistency
        if self._data["value"].dim() == 1:
            self._data["value"] = self._data["value"].unsqueeze(-1)
        if self._data["action_p1"].dim() == 1:
            self._data["action_p1"] = self._data["action_p1"].unsqueeze(-1)
        if self._data["action_p2"].dim() == 1:
            self._data["action_p2"] = self._data["action_p2"].unsqueeze(-1)

        self._n_samples = self._data["observation"].shape[0]

    def __len__(self) -> int:
        """Total number of samples."""
        return self._n_samples

    @property
    def manifest(self) -> TrainingSetManifest:
        """Training set manifest."""
        return self._manifest

    @property
    def width(self) -> int:
        """Maze width."""
        return self._manifest.width

    @property
    def height(self) -> int:
        """Maze height."""
        return self._manifest.height

    def epoch_iter(
        self,
        batch_size: int,
        *,
        augment: bool = True,
        p_swap: float = 0.5,
        shuffle: bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches for one epoch.

        Args:
            batch_size: Number of samples per batch.
            augment: Whether to apply player-swap augmentation.
            p_swap: Probability of swapping each sample (if augment=True).
            shuffle: Whether to shuffle data order.

        Yields:
            Batch dicts with observation, policy_p1, policy_p2, value,
            payout_matrix, action_p1, action_p2 tensors.
        """
        # Apply augmentation to entire dataset (in-place)
        if augment:
            mask = torch.rand(self._n_samples, device=self._device) < p_swap
            if mask.any():
                swap_player_perspective_batch(
                    self._data, mask, self._manifest.width, self._manifest.height
                )

        # Generate shuffled indices
        if shuffle:
            indices = torch.randperm(self._n_samples, device=self._device)
        else:
            indices = torch.arange(self._n_samples, device=self._device)

        # Compute number of complete batches (drop incomplete)
        n_batches = self._n_samples // batch_size

        # Yield batches
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]

            yield {
                "observation": self._data["observation"][batch_indices],
                "policy_p1": self._data["policy_p1"][batch_indices],
                "policy_p2": self._data["policy_p2"][batch_indices],
                "value": self._data["value"][batch_indices],
                "payout_matrix": self._data["payout_matrix"][batch_indices],
                "action_p1": self._data["action_p1"][batch_indices],
                "action_p2": self._data["action_p2"][batch_indices],
            }
