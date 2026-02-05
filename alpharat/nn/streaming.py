"""Streaming dataset for memory-efficient training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alpharat.data.sharding import TrainingSetManifest


class StreamingDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Streams training data from shards with prefetching.

    Memory-efficient alternative to FlatDataset. Loads one shard at a time
    while prefetching the next.

    Data is already globally shuffled at shard creation time by prepare_training_set().
    This dataset shuffles shard order each epoch for additional variety.

    Memory usage:
        Typically ~2 shards (current + prefetched). If batch_size > shard_size,
        DataLoader buffers samples across shards, so memory scales with
        max(2, ceil(batch_size / shard_size) + 1) shards. For large batches,
        increase positions_per_shard at shard creation time.

    Use with DataLoader:
        dataset = StreamingDataset(training_set_dir, shuffle_shards=True, seed=42)
        loader = DataLoader(dataset, batch_size=64, pin_memory=True)
        for batch in loader:
            ...
    """

    def __init__(
        self,
        training_set_dir: Path | str,
        *,
        shuffle_shards: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            training_set_dir: Path to training set with manifest.json and shards.
            shuffle_shards: Whether to shuffle shard order each epoch.
            seed: Random seed for shard shuffling. If None, uses random seed.
        """
        from alpharat.data.sharding import load_training_set_manifest

        self._training_set_dir = Path(training_set_dir)
        self._manifest = load_training_set_manifest(self._training_set_dir)
        self._shuffle_shards = shuffle_shards
        self._seed = seed
        self._epoch = 0

        # Build shard paths
        self._shard_paths = [
            self._training_set_dir / f"shard_{i:04d}.npz" for i in range(self._manifest.shard_count)
        ]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield samples one at a time with shard prefetching.

        Note: Augmentation should be applied after batching using BatchAugmentation.
        """
        # Determine shard order for this epoch
        shard_indices = list(range(len(self._shard_paths)))
        epoch_seed = (self._seed or 0) + self._epoch
        rng = np.random.default_rng(epoch_seed)

        if self._shuffle_shards:
            rng.shuffle(shard_indices)

        self._epoch += 1

        # Stream through shards with prefetching
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_load_shard, self._shard_paths[shard_indices[0]])

            for i, _shard_idx in enumerate(shard_indices):
                # Get current shard
                shard_data = future.result()

                # Prefetch next shard if there is one
                if i + 1 < len(shard_indices):
                    next_idx = shard_indices[i + 1]
                    future = executor.submit(_load_shard, self._shard_paths[next_idx])

                # Yield each sample from this shard
                n_samples = len(shard_data["value_p1"])
                for j in range(n_samples):
                    yield {
                        "observation": torch.from_numpy(shard_data["observations"][j].copy()),
                        "policy_p1": torch.from_numpy(shard_data["policy_p1"][j].copy()),
                        "policy_p2": torch.from_numpy(shard_data["policy_p2"][j].copy()),
                        "value_p1": torch.from_numpy(shard_data["value_p1"][j : j + 1].copy()),
                        "value_p2": torch.from_numpy(shard_data["value_p2"][j : j + 1].copy()),
                        "action_p1": torch.from_numpy(shard_data["action_p1"][j : j + 1].copy()),
                        "action_p2": torch.from_numpy(shard_data["action_p2"][j : j + 1].copy()),
                        # cheese_outcomes: int8 with -1=inactive, 0-3=outcome class
                        "cheese_outcomes": torch.from_numpy(
                            shard_data["cheese_outcomes"][j].copy()
                        ),
                    }

    def __len__(self) -> int:
        """Total positions across all shards."""
        return self._manifest.total_positions

    @property
    def manifest(self) -> TrainingSetManifest:
        """Training set manifest."""
        return self._manifest


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    """Load shard npz file.

    Args:
        path: Path to shard npz file.

    Returns:
        Dict with observations, policies, p1/p2 values, actions, cheese_outcomes.
        cheese_outcomes uses -1 sentinel for inactive cells, 0-3 for outcome classes.
    """
    with np.load(path) as data:
        return {
            "observations": np.array(data["observations"]),
            "policy_p1": np.array(data["policy_p1"]),
            "policy_p2": np.array(data["policy_p2"]),
            "value_p1": np.array(data["value_p1"]),
            "value_p2": np.array(data["value_p2"]),
            "action_p1": np.array(data["action_p1"]),
            "action_p2": np.array(data["action_p2"]),
            "cheese_outcomes": np.array(data["cheese_outcomes"]),
        }
