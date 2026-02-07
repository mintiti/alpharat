"""Streaming dataset for memory-efficient training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import IterableDataset

from alpharat.nn.training.keys import BatchKey

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
                n_samples = len(shard_data[BatchKey.VALUE_P1])
                for j in range(n_samples):
                    yield {
                        BatchKey.OBSERVATION: torch.from_numpy(
                            shard_data[BatchKey.OBSERVATION][j].copy()
                        ),
                        BatchKey.POLICY_P1: torch.from_numpy(
                            shard_data[BatchKey.POLICY_P1][j].copy()
                        ),
                        BatchKey.POLICY_P2: torch.from_numpy(
                            shard_data[BatchKey.POLICY_P2][j].copy()
                        ),
                        BatchKey.VALUE_P1: torch.from_numpy(
                            shard_data[BatchKey.VALUE_P1][j : j + 1].copy()
                        ),
                        BatchKey.VALUE_P2: torch.from_numpy(
                            shard_data[BatchKey.VALUE_P2][j : j + 1].copy()
                        ),
                        BatchKey.ACTION_P1: torch.from_numpy(
                            shard_data[BatchKey.ACTION_P1][j : j + 1].copy()
                        ),
                        BatchKey.ACTION_P2: torch.from_numpy(
                            shard_data[BatchKey.ACTION_P2][j : j + 1].copy()
                        ),
                        BatchKey.CHEESE_OUTCOMES: torch.from_numpy(
                            shard_data[BatchKey.CHEESE_OUTCOMES][j].copy()
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
        Dict with observation, policies, p1/p2 values, actions, cheese_outcomes.
        cheese_outcomes uses -1 sentinel for inactive cells, 0-3 for outcome classes.
    """
    with np.load(path) as data:
        return {
            BatchKey.OBSERVATION: np.array(data[BatchKey.OBSERVATION]),
            BatchKey.POLICY_P1: np.array(data[BatchKey.POLICY_P1]),
            BatchKey.POLICY_P2: np.array(data[BatchKey.POLICY_P2]),
            BatchKey.VALUE_P1: np.array(data[BatchKey.VALUE_P1]),
            BatchKey.VALUE_P2: np.array(data[BatchKey.VALUE_P2]),
            BatchKey.ACTION_P1: np.array(data[BatchKey.ACTION_P1]),
            BatchKey.ACTION_P2: np.array(data[BatchKey.ACTION_P2]),
            BatchKey.CHEESE_OUTCOMES: np.array(data[BatchKey.CHEESE_OUTCOMES]),
        }
