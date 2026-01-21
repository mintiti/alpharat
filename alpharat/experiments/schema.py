"""Pydantic schemas for experiment manifest entries.

These schemas define the structure of entries in the experiments manifest,
tracking lineage and metadata for batches, shards, training runs, and benchmarks.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import Field

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.mcts import MCTSConfig  # noqa: TC001


class BatchEntry(StrictBaseModel):
    """Manifest entry for a sampling batch.

    Tracks the provenance of sampled game data, including the MCTS configuration
    and optional parent checkpoint used for NN-guided sampling.
    """

    group: str
    uuid: str
    created_at: datetime
    parent_checkpoint: str | None
    mcts_config: MCTSConfig = Field(discriminator="variant")
    game: GameConfig
    seed_start: int = 0  # First game uses this seed, increments from there


class ShardEntry(StrictBaseModel):
    """Manifest entry for a processed shard set.

    Tracks processed training data derived from one or more batches,
    including train/val split statistics.
    """

    group: str
    uuid: str
    created_at: datetime
    source_batches: list[str]
    total_positions: int
    train_positions: int
    val_positions: int
    shuffle_seed: int | None = None  # Seed used for train/val split and shuffling


class RunEntry(StrictBaseModel):
    """Manifest entry for a training run.

    Tracks a training run's configuration, data source, and results.
    The config is stored as a dict to avoid circular imports with TrainConfig.
    """

    name: str
    created_at: datetime
    source_shards: str
    parent_checkpoint: str | None
    config: dict[str, Any]
    best_val_loss: float | None = None
    final_epoch: int | None = None


class BenchmarkEntry(StrictBaseModel):
    """Manifest entry for a benchmark or tournament.

    Tracks benchmark configuration and which checkpoints were evaluated.
    """

    name: str
    created_at: datetime
    checkpoints: list[str]
    config: dict[str, Any]


class Manifest(StrictBaseModel):
    """Full experiments manifest tracking all artifacts and their lineage.

    The manifest is the central index for the experiments folder,
    mapping human-readable identifiers to artifact metadata and lineage.
    """

    batches: dict[str, BatchEntry] = {}
    shards: dict[str, ShardEntry] = {}
    runs: dict[str, RunEntry] = {}
    benchmarks: dict[str, BenchmarkEntry] = {}
