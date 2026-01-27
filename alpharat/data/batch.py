"""Batch metadata for organizing game recordings by sampling run."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.mcts import DecoupledPUCTConfig  # noqa: TC001

# --- Batch Metadata ---


class BatchMetadata(StrictBaseModel):
    """Metadata for a batch of sampled games."""

    batch_id: str
    created_at: datetime
    checkpoint_path: str | None
    mcts_config: DecoupledPUCTConfig
    game: GameConfig


class BatchStats(StrictBaseModel):
    """Statistics computed from game files in a batch."""

    game_count: int
    total_positions: int


# --- Functions ---


def create_batch(
    parent_dir: Path | str,
    checkpoint_path: str | None,
    mcts_config: DecoupledPUCTConfig,
    game: GameConfig,
) -> Path:
    """Create a new batch directory with metadata.

    Creates the batch directory structure:
        {parent_dir}/{batch_id}/
            metadata.json
            games/

    Args:
        parent_dir: Parent directory for batches
        checkpoint_path: Path to model checkpoint, or None for random policy
        mcts_config: MCTS algorithm configuration
        game: Game configuration

    Returns:
        Path to the created batch directory
    """
    parent_dir = Path(parent_dir)
    batch_id = str(uuid.uuid4())
    batch_dir = parent_dir / batch_id

    batch_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "games").mkdir()

    metadata = BatchMetadata(
        batch_id=batch_id,
        created_at=datetime.now(UTC),
        checkpoint_path=checkpoint_path,
        mcts_config=mcts_config,
        game=game,
    )

    save_batch_metadata(batch_dir, metadata)
    return batch_dir


def save_batch_metadata(batch_dir: Path | str, metadata: BatchMetadata) -> None:
    """Save batch metadata to metadata.json."""
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2))


def load_batch_metadata(batch_dir: Path | str) -> BatchMetadata:
    """Load batch metadata from metadata.json."""
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    data = json.loads(metadata_path.read_text())
    return BatchMetadata.model_validate(data)


def get_batch_stats(batch_dir: Path | str) -> BatchStats:
    """Compute batch statistics by scanning game files.

    Counts game files and sums positions across all games.
    """
    batch_dir = Path(batch_dir)
    games_dir = batch_dir / "games"

    game_count = 0
    total_positions = 0

    for game_file in games_dir.glob("*.npz"):
        game_count += 1
        with np.load(game_file) as data:
            total_positions += int(data["num_positions"])

    return BatchStats(game_count=game_count, total_positions=total_positions)
