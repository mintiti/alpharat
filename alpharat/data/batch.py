"""Batch metadata for organizing game recordings by sampling run."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import numpy as np
from pydantic import ValidationError

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.data.types import GameFileKey
from alpharat.mcts.config import MCTSConfig, PythonMCTSConfig  # noqa: TC001


class BatchMetadataError(Exception):
    """Raised when batch metadata.json fails validation.

    Provides an actionable message identifying which config fields drifted
    from the current schema, instead of a raw Pydantic dump.
    """


# --- Batch Metadata ---


class BatchMetadata(StrictBaseModel):
    """Metadata for a batch of sampled games."""

    batch_id: str
    created_at: datetime
    checkpoint_path: str | None
    mcts_config: MCTSConfig
    game: GameConfig


class BatchStats(StrictBaseModel):
    """Statistics computed from game files in a batch."""

    game_count: int
    total_positions: int


# --- Functions ---


def create_batch(
    parent_dir: Path | str,
    checkpoint_path: str | None,
    mcts_config: MCTSConfig,
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
    """Load batch metadata from metadata.json.

    Raises:
        BatchMetadataError: If the saved metadata doesn't match the current config
            schemas, with an actionable message identifying which fields drifted.
    """
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    data = json.loads(metadata_path.read_text())

    try:
        return BatchMetadata.model_validate(data)
    except ValidationError as exc:
        _raise_batch_metadata_error(metadata_path, data, cause=exc)


def _field_diff(label: str, model_cls: type[StrictBaseModel], saved: object) -> str:
    """Compare saved dict keys against a model's fields. Returns summary line."""
    if not isinstance(saved, dict):
        return f"  {label}: OK"
    expected = set(model_cls.model_fields)
    actual = set(saved)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    parts: list[str] = []
    if extra:
        parts.append(f"extra fields: {', '.join(extra)}")
    if missing:
        parts.append(f"missing fields: {', '.join(missing)}")
    return f"  {label}: {'; '.join(parts)}" if parts else f"  {label}: OK"


def _raise_batch_metadata_error(
    metadata_path: Path, data: dict[str, object], *, cause: Exception
) -> NoReturn:
    """Build and raise an actionable BatchMetadataError."""
    lines: list[str] = [
        f"Cannot load batch metadata from {metadata_path}",
        "",
        "Config schema mismatch — this batch was created with a different version:",
        _field_diff("mcts_config", PythonMCTSConfig, data.get("mcts_config")),
        _field_diff("game", GameConfig, data.get("game")),
        "",
        "Update the config classes to include these fields, or re-sample with the current config.",
    ]

    raise BatchMetadataError("\n".join(lines)) from cause


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
            total_positions += int(data[GameFileKey.NUM_POSITIONS])

    return BatchStats(game_count=game_count, total_positions=total_positions)
