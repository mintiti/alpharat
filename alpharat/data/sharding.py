"""Training set sharding from game batches."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from alpharat.data.loader import load_game_data
from alpharat.nn.extraction import from_game_arrays
from alpharat.nn.targets import build_targets

if TYPE_CHECKING:
    from pathlib import Path

    from alpharat.nn.builders import ObservationBuilder


class TrainingSetManifest(BaseModel):
    """Manifest for a prepared training set.

    Stored as manifest.json in the training set directory.
    """

    training_set_id: str
    created_at: datetime
    builder_version: str
    source_batches: list[str]
    total_positions: int
    shard_count: int
    positions_per_shard: int
    width: int
    height: int


def prepare_training_set(
    batch_dirs: list[Path],
    output_dir: Path,
    builder: ObservationBuilder,
    *,
    positions_per_shard: int = 10000,
    seed: int | None = None,
) -> Path:
    """Prepare training set from game batches.

    Loads all positions from batches, builds observations and targets,
    shuffles globally, and writes to sharded npz files.

    Directory structure created:
        {output_dir}/{training_set_id}/
            manifest.json
            shard_0000.npz
            shard_0001.npz
            ...

    Each shard contains:
        - observations: from builder.save_to_arrays()
        - policy_p1: float32 (N, 5)
        - policy_p2: float32 (N, 5)
        - value: float32 (N,)

    Args:
        batch_dirs: List of batch directories to process.
        output_dir: Parent directory for training sets.
        builder: ObservationBuilder to use for encoding.
        positions_per_shard: Maximum positions per shard file.
        seed: Random seed for shuffling. If None, uses random seed.

    Returns:
        Path to created training set directory.

    Raises:
        ValueError: If batch_dirs is empty or batches have incompatible dimensions.
    """
    if not batch_dirs:
        raise ValueError("batch_dirs cannot be empty")

    # Load all positions
    all_obs, all_policy_p1, all_policy_p2, all_values, width, height = _load_all_positions(
        batch_dirs, builder
    )

    total_positions = len(all_values)
    if total_positions == 0:
        raise ValueError("No positions found in batch directories")

    # Shuffle all positions
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_positions)

    all_obs = all_obs[indices]
    all_policy_p1 = all_policy_p1[indices]
    all_policy_p2 = all_policy_p2[indices]
    all_values = all_values[indices]

    # Create training set directory
    training_set_id = str(uuid.uuid4())
    training_set_dir = output_dir / training_set_id
    training_set_dir.mkdir(parents=True, exist_ok=False)

    # Write shards
    shard_count = _write_shards(
        training_set_dir,
        all_obs,
        all_policy_p1,
        all_policy_p2,
        all_values,
        positions_per_shard,
    )

    # Write manifest
    manifest = TrainingSetManifest(
        training_set_id=training_set_id,
        created_at=datetime.now(UTC),
        builder_version=builder.version,
        source_batches=[d.name for d in batch_dirs],
        total_positions=total_positions,
        shard_count=shard_count,
        positions_per_shard=positions_per_shard,
        width=width,
        height=height,
    )
    _save_manifest(training_set_dir, manifest)

    return training_set_dir


def load_training_set_manifest(training_set_dir: Path) -> TrainingSetManifest:
    """Load manifest from training set directory.

    Args:
        training_set_dir: Path to training set directory.

    Returns:
        TrainingSetManifest loaded from manifest.json.

    Raises:
        FileNotFoundError: If manifest.json doesn't exist.
    """
    manifest_path = training_set_dir / "manifest.json"
    data = json.loads(manifest_path.read_text())
    return TrainingSetManifest.model_validate(data)


def _load_all_positions(
    batch_dirs: list[Path],
    builder: ObservationBuilder,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load all positions from batches and build observations/targets.

    Args:
        batch_dirs: List of batch directories.
        builder: ObservationBuilder to use.

    Returns:
        Tuple of:
            - observations: float32 (N, obs_dim)
            - policy_p1: float32 (N, 5)
            - policy_p2: float32 (N, 5)
            - values: float32 (N,)
            - width: int
            - height: int

    Raises:
        ValueError: If batches have different dimensions.
    """
    all_observations: list[np.ndarray] = []
    all_policy_p1: list[np.ndarray] = []
    all_policy_p2: list[np.ndarray] = []
    all_values: list[float] = []

    width: int | None = None
    height: int | None = None

    for batch_dir in batch_dirs:
        games_dir = batch_dir / "games"
        if not games_dir.exists():
            continue

        for game_file in games_dir.glob("*.npz"):
            game_data = load_game_data(game_file)

            # Validate dimensions
            if width is None:
                width = game_data.width
                height = game_data.height
            elif game_data.width != width or game_data.height != height:
                raise ValueError(
                    f"Dimension mismatch: expected ({width}, {height}), "
                    f"got ({game_data.width}, {game_data.height}) in {game_file}"
                )

            # Process each position
            for position in game_data.positions:
                obs_input = from_game_arrays(game_data, position)
                obs = builder.build(obs_input)
                targets = build_targets(game_data, position)

                all_observations.append(obs)
                all_policy_p1.append(targets.policy_p1)
                all_policy_p2.append(targets.policy_p2)
                all_values.append(targets.value)

    if width is None or height is None:
        raise ValueError("No games found in batch directories")

    return (
        np.stack(all_observations),
        np.stack(all_policy_p1),
        np.stack(all_policy_p2),
        np.array(all_values, dtype=np.float32),
        width,
        height,
    )


def _write_shards(
    training_set_dir: Path,
    observations: np.ndarray,
    policy_p1: np.ndarray,
    policy_p2: np.ndarray,
    values: np.ndarray,
    positions_per_shard: int,
) -> int:
    """Write shards to training set directory.

    Args:
        training_set_dir: Directory to write shards to.
        observations: All observations, shape (N, obs_dim).
        policy_p1: All P1 policies, shape (N, 5).
        policy_p2: All P2 policies, shape (N, 5).
        values: All values, shape (N,).
        positions_per_shard: Maximum positions per shard.

    Returns:
        Number of shards written.
    """
    n_positions = len(values)
    shard_count = 0

    for start_idx in range(0, n_positions, positions_per_shard):
        end_idx = min(start_idx + positions_per_shard, n_positions)

        shard_path = training_set_dir / f"shard_{shard_count:04d}.npz"
        np.savez_compressed(
            shard_path,
            observations=observations[start_idx:end_idx],
            policy_p1=policy_p1[start_idx:end_idx],
            policy_p2=policy_p2[start_idx:end_idx],
            value=values[start_idx:end_idx],
        )
        shard_count += 1

    return shard_count


def _save_manifest(training_set_dir: Path, manifest: TrainingSetManifest) -> None:
    """Save manifest to training set directory."""
    manifest_path = training_set_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
