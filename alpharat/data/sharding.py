"""Training set sharding from game batches."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from alpharat.data.loader import load_game_data
from alpharat.nn.extraction import from_game_arrays
from alpharat.nn.targets import build_targets

if TYPE_CHECKING:
    from pathlib import Path

    from alpharat.nn.builders import ObservationBuilder

logger = logging.getLogger(__name__)


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
        - payout_matrix: float32 (N, 5, 5)
        - action_p1: int8 (N,)
        - action_p2: int8 (N,)

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
    (
        all_obs,
        all_policy_p1,
        all_policy_p2,
        all_values,
        all_payout_matrices,
        all_action_p1,
        all_action_p2,
        width,
        height,
    ) = _load_all_positions(batch_dirs, builder)

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
    all_payout_matrices = all_payout_matrices[indices]
    all_action_p1 = all_action_p1[indices]
    all_action_p2 = all_action_p2[indices]

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
        all_payout_matrices,
        all_action_p1,
        all_action_p2,
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


def prepare_training_set_with_split(
    batch_dirs: list[Path],
    output_dir: Path,
    builder: ObservationBuilder,
    *,
    val_ratio: float = 0.1,
    positions_per_shard: int = 10000,
    seed: int | None = None,
) -> Path:
    """Prepare training set with game-level train/val split.

    Splits games (not positions) into train and validation sets to prevent
    data leakage between positions in the same game.

    Directory structure created:
        {output_dir}/{training_set_id}/
            train/
                manifest.json
                shard_0000.npz
                ...
            val/
                manifest.json
                shard_0000.npz
                ...

    Args:
        batch_dirs: List of batch directories to process.
        output_dir: Parent directory for training sets.
        builder: ObservationBuilder to use for encoding.
        val_ratio: Fraction of games to use for validation (0.0 to 1.0).
        positions_per_shard: Maximum positions per shard file.
        seed: Random seed for shuffling. If None, uses random seed.

    Returns:
        Path to created training set directory (parent containing train/ and val/).

    Raises:
        ValueError: If batch_dirs is empty, val_ratio invalid, or dimensions mismatch.
    """
    if not batch_dirs:
        raise ValueError("batch_dirs cannot be empty")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    # Collect all game files
    game_files: list[Path] = []
    for batch_dir in batch_dirs:
        games_dir = batch_dir / "games"
        if games_dir.exists():
            game_files.extend(games_dir.glob("*.npz"))

    if not game_files:
        raise ValueError("No game files found in batch directories")

    # Shuffle and split at game level
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(game_files))
    shuffled_files = [game_files[i] for i in indices]

    n_val = int(len(shuffled_files) * val_ratio)
    val_files = shuffled_files[:n_val]
    train_files = shuffled_files[n_val:]

    if not train_files:
        raise ValueError("No games left for training after split")

    # Create parent directory
    training_set_id = str(uuid.uuid4())
    training_set_dir = output_dir / training_set_id
    training_set_dir.mkdir(parents=True, exist_ok=False)

    # Process train split
    logger.info(f"Processing train split: {len(train_files)} games")
    train_dir = training_set_dir / "train"
    train_dir.mkdir()
    _process_game_files_to_shards(
        game_files=train_files,
        output_dir=train_dir,
        builder=builder,
        positions_per_shard=positions_per_shard,
        seed=seed,
        training_set_id=f"{training_set_id}_train",
        source_batches=[d.name for d in batch_dirs],
    )

    # Process val split (if any validation games)
    if val_files:
        logger.info(f"Processing val split: {len(val_files)} games")
        val_dir = training_set_dir / "val"
        val_dir.mkdir()
        # Use different seed for val shuffle
        val_seed = seed + 1 if seed is not None else None
        _process_game_files_to_shards(
            game_files=val_files,
            output_dir=val_dir,
            builder=builder,
            positions_per_shard=positions_per_shard,
            seed=val_seed,
            training_set_id=f"{training_set_id}_val",
            source_batches=[d.name for d in batch_dirs],
        )

    return training_set_dir


def _process_game_files_to_shards(
    game_files: list[Path],
    output_dir: Path,
    builder: ObservationBuilder,
    positions_per_shard: int,
    seed: int | None,
    training_set_id: str,
    source_batches: list[str],
) -> None:
    """Process game files into shards for a single split.

    Args:
        game_files: List of game npz files to process.
        output_dir: Directory to write shards to.
        builder: ObservationBuilder to use.
        positions_per_shard: Maximum positions per shard.
        seed: Random seed for shuffling positions.
        training_set_id: ID for the manifest.
        source_batches: Batch names for the manifest.
    """
    # Load and process all positions
    (
        obs_array,
        policy_p1_array,
        policy_p2_array,
        values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        width,
        height,
    ) = _load_positions_from_files(game_files, builder)

    total_positions = len(values_array)

    # Shuffle positions within this split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_positions)

    obs_array = obs_array[indices]
    policy_p1_array = policy_p1_array[indices]
    policy_p2_array = policy_p2_array[indices]
    values_array = values_array[indices]
    payout_array = payout_array[indices]
    action_p1_array = action_p1_array[indices]
    action_p2_array = action_p2_array[indices]

    # Write shards
    shard_count = _write_shards(
        output_dir,
        obs_array,
        policy_p1_array,
        policy_p2_array,
        values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        positions_per_shard,
    )

    # Write manifest
    manifest = TrainingSetManifest(
        training_set_id=training_set_id,
        created_at=datetime.now(UTC),
        builder_version=builder.version,
        source_batches=source_batches,
        total_positions=total_positions,
        shard_count=shard_count,
        positions_per_shard=positions_per_shard,
        width=width,
        height=height,
    )
    _save_manifest(output_dir, manifest)


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


def _load_positions_from_files(
    game_files: list[Path],
    builder: ObservationBuilder,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Load and process positions from game files into stacked arrays.

    Args:
        game_files: List of game npz files to process.
        builder: ObservationBuilder to use.

    Returns:
        Tuple of:
            - observations: float32 (N, obs_dim)
            - policy_p1: float32 (N, 5)
            - policy_p2: float32 (N, 5)
            - values: float32 (N,)
            - payout_matrices: float32 (N, 5, 5)
            - action_p1: int8 (N,)
            - action_p2: int8 (N,)
            - width: int
            - height: int

    Raises:
        ValueError: If game files have different dimensions or no positions found.
    """
    all_observations: list[np.ndarray] = []
    all_policy_p1: list[np.ndarray] = []
    all_policy_p2: list[np.ndarray] = []
    all_values: list[float] = []
    all_payout_matrices: list[np.ndarray] = []
    all_action_p1: list[int] = []
    all_action_p2: list[int] = []

    width: int | None = None
    height: int | None = None

    for game_file in tqdm(game_files, desc="Loading games", unit="game"):
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
            all_payout_matrices.append(targets.payout_matrix)
            all_action_p1.append(targets.action_p1)
            all_action_p2.append(targets.action_p2)

    if width is None or height is None:
        raise ValueError("No positions found in game files")

    return (
        np.stack(all_observations),
        np.stack(all_policy_p1),
        np.stack(all_policy_p2),
        np.array(all_values, dtype=np.float32),
        np.stack(all_payout_matrices),
        np.array(all_action_p1, dtype=np.int8),
        np.array(all_action_p2, dtype=np.int8),
        width,
        height,
    )


def _load_all_positions(
    batch_dirs: list[Path],
    builder: ObservationBuilder,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    """Load all positions from batches and build observations/targets.

    Args:
        batch_dirs: List of batch directories.
        builder: ObservationBuilder to use.

    Returns:
        Tuple of stacked arrays and dimensions. See _load_positions_from_files.

    Raises:
        ValueError: If batches have different dimensions or no games found.
    """
    # Collect all game files from batch directories
    game_files: list[Path] = []
    for batch_dir in batch_dirs:
        games_dir = batch_dir / "games"
        if games_dir.exists():
            game_files.extend(games_dir.glob("*.npz"))

    if not game_files:
        raise ValueError("No games found in batch directories")

    return _load_positions_from_files(game_files, builder)


def _write_shards(
    training_set_dir: Path,
    observations: np.ndarray,
    policy_p1: np.ndarray,
    policy_p2: np.ndarray,
    values: np.ndarray,
    payout_matrices: np.ndarray,
    action_p1: np.ndarray,
    action_p2: np.ndarray,
    positions_per_shard: int,
) -> int:
    """Write shards to training set directory.

    Args:
        training_set_dir: Directory to write shards to.
        observations: All observations, shape (N, obs_dim).
        policy_p1: All P1 policies, shape (N, 5).
        policy_p2: All P2 policies, shape (N, 5).
        values: All values, shape (N,).
        payout_matrices: All payout matrices, shape (N, 5, 5).
        action_p1: All P1 actions, shape (N,).
        action_p2: All P2 actions, shape (N,).
        positions_per_shard: Maximum positions per shard.

    Returns:
        Number of shards written.
    """
    n_positions = len(values)
    n_shards = (n_positions + positions_per_shard - 1) // positions_per_shard
    shard_count = 0

    for start_idx in tqdm(
        range(0, n_positions, positions_per_shard),
        desc="Writing shards",
        unit="shard",
        total=n_shards,
    ):
        end_idx = min(start_idx + positions_per_shard, n_positions)

        shard_path = training_set_dir / f"shard_{shard_count:04d}.npz"
        np.savez_compressed(
            shard_path,
            observations=observations[start_idx:end_idx],
            policy_p1=policy_p1[start_idx:end_idx],
            policy_p2=policy_p2[start_idx:end_idx],
            value=values[start_idx:end_idx],
            payout_matrix=payout_matrices[start_idx:end_idx],
            action_p1=action_p1[start_idx:end_idx],
            action_p2=action_p2[start_idx:end_idx],
        )
        shard_count += 1

    return shard_count


def _save_manifest(training_set_dir: Path, manifest: TrainingSetManifest) -> None:
    """Save manifest to training set directory."""
    manifest_path = training_set_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
