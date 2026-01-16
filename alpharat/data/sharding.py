"""Training set sharding from game batches."""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from alpharat.config.base import StrictBaseModel
from alpharat.data.loader import is_bundle_file, iter_games_from_bundle, load_game_data
from alpharat.experiments.paths import batch_id_from_path
from alpharat.nn.extraction import from_game_arrays
from alpharat.nn.targets import build_targets

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from alpharat.data.types import GameData
    from alpharat.nn.builders import ObservationBuilder


@dataclass(frozen=True)
class GameRef:
    """Reference to a game, either single-file or within a bundle.

    Allows game-level operations (shuffling, splitting) without loading data.
    """

    file_path: Path
    bundle_index: int | None = None  # None for single-game files, index for bundles


logger = logging.getLogger(__name__)


class TrainingSetManifest(StrictBaseModel):
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


class ShardingResult(StrictBaseModel):
    """Result from preparing a training set with split."""

    shard_id: str
    shard_dir: str  # Path as string for Pydantic
    total_positions: int
    train_positions: int
    val_positions: int


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
        - p1_value: float32 (N,) — P1's actual remaining score
        - p2_value: float32 (N,) — P2's actual remaining score
        - payout_matrix: float32 (N, 2, 5, 5)
        - action_p1: int8 (N,)
        - action_p2: int8 (N,)
        - cheese_outcomes: int8 (N, H, W) — position-level ownership targets.
            Values: -1=inactive (no cheese at this position), 0-3=outcome class.
            Derive loss mask as: cheese_outcomes >= 0

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
        all_p1_values,
        all_p2_values,
        all_payout_matrices,
        all_action_p1,
        all_action_p2,
        all_cheese_outcomes,
        width,
        height,
    ) = _load_all_positions(batch_dirs, builder)

    total_positions = len(all_p1_values)
    if total_positions == 0:
        raise ValueError("No positions found in batch directories")

    # Shuffle all positions
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_positions)

    all_obs = all_obs[indices]
    all_policy_p1 = all_policy_p1[indices]
    all_policy_p2 = all_policy_p2[indices]
    all_p1_values = all_p1_values[indices]
    all_p2_values = all_p2_values[indices]
    all_payout_matrices = all_payout_matrices[indices]
    all_action_p1 = all_action_p1[indices]
    all_action_p2 = all_action_p2[indices]
    all_cheese_outcomes = all_cheese_outcomes[indices]

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
        all_p1_values,
        all_p2_values,
        all_payout_matrices,
        all_action_p1,
        all_action_p2,
        all_cheese_outcomes,
        positions_per_shard,
    )

    # Write manifest
    manifest = TrainingSetManifest(
        training_set_id=training_set_id,
        created_at=datetime.now(UTC),
        builder_version=builder.version,
        source_batches=[batch_id_from_path(d) for d in batch_dirs],
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
) -> ShardingResult:
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
        ShardingResult with shard ID, path, and position counts.

    Raises:
        ValueError: If batch_dirs is empty, val_ratio invalid, or dimensions mismatch.
    """
    if not batch_dirs:
        raise ValueError("batch_dirs cannot be empty")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    # Collect all game refs (individual games, not files)
    game_refs = _collect_game_refs(batch_dirs)
    total_games = len(game_refs)
    logger.info(f"Found {total_games} games across batch directories")

    # Shuffle and split at game level
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_games)
    shuffled_refs = [game_refs[i] for i in indices]

    n_val = int(total_games * val_ratio)
    val_refs = shuffled_refs[:n_val]
    train_refs = shuffled_refs[n_val:]

    if not train_refs:
        raise ValueError("No games left for training after split")

    logger.info(f"Split: {len(train_refs)} train games, {len(val_refs)} val games")

    # Create parent directory
    training_set_id = str(uuid.uuid4())
    training_set_dir = output_dir / training_set_id
    training_set_dir.mkdir(parents=True, exist_ok=False)

    # Process train split
    logger.info(f"Processing train split: {len(train_refs)} games")
    train_dir = training_set_dir / "train"
    train_dir.mkdir()
    train_positions = _process_game_refs_to_shards(
        game_refs=train_refs,
        output_dir=train_dir,
        builder=builder,
        positions_per_shard=positions_per_shard,
        seed=seed,
        training_set_id=f"{training_set_id}_train",
        source_batches=[batch_id_from_path(d) for d in batch_dirs],
    )

    # Process val split (if any validation games)
    val_positions = 0
    if val_refs:
        logger.info(f"Processing val split: {len(val_refs)} games")
        val_dir = training_set_dir / "val"
        val_dir.mkdir()
        # Use different seed for val shuffle
        val_seed = seed + 1 if seed is not None else None
        val_positions = _process_game_refs_to_shards(
            game_refs=val_refs,
            output_dir=val_dir,
            builder=builder,
            positions_per_shard=positions_per_shard,
            seed=val_seed,
            training_set_id=f"{training_set_id}_val",
            source_batches=[batch_id_from_path(d) for d in batch_dirs],
        )

    return ShardingResult(
        shard_id=training_set_id,
        shard_dir=str(training_set_dir),
        total_positions=train_positions + val_positions,
        train_positions=train_positions,
        val_positions=val_positions,
    )


def _process_game_files_to_shards(
    game_files: list[Path],
    output_dir: Path,
    builder: ObservationBuilder,
    positions_per_shard: int,
    seed: int | None,
    training_set_id: str,
    source_batches: list[str],
) -> int:
    """Process game files into shards for a single split.

    Args:
        game_files: List of game npz files to process.
        output_dir: Directory to write shards to.
        builder: ObservationBuilder to use.
        positions_per_shard: Maximum positions per shard.
        seed: Random seed for shuffling positions.
        training_set_id: ID for the manifest.
        source_batches: Batch names for the manifest.

    Returns:
        Number of positions written.
    """
    # Load and process all positions
    (
        obs_array,
        policy_p1_array,
        policy_p2_array,
        p1_values_array,
        p2_values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        cheese_outcomes_array,
        width,
        height,
    ) = _load_positions_from_files(game_files, builder)

    total_positions = len(p1_values_array)

    # Shuffle positions within this split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_positions)

    obs_array = obs_array[indices]
    policy_p1_array = policy_p1_array[indices]
    policy_p2_array = policy_p2_array[indices]
    p1_values_array = p1_values_array[indices]
    p2_values_array = p2_values_array[indices]
    payout_array = payout_array[indices]
    action_p1_array = action_p1_array[indices]
    action_p2_array = action_p2_array[indices]
    cheese_outcomes_array = cheese_outcomes_array[indices]

    # Write shards
    shard_count = _write_shards(
        output_dir,
        obs_array,
        policy_p1_array,
        policy_p2_array,
        p1_values_array,
        p2_values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        cheese_outcomes_array,
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

    return total_positions


def _process_game_refs_to_shards(
    game_refs: list[GameRef],
    output_dir: Path,
    builder: ObservationBuilder,
    positions_per_shard: int,
    seed: int | None,
    training_set_id: str,
    source_batches: list[str],
) -> int:
    """Process game refs into shards for a single split.

    Like _process_game_files_to_shards but takes GameRef list for game-level splitting.

    Args:
        game_refs: List of GameRef objects specifying which games to process.
        output_dir: Directory to write shards to.
        builder: ObservationBuilder to use.
        positions_per_shard: Maximum positions per shard.
        seed: Random seed for shuffling positions.
        training_set_id: ID for the manifest.
        source_batches: Batch names for the manifest.

    Returns:
        Number of positions written.
    """
    # Load and process all positions from game refs
    (
        obs_array,
        policy_p1_array,
        policy_p2_array,
        p1_values_array,
        p2_values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        cheese_outcomes_array,
        width,
        height,
    ) = _load_positions_from_refs(game_refs, builder)

    total_positions = len(p1_values_array)

    # Shuffle positions within this split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_positions)

    obs_array = obs_array[indices]
    policy_p1_array = policy_p1_array[indices]
    policy_p2_array = policy_p2_array[indices]
    p1_values_array = p1_values_array[indices]
    p2_values_array = p2_values_array[indices]
    payout_array = payout_array[indices]
    action_p1_array = action_p1_array[indices]
    action_p2_array = action_p2_array[indices]
    cheese_outcomes_array = cheese_outcomes_array[indices]

    # Write shards
    shard_count = _write_shards(
        output_dir,
        obs_array,
        policy_p1_array,
        policy_p2_array,
        p1_values_array,
        p2_values_array,
        payout_array,
        action_p1_array,
        action_p2_array,
        cheese_outcomes_array,
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

    return total_positions


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


def _collect_game_refs(batch_dirs: list[Path]) -> list[GameRef]:
    """Collect references to all games in batch directories.

    Enumerates games at the individual game level, not file level.
    For bundle files, creates one GameRef per game in the bundle.

    Args:
        batch_dirs: List of batch directories to scan.

    Returns:
        List of GameRef objects, one per game.

    Raises:
        ValueError: If no games found.
    """

    game_refs: list[GameRef] = []

    for batch_dir in batch_dirs:
        games_dir = batch_dir / "games"
        if not games_dir.exists():
            continue

        for game_file in games_dir.glob("*.npz"):
            if is_bundle_file(game_file):
                # Count games in bundle without loading full data
                with np.load(game_file) as data:
                    n_games = len(data["game_lengths"])
                for i in range(n_games):
                    game_refs.append(GameRef(game_file, bundle_index=i))
            else:
                game_refs.append(GameRef(game_file, bundle_index=None))

    if not game_refs:
        raise ValueError("No games found in batch directories")

    return game_refs


def _iter_games_from_refs(
    refs: list[GameRef],
    desc: str = "Loading games",
) -> Iterator[GameData]:
    """Iterate over games from GameRef list, grouping by file for efficiency.

    Groups refs by file path and loads each file only once, yielding games
    in the order specified by refs.

    Args:
        refs: List of GameRef objects specifying which games to load.
        desc: Description for progress bar.

    Yields:
        GameData objects in the order specified by refs.
    """
    # Group refs by file to minimize I/O
    refs_by_file: dict[Path, list[tuple[int, int | None]]] = defaultdict(list)
    for idx, ref in enumerate(refs):
        refs_by_file[ref.file_path].append((idx, ref.bundle_index))

    # Load games, tracking original order
    games_by_idx: dict[int, GameData] = {}

    for file_path in tqdm(refs_by_file.keys(), desc=desc, unit="file"):
        indices_and_bundle_idxs = refs_by_file[file_path]

        if indices_and_bundle_idxs[0][1] is None:
            # Single-game file
            game = load_game_data(file_path)
            for idx, _ in indices_and_bundle_idxs:
                games_by_idx[idx] = game
        else:
            # Bundle file: load once, extract requested games
            # In this branch, bundle_idx is guaranteed to be int (not None)
            needed_bundle_idxs: set[int] = {
                bundle_idx for _, bundle_idx in indices_and_bundle_idxs if bundle_idx is not None
            }
            idx_by_bundle: dict[int, int] = {
                bundle_idx: idx
                for idx, bundle_idx in indices_and_bundle_idxs
                if bundle_idx is not None
            }

            for bundle_idx, game in enumerate(iter_games_from_bundle(file_path)):
                if bundle_idx in needed_bundle_idxs:
                    games_by_idx[idx_by_bundle[bundle_idx]] = game

    # Yield in original order
    for idx in range(len(refs)):
        yield games_by_idx[idx]


def _load_positions_from_refs(
    refs: list[GameRef],
    builder: ObservationBuilder,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Load and process positions from game refs into stacked arrays.

    Args:
        refs: List of GameRef objects specifying which games to load.
        builder: ObservationBuilder to use.

    Returns:
        Same tuple as _load_positions_from_files.
    """
    all_observations: list[np.ndarray] = []
    all_policy_p1: list[np.ndarray] = []
    all_policy_p2: list[np.ndarray] = []
    all_p1_values: list[float] = []
    all_p2_values: list[float] = []
    all_payout_matrices: list[np.ndarray] = []
    all_action_p1: list[int] = []
    all_action_p2: list[int] = []
    all_cheese_outcomes: list[np.ndarray] = []

    width: int | None = None
    height: int | None = None

    for game_data in _iter_games_from_refs(refs):
        # Validate dimensions
        if width is None:
            width = game_data.width
            height = game_data.height
        elif game_data.width != width or game_data.height != height:
            raise ValueError(
                f"Dimension mismatch: expected ({width}, {height}), "
                f"got ({game_data.width}, {game_data.height})"
            )

        # Process each position in the game
        for position in game_data.positions:
            obs_input = from_game_arrays(game_data, position)
            obs = builder.build(obs_input)
            targets = build_targets(game_data, position)

            all_observations.append(obs)
            all_policy_p1.append(targets.policy_p1)
            all_policy_p2.append(targets.policy_p2)
            all_p1_values.append(targets.p1_value)
            all_p2_values.append(targets.p2_value)
            all_payout_matrices.append(targets.payout_matrix)
            all_action_p1.append(targets.action_p1)
            all_action_p2.append(targets.action_p2)
            all_cheese_outcomes.append(targets.cheese_outcomes)

    if width is None or height is None:
        raise ValueError("No positions found in game refs")

    return (
        np.stack(all_observations),
        np.stack(all_policy_p1),
        np.stack(all_policy_p2),
        np.array(all_p1_values, dtype=np.float32),
        np.array(all_p2_values, dtype=np.float32),
        np.stack(all_payout_matrices),
        np.array(all_action_p1, dtype=np.int8),
        np.array(all_action_p2, dtype=np.int8),
        np.stack(all_cheese_outcomes),
        width,
        height,
    )


def _load_positions_from_files(
    game_files: list[Path],
    builder: ObservationBuilder,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """Load and process positions from game files into stacked arrays.

    Handles both single-game .npz files and bundled .npz files.

    Args:
        game_files: List of game npz files to process (single or bundled).
        builder: ObservationBuilder to use.

    Returns:
        Tuple of:
            - observations: float32 (N, obs_dim)
            - policy_p1: float32 (N, 5)
            - policy_p2: float32 (N, 5)
            - p1_values: float32 (N,) — P1's remaining score (actual game outcome)
            - p2_values: float32 (N,) — P2's remaining score (actual game outcome)
            - payout_matrices: float32 (N, 2, 5, 5)
            - action_p1: int8 (N,)
            - action_p2: int8 (N,)
            - cheese_outcomes: int8 (N, H, W) — position-level ownership targets.
                Values: -1=inactive (no cheese), 0-3=outcome class.
            - width: int
            - height: int

    Raises:
        ValueError: If game files have different dimensions or no positions found.
    """
    all_observations: list[np.ndarray] = []
    all_policy_p1: list[np.ndarray] = []
    all_policy_p2: list[np.ndarray] = []
    all_p1_values: list[float] = []
    all_p2_values: list[float] = []
    all_payout_matrices: list[np.ndarray] = []
    all_action_p1: list[int] = []
    all_action_p2: list[int] = []
    all_cheese_outcomes: list[np.ndarray] = []

    width: int | None = None
    height: int | None = None

    def process_game(game_data: GameData, source_file: Path) -> None:
        """Process a single game and append to result lists."""
        nonlocal width, height

        # Validate dimensions
        if width is None:
            width = game_data.width
            height = game_data.height
        elif game_data.width != width or game_data.height != height:
            raise ValueError(
                f"Dimension mismatch: expected ({width}, {height}), "
                f"got ({game_data.width}, {game_data.height}) in {source_file}"
            )

        # Process each position
        for position in game_data.positions:
            obs_input = from_game_arrays(game_data, position)
            obs = builder.build(obs_input)
            targets = build_targets(game_data, position)

            all_observations.append(obs)
            all_policy_p1.append(targets.policy_p1)
            all_policy_p2.append(targets.policy_p2)
            all_p1_values.append(targets.p1_value)
            all_p2_values.append(targets.p2_value)
            all_payout_matrices.append(targets.payout_matrix)
            all_action_p1.append(targets.action_p1)
            all_action_p2.append(targets.action_p2)
            all_cheese_outcomes.append(targets.cheese_outcomes)

    for game_file in tqdm(game_files, desc="Loading files", unit="file"):
        if is_bundle_file(game_file):
            # Bundle file: iterate over all games in the bundle
            for game_data in iter_games_from_bundle(game_file):
                process_game(game_data, game_file)
        else:
            # Single-game file
            game_data = load_game_data(game_file)
            process_game(game_data, game_file)

    if width is None or height is None:
        raise ValueError("No positions found in game files")

    return (
        np.stack(all_observations),
        np.stack(all_policy_p1),
        np.stack(all_policy_p2),
        np.array(all_p1_values, dtype=np.float32),
        np.array(all_p2_values, dtype=np.float32),
        np.stack(all_payout_matrices),
        np.array(all_action_p1, dtype=np.int8),
        np.array(all_action_p2, dtype=np.int8),
        np.stack(all_cheese_outcomes),
        width,
        height,
    )


def _load_all_positions(
    batch_dirs: list[Path],
    builder: ObservationBuilder,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
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
    p1_values: np.ndarray,
    p2_values: np.ndarray,
    payout_matrices: np.ndarray,
    action_p1: np.ndarray,
    action_p2: np.ndarray,
    cheese_outcomes: np.ndarray,
    positions_per_shard: int,
) -> int:
    """Write shards to training set directory.

    Args:
        training_set_dir: Directory to write shards to.
        observations: All observations, shape (N, obs_dim).
        policy_p1: All P1 policies, shape (N, 5).
        policy_p2: All P2 policies, shape (N, 5).
        p1_values: P1's remaining scores, shape (N,).
        p2_values: P2's remaining scores, shape (N,).
        payout_matrices: All payout matrices, shape (N, 2, 5, 5).
        action_p1: All P1 actions, shape (N,).
        action_p2: All P2 actions, shape (N,).
        cheese_outcomes: Position-level ownership targets, shape (N, H, W).
            Values: -1=inactive (no cheese), 0-3=outcome class.
        positions_per_shard: Maximum positions per shard.

    Returns:
        Number of shards written.
    """
    n_positions = len(p1_values)
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
            p1_value=p1_values[start_idx:end_idx],
            p2_value=p2_values[start_idx:end_idx],
            payout_matrix=payout_matrices[start_idx:end_idx],
            action_p1=action_p1[start_idx:end_idx],
            action_p2=action_p2[start_idx:end_idx],
            cheese_outcomes=cheese_outcomes[start_idx:end_idx],
        )
        shard_count += 1

    return shard_count


def _save_manifest(training_set_dir: Path, manifest: TrainingSetManifest) -> None:
    """Save manifest to training set directory."""
    manifest_path = training_set_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
