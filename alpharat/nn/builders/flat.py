"""Flat observation builder for MLP networks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from alpharat.nn.training.keys import BatchKey

if TYPE_CHECKING:
    from alpharat.data.sharding import TrainingSetManifest
    from alpharat.nn.types import ObservationInput

# Normalization constants (exported for tests)
MAX_MUD_COST = 10
MAX_MUD_TURNS = 10
MAX_SCORE = 10

__all__ = [
    "MAX_MUD_COST",
    "MAX_MUD_TURNS",
    "MAX_SCORE",
    "FlatObsLayout",
    "FlatObservationBuilder",
    "FlatDataset",
]


@dataclass(frozen=True)
class FlatObsLayout:
    """Layout of flat observation vector.

    [maze H*W*4][p1_pos H*W][p2_pos H*W][cheese H*W][6 scalars]

    Scalar order: score_diff(0), progress(1), p1_mud(2), p2_mud(3), p1_score(4), p2_score(5)
    """

    width: int
    height: int

    # Scalar sub-indices (relative to scalars_start)
    SCORE_DIFF = 0
    PROGRESS = 1
    P1_MUD = 2
    P2_MUD = 3
    P1_SCORE = 4
    P2_SCORE = 5

    @property
    def spatial(self) -> int:
        return self.width * self.height

    @property
    def maze(self) -> slice:
        return slice(0, self.spatial * 4)

    @property
    def p1_pos(self) -> slice:
        s = self.spatial
        return slice(s * 4, s * 5)

    @property
    def p2_pos(self) -> slice:
        s = self.spatial
        return slice(s * 5, s * 6)

    @property
    def cheese(self) -> slice:
        s = self.spatial
        return slice(s * 6, s * 7)

    @property
    def scalars(self) -> slice:
        s = self.spatial
        return slice(s * 7, s * 7 + 6)

    @property
    def scalars_start(self) -> int:
        return self.spatial * 7

    @property
    def obs_dim(self) -> int:
        return self.spatial * 7 + 6


class FlatObservationBuilder:
    """Flat observation encoding for MLP networks.

    Encodes game state as a 1D float32 vector. For a 5x5 maze:
    - Maze adjacency: H*W*4 = 100 (normalized costs, -1 for walls)
    - P1 position: H*W = 25 (one-hot)
    - P2 position: H*W = 25 (one-hot)
    - Cheese mask: H*W = 25 (binary)
    - Score diff: 1 (raw)
    - Progress: 1 (turn/max_turns)
    - P1 mud: 1 (normalized)
    - P2 mud: 1 (normalized)
    - P1 score: 1 (normalized)
    - P2 score: 1 (normalized)
    Total: 181 floats for 5x5
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize builder for given maze dimensions.

        Args:
            width: Maze width.
            height: Maze height.
        """
        self._width = width
        self._height = height
        self._layout = FlatObsLayout(width, height)

    @property
    def version(self) -> str:
        """Builder version identifier."""
        return "flat_v2"

    @property
    def layout(self) -> FlatObsLayout:
        """Layout descriptor for this builder's observation format."""
        return self._layout

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of observation tensor."""
        return (self._layout.obs_dim,)

    @property
    def width(self) -> int:
        """Maze width."""
        return self._width

    @property
    def height(self) -> int:
        """Maze height."""
        return self._height

    def build(self, input: ObservationInput) -> np.ndarray:
        """Build flat observation from game state.

        Args:
            input: Game state to encode.

        Returns:
            float32 array of shape (obs_dim,).
        """
        h, w = self._height, self._width

        # Maze: normalize costs, keep -1 for walls
        # Shape: (H, W, 4) -> (H*W*4,)
        maze = input.maze.astype(np.float32)
        # Walls stay -1, positive costs get normalized
        mask = maze > 0
        maze[mask] = maze[mask] / MAX_MUD_COST
        maze_flat = maze.flatten()

        # P1 position: one-hot (H, W) -> (H*W,)
        p1_pos = np.zeros((h, w), dtype=np.float32)
        p1_pos[input.p1_pos[1], input.p1_pos[0]] = 1.0
        p1_flat = p1_pos.flatten()

        # P2 position: one-hot (H, W) -> (H*W,)
        p2_pos = np.zeros((h, w), dtype=np.float32)
        p2_pos[input.p2_pos[1], input.p2_pos[0]] = 1.0
        p2_flat = p2_pos.flatten()

        # Cheese: binary mask (H, W) -> (H*W,)
        cheese_flat = input.cheese_mask.astype(np.float32).flatten()

        # Scalars
        score_diff = np.float32(input.p1_score - input.p2_score)
        if input.max_turns > 0:
            progress = np.float32(input.turn / input.max_turns)
        else:
            progress = np.float32(0)
        p1_mud = np.float32(input.p1_mud / MAX_MUD_TURNS)
        p2_mud = np.float32(input.p2_mud / MAX_MUD_TURNS)
        p1_score = np.float32(input.p1_score / MAX_SCORE)
        p2_score = np.float32(input.p2_score / MAX_SCORE)

        # Concatenate all features
        return np.concatenate(
            [
                maze_flat,
                p1_flat,
                p2_flat,
                cheese_flat,
                np.array(
                    [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score],
                    dtype=np.float32,
                ),
            ]
        )

    def save_to_arrays(self, observations: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Stack observations into single array for npz storage.

        Args:
            observations: List of observation arrays from build().

        Returns:
            Dict with single "observation" key containing stacked array.
        """
        return {BatchKey.OBSERVATION: np.stack(observations)}

    def load_from_arrays(self, arrays: dict[str, np.ndarray], idx: int) -> np.ndarray:
        """Load single observation from arrays.

        Args:
            arrays: Loaded npz data.
            idx: Index of observation to load.

        Returns:
            Observation array at index.
        """
        obs: np.ndarray = arrays[BatchKey.OBSERVATION][idx]
        return obs


class FlatDataset:
    """PyTorch-compatible Dataset for flat observations.

    Loads all shards into memory at initialization for fast access.
    Compatible with torch.utils.data.DataLoader.

    Returns numpy arrays from __getitem__ - the DataLoader will handle
    conversion to tensors. This keeps the module torch-agnostic.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = FlatDataset(training_set_dir)
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
        >>> for batch in loader:
        ...     obs = batch["observation"]
        ...     policy_p1 = batch["policy_p1"]
        ...     # train...
    """

    def __init__(self, training_set_dir: Path | str) -> None:
        """Initialize dataset from training set directory.

        Loads manifest and all shards into memory. For large datasets,
        consider using a lazy-loading implementation.

        Args:
            training_set_dir: Path to training set with manifest.json and shards.

        Raises:
            FileNotFoundError: If training set doesn't exist.
        """
        from alpharat.data.sharding import load_training_set_manifest

        training_set_dir = Path(training_set_dir)
        self._manifest = load_training_set_manifest(training_set_dir)

        # Load all shards and concatenate
        observations_list: list[np.ndarray] = []
        policy_p1_list: list[np.ndarray] = []
        policy_p2_list: list[np.ndarray] = []
        value_p1_list: list[np.ndarray] = []
        value_p2_list: list[np.ndarray] = []
        action_p1_list: list[np.ndarray] = []
        action_p2_list: list[np.ndarray] = []

        for i in range(self._manifest.shard_count):
            shard_path = training_set_dir / f"shard_{i:04d}.npz"
            with np.load(shard_path) as data:
                observations_list.append(data[BatchKey.OBSERVATION])
                policy_p1_list.append(data[BatchKey.POLICY_P1])
                policy_p2_list.append(data[BatchKey.POLICY_P2])
                value_p1_list.append(data[BatchKey.VALUE_P1])
                value_p2_list.append(data[BatchKey.VALUE_P2])
                action_p1_list.append(data[BatchKey.ACTION_P1])
                action_p2_list.append(data[BatchKey.ACTION_P2])

        self._observations = np.concatenate(observations_list)
        self._policy_p1 = np.concatenate(policy_p1_list)
        self._policy_p2 = np.concatenate(policy_p2_list)
        self._value_p1 = np.concatenate(value_p1_list)
        self._value_p2 = np.concatenate(value_p2_list)
        self._action_p1 = np.concatenate(action_p1_list)
        self._action_p2 = np.concatenate(action_p2_list)

        # Build observation builder for shape info
        self._builder = FlatObservationBuilder(
            width=self._manifest.width, height=self._manifest.height
        )

    def __len__(self) -> int:
        """Return total number of positions."""
        return len(self._value_p1)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get single training example.

        Args:
            idx: Position index.

        Returns:
            Dict with keys:
                - "observation": float32 (obs_dim,)
                - "policy_p1": float32 (5,)
                - "policy_p2": float32 (5,)
                - "value_p1": float32 (1,) — P1's actual remaining score
                - "value_p2": float32 (1,) — P2's actual remaining score
                - "action_p1": int8 (1,)
                - "action_p2": int8 (1,)
        """
        return {
            BatchKey.OBSERVATION: self._observations[idx],
            BatchKey.POLICY_P1: self._policy_p1[idx],
            BatchKey.POLICY_P2: self._policy_p2[idx],
            BatchKey.VALUE_P1: self._value_p1[idx : idx + 1],  # Keep as 1D for consistency
            BatchKey.VALUE_P2: self._value_p2[idx : idx + 1],
            BatchKey.ACTION_P1: self._action_p1[idx : idx + 1],
            BatchKey.ACTION_P2: self._action_p2[idx : idx + 1],
        }

    @property
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of observation tensors."""
        return self._builder.obs_shape

    @property
    def manifest(self) -> TrainingSetManifest:
        """Training set manifest."""
        return self._manifest
