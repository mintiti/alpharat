"""Spatial observation builder for CNN networks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alpharat.data.sharding import TrainingSetManifest
    from alpharat.nn.types import ObservationInput

# Normalization constants (same as flat builder for consistency)
MAX_MUD_COST = 10
MAX_MUD_TURNS = 10
MAX_SCORE = 10

__all__ = [
    "MAX_MUD_COST",
    "MAX_MUD_TURNS",
    "MAX_SCORE",
    "SpatialObservationBuilder",
    "SpatialDataset",
]


class SpatialObservationBuilder:
    """Spatial observation encoding for CNN networks.

    Encodes game state as a spatial tensor plus per-player side vectors.

    Spatial tensor (7, H, W):
        Channels 0-3: Maze adjacency per direction (normalized costs, -1 for walls)
        Channel 4: P1 position (one-hot)
        Channel 5: P2 position (one-hot)
        Channel 6: Cheese mask (binary)

    Side vectors (3,) each:
        [normalized_score, normalized_mud_turns, progress]

    Position tuples for spatial indexing at player locations.
    """

    def __init__(self, width: int, height: int) -> None:
        """Initialize builder for given maze dimensions.

        Args:
            width: Maze width.
            height: Maze height.
        """
        self._width = width
        self._height = height

    @property
    def version(self) -> str:
        """Builder version identifier."""
        return "spatial_v1"

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Shape of spatial tensor (C, H, W)."""
        return (7, self._height, self._width)

    @property
    def side_dim(self) -> int:
        """Dimension of side vectors."""
        return 3

    @property
    def width(self) -> int:
        """Maze width."""
        return self._width

    @property
    def height(self) -> int:
        """Maze height."""
        return self._height

    def build(self, input: ObservationInput) -> dict[str, np.ndarray]:
        """Build spatial observation from game state.

        Args:
            input: Game state to encode.

        Returns:
            Dict with:
                - "spatial": float32 (7, H, W)
                - "p1_side": float32 (3,)
                - "p2_side": float32 (3,)
                - "p1_pos": int32 (2,) as [x, y]
                - "p2_pos": int32 (2,) as [x, y]
        """
        h, w = self._height, self._width

        # Build spatial tensor (7, H, W)
        spatial = np.zeros((7, h, w), dtype=np.float32)

        # Channels 0-3: Maze adjacency
        # Input maze is (H, W, 4), transpose to (4, H, W)
        maze = input.maze.astype(np.float32).transpose(2, 0, 1)
        # Normalize positive values (mud costs), keep -1 for walls
        mask = maze > 0
        maze[mask] = maze[mask] / MAX_MUD_COST
        spatial[:4] = maze

        # Channel 4: P1 position one-hot
        spatial[4, input.p1_pos[1], input.p1_pos[0]] = 1.0

        # Channel 5: P2 position one-hot
        spatial[5, input.p2_pos[1], input.p2_pos[0]] = 1.0

        # Channel 6: Cheese mask
        spatial[6] = input.cheese_mask.astype(np.float32)

        # Build side vectors (3,) each
        progress = input.turn / input.max_turns if input.max_turns > 0 else 0.0

        p1_side = np.array(
            [
                input.p1_score / MAX_SCORE,
                input.p1_mud / MAX_MUD_TURNS,
                progress,
            ],
            dtype=np.float32,
        )

        p2_side = np.array(
            [
                input.p2_score / MAX_SCORE,
                input.p2_mud / MAX_MUD_TURNS,
                progress,
            ],
            dtype=np.float32,
        )

        # Position arrays for indexing
        p1_pos = np.array(input.p1_pos, dtype=np.int32)
        p2_pos = np.array(input.p2_pos, dtype=np.int32)

        return {
            "spatial": spatial,
            "p1_side": p1_side,
            "p2_side": p2_side,
            "p1_pos": p1_pos,
            "p2_pos": p2_pos,
        }

    def save_to_arrays(self, observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Stack observations into arrays for npz storage.

        Args:
            observations: List of observation dicts from build().

        Returns:
            Dict with stacked arrays for each key.
        """
        return {
            "spatial": np.stack([o["spatial"] for o in observations]),
            "p1_side": np.stack([o["p1_side"] for o in observations]),
            "p2_side": np.stack([o["p2_side"] for o in observations]),
            "p1_pos": np.stack([o["p1_pos"] for o in observations]),
            "p2_pos": np.stack([o["p2_pos"] for o in observations]),
        }

    def load_from_arrays(self, arrays: dict[str, np.ndarray], idx: int) -> dict[str, np.ndarray]:
        """Load single observation from arrays.

        Args:
            arrays: Loaded npz data.
            idx: Index of observation to load.

        Returns:
            Observation dict at index.
        """
        return {
            "spatial": arrays["spatial"][idx],
            "p1_side": arrays["p1_side"][idx],
            "p2_side": arrays["p2_side"][idx],
            "p1_pos": arrays["p1_pos"][idx],
            "p2_pos": arrays["p2_pos"][idx],
        }


class SpatialDataset:
    """PyTorch-compatible Dataset for spatial observations.

    Loads all shards into memory at initialization for fast access.
    Compatible with torch.utils.data.DataLoader.

    Returns numpy arrays from __getitem__ - the DataLoader will handle
    conversion to tensors.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = SpatialDataset(training_set_dir)
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
        >>> for batch in loader:
        ...     spatial = batch["spatial"]  # (batch, 7, H, W)
        ...     p1_side = batch["p1_side"]  # (batch, 3)
        ...     # train...
    """

    def __init__(self, training_set_dir: Path | str) -> None:
        """Initialize dataset from training set directory.

        Args:
            training_set_dir: Path to training set with manifest.json and shards.

        Raises:
            FileNotFoundError: If training set doesn't exist.
        """
        from alpharat.data.sharding import load_training_set_manifest

        training_set_dir = Path(training_set_dir)
        self._manifest = load_training_set_manifest(training_set_dir)

        # Load all shards and concatenate
        spatial_list: list[np.ndarray] = []
        p1_side_list: list[np.ndarray] = []
        p2_side_list: list[np.ndarray] = []
        p1_pos_list: list[np.ndarray] = []
        p2_pos_list: list[np.ndarray] = []
        policy_p1_list: list[np.ndarray] = []
        policy_p2_list: list[np.ndarray] = []
        p1_value_list: list[np.ndarray] = []
        p2_value_list: list[np.ndarray] = []
        payout_matrix_list: list[np.ndarray] = []
        action_p1_list: list[np.ndarray] = []
        action_p2_list: list[np.ndarray] = []

        for i in range(self._manifest.shard_count):
            shard_path = training_set_dir / f"shard_{i:04d}.npz"
            with np.load(shard_path) as data:
                spatial_list.append(data["spatial"])
                p1_side_list.append(data["p1_side"])
                p2_side_list.append(data["p2_side"])
                p1_pos_list.append(data["p1_pos"])
                p2_pos_list.append(data["p2_pos"])
                policy_p1_list.append(data["policy_p1"])
                policy_p2_list.append(data["policy_p2"])
                p1_value_list.append(data["p1_value"])
                p2_value_list.append(data["p2_value"])
                payout_matrix_list.append(data["payout_matrix"])
                action_p1_list.append(data["action_p1"])
                action_p2_list.append(data["action_p2"])

        self._spatial = np.concatenate(spatial_list)
        self._p1_side = np.concatenate(p1_side_list)
        self._p2_side = np.concatenate(p2_side_list)
        self._p1_pos = np.concatenate(p1_pos_list)
        self._p2_pos = np.concatenate(p2_pos_list)
        self._policy_p1 = np.concatenate(policy_p1_list)
        self._policy_p2 = np.concatenate(policy_p2_list)
        self._p1_value = np.concatenate(p1_value_list)
        self._p2_value = np.concatenate(p2_value_list)
        self._payout_matrix = np.concatenate(payout_matrix_list)
        self._action_p1 = np.concatenate(action_p1_list)
        self._action_p2 = np.concatenate(action_p2_list)

        # Build observation builder for shape info
        self._builder = SpatialObservationBuilder(
            width=self._manifest.width, height=self._manifest.height
        )

    def __len__(self) -> int:
        """Return total number of positions."""
        return len(self._p1_value)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Get single training example.

        Args:
            idx: Position index.

        Returns:
            Dict with keys:
                - "spatial": float32 (7, H, W)
                - "p1_side": float32 (3,)
                - "p2_side": float32 (3,)
                - "p1_pos": int32 (2,)
                - "p2_pos": int32 (2,)
                - "policy_p1": float32 (5,)
                - "policy_p2": float32 (5,)
                - "p1_value": float32 (1,)
                - "p2_value": float32 (1,)
                - "payout_matrix": float32 (2, 5, 5)
                - "action_p1": int8 (1,)
                - "action_p2": int8 (1,)
        """
        return {
            "spatial": self._spatial[idx],
            "p1_side": self._p1_side[idx],
            "p2_side": self._p2_side[idx],
            "p1_pos": self._p1_pos[idx],
            "p2_pos": self._p2_pos[idx],
            "policy_p1": self._policy_p1[idx],
            "policy_p2": self._policy_p2[idx],
            "p1_value": self._p1_value[idx : idx + 1],
            "p2_value": self._p2_value[idx : idx + 1],
            "payout_matrix": self._payout_matrix[idx],
            "action_p1": self._action_p1[idx : idx + 1],
            "action_p2": self._action_p2[idx : idx + 1],
        }

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Shape of spatial tensors (C, H, W)."""
        return self._builder.spatial_shape

    @property
    def side_dim(self) -> int:
        """Dimension of side vectors."""
        return self._builder.side_dim

    @property
    def manifest(self) -> TrainingSetManifest:
        """Training set manifest."""
        return self._manifest
