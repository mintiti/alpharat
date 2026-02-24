"""Observation builders for different neural network architectures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from alpharat.nn.builders.flat import FlatDataset, FlatObservationBuilder, FlatObsLayout

if TYPE_CHECKING:
    import numpy as np

    from alpharat.nn.types import ObservationInput


class ObservationBuilder(Protocol):
    """Protocol for observation encoding strategies.

    Each builder owns the full cycle: build observations from game state,
    serialize to npz-friendly format, and deserialize back. Builder modules
    also provide their own Dataset class for training.
    """

    @property
    def version(self) -> str:
        """Identifier for this builder configuration.

        Stored in training set manifest for reproducibility.
        """
        ...

    def build(self, input: ObservationInput) -> Any:
        """Encode game state into observation.

        The return type depends on the builder:
        - Flat: np.ndarray of shape (obs_dim,)
        - GNN: dict with node_features, edge_index, etc.
        """
        ...

    def save_to_arrays(self, observations: list[Any]) -> dict[str, np.ndarray]:
        """Convert list of observations to npz-friendly dict.

        Called by sharding code to prepare a shard for saving.
        """
        ...

    def load_from_arrays(self, arrays: dict[str, np.ndarray], idx: int) -> Any:
        """Load single observation from arrays at index.

        Reconstructs observation from loaded npz data.
        """
        ...


__all__ = [
    "ObservationBuilder",
    "FlatObsLayout",
    "FlatObservationBuilder",
    "FlatDataset",
]
