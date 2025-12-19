"""Neural network utilities for observation encoding and target extraction."""

from alpharat.nn.extraction import from_game_arrays, from_pyrat_game
from alpharat.nn.streaming import StreamingDataset
from alpharat.nn.targets import build_targets
from alpharat.nn.types import ObservationInput, TargetBundle

__all__ = [
    "ObservationInput",
    "StreamingDataset",
    "TargetBundle",
    "build_targets",
    "from_game_arrays",
    "from_pyrat_game",
]
