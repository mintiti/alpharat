"""PyRat Engine Core - Low-level Rust bindings.

This package provides direct access to the Rust game engine implementation.
For most use cases, prefer the high-level API in pyrat_engine.game.
"""

from typing import TYPE_CHECKING

# Import the compiled Rust module (private implementation detail)
import pyrat_engine._core as _impl

# Re-export submodules as module attributes for convenience
# Note: These are not real Python submodules, but attributes on the compiled module
types = _impl.types
game = _impl.game
observation = _impl.observation
builder = _impl.builder

# Re-export commonly used classes at package level
# Rust types from PyO3
Coordinates = _impl.types.Coordinates
Wall = _impl.types.Wall
Mud = _impl.types.Mud

# Direction is a Python IntEnum (not from Rust)
# Must be imported after _impl since types.py depends on it
from pyrat_engine.core.types import Direction  # noqa: E402

# Conditionally import types for type checking to avoid "not valid as a type" errors
if TYPE_CHECKING:
    from pyrat_engine.core.builder import GameBuilder as GameBuilder
    from pyrat_engine.core.builder import GameConfig as GameConfig
    from pyrat_engine.core.game import PyMoveUndo as MoveUndo
    from pyrat_engine.core.game import PyRat
    from pyrat_engine.core.observation import PyGameObservation as GameObservation
    from pyrat_engine.core.observation import PyObservationHandler as ObservationHandler
else:
    PyRat = _impl.game.PyRat
    MoveUndo = _impl.game.PyMoveUndo
    GameObservation = _impl.observation.PyGameObservation
    ObservationHandler = _impl.observation.PyObservationHandler
    GameBuilder = _impl.builder.GameBuilder
    GameConfig = _impl.builder.GameConfig

__all__ = [
    # Submodules
    "types",
    "game",
    "observation",
    "builder",
    # Types
    "Coordinates",
    "Direction",
    "Wall",
    "Mud",
    # Game
    "PyRat",
    "MoveUndo",
    # Observation
    "GameObservation",
    "ObservationHandler",
    # Builder
    "GameBuilder",
    "GameConfig",
]
