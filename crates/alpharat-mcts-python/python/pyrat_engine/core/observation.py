"""Game observation and state tracking classes.

This module re-exports observation classes from the compiled Rust module.
"""

# Import the compiled module directly
import pyrat_engine._core as _impl

# Re-export observation classes with cleaner names
GameObservation = _impl.observation.PyGameObservation
ObservationHandler = _impl.observation.PyObservationHandler

# Keep original names for backward compatibility if needed
PyGameObservation = GameObservation
PyObservationHandler = ObservationHandler

__all__ = [
    "GameObservation",
    "ObservationHandler",
    "PyGameObservation",
    "PyObservationHandler",
]
