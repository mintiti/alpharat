"""Core game state and management classes.

This module re-exports game classes from the compiled Rust module.
"""

# Import the compiled module directly
import pyrat_engine._core as _impl

# Re-export game classes
PyRat = _impl.game.PyRat
MoveUndo = _impl.game.PyMoveUndo

# Alias for backward compatibility
PyMoveUndo = MoveUndo

__all__ = ["MoveUndo", "PyMoveUndo", "PyRat"]
