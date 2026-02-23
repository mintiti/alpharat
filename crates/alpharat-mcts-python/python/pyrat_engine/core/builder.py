"""Game configuration builder for custom games.

This module re-exports the builder classes from the compiled Rust module.
"""

# Import the compiled module directly
import pyrat_engine._core as _impl

# Re-export builder classes
GameBuilder = _impl.builder.GameBuilder
GameConfig = _impl.builder.GameConfig

__all__ = ["GameBuilder", "GameConfig"]
