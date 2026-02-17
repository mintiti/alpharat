"""Game configuration builder for custom games.

This module re-exports the builder class from the compiled Rust module.
"""

# Import the compiled module directly
import pyrat_engine._core as _impl

# Re-export builder class with cleaner name
GameConfigBuilder = _impl.builder.PyGameConfigBuilder

# Keep original name for backward compatibility if needed
PyGameConfigBuilder = GameConfigBuilder

__all__ = ["GameConfigBuilder", "PyGameConfigBuilder"]
