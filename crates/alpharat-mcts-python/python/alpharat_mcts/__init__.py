"""Python bindings for the Rust MCTS search."""

import pyrat_engine._core as _impl  # type: ignore[import-untyped]

SearchResult = _impl.mcts.SearchResult
rust_mcts_search = _impl.mcts.rust_mcts_search

__all__ = ["rust_mcts_search", "SearchResult"]
