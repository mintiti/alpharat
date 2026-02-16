"""Python bindings for the Rust MCTS search."""

import pyrat_engine._core as _impl

SearchResult = _impl.mcts.SearchResult  # type: ignore[attr-defined]
rust_mcts_search = _impl.mcts.rust_mcts_search  # type: ignore[attr-defined]

__all__ = ["rust_mcts_search", "SearchResult"]
