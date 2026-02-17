"""Python bindings for the Rust MCTS search."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat_engine._core.mcts import SearchResult as SearchResult
    from pyrat_engine._core.mcts import rust_mcts_search as rust_mcts_search
else:
    import pyrat_engine._core as _impl

    SearchResult = _impl.mcts.SearchResult
    rust_mcts_search = _impl.mcts.rust_mcts_search

__all__ = ["rust_mcts_search", "SearchResult"]
