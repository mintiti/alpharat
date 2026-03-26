"""Python bindings for the Rust MCGS search."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat_engine._core.mcgs import SearchResult as SearchResult
    from pyrat_engine._core.mcgs import rust_mcgs_search as rust_mcgs_search
else:
    import pyrat_engine._core as _impl

    SearchResult = _impl.mcgs.SearchResult
    rust_mcgs_search = _impl.mcgs.rust_mcgs_search

__all__ = ["rust_mcgs_search", "SearchResult"]
