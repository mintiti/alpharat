"""Python bindings for the Rust self-play sampling pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat_engine._core.sampling import (  # type: ignore[import-not-found]
        SelfPlayProgress as SelfPlayProgress,
    )
    from pyrat_engine._core.sampling import (
        SelfPlayStats as SelfPlayStats,
    )
    from pyrat_engine._core.sampling import (
        rust_self_play as rust_self_play,
    )
else:
    import pyrat_engine._core as _impl

    SelfPlayStats = _impl.sampling.SelfPlayStats
    SelfPlayProgress = _impl.sampling.SelfPlayProgress
    rust_self_play = _impl.sampling.rust_self_play

__all__ = ["rust_self_play", "SelfPlayStats", "SelfPlayProgress"]
