from collections.abc import Callable

import numpy as np
from pyrat_engine.core.game import PyRat

class SearchResult:
    """Result of an MCTS search: policies and values for both players."""

    @property
    def policy_p1(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def policy_p2(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def value_p1(self) -> float: ...
    @property
    def value_p2(self) -> float: ...
    @property
    def visit_counts_p1(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def visit_counts_p2(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def prior_p1(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def prior_p2(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def total_visits(self) -> int: ...
    def __repr__(self) -> str: ...

def rust_mcts_search(
    game: PyRat,
    *,
    predict_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    simulations: int = 100,
    batch_size: int = 8,
    c_puct: float = 1.5,
    fpu_reduction: float = 0.2,
    force_k: float = 2.0,
    noise_epsilon: float = 0.0,
    noise_concentration: float = 10.83,
    seed: int | None = None,
) -> SearchResult: ...
