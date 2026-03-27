from collections.abc import Callable

import numpy as np
from pyrat_engine.core.game import PyRat

class SearchResult:
    """Result of an MCGS search: policies and values for both players."""

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
    def q_values_p1(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def q_values_p2(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...
    @property
    def total_visits(self) -> int: ...
    @property
    def nn_evals(self) -> int: ...
    @property
    def terminals(self) -> int: ...
    @property
    def collisions(self) -> int: ...
    @property
    def tt_stop_hits(self) -> int: ...
    def __repr__(self) -> str: ...

def rust_mcgs_search(
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
    collision_limit_min: int = 1,
    collision_limit_max: int = 256,
    collision_scaling_start: int = 800,
    collision_scaling_end: int = 50000,
    collision_scaling_power: float = 1.0,
    seed: int | None = None,
) -> SearchResult: ...
