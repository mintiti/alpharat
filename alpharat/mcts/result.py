"""Canonical MCTS search result — backend-agnostic output type.

Both Python and Rust MCTS backends produce this identical type.
Consumers (sampling, evaluation, benchmarking) never touch search internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SearchResult:
    """Result of MCTS search containing policy, value, and diagnostic information.

    All arrays are in 5-action space: [UP, RIGHT, DOWN, LEFT, STAY].
    Blocked actions have probability 0 in policies.

    Fields:
        policy_p1: Visit-proportional policy for P1 [5].
        policy_p2: Visit-proportional policy for P2 [5].
        value_p1: Root value estimate for P1 (expected remaining cheese).
        value_p2: Root value estimate for P2 (expected remaining cheese).
        visit_counts_p1: Pruned visit counts for P1 [5].
        visit_counts_p2: Pruned visit counts for P2 [5].
        prior_p1: NN/uniform prior at root for P1 [5].
        prior_p2: NN/uniform prior at root for P2 [5].
        total_visits: Root visit count after search.
    """

    policy_p1: np.ndarray
    policy_p2: np.ndarray
    value_p1: float
    value_p2: float
    visit_counts_p1: np.ndarray
    visit_counts_p2: np.ndarray
    prior_p1: np.ndarray
    prior_p2: np.ndarray
    total_visits: int
