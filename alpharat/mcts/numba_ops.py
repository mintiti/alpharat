"""Numba JIT-compiled operations for MCTS node hot paths.

These functions operate on the reduced (outcome-indexed) arrays stored in MCTSNode.
All arrays use float64 for Numba compatibility.
"""

from __future__ import annotations

import numpy as np
from numba import jit  # type: ignore[import-untyped]

# Large score boost for forced playouts (effectively infinite priority)
FORCED_PLAYOUT_SCORE = 1e20


@jit(cache=True)
def backup_node_scalar(
    q1: np.ndarray,
    q2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    idx1: int,
    idx2: int,
    p1_value: float,
    p2_value: float,
) -> int:
    """Update node statistics after visiting a child - O(1).

    Performs independent incremental mean updates for each player's Q-values.
    This is the decoupled UCT update: Q1[idx1] and Q2[idx2] are updated
    independently based on their respective returns.

    Args:
        q1: P1 Q-values [n1] (modified in place)
        q2: P2 Q-values [n2] (modified in place)
        n1: P1 visit counts [n1] (modified in place)
        n2: P2 visit counts [n2] (modified in place)
        idx1: Outcome index for P1's action
        idx2: Outcome index for P2's action
        p1_value: Return value for P1
        p2_value: Return value for P2

    Returns:
        New total visits (sum of n1 visits).
    """
    # Update P1's Q and N for outcome idx1
    n1_count = n1[idx1]
    n1_plus_1 = n1_count + 1.0
    q1[idx1] += (p1_value - q1[idx1]) / n1_plus_1
    n1[idx1] = n1_plus_1

    # Update P2's Q and N for outcome idx2
    n2_count = n2[idx2]
    n2_plus_1 = n2_count + 1.0
    q2[idx2] += (p2_value - q2[idx2]) / n2_plus_1
    n2[idx2] = n2_plus_1

    return int(np.sum(n1))


@jit(cache=True)
def compute_puct_scores(
    q_values: np.ndarray,
    prior: np.ndarray,
    visit_counts: np.ndarray,
    total_visits: int,
    c_puct: float,
    force_k: float,
    is_root: bool,
) -> np.ndarray:
    """Compute PUCT scores for action selection.

    PUCT = Q + c * prior * sqrt(N) / (1 + n)

    At root with force_k > 0, undervisited actions get a large score boost.

    Args:
        q_values: Q-values for each action [n].
        prior: Prior policy [n].
        visit_counts: Visit count for each action [n].
        total_visits: Total visits at this node.
        c_puct: Exploration constant.
        force_k: Forced playout scaling (0 disables).
        is_root: Whether this is the root node.

    Returns:
        PUCT score for each action [n].
    """
    n = len(q_values)
    scores = np.empty(n, dtype=np.float64)
    sqrt_total = np.sqrt(total_visits)

    for i in range(n):
        exploration = c_puct * prior[i] * sqrt_total / (1.0 + visit_counts[i])
        scores[i] = q_values[i] + exploration

    # Forced playouts at root
    if is_root and force_k > 0:
        for i in range(n):
            threshold = np.sqrt(force_k * prior[i] * total_visits)
            if visit_counts[i] < threshold and prior[i] > 0:
                scores[i] = FORCED_PLAYOUT_SCORE

    return scores


@jit(cache=True)
def select_max_with_tiebreak(scores: np.ndarray) -> int:
    """Select argmax with random tie-breaking (reservoir sampling).

    Args:
        scores: Array of scores [n].

    Returns:
        Index of maximum (random among ties).
    """
    max_val = scores[0]
    max_idx = 0
    n_ties = 1

    for i in range(1, len(scores)):
        if scores[i] > max_val + 1e-9:
            max_val = scores[i]
            max_idx = i
            n_ties = 1
        elif scores[i] > max_val - 1e-9:  # Within tolerance = tie
            n_ties += 1
            if np.random.random() < 1.0 / n_ties:
                max_idx = i

    return max_idx
