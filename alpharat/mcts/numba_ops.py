"""Numba JIT-compiled operations for MCTS node hot paths.

These functions operate on the reduced (outcome-indexed) arrays stored in MCTSNode.
All arrays use float64 for Numba compatibility.
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[import-untyped]


@njit(cache=True)
def backup_node(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    visits: np.ndarray,
    idx1: int,
    idx2: int,
    p1_value: float,
    p2_value: float,
) -> int:
    """Update node statistics after visiting a child - O(1).

    Performs incremental mean update for payout estimates.

    Args:
        payout_p1: P1 payout matrix [n1, n2] (modified in place)
        payout_p2: P2 payout matrix [n1, n2] (modified in place)
        visits: Visit counts [n1, n2] (modified in place)
        idx1: Outcome index for P1's action
        idx2: Outcome index for P2's action
        p1_value: Return value for P1
        p2_value: Return value for P2

    Returns:
        New total visits (visits array is updated in place).
    """
    n = visits[idx1, idx2]
    n_plus_1 = n + 1.0

    # Incremental mean update
    payout_p1[idx1, idx2] += (p1_value - payout_p1[idx1, idx2]) / n_plus_1
    payout_p2[idx1, idx2] += (p2_value - payout_p2[idx1, idx2]) / n_plus_1
    visits[idx1, idx2] = n_plus_1

    return int(np.sum(visits))


@njit(cache=True)
def compute_marginal_q_numba(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    prior_p1: np.ndarray,
    prior_p2: np.ndarray,
    p1_action_to_idx: np.ndarray,
    p2_action_to_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marginalized Q-values for PUCT selection.

    Q1[a1] = sum over a2 of payout[a1, a2] * prior_p2[a2]
    Q2[a2] = sum over a1 of payout[a1, a2] * prior_p1[a1]

    Args:
        payout_p1: P1 payout matrix [n1, n2]
        payout_p2: P2 payout matrix [n1, n2]
        prior_p1: P1 prior over actions [num_actions]
        prior_p2: P2 prior over actions [num_actions]
        p1_action_to_idx: Maps action to outcome index [num_actions]
        p2_action_to_idx: Maps action to outcome index [num_actions]

    Returns:
        Tuple (q1, q2) where q1[a] is P1's Q-value for action a.
    """
    num_actions = len(prior_p1)
    q1 = np.zeros(num_actions, dtype=np.float64)
    q2 = np.zeros(num_actions, dtype=np.float64)

    for a1 in range(num_actions):
        idx1 = p1_action_to_idx[a1]
        total = 0.0
        for a2 in range(num_actions):
            idx2 = p2_action_to_idx[a2]
            total += payout_p1[idx1, idx2] * prior_p2[a2]
        q1[a1] = total

    for a2 in range(num_actions):
        idx2 = p2_action_to_idx[a2]
        total = 0.0
        for a1 in range(num_actions):
            idx1 = p1_action_to_idx[a1]
            total += payout_p2[idx1, idx2] * prior_p1[a1]
        q2[a2] = total

    return q1, q2


@njit(cache=True)
def compute_marginal_visits_numba(
    visits: np.ndarray,
    p1_action_to_idx: np.ndarray,
    p2_action_to_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marginal visit counts for PUCT selection.

    Args:
        visits: Visit counts [n1, n2]
        p1_action_to_idx: Maps action to outcome index [num_actions]
        p2_action_to_idx: Maps action to outcome index [num_actions]

    Returns:
        Tuple (n1, n2) where n1[a] is visit count for P1's action a.
    """
    num_actions = len(p1_action_to_idx)
    n1_shape = visits.shape[0]
    n2_shape = visits.shape[1]

    # Sum over reduced axes
    reduced_n1 = np.zeros(n1_shape, dtype=np.float64)
    reduced_n2 = np.zeros(n2_shape, dtype=np.float64)

    for i in range(n1_shape):
        for j in range(n2_shape):
            reduced_n1[i] += visits[i, j]
            reduced_n2[j] += visits[i, j]

    # Expand to action space
    n1 = np.zeros(num_actions, dtype=np.float64)
    n2 = np.zeros(num_actions, dtype=np.float64)

    for a in range(num_actions):
        n1[a] = reduced_n1[p1_action_to_idx[a]]
        n2[a] = reduced_n2[p2_action_to_idx[a]]

    return n1, n2


@njit(cache=True)
def compute_expected_value_numba(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    prior_p1: np.ndarray,
    prior_p2: np.ndarray,
    p1_action_to_idx: np.ndarray,
    p2_action_to_idx: np.ndarray,
) -> tuple[float, float]:
    """Compute expected value under NN priors for both players.

    E[V] = π₁ᵀ · Payoff · π₂

    Args:
        payout_p1: P1 payout matrix [n1, n2]
        payout_p2: P2 payout matrix [n1, n2]
        prior_p1: P1 prior over actions [num_actions]
        prior_p2: P2 prior over actions [num_actions]
        p1_action_to_idx: Maps action to outcome index [num_actions]
        p2_action_to_idx: Maps action to outcome index [num_actions]

    Returns:
        Tuple (v1, v2) of expected values for P1 and P2.
    """
    num_actions = len(prior_p1)
    v1 = 0.0
    v2 = 0.0

    for a1 in range(num_actions):
        idx1 = p1_action_to_idx[a1]
        p1_prior = prior_p1[a1]
        for a2 in range(num_actions):
            idx2 = p2_action_to_idx[a2]
            weight = p1_prior * prior_p2[a2]
            v1 += weight * payout_p1[idx1, idx2]
            v2 += weight * payout_p2[idx1, idx2]

    return v1, v2


@njit(cache=True)
def build_expanded_payout(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    p1_action_to_idx: np.ndarray,
    p2_action_to_idx: np.ndarray,
) -> np.ndarray:
    """Expand reduced payout matrices to full [2, num_actions, num_actions] shape.

    Args:
        payout_p1: P1 payout matrix [n1, n2]
        payout_p2: P2 payout matrix [n1, n2]
        p1_action_to_idx: Maps action to outcome index [num_actions]
        p2_action_to_idx: Maps action to outcome index [num_actions]

    Returns:
        Expanded payout matrix [2, num_actions, num_actions].
    """
    num_actions = len(p1_action_to_idx)
    expanded = np.zeros((2, num_actions, num_actions), dtype=np.float64)

    for a1 in range(num_actions):
        i = p1_action_to_idx[a1]
        for a2 in range(num_actions):
            j = p2_action_to_idx[a2]
            expanded[0, a1, a2] = payout_p1[i, j]
            expanded[1, a1, a2] = payout_p2[i, j]

    return expanded


@njit(cache=True)
def build_expanded_visits(
    visits: np.ndarray,
    p1_action_to_idx: np.ndarray,
    p2_action_to_idx: np.ndarray,
) -> np.ndarray:
    """Expand reduced visit counts to full [num_actions, num_actions] shape.

    Args:
        visits: Visit counts [n1, n2]
        p1_action_to_idx: Maps action to outcome index [num_actions]
        p2_action_to_idx: Maps action to outcome index [num_actions]

    Returns:
        Expanded visit counts [num_actions, num_actions].
    """
    num_actions = len(p1_action_to_idx)
    expanded = np.zeros((num_actions, num_actions), dtype=np.float64)

    for a1 in range(num_actions):
        i = p1_action_to_idx[a1]
        for a2 in range(num_actions):
            j = p2_action_to_idx[a2]
            expanded[a1, a2] = visits[i, j]

    return expanded
