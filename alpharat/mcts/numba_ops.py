"""Numba JIT-compiled operations for MCTS node hot paths.

These functions operate on the reduced (outcome-indexed) arrays stored in MCTSNode.
All arrays use float64 for Numba compatibility.
"""

from __future__ import annotations

import numpy as np
from numba import jit  # type: ignore[import-untyped]


@jit(cache=True)
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


@jit(cache=True)
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


@jit(cache=True)
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


@jit(cache=True)
def compute_expected_value_reduced(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    prior_p1_reduced: np.ndarray,
    prior_p2_reduced: np.ndarray,
) -> tuple[float, float]:
    """Compute expected value directly in reduced (outcome-indexed) space.

    E[V] = π₁ᵀ · Payoff · π₂

    Args:
        payout_p1: P1 payout matrix [n1, n2]
        payout_p2: P2 payout matrix [n1, n2]
        prior_p1_reduced: P1 prior over outcomes [n1]
        prior_p2_reduced: P2 prior over outcomes [n2]

    Returns:
        Tuple (v1, v2) of expected values for P1 and P2.
    """
    n1 = payout_p1.shape[0]
    n2 = payout_p1.shape[1]
    v1 = 0.0
    v2 = 0.0

    for i in range(n1):
        p1_prior = prior_p1_reduced[i]
        for j in range(n2):
            weight = p1_prior * prior_p2_reduced[j]
            v1 += weight * payout_p1[i, j]
            v2 += weight * payout_p2[i, j]

    return v1, v2


@jit(cache=True)
def compute_marginal_q_reduced(
    payout_p1: np.ndarray,
    payout_p2: np.ndarray,
    prior_p1_reduced: np.ndarray,
    prior_p2_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marginalized Q-values in reduced space.

    Q1[i] = sum over j of payout[i, j] * prior_p2[j]
    Q2[j] = sum over i of payout[i, j] * prior_p1[i]

    Args:
        payout_p1: P1 payout matrix [n1, n2]
        payout_p2: P2 payout matrix [n1, n2]
        prior_p1_reduced: P1 prior over outcomes [n1]
        prior_p2_reduced: P2 prior over outcomes [n2]

    Returns:
        Tuple (q1, q2) where q1[i] is P1's Q-value for outcome i.
    """
    n1 = payout_p1.shape[0]
    n2 = payout_p1.shape[1]

    q1 = np.zeros(n1, dtype=np.float64)
    q2 = np.zeros(n2, dtype=np.float64)

    for i in range(n1):
        total = 0.0
        for j in range(n2):
            total += payout_p1[i, j] * prior_p2_reduced[j]
        q1[i] = total

    for j in range(n2):
        total = 0.0
        for i in range(n1):
            total += payout_p2[i, j] * prior_p1_reduced[i]
        q2[j] = total

    return q1, q2


@jit(cache=True)
def compute_marginal_visits_reduced(
    visits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute marginal visit counts in reduced space.

    Args:
        visits: Visit counts [n1, n2]

    Returns:
        Tuple (n1, n2) where n1[i] is visit count for outcome i.
    """
    n1_size = visits.shape[0]
    n2_size = visits.shape[1]

    n1 = np.zeros(n1_size, dtype=np.float64)
    n2 = np.zeros(n2_size, dtype=np.float64)

    for i in range(n1_size):
        for j in range(n2_size):
            n1[i] += visits[i, j]
            n2[j] += visits[i, j]

    return n1, n2


@jit(cache=True)
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
                scores[i] = 1e20

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


@jit(cache=True)
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
