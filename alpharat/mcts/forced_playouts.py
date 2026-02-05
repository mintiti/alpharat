"""Forced playout pruning for visit-proportional policies.

When using force_k > 0 in PUCT, undervisited actions get "forced" exploration
at the root. These forced visits shouldn't count toward the training policy
since they inflate visit counts beyond what PUCT would naturally select.

This module implements KataGo-style pruning: for each non-best action, cap
its visit count to what would make its PUCT score equal to the best action's
PUCT score.
"""

from __future__ import annotations

import numpy as np


def compute_pruned_visits(
    q_values: np.ndarray,
    prior: np.ndarray,
    visit_counts: np.ndarray,
    total_visits: int,
    c_puct: float,
) -> np.ndarray:
    """Compute pruned visit counts for one player.

    KataGo-style: for each non-best action, cap visits to what would make
    PUCT(action) == PUCT(best). Actions with Q >= PUCT(best) are kept as-is.

    The PUCT formula is: PUCT(i) = Q(i) + c * P(i) * sqrt(N) / (1 + n(i))

    For a non-best action i with Q(i) < PUCT*, we solve for n_min:
        Q(i) + c * P(i) * sqrt(N) / (1 + n_min) = PUCT*
        n_min = c * P(i) * sqrt(N) / (PUCT* - Q(i)) - 1

    Args:
        q_values: Marginal Q-values [n] in reduced space.
        prior: Reduced prior probabilities [n].
        visit_counts: Raw marginal visit counts [n].
        total_visits: Total simulations through this node.
        c_puct: PUCT exploration constant.

    Returns:
        Pruned visit counts [n]. Best action unchanged, high-Q actions
        unchanged, forced visits on low-Q actions capped.
    """
    n = len(q_values)

    if n <= 1 or total_visits == 0:
        return visit_counts.copy()

    # Find best outcome (highest visits)
    best_idx = int(np.argmax(visit_counts))

    # Compute PUCT* for the best action
    sqrt_n = np.sqrt(total_visits)
    exploration_best = c_puct * prior[best_idx] * sqrt_n / (1 + visit_counts[best_idx])
    puct_best = q_values[best_idx] + exploration_best

    # Prune other actions
    pruned = visit_counts.copy()

    for i in range(n):
        if i == best_idx:
            continue

        # If Q >= PUCT*, this action is genuinely good - keep all visits
        if q_values[i] >= puct_best:
            continue

        # Compute n_min: the visit count that would make PUCT(i) == PUCT*
        # PUCT(i) = Q(i) + c * P(i) * sqrt(N) / (1 + n_min) = PUCT*
        # Solving: n_min = c * P(i) * sqrt(N) / (PUCT* - Q(i)) - 1
        gap = puct_best - q_values[i]

        # Handle edge case: zero prior means action shouldn't be visited
        if prior[i] <= 0 or gap <= 0:
            pruned[i] = 0.0
            continue

        n_min = c_puct * prior[i] * sqrt_n / gap - 1

        # Clamp: can't go negative, and can't increase visits
        pruned[i] = min(visit_counts[i], max(0.0, n_min))

    return pruned
