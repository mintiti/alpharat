"""Post-search pruning of forced visits (KataGo-style).

When forced playouts inflate visit counts for exploration, these functions
compute how many visits were "forced" and subtract them before using the
counts for training targets.
"""

from __future__ import annotations

import numpy as np


def compute_pruning_adjustment(
    q_values: np.ndarray,
    prior: np.ndarray,
    marginal_visits: np.ndarray,
    total_visits: int,
    c_puct: float,
) -> np.ndarray:
    """Compute per-outcome visit adjustments for forced playout pruning.

    For each unique outcome, computes how many visits were "forced" (exceeded
    what PUCT would naturally select). These forced visits are subtracted
    before using the visit counts for training.

    All inputs are in reduced (outcome-indexed) space — already over unique
    outcomes, no effective-action filtering needed.

    Algorithm (per player):
    1. Find the best outcome i* (most marginal visits)
    2. Compute PUCT* = Q[i*] + c * P[i*] * sqrt(N) / (1 + M[i*])
    3. For each other outcome i:
       - If Q[i] >= PUCT*: outcome is genuinely good, Δ[i] = 0
       - Else: compute N'_min = c * P[i] * sqrt(N) / (PUCT* - Q[i]) - 1
         This is the visit count that would make PUCT(i) = PUCT*
       - Δ[i] = max(0, M[i] - N'_min)

    Args:
        q_values: Marginalized Q-values per unique outcome [n].
        prior: Prior policy per unique outcome [n].
        marginal_visits: Marginal visit counts per unique outcome [n].
        total_visits: Total visits to the node.
        c_puct: PUCT exploration constant.

    Returns:
        Adjustment array Δ[n]: visits to subtract from each outcome's marginal.
    """
    n = len(q_values)
    delta = np.zeros(n)

    if n <= 1:
        return delta  # Nothing to prune

    # Find best outcome: most marginal visits
    i_star = int(np.argmax(marginal_visits))

    # Compute PUCT* for best outcome
    sqrt_n = np.sqrt(total_visits) if total_visits > 0 else 0.0
    puct_star = q_values[i_star] + c_puct * prior[i_star] * sqrt_n / (1 + marginal_visits[i_star])

    # Compute adjustments for other outcomes
    for a in range(n):
        if a == i_star:
            continue  # Don't touch best outcome

        q_a = q_values[a]
        if q_a >= puct_star:
            # Outcome is genuinely good (high Q), don't prune
            delta[a] = 0.0
        else:
            # Compute minimum visits that would achieve PUCT*
            # PUCT(a) = Q[a] + c * P[a] * sqrt(N) / (1 + M[a])
            # Solve for M[a] when PUCT(a) = PUCT*:
            # M_min = c * P[a] * sqrt(N) / (PUCT* - Q[a]) - 1
            gap = puct_star - q_a
            if gap > 1e-8:  # Avoid division by near-zero
                n_min = c_puct * prior[a] * sqrt_n / gap - 1
                n_min = max(0.0, n_min)  # Can't have negative visits
                delta[a] = max(0.0, marginal_visits[a] - n_min)
            else:
                delta[a] = 0.0

    return delta


def prune_visit_counts(
    action_visits: np.ndarray,
    delta_p1: np.ndarray,
    delta_p2: np.ndarray,
    prior_p1: np.ndarray,
    prior_p2: np.ndarray,
) -> np.ndarray:
    """Apply per-player adjustments to pair visit counts.

    Distributes marginal adjustments to pairs using opponent's prior as weights.
    V'[i,j] = max(0, V[i,j] - Δ1[i] * π2[j] - Δ2[j] * π1[i])

    The intuition: when P1 was forced to outcome i, P2 was selecting via PUCT
    (approximately proportional to π2). So forced visits to pairs (i, ·) are
    distributed as π2.

    All inputs are in reduced (outcome-indexed) space.

    Args:
        action_visits: Original visit counts [n1, n2].
        delta_p1: P1's marginal adjustments [n1].
        delta_p2: P2's marginal adjustments [n2].
        prior_p1: P1's prior policy [n1].
        prior_p2: P2's prior policy [n2].

    Returns:
        Pruned visit counts [n1, n2] (can be fractional, may have zeros).
    """
    # Outer products for distributing adjustments
    # Δ1[i] affects row i, distributed by π2[j]
    p1_adjustment = np.outer(delta_p1, prior_p2)  # [n1, n2]
    # Δ2[j] affects column j, distributed by π1[i]
    p2_adjustment = np.outer(prior_p1, delta_p2)  # [n1, n2]

    pruned = action_visits - p1_adjustment - p2_adjustment
    result: np.ndarray = np.maximum(0.0, pruned)
    return result
