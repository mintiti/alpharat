"""Payout matrix filtering for MCTS.

Filters unreliable cells (low visit counts) before Nash computation and recording.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def filter_low_visit_payout(
    payout: np.ndarray,
    visits: np.ndarray,
    min_visits: int = 2,
) -> np.ndarray:
    """Zero out payout cells with insufficient visits.

    Pessimistic estimate: unexplored action pairs assumed to yield 0 for both players.
    This prevents NN predictions for unexplored cells from polluting Nash computation
    and training data.

    Works with any shape — both expanded [2, 5, 5] and reduced [2, n1, n2].

    Args:
        payout: Shape [2, n1, n2] — P1 and P2 payouts per action pair.
        visits: Shape [n1, n2] — visit count per action pair.
        min_visits: Cells with visits < this are zeroed (default: 2, i.e., filter ≤1).

    Returns:
        Filtered payout matrix (copy, original unchanged).
    """
    filtered = payout.copy()
    mask = visits < min_visits
    filtered[:, mask] = 0.0
    return filtered
