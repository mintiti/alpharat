"""Action selection utilities for MCTS.

Provides forced playout threshold computation for ensuring minimum exploration
of actions proportional to prior probability.
"""

from __future__ import annotations

import numpy as np


def compute_forced_threshold(
    prior: np.ndarray,
    total_visits: int,
    k: float,
) -> np.ndarray:
    """Compute forced visit threshold for one player.

    When marginal visits < threshold, force exploration of that action.
    Based on KataGo's forced playout formula, adapted for decoupled selection.

    Args:
        prior: Prior policy for one player [5].
        total_visits: Total visits to the node.
        k: Scaling constant (2.0 is KataGo default, 0 disables forcing).

    Returns:
        Array of thresholds [5], one per action.
    """
    return np.sqrt(k * prior * total_visits)
