"""Utility functions for AI agents."""

from __future__ import annotations

import numpy as np


def select_action_from_strategy(strategy: np.ndarray, temperature: float = 1.0) -> int:
    """Sample an action from a probability distribution.

    Args:
        strategy: Probability distribution over actions (must sum to 1).
        temperature: Sampling temperature.
            - 0.0 = deterministic argmax
            - 1.0 = sample from distribution
            - >1.0 = more random

    Returns:
        Selected action index.
    """
    if temperature == 0.0:
        return int(np.argmax(strategy))

    if temperature == 1.0:
        return int(np.random.choice(len(strategy), p=strategy))

    # Apply temperature
    log_probs = np.log(strategy + 1e-10)
    tempered_logits = log_probs / temperature
    tempered_probs = np.exp(tempered_logits)
    tempered_probs /= tempered_probs.sum()

    return int(np.random.choice(len(tempered_probs), p=tempered_probs))
