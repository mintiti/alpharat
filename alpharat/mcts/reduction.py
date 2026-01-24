"""Boundary translation between 5-action space and outcome-indexed space.

This module handles the translation between:
- **5-action space**: The external interface (NN outputs, Nash results, recorded data)
- **Outcome space**: The internal MCTS representation (reduced by action equivalence)

Key concepts:
- **Effective action mapping**: `effective[a]` gives the actual outcome when playing action `a`.
  Blocked actions map to STAY (action 4).
- **Outcome indices**: Unique outcomes numbered 0 to n-1, where n = len(set(effective)).
- **Reduction**: Collapsing [5] or [5,5] arrays to [n] or [n1,n2] by summing/selecting.
- **Expansion**: Restoring [n] to [5] by copying values to all actions with same outcome.

The algorithm operates entirely in outcome space. Boundaries handle translation:
- Input boundary: NN predictions [5], [5], [2,5,5] → reduced [n1], [n2], [2,n1,n2]
- Output boundary: Nash strategies [n1], [n2] → expanded [5], [5]
"""

from __future__ import annotations

import numpy as np


def get_unique_outcomes(effective: list[int]) -> list[int]:
    """Get sorted list of unique outcome indices (effective action values).

    Args:
        effective: Maps each action index (0-4) to its effective action.

    Returns:
        Sorted list of unique effective actions.
    """
    return sorted(set(effective))


def build_action_to_outcome_map(effective: list[int]) -> list[int]:
    """Build mapping from action index to outcome index.

    Args:
        effective: Maps each action (0-4) to its effective action.

    Returns:
        List where result[action] gives the outcome index (0 to n-1).
    """
    outcomes = get_unique_outcomes(effective)
    outcome_to_idx = {eff: i for i, eff in enumerate(outcomes)}
    return [outcome_to_idx[effective[a]] for a in range(len(effective))]


def outcome_to_effective(outcome_idx: int, effective: list[int]) -> int:
    """Convert outcome index back to effective action value.

    Args:
        outcome_idx: Outcome index (0 to n-1).
        effective: Maps each action (0-4) to its effective action.

    Returns:
        The effective action value corresponding to this outcome.
    """
    outcomes = get_unique_outcomes(effective)
    return outcomes[outcome_idx]


def reduce_prior(prior_5: np.ndarray, effective: list[int]) -> np.ndarray:
    """Reduce [5] policy to [n] by summing probabilities of equivalent actions.

    When multiple actions map to the same outcome, their probabilities are summed.
    This preserves the total probability mass and maintains a valid distribution.

    Args:
        prior_5: Policy distribution over 5 actions, shape [5].
        effective: Maps each action (0-4) to its effective action.

    Returns:
        Reduced policy over unique outcomes, shape [n].
    """
    outcomes = get_unique_outcomes(effective)
    n = len(outcomes)
    action_to_outcome = build_action_to_outcome_map(effective)

    reduced = np.zeros(n, dtype=prior_5.dtype)
    for a, prob in enumerate(prior_5):
        reduced[action_to_outcome[a]] += prob

    return reduced


def reduce_payout(
    payout_5x5: np.ndarray,
    p1_effective: list[int],
    p2_effective: list[int],
) -> np.ndarray:
    """Reduce [2,5,5] payout matrix to [2,n1,n2] by averaging equivalent pairs.

    When the NN outputs noisy predictions that don't perfectly respect equivalence
    (e.g., payout[blocked, x] != payout[STAY, x]), we average all action pairs
    that map to the same outcome pair. This is more robust than arbitrarily
    selecting one value.

    After MCTS backup, equivalent pairs ARE identical (backup maintains this
    invariant), so averaging degenerates to selection. But for initial NN
    predictions, averaging provides a better estimate.

    Args:
        payout_5x5: Payout matrix shape [2, 5, 5] where [0] is P1's payoffs.
        p1_effective: Maps each P1 action to its effective action.
        p2_effective: Maps each P2 action to its effective action.

    Returns:
        Reduced payout matrix shape [2, n1, n2].
    """
    p1_outcomes = get_unique_outcomes(p1_effective)
    p2_outcomes = get_unique_outcomes(p2_effective)
    n1, n2 = len(p1_outcomes), len(p2_outcomes)

    p1_to_outcome = build_action_to_outcome_map(p1_effective)
    p2_to_outcome = build_action_to_outcome_map(p2_effective)

    # Accumulate sums and counts for averaging
    sums = np.zeros((2, n1, n2), dtype=np.float64)
    counts = np.zeros((n1, n2), dtype=np.int32)

    num_actions_p1 = len(p1_effective)
    num_actions_p2 = len(p2_effective)

    for a1 in range(num_actions_p1):
        i = p1_to_outcome[a1]
        for a2 in range(num_actions_p2):
            j = p2_to_outcome[a2]
            sums[0, i, j] += payout_5x5[0, a1, a2]
            sums[1, i, j] += payout_5x5[1, a1, a2]
            counts[i, j] += 1

    # Average (counts is always > 0 since each outcome has at least one action)
    reduced = sums / counts[np.newaxis, :, :]

    return reduced.astype(payout_5x5.dtype)


def expand_prior(prior_n: np.ndarray, effective: list[int]) -> np.ndarray:
    """Expand [n] policy to [5] by copying outcome probs to equivalent actions.

    Each action gets the probability of its outcome. Non-effective actions
    (those that map to a different action) get 0 probability.

    Note: Only effective actions themselves get probability. If action 0 maps
    to STAY (4), action 0 gets 0 and action 4 gets the STAY outcome probability.

    Args:
        prior_n: Policy distribution over n unique outcomes.
        effective: Maps each action (0-4) to its effective action.

    Returns:
        Expanded policy over 5 actions, shape [5].
    """
    outcomes = get_unique_outcomes(effective)
    num_actions = len(effective)

    expanded = np.zeros(num_actions, dtype=prior_n.dtype)
    for i, outcome_action in enumerate(outcomes):
        # Only the canonical effective action gets probability
        expanded[outcome_action] = prior_n[i]

    return expanded


def expand_payout(
    payout_n1n2: np.ndarray,
    p1_effective: list[int],
    p2_effective: list[int],
) -> np.ndarray:
    """Expand [2,n1,n2] payout to [2,5,5] by copying to equivalent action pairs.

    Each action pair (a1, a2) gets the value of its effective outcome pair.

    Args:
        payout_n1n2: Reduced payout matrix shape [2, n1, n2].
        p1_effective: Maps each P1 action to its effective action.
        p2_effective: Maps each P2 action to its effective action.

    Returns:
        Expanded payout matrix shape [2, 5, 5].
    """
    num_actions_p1 = len(p1_effective)
    num_actions_p2 = len(p2_effective)
    p1_to_outcome = build_action_to_outcome_map(p1_effective)
    p2_to_outcome = build_action_to_outcome_map(p2_effective)

    expanded = np.zeros((2, num_actions_p1, num_actions_p2), dtype=payout_n1n2.dtype)
    for a1 in range(num_actions_p1):
        i = p1_to_outcome[a1]
        for a2 in range(num_actions_p2):
            j = p2_to_outcome[a2]
            expanded[0, a1, a2] = payout_n1n2[0, i, j]
            expanded[1, a1, a2] = payout_n1n2[1, i, j]

    return expanded


def expand_visits(
    visits_n1n2: np.ndarray,
    p1_effective: list[int],
    p2_effective: list[int],
) -> np.ndarray:
    """Expand [n1,n2] visit counts to [5,5] by copying to equivalent action pairs.

    Each action pair (a1, a2) gets the visit count of its effective outcome pair.

    Args:
        visits_n1n2: Reduced visit counts shape [n1, n2].
        p1_effective: Maps each P1 action to its effective action.
        p2_effective: Maps each P2 action to its effective action.

    Returns:
        Expanded visit counts shape [5, 5].
    """
    num_actions_p1 = len(p1_effective)
    num_actions_p2 = len(p2_effective)
    p1_to_outcome = build_action_to_outcome_map(p1_effective)
    p2_to_outcome = build_action_to_outcome_map(p2_effective)

    expanded = np.zeros((num_actions_p1, num_actions_p2), dtype=visits_n1n2.dtype)
    for a1 in range(num_actions_p1):
        i = p1_to_outcome[a1]
        for a2 in range(num_actions_p2):
            j = p2_to_outcome[a2]
            expanded[a1, a2] = visits_n1n2[i, j]

    return expanded
