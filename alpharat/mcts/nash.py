"""Nash equilibrium computation for simultaneous-move games.

This module provides utilities for computing Nash equilibria from payout matrices,
which is essential for action selection in MCTS for simultaneous-move games.

Supports action equivalence: when some actions are equivalent (due to walls, edges,
or mud), the computation reduces to effective actions only. This ensures:
- Unique equilibrium (no arbitrary probability splits between equivalent actions)
- Blocked actions get probability 0
- Effective action gets the full probability mass for its equivalence class
"""

from __future__ import annotations

import nashpy as nash  # type: ignore[import-untyped]
import numpy as np

from alpharat.mcts.equivalence import reduce_and_expand_nash


def compute_nash_equilibrium(
    payout_matrix: np.ndarray,
    p1_effective: list[int] | None = None,
    p2_effective: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium mixed strategies for a zero-sum game.

    Given a payout matrix from Player 1's perspective (higher is better for P1),
    computes the Nash equilibrium mixed strategies for both players.

    For zero-sum games, P2's payoff is the negative of P1's payoff.

    Args:
        payout_matrix: Payout matrix from P1 perspective, shape [p1_actions, p2_actions]
                      Entry [i,j] is P1's payoff when P1 plays action i and P2 plays j
        p1_effective: Optional effective action mapping for P1's actions. If provided,
                     computation reduces to effective actions only.
        p2_effective: Optional effective action mapping for P2's actions.

    Returns:
        Tuple of (p1_strategy, p2_strategy) where each is a probability distribution
        over actions (sums to 1.0). If effective mappings provided, non-effective
        actions will have probability 0.

    Note:
        - For games with multiple Nash equilibria, returns one (implementation dependent)
        - Uses support enumeration from nashpy library
        - Strategies are returned as numpy arrays of shape [num_actions]
        - With equivalence, computation is on reduced matrix for efficiency/uniqueness
    """
    # If effective mappings provided, use equivalence reduction
    if p1_effective is not None and p2_effective is not None:
        return reduce_and_expand_nash(_compute_nash_raw, payout_matrix, p1_effective, p2_effective)

    # No equivalence handling - compute on full matrix
    return _compute_nash_raw(payout_matrix)


def _compute_nash_raw(
    payout_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium on raw matrix without equivalence handling.

    Internal function used by compute_nash_equilibrium.
    """
    # Create zero-sum game
    # P1 wants to maximize payout_matrix
    # P2 wants to minimize payout_matrix (maximize -payout_matrix)
    p1_payoff = payout_matrix
    p2_payoff = -payout_matrix

    # Create nashpy game object
    game = nash.Game(p1_payoff, p2_payoff)

    # Compute Nash equilibrium using support enumeration
    # This returns a generator of equilibria; we take the first one
    equilibria = game.support_enumeration()

    try:
        p1_strategy, p2_strategy = next(equilibria)
    except StopIteration:
        # No equilibrium found (shouldn't happen for valid zero-sum games)
        # Fall back to uniform random
        num_p1_actions, num_p2_actions = payout_matrix.shape
        p1_strategy = np.ones(num_p1_actions) / num_p1_actions
        p2_strategy = np.ones(num_p2_actions) / num_p2_actions

    return p1_strategy, p2_strategy


def compute_nash_value(
    payout_matrix: np.ndarray,
    p1_strategy: np.ndarray,
    p2_strategy: np.ndarray,
) -> float:
    """Compute expected value of a strategy profile.

    Args:
        payout_matrix: Payout matrix from P1 perspective [p1_actions, p2_actions]
        p1_strategy: P1's mixed strategy (probability distribution)
        p2_strategy: P2's mixed strategy (probability distribution)

    Returns:
        Expected payout for P1 when both players play their strategies
    """
    # Expected value = p1^T * M * p2
    return float(p1_strategy @ payout_matrix @ p2_strategy)


def select_action_from_strategy(strategy: np.ndarray, temperature: float = 1.0) -> int:
    """Sample an action from a mixed strategy.

    Args:
        strategy: Probability distribution over actions
        temperature: Temperature for sampling (1.0 = sample from strategy,
                    0.0 = deterministic argmax, >1.0 = more random)

    Returns:
        Selected action index
    """
    if temperature == 0.0:
        # Deterministic: select action with highest probability
        return int(np.argmax(strategy))

    if temperature == 1.0:
        # Sample directly from strategy
        return int(np.random.choice(len(strategy), p=strategy))

    # Apply temperature to strategy
    log_probs = np.log(strategy + 1e-10)  # Add small epsilon to avoid log(0)
    tempered_logits = log_probs / temperature
    # Normalize to get valid probability distribution
    tempered_probs = np.exp(tempered_logits)
    tempered_probs /= tempered_probs.sum()

    return int(np.random.choice(len(tempered_probs), p=tempered_probs))
