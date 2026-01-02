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

import logging
import warnings

import nashpy as nash  # type: ignore[import-untyped]
import numpy as np

from alpharat.mcts.equivalence import reduce_and_expand_nash

logger = logging.getLogger(__name__)


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
    num_p1, num_p2 = payout_matrix.shape

    # Degenerate matrix check: if all entries are nearly equal (constant matrix),
    # any strategy is a Nash equilibrium. Return uniform to avoid arbitrary
    # tie-breaking that systematically biases P1.
    # This catches: all zeros, all ones, any constant value.
    if np.allclose(payout_matrix, payout_matrix.flat[0]):
        logger.debug("Constant payout matrix detected, returning uniform strategies")
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Sparse matrix check: if any row or column is all zeros, the game is under-explored.
    # This commonly occurs with few MCTS simulations and leads to multiple equilibria
    # with arbitrary first-equilibrium selection that biases P1.
    has_zero_row = np.any(np.all(np.isclose(payout_matrix, 0), axis=1))
    has_zero_col = np.any(np.all(np.isclose(payout_matrix, 0), axis=0))
    if has_zero_row or has_zero_col:
        logger.debug("Sparse payout matrix (zero row/col) detected, returning uniform")
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Create zero-sum game
    # P1 wants to maximize payout_matrix
    # P2 wants to minimize payout_matrix (maximize -payout_matrix)
    game = nash.Game(payout_matrix, -payout_matrix)

    # Compute Nash equilibrium using support enumeration
    # Catch warnings about degenerate games for observability
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        equilibria = list(game.support_enumeration())

        if caught:
            logger.debug(
                "Degenerate Nash equilibrium: %d equilibria found for matrix:\n%s",
                len(equilibria),
                payout_matrix,
            )

    if len(equilibria) == 0:
        # No equilibrium found (shouldn't happen for valid zero-sum games)
        logger.warning("No Nash equilibrium found, falling back to uniform")
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Take first equilibrium. For unique games this is the only one.
    # For degenerate games (multiple equilibria), this may be arbitrary,
    # but the constant-matrix check above catches the worst cases.
    p1_strategy, p2_strategy = equilibria[0]
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
