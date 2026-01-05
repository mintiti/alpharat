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
    """Compute Nash equilibrium mixed strategies for a bimatrix game.

    Given separate payout matrices for each player, computes the Nash equilibrium
    mixed strategies for both players.

    Args:
        payout_matrix: Shape [2, p1_actions, p2_actions] where [0] is P1's payoffs
                      and [1] is P2's payoffs. Entry [0,i,j] is P1's payoff when
                      P1 plays action i and P2 plays j.
        p1_effective: Optional effective action mapping for P1's actions. If provided,
                     computation reduces to effective actions only.
        p2_effective: Optional effective action mapping for P2's actions.

    Returns:
        Tuple of (p1_strategy, p2_strategy) where each is a probability distribution
        over actions (sums to 1.0). If effective mappings provided, non-effective
        actions will have probability 0.

    Note:
        - For games with multiple Nash equilibria, returns a random one
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
    """Compute Nash equilibrium on raw bimatrix game without equivalence handling.

    Internal function used by compute_nash_equilibrium.

    Args:
        payout_matrix: Shape [2, p1_actions, p2_actions] where [0] is P1's payoffs
                      and [1] is P2's payoffs.
    """
    p1_payoffs = payout_matrix[0]
    p2_payoffs = payout_matrix[1]
    num_p1, num_p2 = p1_payoffs.shape

    # Optimization: if both matrices are constant, any strategy is Nash
    # Skip computing equilibria - return uniform directly
    p1_constant = np.allclose(p1_payoffs, p1_payoffs.flat[0])
    p2_constant = np.allclose(p2_payoffs, p2_payoffs.flat[0])
    if p1_constant and p2_constant:
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Create bimatrix game
    # P1 wants to maximize p1_payoffs
    # P2 wants to maximize p2_payoffs
    game = nash.Game(p1_payoffs, p2_payoffs)

    # Compute Nash equilibrium using support enumeration
    # Catch warnings about degenerate games for observability
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        equilibria = list(game.support_enumeration())

        if caught:
            logger.debug(
                "Degenerate Nash equilibrium: %d equilibria found for matrices:\nP1:\n%s\nP2:\n%s",
                len(equilibria),
                p1_payoffs,
                p2_payoffs,
            )

    if len(equilibria) == 0:
        # No equilibrium found (shouldn't happen for valid games)
        logger.warning("No Nash equilibrium found, falling back to uniform")
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Return a random equilibrium
    # Note: For general-sum games, centroid of equilibria is NOT guaranteed to be
    # an equilibrium itself. Random selection avoids arbitrary bias while ensuring
    # we always return a valid Nash equilibrium.
    idx = np.random.randint(len(equilibria))
    p1_strat, p2_strat = equilibria[idx]
    return np.asarray(p1_strat), np.asarray(p2_strat)


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
