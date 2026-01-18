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

from alpharat.mcts.equivalence import reduce_matrix
from alpharat.mcts.reduction import reduce_prior

logger = logging.getLogger(__name__)


def aggregate_equilibria(
    equilibria: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate multiple Nash equilibria into a single policy pair.

    Computes the centroid (arithmetic mean) of each player's equilibrium
    strategies independently. For constant-sum games like PyRat, this is
    guaranteed to be a valid equilibrium due to interchangeability.

    Args:
        equilibria: List of (p1_strategy, p2_strategy) Nash equilibria.
                    Must be non-empty.

    Returns:
        (p1_policy, p2_policy) — the aggregated policy for each player

    Raises:
        ValueError: If equilibria is empty
    """
    if not equilibria:
        raise ValueError("Cannot aggregate empty equilibria list")

    p1_strategies = np.array([eq[0] for eq in equilibria])
    p2_strategies = np.array([eq[1] for eq in equilibria])

    return p1_strategies.mean(axis=0), p2_strategies.mean(axis=0)


def _reduce_visits(
    visits: np.ndarray,
    p1_actions: list[int],
    p2_actions: list[int],
) -> np.ndarray:
    """Reduce visit matrix to effective actions only.

    Args:
        visits: Full visit matrix [5, 5].
        p1_actions: Effective actions for P1.
        p2_actions: Effective actions for P2.

    Returns:
        Reduced visit matrix [len(p1_actions), len(p2_actions)].
    """
    return visits[np.ix_(p1_actions, p2_actions)]


def _filter_by_visits(
    matrix: np.ndarray,
    visits: np.ndarray,
    p1_actions: list[int],
    p2_actions: list[int],
    min_visits: int = 5,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Filter actions with insufficient exploration.

    Actions with total visits below threshold are removed from Nash computation.
    This prevents computing equilibrium on unreliable payoff estimates.

    Args:
        matrix: Reduced payout matrix [2, m, n].
        visits: Reduced visit matrix [m, n].
        p1_actions: Effective actions for P1.
        p2_actions: Effective actions for P2.
        min_visits: Minimum visits required per player action.

    Returns:
        Tuple of:
        - filtered_matrix: [2, m', n'] with low-visit actions removed
        - p1_kept: subset of p1_actions that survived filtering
        - p2_kept: subset of p2_actions that survived filtering
    """
    p1_visits = visits.sum(axis=1)  # [m]
    p2_visits = visits.sum(axis=0)  # [n]

    p1_mask = p1_visits >= min_visits
    p2_mask = p2_visits >= min_visits

    # Edge case: if all actions filtered, keep the most-visited
    if not p1_mask.any():
        p1_mask[p1_visits.argmax()] = True
    if not p2_mask.any():
        p2_mask[p2_visits.argmax()] = True

    p1_kept = [a for a, keep in zip(p1_actions, p1_mask, strict=True) if keep]
    p2_kept = [a for a, keep in zip(p2_actions, p2_mask, strict=True) if keep]

    filtered = matrix[:, p1_mask, :][:, :, p2_mask]
    return filtered, p1_kept, p2_kept


def _expand_strategy(
    strategy: np.ndarray,
    from_actions: list[int],
    to_size: int,
) -> np.ndarray:
    """Expand strategy from subset of actions to full action space.

    Args:
        strategy: Strategy over subset of actions.
        from_actions: Action indices in the strategy.
        to_size: Size of full action space (typically 5).

    Returns:
        Full strategy with 0 for actions not in from_actions.
    """
    result = np.zeros(to_size)
    for i, action in enumerate(from_actions):
        result[action] = strategy[i]
    return result


def compute_nash_equilibrium(
    payout_matrix: np.ndarray,
    p1_effective: list[int] | None = None,
    p2_effective: list[int] | None = None,
    prior_p1: np.ndarray | None = None,
    prior_p2: np.ndarray | None = None,
    action_visits: np.ndarray | None = None,
    min_visits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium mixed strategies for a bimatrix game.

    Given separate payout matrices for each player, computes the Nash equilibrium
    mixed strategies for both players.

    The computation flow:
    1. Reduce to effective actions (blocked moves filtered out)
    2. Filter by visits (actions with < min_visits removed for reliability)
    3. Compute Nash on filtered matrix
    4. Expand back to full 5-action space

    Args:
        payout_matrix: Shape [2, p1_actions, p2_actions] where [0] is P1's payoffs
                      and [1] is P2's payoffs. Entry [0,i,j] is P1's payoff when
                      P1 plays action i and P2 plays j.
        p1_effective: Optional effective action mapping for P1's actions. If provided,
                     computation reduces to effective actions only.
        p2_effective: Optional effective action mapping for P2's actions.
        prior_p1: Optional NN prior policy for P1 (for logging on fallback).
        prior_p2: Optional NN prior policy for P2 (for logging on fallback).
        action_visits: Optional visit counts per action pair. If provided, actions
                      with insufficient visits are filtered before Nash computation.
        min_visits: Minimum visits required per player action (default 5).
                   Actions below this threshold are excluded from Nash computation.

    Returns:
        Tuple of (p1_strategy, p2_strategy) where each is a probability distribution
        over actions (sums to 1.0). Blocked and under-explored actions have probability 0.

    Note:
        - For games with multiple Nash equilibria, returns the centroid (mean)
          of each player's strategies. This is valid for constant-sum games.
        - Uses support enumeration from nashpy library
        - Strategies are returned as numpy arrays of shape [num_actions]
        - With equivalence, computation is on reduced matrix for efficiency/uniqueness
    """
    num_actions = payout_matrix.shape[1]

    # No effective mappings → compute on full matrix (no filtering)
    if p1_effective is None or p2_effective is None:
        return _compute_nash_raw(
            payout_matrix,
            prior_p1=prior_p1,
            prior_p2=prior_p2,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
            action_visits=action_visits,
            original_payout_matrix=payout_matrix,
        )

    # Step 1: Reduce by effective actions
    reduced_matrix, p1_eff_actions, p2_eff_actions = reduce_matrix(
        payout_matrix, p1_effective, p2_effective
    )

    # Step 2: Reduce priors to match effective actions
    reduced_p1_prior = reduce_prior(prior_p1, p1_effective) if prior_p1 is not None else None
    reduced_p2_prior = reduce_prior(prior_p2, p2_effective) if prior_p2 is not None else None

    # Step 3: Filter by visits (if provided)
    if action_visits is not None:
        reduced_visits = _reduce_visits(action_visits, p1_eff_actions, p2_eff_actions)
        nash_matrix, p1_nash_actions, p2_nash_actions = _filter_by_visits(
            reduced_matrix, reduced_visits, p1_eff_actions, p2_eff_actions, min_visits
        )
        # Also filter priors to match the visit-filtered actions
        if reduced_p1_prior is not None:
            # Map effective actions to indices in reduced prior
            eff_to_idx = {a: i for i, a in enumerate(p1_eff_actions)}
            nash_p1_prior = np.array([reduced_p1_prior[eff_to_idx[a]] for a in p1_nash_actions])
        else:
            nash_p1_prior = None
        if reduced_p2_prior is not None:
            eff_to_idx = {a: i for i, a in enumerate(p2_eff_actions)}
            nash_p2_prior = np.array([reduced_p2_prior[eff_to_idx[a]] for a in p2_nash_actions])
        else:
            nash_p2_prior = None
    else:
        nash_matrix = reduced_matrix
        p1_nash_actions, p2_nash_actions = p1_eff_actions, p2_eff_actions
        nash_p1_prior, nash_p2_prior = reduced_p1_prior, reduced_p2_prior

    # Step 4: Compute Nash
    p1_strat, p2_strat = _compute_nash_raw(
        nash_matrix,
        prior_p1=nash_p1_prior,
        prior_p2=nash_p2_prior,
        p1_effective=p1_effective,
        p2_effective=p2_effective,
        action_visits=action_visits,
        original_payout_matrix=payout_matrix,
    )

    # Step 5: Expand to full action space
    full_p1 = _expand_strategy(p1_strat, p1_nash_actions, num_actions)
    full_p2 = _expand_strategy(p2_strat, p2_nash_actions, num_actions)

    return full_p1, full_p2


def _compute_nash_raw(
    payout_matrix: np.ndarray,
    prior_p1: np.ndarray | None = None,
    prior_p2: np.ndarray | None = None,
    p1_effective: list[int] | None = None,
    p2_effective: list[int] | None = None,
    action_visits: np.ndarray | None = None,
    original_payout_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium on raw bimatrix game without equivalence handling.

    Internal function used by compute_nash_equilibrium.

    Args:
        payout_matrix: Shape [2, p1_actions, p2_actions] where [0] is P1's payoffs
                      and [1] is P2's payoffs. May be reduced if using equivalence.
        prior_p1: Optional NN prior policy for P1 (for logging on fallback).
        prior_p2: Optional NN prior policy for P2 (for logging on fallback).
        p1_effective: Optional effective action mapping for P1 (for logging on fallback).
        p2_effective: Optional effective action mapping for P2 (for logging on fallback).
        action_visits: Optional visit counts per action pair (for logging on fallback).
        original_payout_matrix: Optional original (non-reduced) payout matrix for logging.
    """
    p1_payoffs = payout_matrix[0]
    p2_payoffs = payout_matrix[1]
    num_p1, num_p2 = p1_payoffs.shape

    # Optimization: if both matrices are constant, any strategy is Nash
    # Use NN prior if available (it's what we started with, MCTS couldn't find better)
    p1_constant = np.allclose(p1_payoffs, p1_payoffs.flat[0])
    p2_constant = np.allclose(p2_payoffs, p2_payoffs.flat[0])
    if p1_constant and p2_constant:
        if prior_p1 is not None and prior_p2 is not None:
            # Normalize priors (should already sum to 1, but defensive)
            return prior_p1 / prior_p1.sum(), prior_p2 / prior_p2.sum()
        # No priors available (e.g., no NN) - fall back to uniform
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
        # Use NN prior if available, otherwise uniform
        log_matrix = original_payout_matrix if original_payout_matrix is not None else payout_matrix
        fallback_type = "prior" if (prior_p1 is not None and prior_p2 is not None) else "uniform"
        logger.warning(
            "No Nash equilibrium found, falling back to %s.\n"
            "P1 payoffs:\n%s\nP2 payoffs:\n%s\n"
            "P1 prior: %s\nP2 prior: %s\n"
            "P1 effective: %s\nP2 effective: %s\n"
            "Action visits:\n%s",
            fallback_type,
            log_matrix[0],
            log_matrix[1],
            prior_p1,
            prior_p2,
            p1_effective,
            p2_effective,
            action_visits,
        )
        if prior_p1 is not None and prior_p2 is not None:
            return prior_p1 / prior_p1.sum(), prior_p2 / prior_p2.sum()
        return np.ones(num_p1) / num_p1, np.ones(num_p2) / num_p2

    # Aggregate equilibria via centroid
    # For constant-sum games like PyRat, the centroid is guaranteed to be a valid
    # equilibrium due to interchangeability: any convex combination of equilibrium
    # strategies is also an equilibrium.
    return aggregate_equilibria(equilibria)


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
