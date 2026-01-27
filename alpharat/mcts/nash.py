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
    strategies independently. For constant-sum games this is guaranteed to be
    a valid equilibrium due to interchangeability. PyRat is approximately
    constant-sum (exact under infinite horizon, approximate under turn limits).

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
    """Remove entire actions with insufficient marginal visits from Nash computation.

    This is the second of two filtering stages in the pipeline:
    1. ``filter_low_visit_payout`` (in payout_filter.py) zeros individual cells
       where a specific action *pair* has < 2 visits — pessimistic per-cell cleanup.
    2. This function removes entire rows/columns where a player's *marginal*
       visits (summed across opponent actions) fall below threshold — structural
       reduction of the matrix dimensions before Nash.

    Args:
        matrix: Reduced payout matrix [2, m, n].
        visits: Reduced visit matrix [m, n].
        p1_actions: Action labels for P1's rows.
        p2_actions: Action labels for P2's columns.
        min_visits: Minimum marginal visits required per player action.

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
        prior_p1: Optional prior policy for P1 (used as fallback for constant
                  or degenerate matrices).
        prior_p2: Optional prior policy for P2 (used as fallback).
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
        # Also filter priors and marginal visits to match the visit-filtered actions
        eff_to_idx_p1 = {a: i for i, a in enumerate(p1_eff_actions)}
        eff_to_idx_p2 = {a: i for i, a in enumerate(p2_eff_actions)}

        if reduced_p1_prior is not None:
            nash_p1_prior = np.array([reduced_p1_prior[eff_to_idx_p1[a]] for a in p1_nash_actions])
        else:
            nash_p1_prior = None
        if reduced_p2_prior is not None:
            nash_p2_prior = np.array([reduced_p2_prior[eff_to_idx_p2[a]] for a in p2_nash_actions])
        else:
            nash_p2_prior = None

        # Compute marginal visits filtered to nash actions
        p1_marginals = reduced_visits.sum(axis=1)  # [m]
        p2_marginals = reduced_visits.sum(axis=0)  # [n]
        nash_p1_marginals = np.array([p1_marginals[eff_to_idx_p1[a]] for a in p1_nash_actions])
        nash_p2_marginals = np.array([p2_marginals[eff_to_idx_p2[a]] for a in p2_nash_actions])
    else:
        nash_matrix = reduced_matrix
        p1_nash_actions, p2_nash_actions = p1_eff_actions, p2_eff_actions
        nash_p1_prior, nash_p2_prior = reduced_p1_prior, reduced_p2_prior
        nash_p1_marginals, nash_p2_marginals = None, None

    # Step 4: Compute Nash
    p1_strat, p2_strat = _compute_nash_raw(
        nash_matrix,
        prior_p1=nash_p1_prior,
        prior_p2=nash_p2_prior,
        marginal_visits_p1=nash_p1_marginals,
        marginal_visits_p2=nash_p2_marginals,
    )

    # Step 5: Expand to full action space
    full_p1 = _expand_strategy(p1_strat, p1_nash_actions, num_actions)
    full_p2 = _expand_strategy(p2_strat, p2_nash_actions, num_actions)

    return full_p1, full_p2


def _resolve_fallback(
    num_actions: int,
    marginal_visits: np.ndarray | None,
    prior: np.ndarray | None,
) -> np.ndarray:
    """Pick best available fallback: marginal visits → prior → uniform."""
    if marginal_visits is not None:
        total = marginal_visits.sum()
        if total > 0:
            result: np.ndarray = marginal_visits / total
            return result
    if prior is not None:
        normalized: np.ndarray = prior / prior.sum()
        return normalized
    return np.ones(num_actions) / num_actions


def _compute_nash_raw(
    payout_matrix: np.ndarray,
    prior_p1: np.ndarray | None = None,
    prior_p2: np.ndarray | None = None,
    marginal_visits_p1: np.ndarray | None = None,
    marginal_visits_p2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium on raw bimatrix game without equivalence handling.

    Internal function used by compute_nash_equilibrium and compute_nash_from_reduced.

    Args:
        payout_matrix: Shape [2, p1_actions, p2_actions] where [0] is P1's payoffs
                      and [1] is P2's payoffs. May be reduced if using equivalence.
        prior_p1: Optional prior policy for P1 (used as fallback for constant or
                  degenerate matrices).
        prior_p2: Optional prior policy for P2 (used as fallback).
        marginal_visits_p1: Optional marginal visit counts for P1's actions.
                           Preferred over prior when available (more informed fallback).
        marginal_visits_p2: Optional marginal visit counts for P2's actions.
    """
    p1_payoffs = payout_matrix[0]
    p2_payoffs = payout_matrix[1]
    num_p1, num_p2 = p1_payoffs.shape

    # Optimization: if both matrices are constant, any strategy is Nash
    # Fallback chain: marginal visits → prior → uniform
    p1_constant = np.allclose(p1_payoffs, p1_payoffs.flat[0])
    p2_constant = np.allclose(p2_payoffs, p2_payoffs.flat[0])
    if p1_constant and p2_constant:
        return (
            _resolve_fallback(num_p1, marginal_visits_p1, prior_p1),
            _resolve_fallback(num_p2, marginal_visits_p2, prior_p2),
        )

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
        # Fallback chain: marginal visits → prior → uniform
        p1_fallback = _resolve_fallback(num_p1, marginal_visits_p1, prior_p1)
        p2_fallback = _resolve_fallback(num_p2, marginal_visits_p2, prior_p2)

        # Determine fallback type for logging
        def _fallback_label(marginals: np.ndarray | None, prior: np.ndarray | None) -> str:
            if marginals is not None and marginals.sum() > 0:
                return "marginal_visits"
            if prior is not None:
                return "prior"
            return "uniform"

        p1_label = _fallback_label(marginal_visits_p1, prior_p1)
        p2_label = _fallback_label(marginal_visits_p2, prior_p2)
        logger.warning(
            "No Nash equilibrium found, falling back to P1=%s, P2=%s.\n"
            "P1 payoffs:\n%s\nP2 payoffs:\n%s\n"
            "P1 prior: %s\nP2 prior: %s",
            p1_label,
            p2_label,
            p1_payoffs,
            p2_payoffs,
            prior_p1,
            prior_p2,
        )
        return p1_fallback, p2_fallback

    # Aggregate equilibria via centroid
    # For constant-sum games, the centroid is a valid equilibrium due to
    # interchangeability. PyRat is approximately constant-sum (exact under
    # infinite horizon, approximate under turn limits).
    return aggregate_equilibria(equilibria)


def compute_nash_from_reduced(
    reduced_payout: np.ndarray,
    reduced_prior_p1: np.ndarray | None = None,
    reduced_prior_p2: np.ndarray | None = None,
    reduced_visits: np.ndarray | None = None,
    min_visits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium from already-reduced payout matrix.

    Like compute_nash_equilibrium but skips the reduce_matrix step — the
    input is already in outcome-indexed space [2, n1, n2]. Returns strategies
    in the same reduced space [n1], [n2].

    Args:
        reduced_payout: Payout matrix [2, n1, n2] already in reduced space.
        reduced_prior_p1: Prior policy [n1] in reduced space.
        reduced_prior_p2: Prior policy [n2] in reduced space.
        reduced_visits: Visit counts [n1, n2] in reduced space.
        min_visits: Minimum visits per action for visit filtering.

    Returns:
        (p1_strategy, p2_strategy) — each [n1] and [n2] in reduced space.
    """
    n1 = reduced_payout.shape[1]
    n2 = reduced_payout.shape[2]

    # Outcome indices as labels for _filter_by_visits / _expand_strategy
    p1_indices = list(range(n1))
    p2_indices = list(range(n2))

    # Filter by visits if provided
    if reduced_visits is not None:
        nash_matrix, p1_kept, p2_kept = _filter_by_visits(
            reduced_payout, reduced_visits, p1_indices, p2_indices, min_visits
        )
        # Filter priors to match
        if reduced_prior_p1 is not None:
            nash_p1_prior = np.array([reduced_prior_p1[i] for i in p1_kept])
        else:
            nash_p1_prior = None
        if reduced_prior_p2 is not None:
            nash_p2_prior = np.array([reduced_prior_p2[i] for i in p2_kept])
        else:
            nash_p2_prior = None

        # Compute marginal visits filtered to kept actions
        p1_marginals = reduced_visits.sum(axis=1)  # [n1]
        p2_marginals = reduced_visits.sum(axis=0)  # [n2]
        nash_p1_marginals = np.array([p1_marginals[i] for i in p1_kept])
        nash_p2_marginals = np.array([p2_marginals[i] for i in p2_kept])
    else:
        nash_matrix = reduced_payout
        p1_kept, p2_kept = p1_indices, p2_indices
        nash_p1_prior = reduced_prior_p1
        nash_p2_prior = reduced_prior_p2
        nash_p1_marginals, nash_p2_marginals = None, None

    # Compute Nash
    p1_strat, p2_strat = _compute_nash_raw(
        nash_matrix,
        prior_p1=nash_p1_prior,
        prior_p2=nash_p2_prior,
        marginal_visits_p1=nash_p1_marginals,
        marginal_visits_p2=nash_p2_marginals,
    )

    # Expand from filtered subset back to full reduced space [n1], [n2]
    full_p1 = _expand_strategy(p1_strat, p1_kept, n1)
    full_p2 = _expand_strategy(p2_strat, p2_kept, n2)

    return full_p1, full_p2


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
