"""Policy strategies for deriving SearchResult policies from MCTS outputs.

Policy strategies determine how final policies are derived from MCTS search.
They apply to both acting (move selection) and learning (NN training).
The strategy is configured at MCTS search time, not at sharding time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol

import numpy as np
from pydantic import Discriminator, Field

from alpharat.config.base import StrictBaseModel
from alpharat.mcts.forced_playouts import compute_pruning_adjustment, prune_visit_counts
from alpharat.mcts.nash import compute_nash_from_reduced
from alpharat.mcts.payout_filter import filter_low_visit_payout

if TYPE_CHECKING:
    from alpharat.mcts.node import MCTSNode


class PolicyStrategy(Protocol):
    """Derives policy from MCTS search outputs.

    Applied inside _make_result() to determine SearchResult policies.
    Each strategy owns its full computation, including forced visit pruning
    if applicable.
    """

    def derive_policies(
        self,
        node: MCTSNode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return final policies (p1, p2) in reduced space."""
        ...


def get_effective_visits(
    node: MCTSNode,
    force_k: float,
    c_puct: float,
) -> np.ndarray:
    """Get effective visits, optionally with forced playout pruning.

    Args:
        node: Node to get visits from.
        force_k: Forced playout coefficient. 0 = no pruning.
        c_puct: PUCT exploration constant.

    Returns:
        Visit counts [n1, n2] in reduced space.
    """
    if force_k <= 0:
        return node._visits.copy()

    # Compute pruning adjustments
    q1, q2 = node.compute_marginal_q_reduced()
    n1, n2 = node.compute_marginal_visits_reduced()
    n_total = node.total_visits

    delta_p1 = compute_pruning_adjustment(q1, node.prior_p1_reduced, n1, n_total, c_puct)
    delta_p2 = compute_pruning_adjustment(q2, node.prior_p2_reduced, n2, n_total, c_puct)

    return prune_visit_counts(
        node._visits.copy(),
        delta_p1,
        delta_p2,
        node.prior_p1_reduced,
        node.prior_p2_reduced,
    )


class NashPolicyStrategy:
    """Use Nash equilibrium computed from node's payout matrix.

    Reads node, filters payout by visits, computes Nash equilibrium.
    Game-theoretically optimal but can be sharp/overconfident.
    """

    def __init__(self, force_k: float = 0.0, c_puct: float = 1.5) -> None:
        """Initialize Nash policy strategy.

        Args:
            force_k: Forced playout coefficient for visit pruning. 0 = no pruning.
            c_puct: PUCT exploration constant used in pruning calculation.
        """
        self._force_k = force_k
        self._c_puct = c_puct

    def derive_policies(
        self,
        node: MCTSNode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Nash equilibrium from node's payout matrix.

        Returns:
            (p1_policy, p2_policy) in reduced space [n1], [n2].
        """
        visits = get_effective_visits(node, self._force_k, self._c_puct)
        payout = node.get_reduced_payout()
        filtered = filter_low_visit_payout(payout, visits, min_visits=2)

        return compute_nash_from_reduced(
            filtered,
            reduced_prior_p1=node.prior_p1_reduced,
            reduced_prior_p2=node.prior_p2_reduced,
            reduced_visits=visits,
        )


class VisitPolicyStrategy:
    """KataGo-style visit-based policy with temperature.

    Derives policy from marginal visit counts instead of Nash.
    Preserves MCTS uncertainty and avoids sharp distributions.
    Skips Nash computation entirely.
    """

    def __init__(
        self,
        temperature: float,
        temperature_only_below_prob: float,
        prune_threshold: float,
        subtract_visits: float,
        force_k: float = 0.0,
        c_puct: float = 1.5,
    ) -> None:
        """Initialize visit policy strategy.

        Args:
            temperature: Temperature for softening. 1.0 = proportional to visits.
            temperature_only_below_prob: Only apply temperature to moves below this prob.
            prune_threshold: Zero out moves with fewer than this many visits.
            subtract_visits: Subtract this constant from all visits.
            force_k: Forced playout coefficient for visit pruning. 0 = no pruning.
            c_puct: PUCT exploration constant used in pruning calculation.
        """
        self.temperature = temperature
        self.temperature_only_below_prob = temperature_only_below_prob
        self.prune_threshold = prune_threshold
        self.subtract_visits = subtract_visits
        self._force_k = force_k
        self._c_puct = c_puct

    def derive_policies(
        self,
        node: MCTSNode,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive policy from marginal visit counts with KataGo temperature.

        Returns:
            (p1_policy, p2_policy) in reduced space [n1], [n2].
        """
        visits = get_effective_visits(node, self._force_k, self._c_puct)

        marginal_p1 = visits.sum(axis=1)  # [n1]
        marginal_p2 = visits.sum(axis=0)  # [n2]

        return (
            apply_katago_temperature(
                marginal_p1,
                self.temperature,
                self.temperature_only_below_prob,
                self.prune_threshold,
                self.subtract_visits,
            ),
            apply_katago_temperature(
                marginal_p2,
                self.temperature,
                self.temperature_only_below_prob,
                self.prune_threshold,
                self.subtract_visits,
            ),
        )


def apply_katago_temperature(
    visits: np.ndarray,
    temperature: float,
    only_below_prob: float,
    prune_threshold: float,
    subtract_visits: float,
) -> np.ndarray:
    """KataGo's piecewise log-linear temperature with pruning/subtraction.

    Based on KataGo's searchhelpers.cpp:chooseIndexWithTemperature.

    Args:
        visits: Raw visit counts (n,).
        temperature: Temperature for softening. 1.0 = proportional to visits.
        only_below_prob: Only apply temperature to moves below this prob threshold.
            1.0 = apply to all moves. 0.1 = only dampen moves with <10% probability.
        prune_threshold: Zero out moves with fewer than this many visits.
        subtract_visits: Subtract this constant from all visits before temperature.

    Returns:
        Normalized policy (n,) as float32.
    """
    visits = visits.astype(float).copy()

    # 1. Prune low-visit moves
    if prune_threshold > 0:
        visits[visits < prune_threshold] = 0

    # 2. Subtract constant (capped at max/64 like KataGo)
    if subtract_visits > 0:
        max_v = visits.max()
        if max_v > 0:
            amount = min(subtract_visits, max_v / 64)
            visits = np.maximum(visits - amount, 0)

    # 3. Handle edge cases
    max_v = visits.max()
    sum_v = visits.sum()

    if max_v <= 0:
        # No visits — uniform fallback
        return np.ones(len(visits), dtype=np.float32) / len(visits)

    # 4. Near-zero temp = argmax
    if temperature <= 1e-4:
        result = np.zeros(len(visits), dtype=np.float32)
        result[np.argmax(visits)] = 1.0
        return result

    # 5. KataGo's piecewise log-linear transform
    log_max = np.log(max_v)
    log_sum = np.log(sum_v)
    log_threshold = np.log(max(1e-50, only_below_prob))
    threshold = min(0.0, log_threshold + log_sum - log_max)

    result = np.zeros(len(visits), dtype=np.float64)
    for i, v in enumerate(visits):
        if v <= 0:
            result[i] = 0
        else:
            log_v = np.log(v) - log_max
            if log_v > threshold:
                # Top actions: no dampening
                new_log_v = log_v
            else:
                # Lower actions: apply temperature
                new_log_v = (log_v - threshold) / temperature + threshold
            result[i] = np.exp(new_log_v)

    total = result.sum()
    if total > 0:
        result /= total

    return result.astype(np.float32)


# --- Config Classes with Validation ---


class NashPolicyConfig(StrictBaseModel):
    """Config for Nash equilibrium policy (default behavior).

    Uses the Nash equilibrium computed from the payout matrix.
    Game-theoretically optimal but may be overconfident when
    payout estimates are noisy.
    """

    strategy: Literal["nash"] = "nash"

    def build(self, force_k: float = 0.0, c_puct: float = 1.5) -> PolicyStrategy:
        """Build the Nash policy strategy.

        Args:
            force_k: Forced playout coefficient for visit pruning.
            c_puct: PUCT exploration constant.
        """
        return NashPolicyStrategy(force_k=force_k, c_puct=c_puct)


class VisitPolicyConfig(StrictBaseModel):
    """Config for visit-based policy with KataGo-style temperature.

    Derives policy from marginal visit counts instead of Nash equilibrium.
    Preserves MCTS uncertainty and avoids the sharp distributions that
    Nash produces from noisy payout estimates.
    """

    strategy: Literal["visits"] = "visits"

    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Temperature for visit-based policy. "
            "1.0 = proportional to visits. "
            "<1.0 = sharper (approaches argmax). "
            ">1.0 = flatter (approaches uniform)."
        ),
    )

    temperature_only_below_prob: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Only apply temperature to moves below this probability threshold. "
            "1.0 = apply to all moves (simple mode). "
            "0.1 = only dampen moves with <10% probability, preserving top moves."
        ),
    )

    prune_threshold: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Zero out moves with fewer than this many visits before temperature. "
            "0.0 = no pruning. 1.0 = ignore moves with <1 visit."
        ),
    )

    subtract_visits: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Subtract this constant from all visit counts before temperature. "
            "Reduces temptation to play rarely-visited moves. 0.0 = no subtraction."
        ),
    )

    def build(self, force_k: float = 0.0, c_puct: float = 1.5) -> PolicyStrategy:
        """Build the visit policy strategy.

        Args:
            force_k: Forced playout coefficient for visit pruning.
            c_puct: PUCT exploration constant.
        """
        return VisitPolicyStrategy(
            temperature=self.temperature,
            temperature_only_below_prob=self.temperature_only_below_prob,
            prune_threshold=self.prune_threshold,
            subtract_visits=self.subtract_visits,
            force_k=force_k,
            c_puct=c_puct,
        )


# Discriminated union
PolicyConfig = Annotated[
    NashPolicyConfig | VisitPolicyConfig,
    Discriminator("strategy"),
]
