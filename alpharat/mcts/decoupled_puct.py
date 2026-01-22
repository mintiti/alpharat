"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with Q-values
marginalized over the opponent's prior policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from alpharat.config.base import StrictBaseModel
from alpharat.mcts.nash import compute_nash_equilibrium
from alpharat.mcts.numba_ops import compute_puct_scores, select_max_with_tiebreak
from alpharat.mcts.payout_filter import filter_low_visit_payout
from alpharat.mcts.policy_strategy import NashPolicyConfig, PolicyConfig, PolicyStrategy
from alpharat.mcts.selection import compute_pruning_adjustment, prune_visit_counts

if TYPE_CHECKING:
    import numpy as np

    from alpharat.mcts.node import MCTSNode
    from alpharat.mcts.tree import MCTSTree


@dataclass
class SearchResult:
    """Result of MCTS search containing policy and value information.

    Attributes:
        payout_matrix: Root's payout matrix after search (filtered for low visits).
        policy_p1: Acting policy for player 1 (agent samples from this).
        policy_p2: Acting policy for player 2 (agent samples from this).
        learning_policy_p1: Learning target for player 1 (NN trained on this).
        learning_policy_p2: Learning target for player 2 (NN trained on this).
        action_visits: Visit counts per action pair [5, 5].
    """

    payout_matrix: np.ndarray
    policy_p1: np.ndarray
    policy_p2: np.ndarray
    learning_policy_p1: np.ndarray
    learning_policy_p2: np.ndarray
    action_visits: np.ndarray


class DecoupledPUCTConfig(StrictBaseModel):
    """Configuration for decoupled PUCT search."""

    simulations: int = 100
    gamma: float = 1.0
    c_puct: float = 1.5
    force_k: float = 2.0
    policy: PolicyConfig = Field(
        default_factory=NashPolicyConfig,
        description=(
            "Strategy for deriving policies from search results. "
            "Configures BOTH acting and learning policies (coupled for simplicity). "
            "'nash' = Nash equilibrium (default). "
            "'visits' = marginal visit counts with temperature."
        ),
    )

    def build(self, tree: MCTSTree) -> DecoupledPUCTSearch:
        """Construct a search instance with these settings."""
        return DecoupledPUCTSearch(tree, self)


class DecoupledPUCTSearch:
    """Decoupled PUCT search for simultaneous-move MCTS.

    Each player independently selects actions via PUCT formula with Q-values
    marginalized over the opponent's prior policy.

    Args:
        tree: MCTSTree to search
        config: Search configuration
    """

    def __init__(self, tree: MCTSTree, config: DecoupledPUCTConfig) -> None:
        self.tree = tree
        self._n_sims = config.simulations
        self._c_puct = config.c_puct
        self._force_k = config.force_k

        # Two separate slots, but same instance when config couples them
        strategy: PolicyStrategy = config.policy.build()
        self._acting_strategy = strategy
        self._learning_strategy = strategy

    def search(self) -> SearchResult:
        """Run MCTS search and return the result.

        Returns:
            SearchResult with Nash policies and payout matrix.
        """
        if self._n_sims == 0:
            return self._pure_nn_result()

        for _ in range(self._n_sims):
            self._simulate()

        return self._make_result()

    def _pure_nn_result(self) -> SearchResult:
        """Return NN priors directly without tree search."""
        root = self.tree.root
        return SearchResult(
            payout_matrix=root.payout_matrix.copy(),
            policy_p1=root.prior_policy_p1.copy(),
            policy_p2=root.prior_policy_p2.copy(),
            learning_policy_p1=root.prior_policy_p1.copy(),
            learning_policy_p2=root.prior_policy_p2.copy(),
            action_visits=root.action_visits.copy(),
        )

    def _make_result(self) -> SearchResult:
        """Compute policies from root statistics.

        When forced playouts are enabled (force_k > 0), prunes artificial visits
        before filtering. This gives cleaner training targets by removing visits
        that were forced for exploration rather than naturally selected by PUCT.

        Filters low-visit cells to 0 before Nash computation — pessimistic estimate
        for unexplored action pairs. This ensures Nash and training data use only
        trustworthy values.
        """
        root = self.tree.root
        visits_for_filtering = root.action_visits

        # Prune forced visits if forced playouts were used
        if self._force_k > 0:
            visits_for_filtering = self._prune_forced_visits(root)

        # Filter unreliable cells before Nash and recording
        filtered_payout = filter_low_visit_payout(
            root.payout_matrix, visits_for_filtering, min_visits=2
        )

        nash_p1, nash_p2 = compute_nash_equilibrium(
            filtered_payout,
            root.p1_effective,
            root.p2_effective,
            prior_p1=root.prior_policy_p1,
            prior_p2=root.prior_policy_p2,
            action_visits=visits_for_filtering,
        )

        # Compute marginal visits for policy derivation
        marginal_p1 = visits_for_filtering.sum(axis=1)
        marginal_p2 = visits_for_filtering.sum(axis=0)

        # Acting policies (agent samples from these)
        acting_p1 = self._acting_strategy.derive_policy(marginal_p1, nash_p1)
        acting_p2 = self._acting_strategy.derive_policy(marginal_p2, nash_p2)

        # Learning policies (recorded for NN training)
        learning_p1 = self._learning_strategy.derive_policy(marginal_p1, nash_p1)
        learning_p2 = self._learning_strategy.derive_policy(marginal_p2, nash_p2)

        return SearchResult(
            payout_matrix=filtered_payout,
            policy_p1=acting_p1,
            policy_p2=acting_p2,
            learning_policy_p1=learning_p1,
            learning_policy_p2=learning_p2,
            action_visits=visits_for_filtering.copy()
            if visits_for_filtering is not root.action_visits
            else root.action_visits.copy(),
        )

    def _simulate(self) -> None:
        """Run a single MCTS simulation.

        Selection: Walk tree selecting actions via decoupled PUCT.
        Expansion: Create child node when reaching unexpanded action pair.
        Backup: Propagate discounted values up the path.
        """
        path: list[tuple[MCTSNode, int, int, tuple[float, float]]] = []
        current = self.tree.root
        leaf_child: MCTSNode | None = None

        # Selection phase: walk until we expand or hit terminal
        while True:
            if current.is_terminal:
                break

            a1, a2 = self._select_actions(current)

            # Check if this action pair leads to a new node (expansion)
            # Children are keyed by outcome indices
            outcome_i = current.action_to_outcome(1, a1)
            outcome_j = current.action_to_outcome(2, a2)
            outcome_pair = (outcome_i, outcome_j)
            is_expansion = outcome_pair not in current.children

            # Make move (creates child if needed)
            child, reward = self.tree.make_move_from(current, a1, a2)

            # Record step BEFORE potential expansion
            path.append((current, a1, a2, reward))

            if is_expansion:
                leaf_child = child
                break

            current = child

        # Backup
        if not path:
            return  # Root was terminal

        # Compute leaf value as tuple for both players
        if leaf_child is None or leaf_child.is_terminal:
            g: tuple[float, float] = (0.0, 0.0)
        else:
            # NN's expected value under its own policy for each player
            g = leaf_child.compute_expected_value()

        self.tree.backup(path, g=g)

    def _select_actions(self, node: MCTSNode) -> tuple[int, int]:
        """Select actions for both players using decoupled PUCT.

        Works entirely in reduced (outcome-indexed) space for efficiency.
        Each outcome index corresponds to a unique game outcome, so no
        complex tie-breaking for equivalent actions is needed.

        Args:
            node: Current node to select from.

        Returns:
            Tuple of (action_p1, action_p2) selected via PUCT formula.
        """
        # Work in reduced space: [n1], [n2] arrays
        q1, q2 = node.compute_marginal_q_reduced()
        n1, n2 = node.compute_marginal_visits_reduced()
        n_total = node.total_visits
        is_root = node == self.tree.root

        # Compute PUCT in reduced space (JIT-compiled)
        puct1 = compute_puct_scores(
            q1, node.prior_p1_reduced, n1, n_total, self._c_puct, self._force_k, is_root
        )
        puct2 = compute_puct_scores(
            q2, node.prior_p2_reduced, n2, n_total, self._c_puct, self._force_k, is_root
        )

        # Select outcome indices (JIT-compiled random tie-break)
        idx1 = select_max_with_tiebreak(puct1)
        idx2 = select_max_with_tiebreak(puct2)

        # Map outcome index back to action
        a1 = node.p1_outcomes[idx1]
        a2 = node.p2_outcomes[idx2]

        return a1, a2

    def _prune_forced_visits(self, node: MCTSNode) -> np.ndarray:
        """Prune forced playout visits from the visit counts.

        Computes how many visits were "forced" (exceeded PUCT-justified amount)
        and subtracts them, distributing the adjustment across pairs using
        opponent's prior as weights.

        Args:
            node: Node to prune visits for (typically root).

        Returns:
            Pruned visit counts [5, 5]. Can be fractional.
        """
        # Marginal Q-values (expanded space)
        q1 = node.payout_matrix[0] @ node.prior_policy_p2
        q2 = node.payout_matrix[1].T @ node.prior_policy_p1

        # Marginal visit counts
        n1 = node.action_visits.sum(axis=1)
        n2 = node.action_visits.sum(axis=0)
        n_total = node.total_visits

        # Compute adjustments for each player
        delta_p1 = compute_pruning_adjustment(
            q1, node.prior_policy_p1, n1, n_total, self._c_puct, node.p1_effective
        )
        delta_p2 = compute_pruning_adjustment(
            q2, node.prior_policy_p2, n2, n_total, self._c_puct, node.p2_effective
        )

        # Apply to pair visits
        return prune_visit_counts(
            node.action_visits, delta_p1, delta_p2, node.prior_policy_p1, node.prior_policy_p2
        )
