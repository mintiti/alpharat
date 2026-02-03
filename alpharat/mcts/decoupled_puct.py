"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with Q-values
marginalized over the opponent's prior policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field

from alpharat.config.base import StrictBaseModel
from alpharat.mcts.numba_ops import compute_puct_scores, select_max_with_tiebreak
from alpharat.mcts.payout_filter import filter_low_visit_payout
from alpharat.mcts.policy_strategy import (
    NashPolicyConfig,
    PolicyConfig,
    PolicyStrategy,
    get_effective_visits,
)
from alpharat.mcts.reduction import expand_payout, expand_prior, expand_visits

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

        # Build strategy with search params (force_k, c_puct)
        # Agent samples from acting, recorder saves learning targets.
        # Two slots so they can diverge later without changing the data flow.
        strategy: PolicyStrategy = config.policy.build(force_k=config.force_k, c_puct=config.c_puct)
        self._acting_strategy = strategy
        self._learning_strategy = strategy

    def search(self) -> SearchResult:
        """Run MCTS search and return the result.

        Returns:
            SearchResult with policies and payout matrix.
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

        All decision computation (pruning, filtering, Nash, policy derivation)
        is delegated to the policy strategy. Expansion to 5-action space only
        at the SearchResult boundary for consumers (recorder, agent, NN).
        """
        root = self.tree.root

        # --- Delegate policy computation to strategies ---
        acting_p1, acting_p2 = self._acting_strategy.derive_policies(root)
        learning_p1, learning_p2 = self._learning_strategy.derive_policies(root)

        # --- Compute filtered payout for recording (NN training) ---
        # Use pruned visits (same as policy strategies) for consistency.
        visits_reduced = get_effective_visits(root, self._force_k, self._c_puct)
        payout_reduced = root.get_reduced_payout()  # [2, n1, n2]
        filtered_payout_reduced = filter_low_visit_payout(
            payout_reduced, visits_reduced, min_visits=2
        )

        # --- Boundary: expand to 5-action space for consumers ---
        return SearchResult(
            payout_matrix=expand_payout(
                filtered_payout_reduced, root.p1_effective, root.p2_effective
            ),
            policy_p1=expand_prior(acting_p1, root.p1_effective),
            policy_p2=expand_prior(acting_p2, root.p2_effective),
            learning_policy_p1=expand_prior(learning_p1, root.p1_effective),
            learning_policy_p2=expand_prior(learning_p2, root.p2_effective),
            action_visits=expand_visits(visits_reduced, root.p1_effective, root.p2_effective),
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
