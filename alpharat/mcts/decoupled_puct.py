"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with marginal
Q-values. Returns visit-proportional policies instead of Nash equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from alpharat.config.base import StrictBaseModel
from alpharat.mcts.forced_playouts import compute_pruned_visits
from alpharat.mcts.numba_ops import compute_puct_scores, select_max_with_tiebreak
from alpharat.mcts.reduction import expand_prior

if TYPE_CHECKING:
    from alpharat.mcts.node import MCTSNode
    from alpharat.mcts.tree import MCTSTree


@dataclass
class SearchResult:
    """Result of MCTS search containing policy and value information.

    policy_p1: np.ndarray  # [5] visit-proportional policy
    policy_p2: np.ndarray  # [5] visit-proportional policy
    value_p1: float  # Root value estimate for P1
    value_p2: float  # Root value estimate for P2
    visit_counts_p1: np.ndarray  # [5] pruned visit counts for P1
    visit_counts_p2: np.ndarray  # [5] pruned visit counts for P2
    """

    policy_p1: np.ndarray
    policy_p2: np.ndarray
    value_p1: float
    value_p2: float
    visit_counts_p1: np.ndarray
    visit_counts_p2: np.ndarray


class DecoupledPUCTConfig(StrictBaseModel):
    """Configuration for decoupled PUCT search."""

    simulations: int = 100
    gamma: float = 1.0
    c_puct: float = 1.5
    force_k: float = 2.0

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

    def search(self) -> SearchResult:
        """Run MCTS search and return the result.

        Returns:
            SearchResult with policies and scalar value estimates.
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
            policy_p1=root.prior_policy_p1.copy(),
            policy_p2=root.prior_policy_p2.copy(),
            value_p1=root.v1,
            value_p2=root.v2,
            visit_counts_p1=np.zeros(5, dtype=np.float64),
            visit_counts_p2=np.zeros(5, dtype=np.float64),
        )

    def _make_result(self) -> SearchResult:
        """Compute visit-proportional policy from root statistics."""
        root = self.tree.root

        # Get raw Q-values and visit counts (reduced space)
        q1, q2 = root.get_q_values(gamma=self.tree.gamma)
        n1, n2 = root.get_visit_counts()

        # Normalize Q for pruning (PUCT sees [0, 1] values)
        rc = root.remaining_cheese
        q1_norm = q1 / rc
        q2_norm = q2 / rc

        # Prune forced visits (per-player, in reduced space)
        n1_pruned = compute_pruned_visits(
            q1_norm, root.prior_p1_reduced, n1, root.total_visits, self._c_puct
        )
        n2_pruned = compute_pruned_visits(
            q2_norm, root.prior_p2_reduced, n2, root.total_visits, self._c_puct
        )

        # Expand to [5] action space
        n1_expanded = expand_prior(n1_pruned, root.p1_effective)
        n2_expanded = expand_prior(n2_pruned, root.p2_effective)

        # Normalize to get visit-proportional policy
        n1_sum = n1_expanded.sum()
        n2_sum = n2_expanded.sum()

        policy_p1 = n1_expanded / n1_sum if n1_sum > 0 else root.prior_policy_p1.copy()
        policy_p2 = n2_expanded / n2_sum if n2_sum > 0 else root.prior_policy_p2.copy()

        # Compute root value from raw Q-values weighted by visit counts
        n1_total = n1.sum()
        n2_total = n2.sum()

        value_p1 = float(np.dot(q1, n1) / n1_total) if n1_total > 0 else root.v1
        value_p2 = float(np.dot(q2, n2) / n2_total) if n2_total > 0 else root.v2

        return SearchResult(
            policy_p1=policy_p1.astype(np.float64),
            policy_p2=policy_p2.astype(np.float64),
            value_p1=value_p1,
            value_p2=value_p2,
            visit_counts_p1=n1_expanded.astype(np.float64),
            visit_counts_p2=n2_expanded.astype(np.float64),
        )

    def _simulate(self) -> None:
        """Run a single MCTS simulation.

        Selection: Walk tree selecting actions via decoupled PUCT.
        Expansion: Create child node when reaching unexpanded action pair.
        Backup: Propagate discounted values up the path.
        """
        path: list[tuple[MCTSNode, int, int]] = []
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
            child, _reward = self.tree.make_move_from(current, a1, a2)

            # Record step BEFORE potential expansion
            path.append((current, a1, a2))

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
            # NN's scalar value estimates for the leaf position
            g = (leaf_child.v1, leaf_child.v2)

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
        # Decoupled UCT: each player has independent Q and N
        q1, q2 = node.get_q_values(gamma=self.tree.gamma)
        n1, n2 = node.get_visit_counts()
        n_total = node.total_visits
        is_root = node == self.tree.root

        # Normalize Q by remaining cheese at this node → [0, 1] at every depth
        rc = node.remaining_cheese
        q1_norm = q1 / rc
        q2_norm = q2 / rc

        # Compute PUCT in reduced space (JIT-compiled)
        puct1 = compute_puct_scores(
            q1_norm, node.prior_p1_reduced, n1, n_total, self._c_puct, self._force_k, is_root
        )
        puct2 = compute_puct_scores(
            q2_norm, node.prior_p2_reduced, n2, n_total, self._c_puct, self._force_k, is_root
        )

        # Select outcome indices (JIT-compiled random tie-break)
        idx1 = select_max_with_tiebreak(puct1)
        idx2 = select_max_with_tiebreak(puct2)

        # Map outcome index back to action
        a1 = node.p1_outcomes[idx1]
        a2 = node.p2_outcomes[idx2]

        return a1, a2
