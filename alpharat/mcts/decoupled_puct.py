"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with Q-values
marginalized over the opponent's prior policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from alpharat.config.base import StrictBaseModel
from alpharat.mcts.nash import compute_nash_equilibrium
from alpharat.mcts.selection import compute_forced_threshold

if TYPE_CHECKING:
    from alpharat.mcts.node import MCTSNode
    from alpharat.mcts.tree import MCTSTree


@dataclass
class SearchResult:
    """Result of MCTS search containing policy and value information."""

    policy_p1: np.ndarray
    policy_p2: np.ndarray
    payout_matrix: np.ndarray


class DecoupledPUCTConfig(StrictBaseModel):
    """Configuration for decoupled PUCT search."""

    variant: Literal["decoupled_puct"] = "decoupled_puct"
    simulations: int = 100
    gamma: float = 1.0
    c_puct: float = 1.5
    force_k: float = 2.0
    policy: dict | None = None  # Legacy field, ignored (for manifest compat)

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

    def __init__(self, tree: MCTSTree, config: DecoupledPUCTConfig):
        self.tree = tree
        self._n_sims = config.simulations
        self._c_puct = config.c_puct
        self._force_k = config.force_k

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
            policy_p1=root.prior_policy_p1.copy(),
            policy_p2=root.prior_policy_p2.copy(),
            payout_matrix=root.payout_matrix.copy(),
        )

    def _make_result(self) -> SearchResult:
        """Compute Nash equilibrium from root statistics."""
        root = self.tree.root
        payout_matrix = root.payout_matrix

        policy_p1, policy_p2 = compute_nash_equilibrium(
            payout_matrix,
            root.p1_effective,
            root.p2_effective,
        )

        return SearchResult(
            policy_p1=policy_p1,
            policy_p2=policy_p2,
            payout_matrix=payout_matrix.copy(),
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

        # Compute PUCT in reduced space
        puct1 = self._compute_puct_scores(q1, node._prior_p1, n1, n_total, is_root)
        puct2 = self._compute_puct_scores(q2, node._prior_p2, n2, n_total, is_root)

        # Select outcome indices (simple random tie-break)
        idx1 = self._select_max_with_tiebreak(puct1)
        idx2 = self._select_max_with_tiebreak(puct2)

        # Map outcome index back to action
        a1 = node._p1_outcomes[idx1]
        a2 = node._p2_outcomes[idx2]

        return a1, a2

    def _select_max_with_tiebreak(self, scores: np.ndarray) -> int:
        """Select argmax with random tie-breaking.

        Args:
            scores: Array of scores.

        Returns:
            Index of maximum (random among ties).
        """
        max_val = scores[0]
        max_idx = 0
        n_ties = 1

        # Single pass: find max and count ties
        for i in range(1, len(scores)):
            if scores[i] > max_val + 1e-9:
                max_val = scores[i]
                max_idx = i
                n_ties = 1
            elif abs(scores[i] - max_val) < 1e-9:
                n_ties += 1
                # Reservoir sampling: replace with probability 1/n_ties
                if np.random.random() < 1.0 / n_ties:
                    max_idx = i

        return max_idx

    def _compute_puct_scores(
        self,
        q_values: np.ndarray,
        prior: np.ndarray,
        visit_counts: np.ndarray,
        total_visits: int,
        is_root: bool = False,
    ) -> np.ndarray:
        """Compute PUCT scores for action selection.

        PUCT = Q + c * prior * sqrt(N) / (1 + n)

        At root with force_k > 0, undervisited actions get a large score boost
        to ensure exploration proportional to prior (KataGo-style forced playouts).

        Args:
            q_values: Q-values for each action.
            prior: Prior policy.
            visit_counts: Visit count for each action.
            total_visits: Total visits at this node.
            is_root: Whether this is the root node.

        Returns:
            PUCT score for each action.
        """
        exploration = self._c_puct * prior * np.sqrt(total_visits) / (1 + visit_counts)
        scores: np.ndarray = q_values + exploration

        # Forced playouts at root: boost undervisited actions
        if is_root and self._force_k > 0:
            threshold = compute_forced_threshold(prior, total_visits, self._force_k)
            forced = (visit_counts < threshold) & (prior > 0)
            scores[forced] = 1e20

        return scores
