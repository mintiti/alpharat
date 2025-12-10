"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with Q-values
marginalized over the opponent's prior policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

from alpharat.mcts.nash import compute_nash_equilibrium

if TYPE_CHECKING:
    from alpharat.mcts.node import MCTSNode
    from alpharat.mcts.search import SearchResult
    from alpharat.mcts.tree import MCTSTree


class DecoupledPUCTConfig(BaseModel):
    """Config for decoupled PUCT MCTS search."""

    variant: Literal["decoupled_puct"] = "decoupled_puct"
    simulations: int
    gamma: float = 1.0
    c_puct: float = 1.5

    def build(self, tree: MCTSTree) -> DecoupledPUCTSearch:
        """Construct a DecoupledPUCTSearch from this config."""
        return DecoupledPUCTSearch(tree, self)


class DecoupledPUCTSearch:
    """MCTS search using decoupled PUCT action selection.

    Each player independently selects actions by maximizing:
        Q(a) + c * P(a) * sqrt(N_total) / (1 + N(a))

    Where Q-values are marginalized over the opponent's prior policy.

    Attributes:
        tree: The MCTS tree to search.
    """

    def __init__(self, tree: MCTSTree, config: DecoupledPUCTConfig) -> None:
        """Initialize decoupled PUCT search.

        Args:
            tree: MCTS tree with root node and game state.
            config: Configuration including simulations, gamma, c_puct.
        """
        self.tree = tree
        self._n_sims = config.simulations
        self._c_puct = config.c_puct

    def search(self) -> SearchResult:
        """Run MCTS search and return result.

        If root is terminal, returns current state without simulations.

        Returns:
            SearchResult with updated payout matrix and Nash strategies.
        """
        if self.tree.root.is_terminal:
            return self._make_result()

        for _ in range(self._n_sims):
            self._simulate()

        return self._make_result()

    def _simulate(self) -> None:
        """Run a single MCTS simulation.

        Selection: Walk tree selecting actions via decoupled PUCT.
        Expansion: Create child node when reaching unexpanded action pair.
        Backup: Propagate discounted values up the path.
        """
        path: list[tuple[MCTSNode, int, int, float]] = []
        current = self.tree.root
        leaf_child: MCTSNode | None = None

        # Selection + Expansion
        while not current.is_terminal:
            # Select actions via decoupled PUCT
            a1, a2 = self._select_actions(current)

            # Check if this is expansion
            effective_pair = (current.p1_effective[a1], current.p2_effective[a2])
            is_expansion = effective_pair not in current.children

            # Execute move
            child, reward = self.tree.make_move_from(current, a1, a2)
            path.append((current, a1, a2, reward))

            if is_expansion:
                leaf_child = child
                break

            current = child

        # Backup
        if not path:
            return  # Root was terminal

        # Compute leaf value
        if leaf_child is None or leaf_child.is_terminal:
            g = 0.0
        else:
            g = float(
                leaf_child.prior_policy_p1 @ leaf_child.payout_matrix @ leaf_child.prior_policy_p2
            )

        self.tree.backup(path, g=g)

    def _select_actions(self, node: MCTSNode) -> tuple[int, int]:
        """Select actions for both players using decoupled PUCT.

        Args:
            node: Current node to select from.

        Returns:
            Tuple of (action_p1, action_p2) selected via PUCT formula.
        """
        # Marginal Q-values
        # P1 maximizes payout_matrix, expects P2 to play prior_p2
        q1 = node.payout_matrix @ node.prior_policy_p2

        # P2 minimizes payout_matrix (zero-sum), expects P1 to play prior_p1
        q2 = -(node.payout_matrix.T @ node.prior_policy_p1)

        # Marginal visit counts
        n1 = node.action_visits.sum(axis=1)  # sum over P2 actions
        n2 = node.action_visits.sum(axis=0)  # sum over P1 actions

        n_total = node.total_visits

        # PUCT scores
        puct1 = self._compute_puct_scores(q1, node.prior_policy_p1, n1, n_total)
        puct2 = self._compute_puct_scores(q2, node.prior_policy_p2, n2, n_total)

        return int(np.argmax(puct1)), int(np.argmax(puct2))

    def _compute_puct_scores(
        self,
        q_values: np.ndarray,
        priors: np.ndarray,
        marginal_visits: np.ndarray,
        total_visits: int,
    ) -> np.ndarray:
        """Compute PUCT scores: Q(a) + c * P(a) * sqrt(N_total) / (1 + N(a)).

        Args:
            q_values: Marginalized Q-values for this player [5].
            priors: NN prior policy [5].
            marginal_visits: Marginal visit counts [5].
            total_visits: Total visits to this node.

        Returns:
            PUCT scores for each action [5].
        """
        exploration = self._c_puct * priors * np.sqrt(total_visits) / (1 + marginal_visits)
        result: np.ndarray = q_values + exploration
        return result

    def _make_result(self) -> SearchResult:
        """Create SearchResult from current root state.

        Returns:
            SearchResult with payout matrix and Nash equilibrium strategies.
        """
        from alpharat.mcts.search import SearchResult

        root = self.tree.root
        p1_strat, p2_strat = compute_nash_equilibrium(
            root.payout_matrix,
            root.p1_effective,
            root.p2_effective,
        )
        return SearchResult(
            payout_matrix=root.payout_matrix.copy(),
            policy_p1=p1_strat,
            policy_p2=p2_strat,
        )
