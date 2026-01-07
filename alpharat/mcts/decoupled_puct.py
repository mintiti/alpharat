"""Decoupled PUCT search for simultaneous-move MCTS.

Each player independently selects actions via PUCT formula with Q-values
marginalized over the opponent's prior policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

from alpharat.mcts.nash import compute_nash_equilibrium
from alpharat.mcts.selection import compute_forced_threshold

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
    force_k: float = 0.0  # Forced playout scaling (0 disables, 2.0 is KataGo default)

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
            config: Configuration including simulations, gamma, c_puct, force_k.
        """
        self.tree = tree
        self._n_sims = config.simulations
        self._c_puct = config.c_puct
        self._force_k = config.force_k

    def search(self) -> SearchResult:
        """Run MCTS search and return result.

        If simulations=0, returns raw NN priors (pure NN mode).
        If root is terminal, returns current state without simulations.

        Returns:
            SearchResult with updated payout matrix and Nash/NN strategies.
        """
        # Pure NN mode: return raw priors, skip MCTS
        if self._n_sims == 0:
            return self._make_nn_result()

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
        path: list[tuple[MCTSNode, int, int, tuple[float, float]]] = []
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

        # Compute leaf value as tuple for both players
        if leaf_child is None or leaf_child.is_terminal:
            g: tuple[float, float] = (0.0, 0.0)
        else:
            # NN's expected value under its own policy for each player
            g = (
                float(
                    leaf_child.prior_policy_p1
                    @ leaf_child.payout_matrix[0]
                    @ leaf_child.prior_policy_p2
                ),
                float(
                    leaf_child.prior_policy_p1
                    @ leaf_child.payout_matrix[1]
                    @ leaf_child.prior_policy_p2
                ),
            )

        self.tree.backup(path, g=g)

    def _select_actions(self, node: MCTSNode) -> tuple[int, int]:
        """Select actions for both players using decoupled PUCT.

        Args:
            node: Current node to select from.

        Returns:
            Tuple of (action_p1, action_p2) selected via PUCT formula.
        """
        # Marginal Q-values from bimatrix payout
        # P1 maximizes payout_matrix[0], expects P2 to play prior_p2
        q1 = node.payout_matrix[0] @ node.prior_policy_p2

        # P2 maximizes payout_matrix[1], expects P1 to play prior_p1
        q2 = node.payout_matrix[1].T @ node.prior_policy_p1

        # Marginal visit counts
        n1 = node.action_visits.sum(axis=1)  # sum over P2 actions
        n2 = node.action_visits.sum(axis=0)  # sum over P1 actions

        n_total = node.total_visits

        # PUCT scores
        puct1 = self._compute_puct_scores(q1, node.prior_policy_p1, n1, n_total)
        puct2 = self._compute_puct_scores(q2, node.prior_policy_p2, n2, n_total)

        # Forced playouts: set PUCT=inf for undervisited actions (root only)
        if node == self.tree.root and self._force_k > 0:
            thresh1 = compute_forced_threshold(node.prior_policy_p1, n_total, self._force_k)
            thresh2 = compute_forced_threshold(node.prior_policy_p2, n_total, self._force_k)
            puct1[n1 < thresh1] = np.inf
            puct2[n2 < thresh2] = np.inf

        a1 = self._select_with_tiebreak(puct1, node.p1_effective, node.prior_policy_p1)
        a2 = self._select_with_tiebreak(puct2, node.p2_effective, node.prior_policy_p2)
        return a1, a2

    def _select_with_tiebreak(
        self, puct_scores: np.ndarray, effective: list[int], prior: np.ndarray
    ) -> int:
        """Select action with random tie-breaking among effective actions.

        When PUCT scores are tied (common on first simulation when all are 0),
        samples from the prior distribution over effective actions to ensure
        symmetric exploration.

        Args:
            puct_scores: PUCT scores for each action [5].
            effective: Effective action mapping (blocked actions map to STAY).
            prior: Prior policy for this player [5].

        Returns:
            Selected action index.
        """
        # Effective actions are those that map to themselves
        effective_actions = [a for a in range(5) if effective[a] == a]

        if not effective_actions:
            # Shouldn't happen, but fallback to STAY
            return 4

        # Get PUCT scores for effective actions
        effective_puct = [puct_scores[a] for a in effective_actions]
        max_puct = max(effective_puct)

        # Find all actions tied at max
        best_actions = [
            a for a, p in zip(effective_actions, effective_puct, strict=True) if p == max_puct
        ]

        if len(best_actions) == 1:
            return best_actions[0]

        # Multiple tied — sample from prior restricted to tied actions
        tied_prior = np.array([prior[a] for a in best_actions])
        tied_prior /= tied_prior.sum()
        return int(np.random.choice(best_actions, p=tied_prior))

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
            prior_p1=root.prior_policy_p1,
            prior_p2=root.prior_policy_p2,
            action_visits=root.action_visits,
        )
        return SearchResult(
            payout_matrix=root.payout_matrix.copy(),
            policy_p1=p1_strat,
            policy_p2=p2_strat,
        )

    def _make_nn_result(self) -> SearchResult:
        """Create SearchResult using raw NN priors (pure NN mode).

        Used when simulations=0 to skip MCTS and return the NN policy directly.
        At this point, payout_matrix still equals the initial NN prediction.

        Returns:
            SearchResult with NN payout prediction and prior policies.
        """
        from alpharat.mcts.search import SearchResult

        root = self.tree.root
        return SearchResult(
            payout_matrix=root.payout_matrix.copy(),
            policy_p1=root.prior_policy_p1.copy(),
            policy_p2=root.prior_policy_p2.copy(),
        )
