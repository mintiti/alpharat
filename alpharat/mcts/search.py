"""MCTS Search implementation for simultaneous-move games.

This module provides the search loop that ties together tree navigation,
node expansion, and value backup for Monte Carlo Tree Search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel

from alpharat.mcts.nash import compute_nash_equilibrium

if TYPE_CHECKING:
    from alpharat.mcts.node import MCTSNode
    from alpharat.mcts.tree import MCTSTree


class PriorSamplingConfig(BaseModel):
    """Config for prior sampling MCTS search."""

    variant: Literal["prior_sampling"] = "prior_sampling"
    simulations: int
    gamma: float = 1.0

    def build(self, tree: MCTSTree) -> MCTSSearch:
        """Construct a PriorSamplingSearch from this config."""
        return MCTSSearch(tree, self.simulations)


@dataclass
class SearchResult:
    """Result of an MCTS search.

    Attributes:
        payout_matrix: Root's updated payout matrix after search.
        policy_p1: Nash equilibrium strategy for player 1.
        policy_p2: Nash equilibrium strategy for player 2.
    """

    payout_matrix: np.ndarray
    policy_p1: np.ndarray
    policy_p2: np.ndarray


class MCTSSearch:
    """MCTS search for simultaneous-move games.

    Runs simulations to refine value estimates at the root node,
    then returns Nash equilibrium strategies based on the updated
    payout matrix.

    Selection uses prior-based sampling: actions are sampled from
    the neural network's policy priors at each node.

    Attributes:
        tree: The MCTS tree to search.
        n_sims: Number of simulations to run.
    """

    def __init__(self, tree: MCTSTree, n_sims: int) -> None:
        """Initialize MCTS search.

        Args:
            tree: MCTS tree with root node and game state.
            n_sims: Number of simulations to run.
        """
        self.tree = tree
        self.n_sims = n_sims

    def search(self) -> SearchResult:
        """Run MCTS search and return result.

        If simulations=0, returns raw NN priors (pure NN mode).
        If root is terminal, returns current state without simulations.

        Returns:
            SearchResult with updated payout matrix and Nash/NN strategies.
        """
        # Pure NN mode: return raw priors, skip MCTS
        if self.n_sims == 0:
            return self._make_nn_result()

        # If root is terminal, return current state
        if self.tree.root.is_terminal:
            return self._make_result()

        for _ in range(self.n_sims):
            self._simulate()

        return self._make_result()

    def _simulate(self) -> None:
        """Run a single MCTS simulation.

        Selection: Walk tree sampling actions from prior policies.
        Expansion: Create child node when reaching unexpanded action pair.
        Backup: Propagate discounted values up the path.
        """
        path: list[tuple[MCTSNode, int, int, tuple[float, float]]] = []
        current = self.tree.root
        leaf_child: MCTSNode | None = None

        # Selection + Expansion
        while not current.is_terminal:
            # Sample actions from priors
            a1 = int(np.random.choice(len(current.prior_policy_p1), p=current.prior_policy_p1))
            a2 = int(np.random.choice(len(current.prior_policy_p2), p=current.prior_policy_p2))

            # Check if this is expansion (child doesn't exist yet)
            effective_pair = (current.p1_effective[a1], current.p2_effective[a2])
            is_expansion = effective_pair not in current.children

            # Execute move (handles navigation, creates child if needed)
            child, reward = self.tree.make_move_from(current, a1, a2)
            path.append((current, a1, a2, reward))

            if is_expansion:
                leaf_child = child
                break

            current = child

        # Backup
        if not path:
            return  # Root was terminal

        # Compute leaf value (g) as tuple for both players
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

    def _make_result(self) -> SearchResult:
        """Create SearchResult from current root state.

        Returns:
            SearchResult with payout matrix and Nash equilibrium strategies.
        """
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

    def _make_nn_result(self) -> SearchResult:
        """Create SearchResult using raw NN priors (pure NN mode).

        Used when simulations=0 to skip MCTS and return the NN policy directly.
        At this point, payout_matrix still equals the initial NN prediction.

        Returns:
            SearchResult with NN payout prediction and prior policies.
        """
        root = self.tree.root
        return SearchResult(
            payout_matrix=root.payout_matrix.copy(),
            policy_p1=root.prior_policy_p1.copy(),
            policy_p2=root.prior_policy_p2.copy(),
        )
