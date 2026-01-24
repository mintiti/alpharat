"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. Statistics are indexed by
unique effective outcomes rather than raw actions, giving O(1) backup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpharat.mcts.reduction import (
    expand_payout,
    expand_prior,
    expand_visits,
    reduce_payout,
    reduce_prior,
)


class MCTSNode:
    """MCTS node for simultaneous-move games with O(1) backup.

    Maintains statistics indexed by unique effective outcomes rather than
    raw actions. This makes backup O(1) instead of O(k²) for k equivalent actions.

    The node stores:
    - Reduced priors [n1], [n2] where n = count of unique outcomes
    - Reduced payout matrices [n1, n2] for each player
    - Reduced visit counts [n1, n2]
    - Children keyed by outcome index pairs (i, j)

    Attributes:
        game_state: The game state this node represents
        is_terminal: Whether this is a terminal game state
        parent: Parent node in the search tree
        move_undo: MoveUndo object to reach this node from parent (None for root)
        depth: Depth of this node in the tree (0 for root)
        p1_mud_turns_remaining: Turns player 1 is stuck in mud
        p2_mud_turns_remaining: Turns player 2 is stuck in mud
        p1_effective: Maps each P1 action index to its effective action
        p2_effective: Maps each P2 action index to its effective action
    """

    def __init__(
        self,
        game_state: Any,
        prior_policy_p1: np.ndarray,
        prior_policy_p2: np.ndarray,
        nn_payout_prediction: np.ndarray,
        parent: MCTSNode | None = None,
        p1_mud_turns_remaining: int = 0,
        p2_mud_turns_remaining: int = 0,
        move_undo: Any = None,
        parent_action: tuple[int, int] | None = None,
        p1_effective: list[int] | None = None,
        p2_effective: list[int] | None = None,
    ) -> None:
        """Initialize MCTS node.

        Args:
            game_state: The game state this node represents
            prior_policy_p1: Policy prior for player 1 [5] (will be reduced)
            prior_policy_p2: Policy prior for player 2 [5] (will be reduced)
            nn_payout_prediction: Initial payout matrix from NN [2, 5, 5] (will be reduced)
            parent: Parent node (None for root)
            p1_mud_turns_remaining: Turns P1 is stuck in mud
            p2_mud_turns_remaining: Turns P2 is stuck in mud
            move_undo: MoveUndo object to reach this node from parent (None for root)
            parent_action: Outcome index pair taken from parent to reach this node
            p1_effective: Maps each P1 action to its effective action (blocked -> STAY)
            p2_effective: Maps each P2 action to its effective action (blocked -> STAY)
        """
        self.game_state = game_state
        self.is_terminal = False
        self.parent = parent
        self.move_undo = move_undo
        self.parent_action = parent_action

        # Calculate depth from parent
        self.depth: int = 0 if parent is None else parent.depth + 1

        # Mud state
        self.p1_mud_turns_remaining = p1_mud_turns_remaining
        self.p2_mud_turns_remaining = p2_mud_turns_remaining

        # Action equivalence
        num_actions = len(prior_policy_p1)
        stay_action = num_actions - 1

        if p1_mud_turns_remaining > 0:
            self.p1_effective = [stay_action] * num_actions
        elif p1_effective is not None:
            self.p1_effective = p1_effective
        else:
            self.p1_effective = list(range(num_actions))

        if p2_mud_turns_remaining > 0:
            self.p2_effective = [stay_action] * num_actions
        elif p2_effective is not None:
            self.p2_effective = p2_effective
        else:
            self.p2_effective = list(range(num_actions))

        # Compute unique outcomes and mappings
        self._p1_outcomes = sorted(set(self.p1_effective))
        self._p2_outcomes = sorted(set(self.p2_effective))
        self._n1 = len(self._p1_outcomes)
        self._n2 = len(self._p2_outcomes)

        # Effective action -> outcome index mappings
        self._p1_outcome_idx = {eff: i for i, eff in enumerate(self._p1_outcomes)}
        self._p2_outcome_idx = {eff: i for i, eff in enumerate(self._p2_outcomes)}

        # Pre-computed action -> outcome index for fast backup
        self._p1_action_to_idx = [
            self._p1_outcome_idx[self.p1_effective[a]] for a in range(num_actions)
        ]
        self._p2_action_to_idx = [
            self._p2_outcome_idx[self.p2_effective[a]] for a in range(num_actions)
        ]

        # Store priors as reduced [n1], [n2]
        self._prior_p1 = reduce_prior(prior_policy_p1, self.p1_effective)
        self._prior_p2 = reduce_prior(prior_policy_p2, self.p2_effective)

        # Initialize reduced statistics with NN predictions
        reduced_payout = reduce_payout(nn_payout_prediction, self.p1_effective, self.p2_effective)
        self._payout_p1: list[list[float]] = [
            [float(reduced_payout[0, i, j]) for j in range(self._n2)] for i in range(self._n1)
        ]
        self._payout_p2: list[list[float]] = [
            [float(reduced_payout[1, i, j]) for j in range(self._n2)] for i in range(self._n1)
        ]
        self._visits: list[list[int]] = [[0] * self._n2 for _ in range(self._n1)]

        # Cached total visits for O(1) access
        self._total_visits: int = 0

        # Tree structure - children keyed by outcome index pairs (i, j)
        self.children: dict[tuple[int, int], MCTSNode] = {}

    @property
    def n1(self) -> int:
        """Number of unique outcomes for player 1."""
        return self._n1

    @property
    def n2(self) -> int:
        """Number of unique outcomes for player 2."""
        return self._n2

    def outcome_to_effective(self, player: int, outcome_idx: int) -> int:
        """Convert outcome index to effective action value.

        Args:
            player: 1 for P1, 2 for P2
            outcome_idx: Outcome index (0 to n-1)

        Returns:
            Effective action value (0-4)
        """
        if player == 1:
            return self._p1_outcomes[outcome_idx]
        return self._p2_outcomes[outcome_idx]

    def action_to_outcome(self, player: int, action: int) -> int:
        """Convert action index to outcome index.

        Args:
            player: 1 for P1, 2 for P2
            action: Action index (0-4)

        Returns:
            Outcome index (0 to n-1)
        """
        if player == 1:
            return self._p1_action_to_idx[action]
        return self._p2_action_to_idx[action]

    def update_effective_actions(
        self, p1_effective: list[int] | None = None, p2_effective: list[int] | None = None
    ) -> None:
        """Update effective action mappings and recompute internal structures.

        Call this when effective actions change after construction (e.g., when
        the tree initializes root with game-derived wall information).

        Args:
            p1_effective: New effective action mapping for P1, or None to keep current.
            p2_effective: New effective action mapping for P2, or None to keep current.
        """
        if p1_effective is not None:
            self.p1_effective = p1_effective
        if p2_effective is not None:
            self.p2_effective = p2_effective

        # Recompute outcome mappings
        self._p1_outcomes = sorted(set(self.p1_effective))
        self._p2_outcomes = sorted(set(self.p2_effective))
        self._n1 = len(self._p1_outcomes)
        self._n2 = len(self._p2_outcomes)

        self._p1_outcome_idx = {eff: i for i, eff in enumerate(self._p1_outcomes)}
        self._p2_outcome_idx = {eff: i for i, eff in enumerate(self._p2_outcomes)}

        num_actions = len(self.p1_effective)
        self._p1_action_to_idx = [
            self._p1_outcome_idx[self.p1_effective[a]] for a in range(num_actions)
        ]
        self._p2_action_to_idx = [
            self._p2_outcome_idx[self.p2_effective[a]] for a in range(num_actions)
        ]

        # Re-reduce priors with new effective mappings
        # Expand current priors first to get [5] arrays, then reduce
        expanded_p1 = expand_prior(self._prior_p1, self.p1_effective)
        expanded_p2 = expand_prior(self._prior_p2, self.p2_effective)
        self._prior_p1 = reduce_prior(expanded_p1, self.p1_effective)
        self._prior_p2 = reduce_prior(expanded_p2, self.p2_effective)

        # Reinitialize reduced statistics to match new structure
        self._payout_p1 = [[0.0] * self._n2 for _ in range(self._n1)]
        self._payout_p2 = [[0.0] * self._n2 for _ in range(self._n1)]
        self._visits = [[0] * self._n2 for _ in range(self._n1)]
        self._total_visits = 0

    @property
    def total_visits(self) -> int:
        """Total number of simulations through this node (O(1))."""
        return self._total_visits

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    @property
    def prior_policy_p1(self) -> np.ndarray:
        """Get expanded prior policy for P1 [5] (for compatibility)."""
        return expand_prior(self._prior_p1, self.p1_effective)

    @prior_policy_p1.setter
    def prior_policy_p1(self, value: np.ndarray) -> None:
        """Set prior policy for P1 from [5] array."""
        self._prior_p1 = reduce_prior(value, self.p1_effective)

    @property
    def prior_policy_p2(self) -> np.ndarray:
        """Get expanded prior policy for P2 [5] (for compatibility)."""
        return expand_prior(self._prior_p2, self.p2_effective)

    @prior_policy_p2.setter
    def prior_policy_p2(self, value: np.ndarray) -> None:
        """Set prior policy for P2 from [5] array."""
        self._prior_p2 = reduce_prior(value, self.p2_effective)

    @property
    def payout_matrix(self) -> np.ndarray:
        """Get expanded payout matrix [2, 5, 5] for compatibility.

        This reconstructs the full matrix from reduced storage.
        Use sparingly - prefer reduced access for hot paths.
        """
        return self.get_expanded_payout_matrix()

    @payout_matrix.setter
    def payout_matrix(self, value: np.ndarray) -> None:
        """Set payout matrix from [2, 5, 5] array.

        This reinitializes the reduced payout matrices from the full matrix.
        Used by MCTSTree._init_root_priors() to set NN predictions.
        """
        reduced = reduce_payout(value, self.p1_effective, self.p2_effective)
        self._payout_p1 = [
            [float(reduced[0, i, j]) for j in range(self._n2)] for i in range(self._n1)
        ]
        self._payout_p2 = [
            [float(reduced[1, i, j]) for j in range(self._n2)] for i in range(self._n1)
        ]

    @property
    def action_visits(self) -> np.ndarray:
        """Get expanded visit counts [5, 5] for compatibility.

        This reconstructs the full matrix from reduced storage.
        Use sparingly - prefer reduced access for hot paths.
        """
        return self.get_expanded_visits()

    def backup(self, action_p1: int, action_p2: int, value: tuple[float, float]) -> None:
        """Update statistics after visiting a child node.

        O(1) operation - directly indexes into reduced matrices.

        Args:
            action_p1: Player 1's action that led to child (0-4)
            action_p2: Player 2's action that led to child (0-4)
            value: Tuple of (p1_value, p2_value) representing returns for each player
        """
        idx1 = self._p1_action_to_idx[action_p1]
        idx2 = self._p2_action_to_idx[action_p2]

        n = self._visits[idx1][idx2]
        n_plus_1 = n + 1

        p1_value, p2_value = value

        # Incremental mean update
        self._payout_p1[idx1][idx2] += (p1_value - self._payout_p1[idx1][idx2]) / n_plus_1
        self._payout_p2[idx1][idx2] += (p2_value - self._payout_p2[idx1][idx2]) / n_plus_1
        self._visits[idx1][idx2] = n_plus_1

        # Update cached total
        self._total_visits += 1

    def get_expanded_payout_matrix(self) -> np.ndarray:
        """Expand reduced payout matrices to full [2, 5, 5] shape.

        Each action maps to its effective outcome's payout value.
        Equivalent actions get identical values (as expected by Nash computation).

        Returns:
            Payout matrix with shape [2, 5, 5].
        """
        # Build reduced numpy array first
        reduced = np.zeros((2, self._n1, self._n2), dtype=np.float64)
        for i in range(self._n1):
            for j in range(self._n2):
                reduced[0, i, j] = self._payout_p1[i][j]
                reduced[1, i, j] = self._payout_p2[i][j]

        return expand_payout(reduced, self.p1_effective, self.p2_effective)

    def get_expanded_visits(self) -> np.ndarray:
        """Expand reduced visit counts to full [5, 5] shape.

        Each action maps to its effective outcome's visit count.
        Equivalent actions get identical values.

        Returns:
            Visit count matrix with shape [5, 5].
        """
        # Build reduced numpy array first
        reduced = np.zeros((self._n1, self._n2), dtype=np.int32)
        for i in range(self._n1):
            for j in range(self._n2):
                reduced[i, j] = self._visits[i][j]

        return expand_visits(reduced, self.p1_effective, self.p2_effective)

    def get_reduced_payout(self) -> np.ndarray:
        """Get reduced payout matrix [2, n1, n2].

        Returns the internal reduced representation without expansion.

        Returns:
            Payout matrix with shape [2, n1, n2].
        """
        reduced = np.zeros((2, self._n1, self._n2), dtype=np.float64)
        for i in range(self._n1):
            for j in range(self._n2):
                reduced[0, i, j] = self._payout_p1[i][j]
                reduced[1, i, j] = self._payout_p2[i][j]
        return reduced

    def get_reduced_visits(self) -> np.ndarray:
        """Get reduced visit counts [n1, n2].

        Returns the internal reduced representation without expansion.

        Returns:
            Visit counts with shape [n1, n2].
        """
        reduced = np.zeros((self._n1, self._n2), dtype=np.int32)
        for i in range(self._n1):
            for j in range(self._n2):
                reduced[i, j] = self._visits[i][j]
        return reduced

    def compute_marginal_q_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute marginalized Q-values for PUCT selection.

        Returns Q-values for each action, marginalized over opponent's prior.

        Returns:
            Tuple of (q1, q2) where q1[a] is P1's Q-value for action a.
        """
        num_actions = len(self.p1_effective)

        # Get expanded priors for weighting
        prior_p1 = self.prior_policy_p1
        prior_p2 = self.prior_policy_p2

        # P1's Q-values: Q1[a1] = sum over a2 of payout[0, a1, a2] * prior_p2[a2]
        q1 = np.zeros(num_actions)
        for a1 in range(num_actions):
            idx1 = self._p1_action_to_idx[a1]
            total = 0.0
            for a2 in range(num_actions):
                idx2 = self._p2_action_to_idx[a2]
                total += self._payout_p1[idx1][idx2] * prior_p2[a2]
            q1[a1] = total

        # P2's Q-values: Q2[a2] = sum over a1 of payout[1, a1, a2] * prior_p1[a1]
        q2 = np.zeros(num_actions)
        for a2 in range(num_actions):
            idx2 = self._p2_action_to_idx[a2]
            total = 0.0
            for a1 in range(num_actions):
                idx1 = self._p1_action_to_idx[a1]
                total += self._payout_p2[idx1][idx2] * prior_p1[a1]
            q2[a2] = total

        return q1, q2

    def compute_marginal_visits(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute marginal visit counts for PUCT selection.

        Returns visit counts marginalized over opponent's actions,
        properly handling action equivalence (no double-counting).

        Returns:
            Tuple of (n1, n2) where n1[a] is visit count for P1's action a.
        """
        num_actions = len(self.p1_effective)

        # Sum over reduced indices, then expand
        reduced_n1 = [sum(self._visits[i]) for i in range(self._n1)]
        reduced_n2 = [sum(self._visits[i][j] for i in range(self._n1)) for j in range(self._n2)]

        # Expand to action space
        n1 = np.array([reduced_n1[self._p1_action_to_idx[a]] for a in range(num_actions)])
        n2 = np.array([reduced_n2[self._p2_action_to_idx[a]] for a in range(num_actions)])

        return n1, n2

    def compute_expected_value(self) -> tuple[float, float]:
        """Compute expected value under NN priors for both players.

        This is the NN's expected payoff: E[V] = π₁ᵀ · Payoff · π₂

        Returns:
            Tuple (v1, v2) of expected values for P1 and P2.
        """
        v1 = 0.0
        v2 = 0.0
        num_actions = len(self.p1_effective)

        # Get expanded priors
        prior_p1 = self.prior_policy_p1
        prior_p2 = self.prior_policy_p2

        for a1 in range(num_actions):
            idx1 = self._p1_action_to_idx[a1]
            p1_prior = prior_p1[a1]
            for a2 in range(num_actions):
                idx2 = self._p2_action_to_idx[a2]
                p2_prior = prior_p2[a2]
                weight = p1_prior * p2_prior
                v1 += weight * self._payout_p1[idx1][idx2]
                v2 += weight * self._payout_p2[idx1][idx2]

        return v1, v2

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.total_visits}, "
            f"outcomes=({self._n1}x{self._n2}), "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
