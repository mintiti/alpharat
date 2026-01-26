"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. Statistics are indexed by
unique effective outcomes rather than raw actions, giving O(1) backup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpharat.mcts.numba_ops import (
    backup_node,
    build_expanded_payout,
    build_expanded_visits,
    compute_expected_value_reduced,
    compute_marginal_q_reduced,
    compute_marginal_visits_reduced,
)
from alpharat.mcts.reduction import (
    expand_prior,
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
        self.p1_outcomes = sorted(set(self.p1_effective))
        self.p2_outcomes = sorted(set(self.p2_effective))
        self._n1 = len(self.p1_outcomes)
        self._n2 = len(self.p2_outcomes)

        # Effective action -> outcome index mappings
        self._p1_outcome_idx = {eff: i for i, eff in enumerate(self.p1_outcomes)}
        self._p2_outcome_idx = {eff: i for i, eff in enumerate(self.p2_outcomes)}

        # Pre-computed action -> outcome index as numpy arrays for Numba
        self._p1_action_to_idx = np.array(
            [self._p1_outcome_idx[self.p1_effective[a]] for a in range(num_actions)],
            dtype=np.int64,
        )
        self._p2_action_to_idx = np.array(
            [self._p2_outcome_idx[self.p2_effective[a]] for a in range(num_actions)],
            dtype=np.int64,
        )

        # Store priors as reduced [n1], [n2] - vectorized reduction
        self.prior_p1_reduced = np.zeros(self._n1, dtype=prior_policy_p1.dtype)
        np.add.at(self.prior_p1_reduced, self._p1_action_to_idx, prior_policy_p1)

        self.prior_p2_reduced = np.zeros(self._n2, dtype=prior_policy_p2.dtype)
        np.add.at(self.prior_p2_reduced, self._p2_action_to_idx, prior_policy_p2)

        # Cache expanded priors - only canonical actions get probability
        p1_outcomes_arr = np.array(self.p1_outcomes, dtype=np.int64)
        p2_outcomes_arr = np.array(self.p2_outcomes, dtype=np.int64)

        self._expanded_prior_p1 = np.zeros(num_actions, dtype=prior_policy_p1.dtype)
        self._expanded_prior_p1[p1_outcomes_arr] = self.prior_p1_reduced

        self._expanded_prior_p2 = np.zeros(num_actions, dtype=prior_policy_p2.dtype)
        self._expanded_prior_p2[p2_outcomes_arr] = self.prior_p2_reduced

        # Initialize reduced payout matrices - vectorized reduction with averaging
        sums_p1 = np.zeros((self._n1, self._n2), dtype=np.float64)
        sums_p2 = np.zeros((self._n1, self._n2), dtype=np.float64)
        counts = np.zeros((self._n1, self._n2), dtype=np.int32)

        # Use meshgrid to create all (i, j) index pairs
        i_idx = self._p1_action_to_idx  # [num_actions]
        j_idx = self._p2_action_to_idx  # [num_actions]
        ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")  # [5, 5] each
        np.add.at(sums_p1, (ii, jj), nn_payout_prediction[0])
        np.add.at(sums_p2, (ii, jj), nn_payout_prediction[1])
        np.add.at(counts, (ii, jj), 1)

        self._payout_p1 = sums_p1 / counts
        self._payout_p2 = sums_p2 / counts
        self._visits = np.zeros((self._n1, self._n2), dtype=np.float64)

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
            return self.p1_outcomes[outcome_idx]
        return self.p2_outcomes[outcome_idx]

    def action_to_outcome(self, player: int, action: int) -> int:
        """Convert action index to outcome index.

        Args:
            player: 1 for P1, 2 for P2
            action: Action index (0-4)

        Returns:
            Outcome index (0 to n-1)
        """
        if player == 1:
            return int(self._p1_action_to_idx[action])
        return int(self._p2_action_to_idx[action])

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
        self.p1_outcomes = sorted(set(self.p1_effective))
        self.p2_outcomes = sorted(set(self.p2_effective))
        self._n1 = len(self.p1_outcomes)
        self._n2 = len(self.p2_outcomes)

        self._p1_outcome_idx = {eff: i for i, eff in enumerate(self.p1_outcomes)}
        self._p2_outcome_idx = {eff: i for i, eff in enumerate(self.p2_outcomes)}

        num_actions = len(self.p1_effective)
        self._p1_action_to_idx = np.array(
            [self._p1_outcome_idx[self.p1_effective[a]] for a in range(num_actions)],
            dtype=np.int64,
        )
        self._p2_action_to_idx = np.array(
            [self._p2_outcome_idx[self.p2_effective[a]] for a in range(num_actions)],
            dtype=np.int64,
        )

        # Re-reduce priors with new effective mappings
        # Expand current priors first to get [5] arrays, then reduce
        expanded_p1 = expand_prior(self.prior_p1_reduced, self.p1_effective)
        expanded_p2 = expand_prior(self.prior_p2_reduced, self.p2_effective)
        self.prior_p1_reduced = reduce_prior(expanded_p1, self.p1_effective)
        self.prior_p2_reduced = reduce_prior(expanded_p2, self.p2_effective)

        # Update cached expanded priors
        self._expanded_prior_p1 = expand_prior(self.prior_p1_reduced, self.p1_effective)
        self._expanded_prior_p2 = expand_prior(self.prior_p2_reduced, self.p2_effective)

        # Reinitialize reduced statistics to match new structure
        self._payout_p1 = np.zeros((self._n1, self._n2), dtype=np.float64)
        self._payout_p2 = np.zeros((self._n1, self._n2), dtype=np.float64)
        self._visits = np.zeros((self._n1, self._n2), dtype=np.float64)
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
        """Get expanded prior policy for P1 [5] (cached)."""
        return self._expanded_prior_p1

    @prior_policy_p1.setter
    def prior_policy_p1(self, value: np.ndarray) -> None:
        """Set prior policy for P1 from [5] array."""
        self.prior_p1_reduced = reduce_prior(value, self.p1_effective)
        self._expanded_prior_p1 = expand_prior(self.prior_p1_reduced, self.p1_effective)

    @property
    def prior_policy_p2(self) -> np.ndarray:
        """Get expanded prior policy for P2 [5] (cached)."""
        return self._expanded_prior_p2

    @prior_policy_p2.setter
    def prior_policy_p2(self, value: np.ndarray) -> None:
        """Set prior policy for P2 from [5] array."""
        self.prior_p2_reduced = reduce_prior(value, self.p2_effective)
        self._expanded_prior_p2 = expand_prior(self.prior_p2_reduced, self.p2_effective)

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
        self._payout_p1 = np.ascontiguousarray(reduced[0], dtype=np.float64)
        self._payout_p2 = np.ascontiguousarray(reduced[1], dtype=np.float64)

    @property
    def action_visits(self) -> np.ndarray:
        """Get expanded visit counts [5, 5] for compatibility.

        This reconstructs the full matrix from reduced storage.
        Use sparingly - prefer reduced access for hot paths.
        """
        return self.get_expanded_visits()

    def backup(self, action_p1: int, action_p2: int, value: tuple[float, float]) -> None:
        """Update statistics after visiting a child node.

        O(1) operation - directly indexes into reduced matrices via Numba JIT.

        Args:
            action_p1: Player 1's action that led to child (0-4)
            action_p2: Player 2's action that led to child (0-4)
            value: Tuple of (p1_value, p2_value) representing returns for each player
        """
        idx1 = int(self._p1_action_to_idx[action_p1])
        idx2 = int(self._p2_action_to_idx[action_p2])
        p1_value, p2_value = value

        self._total_visits = backup_node(
            self._payout_p1,
            self._payout_p2,
            self._visits,
            idx1,
            idx2,
            p1_value,
            p2_value,
        )

    def get_expanded_payout_matrix(self) -> np.ndarray:
        """Expand reduced payout matrices to full [2, 5, 5] shape.

        Each action maps to its effective outcome's payout value.
        Equivalent actions get identical values (as expected by Nash computation).

        Returns:
            Payout matrix with shape [2, 5, 5].
        """
        result: np.ndarray = build_expanded_payout(
            self._payout_p1,
            self._payout_p2,
            self._p1_action_to_idx,
            self._p2_action_to_idx,
        )
        return result

    def get_expanded_visits(self) -> np.ndarray:
        """Expand reduced visit counts to full [5, 5] shape.

        Each action maps to its effective outcome's visit count.
        Equivalent actions get identical values.

        Returns:
            Visit count matrix with shape [5, 5].
        """
        result: np.ndarray = build_expanded_visits(
            self._visits,
            self._p1_action_to_idx,
            self._p2_action_to_idx,
        )
        return result

    def get_reduced_payout(self) -> np.ndarray:
        """Get reduced payout matrix [2, n1, n2].

        Returns the internal reduced representation without expansion.

        Returns:
            Payout matrix with shape [2, n1, n2].
        """
        reduced = np.zeros((2, self._n1, self._n2), dtype=np.float64)
        reduced[0] = self._payout_p1
        reduced[1] = self._payout_p2
        return reduced

    def get_reduced_visits(self) -> np.ndarray:
        """Get reduced visit counts [n1, n2].

        Returns the internal reduced representation without expansion.

        Returns:
            Visit counts with shape [n1, n2].
        """
        return self._visits.astype(np.int32)

    def compute_marginal_q_reduced(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute marginalized Q-values in reduced (outcome-indexed) space.

        Returns:
            Tuple of (q1, q2) where q1[i] is P1's Q-value for outcome i.
            Shapes are [n1] and [n2].
        """
        result: tuple[np.ndarray, np.ndarray] = compute_marginal_q_reduced(
            self._payout_p1,
            self._payout_p2,
            self.prior_p1_reduced,
            self.prior_p2_reduced,
        )
        return result

    def compute_marginal_visits_reduced(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute marginal visit counts in reduced (outcome-indexed) space.

        Returns:
            Tuple of (n1, n2) where n1[i] is visit count for outcome i.
            Shapes are [n1] and [n2].
        """
        result: tuple[np.ndarray, np.ndarray] = compute_marginal_visits_reduced(
            self._visits,
        )
        return result

    def compute_expected_value(self) -> tuple[float, float]:
        """Compute expected value under NN priors for both players.

        This is the NN's expected payoff: E[V] = π₁ᵀ · Payoff · π₂

        Returns:
            Tuple (v1, v2) of expected values for P1 and P2.
        """
        result: tuple[float, float] = compute_expected_value_reduced(
            self._payout_p1,
            self._payout_p2,
            self.prior_p1_reduced,
            self.prior_p2_reduced,
        )
        return result

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.total_visits}, "
            f"outcomes=({self._n1}x{self._n2}), "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
