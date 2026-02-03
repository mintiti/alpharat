"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. Statistics are indexed by
unique effective outcomes rather than raw actions, giving O(1) backup.

Each node stores scalar values (v1, v2) like LC0/KataGo. Marginal Q-values
for PUCT selection are computed on-the-fly from children.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpharat.mcts.reduction import (
    expand_prior,
    reduce_prior,
)


class MCTSNode:
    """MCTS node for simultaneous-move games (LC0-style scalar values).

    Each node stores scalar values (v1, v2) representing expected returns.
    Marginal Q-values for PUCT selection are computed on-the-fly from children,
    following the LC0/KataGo pattern where Q(action) = child.value.

    The node stores:
    - Reduced priors [n1], [n2] where n = count of unique outcomes
    - Scalar values (v1, v2) - NN estimate initially, updated via backup
    - Marginal visit counts [n1], [n2] for policy output
    - Edge visits (_visits) - how many times parent backed up through this node
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
        nn_value_p1: float,
        nn_value_p2: float,
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
            nn_value_p1: NN's scalar value estimate for P1
            nn_value_p2: NN's scalar value estimate for P2
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

        # Node values: NN estimate initially, updated via backup
        # V = expected future value from this position
        self._v1 = float(nn_value_p1)
        self._v2 = float(nn_value_p2)

        # Edge rewards: reward collected when transitioning to this node from parent
        # Set by make_move_from(), used to compute Q = reward + gamma * V
        self._edge_r1: float = 0.0
        self._edge_r2: float = 0.0

        # Edge visits: how many times parent backed up through this node
        self._visits: int = 0

        # Marginal visit counts for policy output (visit-proportional policy)
        self._n1_visits = np.zeros(self._n1, dtype=np.float64)
        self._n2_visits = np.zeros(self._n2, dtype=np.float64)

        # Total visits through this node (for incremental averaging)
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

    @property
    def v1(self) -> float:
        """Scalar value estimate for P1 (NN initial, updated via backup)."""
        return self._v1

    @property
    def v2(self) -> float:
        """Scalar value estimate for P2 (NN initial, updated via backup)."""
        return self._v2

    @property
    def visits(self) -> int:
        """Edge visits: how many times parent backed up through this node."""
        return self._visits

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

        # Reinitialize visit arrays for new structure (node values stay as-is)
        self._n1_visits = np.zeros(self._n1, dtype=np.float64)
        self._n2_visits = np.zeros(self._n2, dtype=np.float64)
        self._total_visits = 0
        self._visits = 0
        self._edge_r1 = 0.0
        self._edge_r2 = 0.0

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

    def set_values(self, v1: float, v2: float) -> None:
        """Set scalar value estimates.

        Used by MCTSTree._init_root_priors() to set NN predictions.

        Args:
            v1: Value estimate for P1
            v2: Value estimate for P2
        """
        self._v1 = float(v1)
        self._v2 = float(v2)

    def backup(
        self,
        action_p1: int,
        action_p2: int,
        q_value: tuple[float, float],
    ) -> None:
        """Update statistics after visiting a child node.

        O(1) operation using incremental averaging. Updates this node's value
        and marginal visit counts. Child node values are updated separately
        in tree.backup() to avoid double-counting.

        Args:
            action_p1: Player 1's action that led to child (0-4)
            action_p2: Player 2's action that led to child (0-4)
            q_value: Discounted return Q = r + gamma * child_value (updates self.V)
        """
        idx1 = int(self._p1_action_to_idx[action_p1])
        idx2 = int(self._p2_action_to_idx[action_p2])

        # Increment child's edge visit count (for weighting in get_q_values)
        child_key = (idx1, idx2)
        if child_key in self.children:
            child = self.children[child_key]
            child._visits += 1

        # Update marginal visit counts (for policy output)
        self._n1_visits[idx1] += 1
        self._n2_visits[idx2] += 1

        # Update this node's value with the observed return
        # V = E[Q] = average discounted return from this position
        self._total_visits += 1
        self._v1 += (q_value[0] - self._v1) / self._total_visits
        self._v2 += (q_value[1] - self._v2) / self._total_visits

    def get_q_values(self, gamma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Compute marginal Q-values from children (LC0-style).

        Q = reward + gamma * V, where V is the child's value.
        Q1(i) = weighted average of (r1 + gamma * child.v1) for children (i, *).
        Q2(j) = weighted average of (r2 + gamma * child.v2) for children (*, j).
        Weight = child's edge visits.
        FPU (first play urgency) = this node's v1/v2 for unvisited outcomes.

        Args:
            gamma: Discount factor for computing Q = r + gamma * V.

        Returns:
            Tuple of (q1, q2) where q1[i] is P1's Q-value for outcome i.
            Shapes are [n1] and [n2].
        """
        # FPU = node's own value for unvisited outcomes
        q1 = np.full(self._n1, self._v1, dtype=np.float64)
        q2 = np.full(self._n2, self._v2, dtype=np.float64)

        if not self.children:
            return q1, q2

        # Accumulate weighted Q values: Q = r + gamma * V
        w1 = np.zeros(self._n1, dtype=np.float64)
        w2 = np.zeros(self._n2, dtype=np.float64)
        n1_sum = np.zeros(self._n1, dtype=np.float64)
        n2_sum = np.zeros(self._n2, dtype=np.float64)

        for (i, j), child in self.children.items():
            if child._visits > 0:
                q1_child = child._edge_r1 + gamma * child._v1
                q2_child = child._edge_r2 + gamma * child._v2
                w1[i] += child._visits * q1_child
                w2[j] += child._visits * q2_child
                n1_sum[i] += child._visits
                n2_sum[j] += child._visits

        # Replace FPU where we have data
        mask1 = n1_sum > 0
        mask2 = n2_sum > 0
        q1[mask1] = w1[mask1] / n1_sum[mask1]
        q2[mask2] = w2[mask2] / n2_sum[mask2]

        return q1, q2

    def get_visit_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Get visit counts in reduced (outcome-indexed) space.

        Returns:
            Tuple of (n1, n2) where n1[i] is visit count for outcome i.
            Shapes are [n1] and [n2].
        """
        return self._n1_visits.copy(), self._n2_visits.copy()

    def get_marginal_visits_expanded(self) -> tuple[np.ndarray, np.ndarray]:
        """Get marginal visit counts expanded to [5] action space.

        Returns:
            Tuple of (n1, n2) with shape [5] each.
        """
        num_actions = len(self.p1_effective)
        n1_expanded = np.zeros(num_actions, dtype=np.float64)
        n2_expanded = np.zeros(num_actions, dtype=np.float64)

        # Only canonical actions get visit counts
        for i, outcome in enumerate(self.p1_outcomes):
            n1_expanded[outcome] = self._n1_visits[i]
        for i, outcome in enumerate(self.p2_outcomes):
            n2_expanded[outcome] = self._n2_visits[i]

        return n1_expanded, n2_expanded

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.total_visits}, "
            f"outcomes=({self._n1}x{self._n2}), "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
