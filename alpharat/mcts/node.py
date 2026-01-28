"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. Statistics are indexed by
unique effective outcomes rather than raw actions, giving O(1) backup.

Each player maintains independent Q-values and visit counts (decoupled UCT).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpharat.mcts.numba_ops import (
    backup_node_scalar,
)
from alpharat.mcts.reduction import (
    expand_prior,
    reduce_prior,
)


class MCTSNode:
    """MCTS node for simultaneous-move games with O(1) backup.

    Uses decoupled UCT: each player maintains independent Q-values and visit counts
    indexed by unique effective outcomes. This makes backup O(1) and simplifies
    the algorithm compared to joint payout matrices.

    The node stores:
    - Reduced priors [n1], [n2] where n = count of unique outcomes
    - Marginal Q-values [n1], [n2] for each player (decoupled)
    - Marginal visit counts [n1], [n2] for each player
    - Scalar value estimates (init_v1, init_v2) from NN
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

        # Store NN's scalar value estimates
        self._init_v1 = nn_value_p1
        self._init_v2 = nn_value_p2

        # Decoupled UCT: marginal Q-values and visit counts per outcome
        # Q initialized to NN value, N initialized to 0
        self._q1 = np.full(self._n1, nn_value_p1, dtype=np.float64)
        self._q2 = np.full(self._n2, nn_value_p2, dtype=np.float64)
        self._n1_visits = np.zeros(self._n1, dtype=np.float64)
        self._n2_visits = np.zeros(self._n2, dtype=np.float64)

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

    @property
    def init_v1(self) -> float:
        """NN's initial scalar value estimate for P1."""
        return self._init_v1

    @property
    def init_v2(self) -> float:
        """NN's initial scalar value estimate for P2."""
        return self._init_v2

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

        # Reinitialize Q and N arrays for new structure
        self._q1 = np.full(self._n1, self._init_v1, dtype=np.float64)
        self._q2 = np.full(self._n2, self._init_v2, dtype=np.float64)
        self._n1_visits = np.zeros(self._n1, dtype=np.float64)
        self._n2_visits = np.zeros(self._n2, dtype=np.float64)
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

    def set_init_values(self, v1: float, v2: float) -> None:
        """Set initial value estimates and reinitialize Q arrays.

        Used by MCTSTree._init_root_priors() to set NN predictions.

        Args:
            v1: Value estimate for P1
            v2: Value estimate for P2
        """
        self._init_v1 = v1
        self._init_v2 = v2
        self._q1 = np.full(self._n1, v1, dtype=np.float64)
        self._q2 = np.full(self._n2, v2, dtype=np.float64)

    def backup(self, action_p1: int, action_p2: int, value: tuple[float, float]) -> None:
        """Update statistics after visiting a child node.

        O(1) operation - independently updates marginal Q1[idx1] and Q2[idx2].

        Args:
            action_p1: Player 1's action that led to child (0-4)
            action_p2: Player 2's action that led to child (0-4)
            value: Tuple of (p1_value, p2_value) representing returns for each player
        """
        idx1 = int(self._p1_action_to_idx[action_p1])
        idx2 = int(self._p2_action_to_idx[action_p2])
        p1_value, p2_value = value

        self._total_visits = backup_node_scalar(
            self._q1,
            self._q2,
            self._n1_visits,
            self._n2_visits,
            idx1,
            idx2,
            p1_value,
            p2_value,
        )

    def get_q_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Get Q-values in reduced (outcome-indexed) space.

        Returns:
            Tuple of (q1, q2) where q1[i] is P1's Q-value for outcome i.
            Shapes are [n1] and [n2].
        """
        return self._q1.copy(), self._q2.copy()

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
