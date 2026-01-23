"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. The node maintains payout
matrices instead of single Q-values to properly handle game-theoretic aspects.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpharat.mcts.equivalence import compute_effective_total_visits


class MCTSNode:
    """MCTS node for simultaneous-move games.

    Maintains statistics as separate payout matrices for each player to handle
    simultaneous actions. Shape is [2, p1_actions, p2_actions] where index 0 is
    P1's payoffs and index 1 is P2's payoffs.

    Attributes:
        game_state: The game state this node represents
        is_terminal: Whether this is a terminal game state
        payout_matrix: Accumulated payouts [2, p1_actions, p2_actions]
        action_visits: Visit counts for each action pair [p1_actions, p2_actions]
        prior_policy_p1: Neural network policy prior for player 1
        prior_policy_p2: Neural network policy prior for player 2
        children: Dictionary mapping action pairs to child nodes
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
            prior_policy_p1: Policy prior for player 1 [num_actions_p1]
            prior_policy_p2: Policy prior for player 2 [num_actions_p2]
            nn_payout_prediction: Initial payout matrix from NN [num_actions_p1, num_actions_p2]
            parent: Parent node (None for root)
            p1_mud_turns_remaining: Turns P1 is stuck in mud
            p2_mud_turns_remaining: Turns P2 is stuck in mud
            move_undo: MoveUndo object to reach this node from parent (None for root)
            parent_action: Action pair taken from parent to reach this node
            p1_effective: Maps each P1 action to its effective action (blocked -> STAY)
            p2_effective: Maps each P2 action to its effective action (blocked -> STAY)
        """
        self.game_state = game_state
        self.is_terminal = False  # Will be set based on game_state
        self.parent = parent
        self.move_undo = move_undo
        # Action pair taken from parent to reach this node (None for root)
        self.parent_action = parent_action

        # Calculate depth from parent
        self.depth: int = 0 if parent is None else parent.depth + 1

        # Mud state
        self.p1_mud_turns_remaining = p1_mud_turns_remaining
        self.p2_mud_turns_remaining = p2_mud_turns_remaining

        # Action equivalence: maps each action to what actually happens
        # Mud overrides any provided effective mapping - all actions become STAY
        num_actions = len(prior_policy_p1)
        stay_action = num_actions - 1  # STAY is always the last action

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

        # Neural network priors
        self.prior_policy_p1 = prior_policy_p1
        self.prior_policy_p2 = prior_policy_p2

        # Initialize statistics with NN predictions
        num_actions_p1 = len(prior_policy_p1)
        num_actions_p2 = len(prior_policy_p2)

        self.payout_matrix = nn_payout_prediction.copy()
        self.action_visits = np.zeros((num_actions_p1, num_actions_p2), dtype=np.int32)

        # Tree structure
        self.children: dict[tuple[int, int], MCTSNode] = {}

    @property
    def total_visits(self) -> int:
        """Total number of simulations through this node (equivalence-aware)."""
        return compute_effective_total_visits(
            self.action_visits, self.p1_effective, self.p2_effective
        )

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    def backup(self, action_p1: int, action_p2: int, value: tuple[float, float]) -> None:
        """Update statistics after visiting a child node.

        Uses incremental mean update formula: Q_new = Q_old + (G - Q_old) / (n + 1)

        Updates all equivalent action pairs: actions that map to the same effective
        action (due to walls, edges, or mud) share statistics.

        Args:
            action_p1: Player 1's action that led to child
            action_p2: Player 2's action that led to child
            value: Tuple of (p1_value, p2_value) representing returns for each player
        """
        # Find all actions equivalent to the given actions
        p1_equiv = self._get_equivalent_actions(action_p1, player=1)
        p2_equiv = self._get_equivalent_actions(action_p2, player=2)

        # Update the rectangular region of equivalent action pairs for both players
        self._update_region(p1_equiv, p2_equiv, value)

    def _get_equivalent_actions(self, action: int, player: int) -> list[int]:
        """Get all actions equivalent to the given action for a player.

        Args:
            action: The action index
            player: 1 for player 1, 2 for player 2

        Returns:
            List of action indices that share the same effective action
        """
        effective_map = self.p1_effective if player == 1 else self.p2_effective
        target_effective = effective_map[action]
        return [a for a, e in enumerate(effective_map) if e == target_effective]

    def _update_region(
        self,
        row_idx: int | slice | list[int],
        col_idx: int | slice | list[int],
        value: tuple[float, float],
    ) -> None:
        """Update a region of the payout matrix using incremental mean formula.

        Args:
            row_idx: Row index, slice, or list of indices to update
            col_idx: Column index, slice, or list of indices to update
            value: Tuple of (p1_value, p2_value) to incorporate
        """
        p1_value, p2_value = value

        # Build the index for numpy array access (for the action dimensions)
        if isinstance(row_idx, list) and isinstance(col_idx, list):
            # Both are lists - use np.ix_ for outer product indexing
            action_idx: tuple[np.ndarray, ...] | tuple[int | slice | list[int], ...] = np.ix_(
                row_idx, col_idx
            )
        else:
            # At least one is int or slice - use tuple directly
            action_idx = (row_idx, col_idx)

        # Get current visit counts for the region
        current_n = self.action_visits[action_idx]

        # Update P1's payout matrix (index 0)
        current_q_p1 = self.payout_matrix[(0,) + action_idx]
        new_q_p1 = current_q_p1 + (p1_value - current_q_p1) / (current_n + 1)
        self.payout_matrix[(0,) + action_idx] = new_q_p1

        # Update P2's payout matrix (index 1)
        current_q_p2 = self.payout_matrix[(1,) + action_idx]
        new_q_p2 = current_q_p2 + (p2_value - current_q_p2) / (current_n + 1)
        self.payout_matrix[(1,) + action_idx] = new_q_p2

        # Increment visit counts
        self.action_visits[action_idx] = current_n + 1

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.total_visits}, "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
