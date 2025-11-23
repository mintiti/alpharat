"""MCTS Node implementation for simultaneous-move games.

This module implements the core node structure for Monte Carlo Tree Search
adapted for simultaneous-move games like PyRat. The node maintains payout
matrices instead of single Q-values to properly handle game-theoretic aspects.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MCTSNode:
    """MCTS node for simultaneous-move games.

    Maintains statistics as payout matrices to handle simultaneous actions
    from both players. All values are from Player 1's perspective in a
    zero-sum formulation (score_p1 - score_p2).

    Attributes:
        game_state: The game state this node represents
        is_terminal: Whether this is a terminal game state
        payout_matrix: Accumulated payouts for each action pair [p1_actions, p2_actions]
        action_visits: Visit counts for each action pair [p1_actions, p2_actions]
        prior_policy_p1: Neural network policy prior for player 1
        prior_policy_p2: Neural network policy prior for player 2
        children: Dictionary mapping action pairs to child nodes
        parent: Parent node in the search tree
        move_undo: MoveUndo object to reach this node from parent (None for root)
        depth: Depth of this node in the tree (0 for root)
        p1_mud_turns_remaining: Turns player 1 is stuck in mud
        p2_mud_turns_remaining: Turns player 2 is stuck in mud
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
        """Total number of visits to this node across all action pairs."""
        return int(np.sum(self.action_visits))

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    def backup(self, action_p1: int, action_p2: int, value: float) -> None:
        """Update statistics after visiting a child node.

        Uses incremental mean update formula: Q_new = Q_old + (G - Q_old) / (n + 1)

        Handles mud states by updating appropriate regions of the matrix:
        - No mud: Update single entry [action_p1, action_p2]
        - P1 in mud: Update entire column [:, action_p2]
        - P2 in mud: Update entire row [action_p1, :]
        - Both in mud: Update entire matrix

        Args:
            action_p1: Player 1's action that led to child
            action_p2: Player 2's action that led to child
            value: Return value from child (from P1's perspective, zero-sum)
        """
        # Determine which entries to update based on mud state
        if self.p1_mud_turns_remaining > 0 and self.p2_mud_turns_remaining > 0:
            # Both players stuck: all actions lead to same outcome
            # Update entire matrix
            self._update_region(slice(None), slice(None), value)

        elif self.p1_mud_turns_remaining > 0:
            # P1 stuck: P1's action doesn't matter, update entire column for P2's action
            self._update_region(slice(None), action_p2, value)

        elif self.p2_mud_turns_remaining > 0:
            # P2 stuck: P2's action doesn't matter, update entire row for P1's action
            self._update_region(action_p1, slice(None), value)

        else:
            # No mud: update single action pair
            self._update_region(action_p1, action_p2, value)

    def _update_region(
        self,
        row_idx: int | slice,
        col_idx: int | slice,
        value: float,
    ) -> None:
        """Update a region of the payout matrix using incremental mean formula.

        Args:
            row_idx: Row index or slice to update
            col_idx: Column index or slice to update
            value: New value to incorporate
        """
        # Get current Q-values and visit counts for the region
        current_q = self.payout_matrix[row_idx, col_idx]
        current_n = self.action_visits[row_idx, col_idx]

        # Incremental mean update: Q_new = Q_old + (G - Q_old) / (n + 1)
        new_q = current_q + (value - current_q) / (current_n + 1)

        # Update the payout matrix and increment visit counts
        self.payout_matrix[row_idx, col_idx] = new_q
        self.action_visits[row_idx, col_idx] = current_n + 1

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MCTSNode(visits={self.total_visits}, "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
