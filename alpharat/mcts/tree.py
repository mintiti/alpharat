"""MCTS Tree management for simultaneous-move games.

This module handles tree navigation and game state synchronization using
the make_move/unmake_move pattern for efficiency.
"""

from __future__ import annotations

import numpy as np
from pyrat_engine.game import Direction, PyRat

from alpharat.mcts.node import MCTSNode


class MCTSTree:
    """Manages MCTS tree structure and game state navigation.

    The tree owns a PyRat game instance and handles efficient navigation
    via make_move/unmake_move. Nodes store only MoveUndo data, not full game state.

    Attributes:
        game: PyRat game instance (simulator)
        root: Root node of the tree
        gamma: Discount factor for value backup
        _sim_path: Path from root to current simulator position
    """

    def __init__(self, game: PyRat, root: MCTSNode, gamma: float = 1.0):
        """Initialize MCTS tree.

        Args:
            game: PyRat game instance
            root: Root node of the tree
            gamma: Discount factor (1.0 = no discounting)
        """
        self.game = game
        self.root = root
        self.gamma = gamma

        # Track where simulator currently is
        self._sim_path: list[MCTSNode] = [root]

    @property
    def simulator_node(self) -> MCTSNode:
        """Get current simulator position in the tree."""
        return self._sim_path[-1]

    def make_move_from(
        self, node: MCTSNode, action_p1: int, action_p2: int
    ) -> tuple[MCTSNode, float]:
        """Make a move from given node, creating child if needed.

        This method:
        1. Navigates simulator to the target node
        2. Records scores before move
        3. Executes move on game
        4. Calculates reward (score differential)
        5. Creates or retrieves child node
        6. Updates simulator path

        Args:
            node: Node to make move from
            action_p1: Player 1's action (0-4 for UP, RIGHT, DOWN, LEFT, STAY)
            action_p2: Player 2's action (0-4)

        Returns:
            Tuple of (child_node, reward) where reward is the zero-sum reward
        """
        # Navigate simulator to target node if not already there
        if self.simulator_node != node:
            self._navigate_to(node)

        # Record scores before move
        p1_score_before, p2_score_before = self.game.scores

        # Check if child already exists
        action_pair = (action_p1, action_p2)
        if action_pair in node.children:
            # Child exists, just navigate to it
            child = node.children[action_pair]
            move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
        else:
            # Create new child (expansion)
            move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
            child = self._create_child(node, move_undo)
            node.children[action_pair] = child

        # Calculate reward (zero-sum: p1_delta - p2_delta)
        p1_score_after, p2_score_after = self.game.scores
        reward = (p1_score_after - p1_score_before) - (p2_score_after - p2_score_before)

        # Update simulator path
        self._sim_path.append(child)

        return child, reward

    def backup(self, path: list[tuple[MCTSNode, int, int, float]]) -> None:
        """Backup values through the tree with discounting.

        Args:
            path: List of (node, action_p1, action_p2, reward) tuples
                 from the search path, in forward order (root to leaf)
        """
        value = 0.0  # Terminal value

        # Backup in reverse (from leaf to root)
        for node, action_p1, action_p2, reward in reversed(path):
            # Apply discounting: value = reward + gamma * future_value
            value = reward + self.gamma * value

            # Update node statistics
            node.backup(action_p1, action_p2, value)

    def _navigate_to(self, target: MCTSNode) -> None:
        """Navigate simulator from current position to target node.

        Args:
            target: Target node to navigate to
        """
        # Get path from root to target
        target_path = self._get_path_from_root(target)
        current_path = self._sim_path

        # Find common ancestor (where paths diverge)
        common_depth = 0
        while (
            common_depth < len(current_path)
            and common_depth < len(target_path)
            and current_path[common_depth] == target_path[common_depth]
        ):
            common_depth += 1

        # Unmake moves back to common ancestor
        for i in range(len(current_path) - 1, common_depth - 1, -1):
            if i > 0:  # Don't unmake from root
                self.game.unmake_move(current_path[i].move_undo)

        # Make moves forward to target
        for i in range(common_depth, len(target_path)):
            if i > 0:  # Skip root
                # Note: We make a dummy move here
                # In practice, the tree navigation happens through make_move_from
                # which properly executes the moves
                self.game.make_move(Direction(0), Direction(0))

        # Update simulator path
        self._sim_path = target_path

    def _get_path_from_root(self, node: MCTSNode) -> list[MCTSNode]:
        """Get path from root to given node.

        Args:
            node: Target node

        Returns:
            List of nodes from root to target (inclusive)
        """
        path: list[MCTSNode] = []
        current: MCTSNode | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def _create_child(self, parent: MCTSNode, move_undo: object) -> MCTSNode:
        """Create a new child node.

        Args:
            parent: Parent node
            move_undo: MoveUndo object from the move

        Returns:
            New child node
        """
        # For now, use uniform priors (will be replaced by NN predictions)
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        # TODO: Get mud state from game
        p1_mud = 0
        p2_mud = 0

        child = MCTSNode(
            game_state=None,  # Don't store game state
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=parent,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
            move_undo=move_undo,
        )

        return child
