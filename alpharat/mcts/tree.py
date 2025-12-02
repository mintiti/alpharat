"""MCTS Tree management for simultaneous-move games.

This module handles tree navigation and game state synchronization using
the make_move/unmake_move pattern for efficiency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pyrat_engine.core.types import Direction

from alpharat.mcts.node import MCTSNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrat_engine.core.game import PyRat


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

    def __init__(
        self,
        game: PyRat,
        root: MCTSNode,
        gamma: float = 1.0,
        predict_fn: Callable[[Any], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ):
        """Initialize MCTS tree.

        Args:
            game: PyRat game instance
            root: Root node of the tree
            gamma: Discount factor (1.0 = no discounting)
            predict_fn: Callable taking an observation and returning
                        (prior_p1, prior_p2, payout_matrix) as numpy arrays.
        """
        self.game = game
        self.root = root
        self.gamma = gamma
        self._predict_fn = predict_fn

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
        p1_score_before, p2_score_before = self.game.player1_score, self.game.player2_score

        # Make the move and get mud information from MoveUndo
        action_pair = (action_p1, action_p2)
        move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
        p1_mud = move_undo.p1_mud
        p2_mud = move_undo.p2_mud

        # Check if child already exists
        if action_pair in node.children:
            # Child exists, just navigate to it and refresh metadata
            child = node.children[action_pair]
            child.move_undo = move_undo
            child.parent_action = action_pair
        else:
            # Create new child (expansion)
            child = self._create_child(
                parent=node,
                move_undo=move_undo,
                action_p1=action_p1,
                action_p2=action_p2,
                p1_mud=p1_mud,
                p2_mud=p2_mud,
            )
            node.children[action_pair] = child

        # Calculate reward (zero-sum: p1_delta - p2_delta)
        p1_score_after, p2_score_after = self.game.player1_score, self.game.player2_score
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
        forward_start = max(common_depth, 1)
        for i in range(forward_start, len(target_path)):
            child = target_path[i]
            if child.parent_action is None:
                msg = "Cannot navigate: child missing parent_action metadata"
                raise RuntimeError(msg)

            action_p1, action_p2 = child.parent_action
            move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
            child.move_undo = move_undo

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

    def _create_child(
        self,
        parent: MCTSNode,
        move_undo: object,
        action_p1: int,
        action_p2: int,
        p1_mud: int,
        p2_mud: int,
    ) -> MCTSNode:
        """Create a new child node.

        Args:
            parent: Parent node
            move_undo: MoveUndo object from the move

        Returns:
            New child node
        """
        prior_p1, prior_p2, nn_payout = self._predict()

        child = MCTSNode(
            game_state=None,  # Don't store game state
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=parent,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
            move_undo=move_undo,
            parent_action=(action_p1, action_p2),
        )

        return child

    def _predict(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run policy/value network to obtain priors and payout prediction."""
        if self._predict_fn is None:
            # Fallback to uniform priors and zero payout for testing/sanity
            prior_p1 = np.ones(5) / 5
            prior_p2 = np.ones(5) / 5
            payout = np.zeros((5, 5))
            return prior_p1, prior_p2, payout

        observation = self._get_observation()
        return self._predict_fn(observation)

    def _get_observation(self) -> Any:
        """Obtain observation to feed the predictor.

        Tries to use the underlying core state's get_observation if available;
        otherwise returns the game object itself for custom predictors.
        """
        return self.game.get_observation(is_player_one=True)
