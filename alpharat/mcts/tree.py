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
        predict_fn: Callable[[Any], tuple[np.ndarray, np.ndarray, float, float]] | None = None,
    ):
        """Initialize MCTS tree.

        Args:
            game: PyRat game instance
            root: Root node of the tree
            gamma: Discount factor (1.0 = no discounting)
            predict_fn: Callable taking an observation and returning
                        (prior_p1, prior_p2, v1, v2) as numpy arrays and floats.
        """
        self.game = game
        self.root = root
        self.gamma = gamma
        self._predict_fn = predict_fn

        # Cache NN predictions keyed on game state (skip redundant forward passes)
        self._prediction_cache: dict[tuple, tuple[np.ndarray, np.ndarray, float, float]] = {}

        # Track where simulator currently is
        self._sim_path: list[MCTSNode] = [root]

        # Initialize root node with effective mappings first (needed for smart priors),
        # then NN predictions (root may have been created without game context)
        self._init_root_effective()
        self._init_root_priors()

    @property
    def simulator_node(self) -> MCTSNode:
        """Get current simulator position in the tree."""
        return self._sim_path[-1]

    def make_move_from(
        self, node: MCTSNode, action_p1: int, action_p2: int
    ) -> tuple[MCTSNode, tuple[float, float]]:
        """Make a move from given node, creating child if needed.

        This method:
        1. Navigates simulator to the target node
        2. Records scores before move
        3. Executes move on game
        4. Calculates reward (separate for each player)
        5. Creates or retrieves child node
        6. Updates simulator path

        Args:
            node: Node to make move from
            action_p1: Player 1's action (0-4 for UP, RIGHT, DOWN, LEFT, STAY)
            action_p2: Player 2's action (0-4)

        Returns:
            Tuple of (child_node, reward) where reward is (p1_delta, p2_delta)
        """
        # Navigate simulator to target node if not already there
        if self.simulator_node != node:
            self._navigate_to(node)

        # Record scores before move
        p1_score_before, p2_score_before = self.game.player1_score, self.game.player2_score

        # Make the move and get mud information from MoveUndo
        move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
        p1_mud = move_undo.p1_mud
        p2_mud = move_undo.p2_mud

        # Get effective actions and outcome indices
        effective_a1 = node.p1_effective[action_p1]
        effective_a2 = node.p2_effective[action_p2]
        outcome_i = node.action_to_outcome(1, action_p1)
        outcome_j = node.action_to_outcome(2, action_p2)
        outcome_pair = (outcome_i, outcome_j)

        # Check if child already exists for this outcome pair
        if outcome_pair in node.children:
            # Child exists, just navigate to it and refresh metadata
            child = node.children[outcome_pair]
            child.move_undo = move_undo
            # Keep effective actions as parent_action for navigation consistency
            child.parent_action = (effective_a1, effective_a2)
        else:
            # Create new child (expansion)
            child = self._create_child(
                parent=node,
                move_undo=move_undo,
                action_p1=effective_a1,
                action_p2=effective_a2,
                p1_mud=p1_mud,
                p2_mud=p2_mud,
            )
            node.children[outcome_pair] = child

        # Calculate separate rewards for each player
        p1_score_after, p2_score_after = self.game.player1_score, self.game.player2_score
        reward = (p1_score_after - p1_score_before, p2_score_after - p2_score_before)

        # Store edge reward on child (used for Q = r + gamma * V computation)
        child._edge_r1 = reward[0]
        child._edge_r2 = reward[1]

        # Update simulator path
        self._sim_path.append(child)

        return child, reward

    def backup(
        self,
        path: list[tuple[MCTSNode, int, int]],
        g: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Backup discounted returns through the tree.

        Uses edge_r stored on child nodes as the source of truth for rewards.

        Args:
            path: List of (node, action_p1, action_p2) tuples from the search
                 path, in forward order (root to leaf).
            g: Leaf value tuple (p1_value, p2_value) from NN estimate.
               Defaults to (0.0, 0.0) for terminal states.
        """
        child_value = g  # Start with leaf's expected return
        is_leaf = True  # First child in reversed path is the leaf

        # Backup in reverse (from leaf to root)
        for node, action_p1, action_p2 in reversed(path):
            idx1 = int(node._p1_action_to_idx[action_p1])
            idx2 = int(node._p2_action_to_idx[action_p2])
            child = node.children[(idx1, idx2)]

            # Only update the leaf's value here; intermediate nodes are
            # updated via node.backup() to avoid double-counting
            if is_leaf:
                child._total_visits += 1
                child._v1 += (child_value[0] - child._v1) / child._total_visits
                child._v2 += (child_value[1] - child._v2) / child._total_visits
                is_leaf = False

            # Q = edge_r + gamma * child_value (edge_r is source of truth)
            q_value = (
                child._edge_r1 + self.gamma * child_value[0],
                child._edge_r2 + self.gamma * child_value[1],
            )

            node.backup(action_p1, action_p2, q_value)
            child_value = q_value

    def advance_root(self, action_p1: int, action_p2: int) -> None:
        """Advance the tree's root to the child reached by the given actions.

        Used for tree reuse between turns: after a real game move is made,
        advance the root so accumulated statistics are preserved.

        Args:
            action_p1: Player 1's action (0-4)
            action_p2: Player 2's action (0-4)
        """
        # Map to outcome indices for child lookup
        outcome_i = self.root.action_to_outcome(1, action_p1)
        outcome_j = self.root.action_to_outcome(2, action_p2)
        outcome_pair = (outcome_i, outcome_j)

        # Get or create child
        if outcome_pair in self.root.children:
            new_root = self.root.children[outcome_pair]
            # If simulator is at current root, advance game state to stay in sync
            if self.simulator_node == self.root:
                move_undo = self.game.make_move(Direction(action_p1), Direction(action_p2))
                new_root.move_undo = move_undo
        else:
            # Child doesn't exist - create via make_move_from (handles game advancement)
            new_root, _ = self.make_move_from(self.root, action_p1, action_p2)

        # Update tree state (keep parent refs intact)
        self.root = new_root
        self._sim_path = [new_root]

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
            action_p1: Player 1's action that led here
            action_p2: Player 2's action that led here
            p1_mud: Player 1's mud turns remaining
            p2_mud: Player 2's mud turns remaining

        Returns:
            New child node with effective action mappings computed
        """
        # Compute effective action mappings from current game state
        # (simulator is at the child's state after make_move)
        p1_effective = self._compute_effective_actions(self.game.player1_position)
        p2_effective = self._compute_effective_actions(self.game.player2_position)

        # Get priors and values (smart uniform uses effective mappings)
        prior_p1, prior_p2, v1, v2 = self._predict(p1_effective, p2_effective)

        child = MCTSNode(
            game_state=None,  # Don't store game state
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=v1,
            nn_value_p2=v2,
            parent=parent,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
            move_undo=move_undo,
            parent_action=(action_p1, action_p2),
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

        # Check if this child represents a terminal state
        child.is_terminal = self._check_terminal()

        return child

    def _make_state_key(self) -> tuple:
        """Build a hashable key from the current game state for prediction caching.

        Captures everything that affects NN output: positions, cheese layout,
        mud status, and turn number.
        """
        game = self.game
        p1 = game.player1_position
        p2 = game.player2_position
        cheese = frozenset((c[0], c[1]) for c in game.cheese_positions())
        return (
            (p1[0], p1[1]),
            (p2[0], p2[1]),
            cheese,
            game.player1_mud_turns,
            game.player2_mud_turns,
            game.turn,
        )

    def _predict(
        self,
        p1_effective: list[int] | None = None,
        p2_effective: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Run policy/value network to obtain priors and value prediction.

        Uses a transposition cache to skip redundant NN calls when the same
        game state is reached via different move sequences.

        Args:
            p1_effective: Effective action mapping for P1 (used for smart uniform priors).
            p2_effective: Effective action mapping for P2 (used for smart uniform priors).

        Returns:
            Tuple of (prior_p1, prior_p2, v1, v2) where v1, v2 are scalar values.
        """
        if self._predict_fn is None:
            # Smart uniform: only spread probability over actions that do different things
            prior_p1 = self._smart_uniform_prior(p1_effective)
            prior_p2 = self._smart_uniform_prior(p2_effective)
            return prior_p1, prior_p2, 0.0, 0.0

        key = self._make_state_key()
        cached = self._prediction_cache.get(key)
        if cached is not None:
            # Return copies of arrays — node init mutates arrays via np.add.at
            return cached[0].copy(), cached[1].copy(), cached[2], cached[3]

        result = self._predict_fn(self._get_observation())
        self._prediction_cache[key] = result
        return result[0], result[1], result[2], result[3]

    def _smart_uniform_prior(self, effective: list[int] | None) -> np.ndarray:
        """Create uniform prior over unique effective actions only.

        Instead of spreading probability over all 5 actions (where blocked moves
        waste probability on STAY duplicates), only give probability to the
        canonical effective actions themselves.

        Args:
            effective: Mapping from action -> effective action. If None, returns
                      standard uniform over all 5 actions.

        Returns:
            Prior distribution with uniform weight on unique effective actions.
        """
        if effective is None:
            return np.ones(5) / 5

        # Put probability only on the effective actions themselves
        # e.g., if effective=[0,4,4,4,4], only actions 0 and 4 get probability
        unique_effective = set(effective)
        prior = np.zeros(5)
        for eff_action in unique_effective:
            prior[eff_action] = 1.0

        # Normalize
        prior /= prior.sum()
        return prior

    def _get_observation(self) -> Any:
        """Obtain observation to feed the predictor.

        Tries to use the underlying core state's get_observation if available;
        otherwise returns the game object itself for custom predictors.
        """
        return self.game.get_observation(is_player_one=True)

    def _init_root_priors(self) -> None:
        """Initialize root node with NN predictions.

        Called during tree initialization to ensure root has proper
        policy priors and value predictions from the neural network,
        consistent with how child nodes are created.

        Note: Must be called after _init_root_effective so smart uniform
        priors can use the effective action mappings.
        """
        prior_p1, prior_p2, v1, v2 = self._predict(self.root.p1_effective, self.root.p2_effective)
        self.root.prior_policy_p1 = prior_p1
        self.root.prior_policy_p2 = prior_p2
        self.root.set_values(v1, v2)

    def _init_root_effective(self) -> None:
        """Initialize effective action mappings for the root node.

        Called during tree initialization to ensure root has proper
        effective mappings based on the game's initial state.
        """
        # Only update if root has default identity mapping (no walls detected)
        # This preserves any mud-based mapping already applied
        p1_eff = None
        p2_eff = None

        if self.root.p1_mud_turns_remaining == 0:
            p1_eff = self._compute_effective_actions(self.game.player1_position)
        if self.root.p2_mud_turns_remaining == 0:
            p2_eff = self._compute_effective_actions(self.game.player2_position)

        if p1_eff is not None or p2_eff is not None:
            self.root.update_effective_actions(p1_eff, p2_eff)

    def _compute_effective_actions(self, position: Any) -> list[int]:
        """Compute effective action mapping for a player at given position.

        Uses get_valid_moves to determine which actions result in actual movement.
        Actions blocked by walls/edges map to STAY (action 4).

        Args:
            position: The player's current position (Coordinates type from pyrat)

        Returns:
            List of 5 ints mapping each action to its effective action.
            Valid moves map to themselves, blocked moves map to STAY.
        """
        stay_action = 4
        valid_moves = set(self.game.get_valid_moves(position))  # type: ignore[attr-defined]

        effective = []
        for action in range(5):
            if action == stay_action:
                # STAY always maps to itself
                effective.append(stay_action)
            elif action in valid_moves:
                # Valid movement maps to itself
                effective.append(action)
            else:
                # Blocked movement is equivalent to STAY
                effective.append(stay_action)

        return effective

    def _check_terminal(self) -> bool:
        """Check if current game state is terminal.

        Terminal conditions:
        - Turn limit reached (turn >= max_turns)
        - All cheese collected
        - One player has majority of cheese (> total/2)

        Returns:
            True if game is over, False otherwise.
        """
        # Turn limit reached
        if self.game.turn >= self.game.max_turns:
            return True

        # All cheese collected
        remaining = len(self.game.cheese_positions())
        if remaining == 0:
            return True

        # Majority win check
        total = self.game.player1_score + self.game.player2_score + remaining
        return self.game.player1_score > total / 2 or self.game.player2_score > total / 2
