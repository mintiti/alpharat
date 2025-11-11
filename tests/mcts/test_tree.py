"""Tests for MCTS Tree implementation."""

import numpy as np
import pytest
from pyrat_engine.game import PyRat

from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree


@pytest.fixture
def game() -> PyRat:
    """Create a small PyRat game for testing."""
    return PyRat(width=5, height=5, cheese_count=3, seed=42)


@pytest.fixture
def root_node() -> MCTSNode:
    """Create a root node with uniform priors."""
    prior_p1 = np.ones(5) / 5
    prior_p2 = np.ones(5) / 5
    nn_payout = np.zeros((5, 5))

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
        move_undo=None,
    )


@pytest.fixture
def tree(game: PyRat, root_node: MCTSNode) -> MCTSTree:
    """Create an MCTS tree for testing."""
    return MCTSTree(game=game, root=root_node, gamma=1.0)


class TestTreeInitialization:
    """Tests for tree initialization."""

    def test_tree_starts_at_root(self, tree: MCTSTree, root_node: MCTSNode) -> None:
        """Tree should start with simulator at root."""
        assert tree.root == root_node
        assert tree.simulator_node == root_node
        assert len(tree._sim_path) == 1
        assert tree._sim_path[0] == root_node

    def test_root_has_zero_depth(self, root_node: MCTSNode) -> None:
        """Root node should have depth 0."""
        assert root_node.depth == 0
        assert root_node.parent is None
        assert root_node.move_undo is None


class TestMakeMoveFrom:
    """Tests for making moves and expanding the tree."""

    def test_make_move_from_root(self, tree: MCTSTree) -> None:
        """Make a move from root should create child and advance simulator."""
        # Record initial game state
        initial_p1_pos = tree.game.player1_pos
        initial_p2_pos = tree.game.player2_pos

        # Make move from root
        child, reward = tree.make_move_from(tree.root, action_p1=1, action_p2=1)

        # Child should be created
        assert (1, 1) in tree.root.children
        assert tree.root.children[(1, 1)] == child

        # Child should have correct properties
        assert child.parent == tree.root
        assert child.depth == 1
        assert child.move_undo is not None

        # Simulator should be at child
        assert tree.simulator_node == child
        assert len(tree._sim_path) == 2
        assert tree._sim_path == [tree.root, child]

        # Game state should have changed
        assert (
            tree.game.player1_pos != initial_p1_pos
            or tree.game.player2_pos != initial_p2_pos
            or tree.game.turn == 1
        )

    def test_reward_calculation_no_cheese(self, tree: MCTSTree) -> None:
        """Reward should be 0 when no cheese is collected."""
        # Make a move that doesn't collect cheese
        child, reward = tree.make_move_from(tree.root, action_p1=4, action_p2=4)  # STAY

        # No cheese collected, reward should be 0
        assert reward == pytest.approx(0.0)

    def test_make_move_from_child_already_there(self, tree: MCTSTree) -> None:
        """Making move when simulator is already at node should not navigate."""
        # First move
        child1, _ = tree.make_move_from(tree.root, action_p1=0, action_p2=0)

        # Simulator at child1, make another move from child1
        child2, _ = tree.make_move_from(child1, action_p1=1, action_p2=1)

        # Should create grandchild
        assert child2.parent == child1
        assert child2.depth == 2

        # Simulator path should be root → child1 → child2
        assert tree._sim_path == [tree.root, child1, child2]
        assert tree.simulator_node == child2

    def test_make_move_from_sibling_requires_navigation(self, tree: MCTSTree) -> None:
        """Making move from sibling should navigate back and forth."""
        # Create first child
        child1, _ = tree.make_move_from(tree.root, action_p1=0, action_p2=0)
        assert tree.simulator_node == child1

        # Now make move from root (sibling of child1)
        # This requires navigating back to root
        child2, _ = tree.make_move_from(tree.root, action_p1=1, action_p2=1)

        # Should have two children of root
        assert (0, 0) in tree.root.children
        assert (1, 1) in tree.root.children

        # Simulator should be at child2
        assert tree.simulator_node == child2
        assert tree._sim_path == [tree.root, child2]

    def test_child_reuse_when_already_exists(self, tree: MCTSTree) -> None:
        """Making move to existing child should reuse it, not create new one."""
        # Create child
        child1, _ = tree.make_move_from(tree.root, action_p1=0, action_p2=0)

        # Navigate back to root
        tree.make_move_from(tree.root, action_p1=1, action_p2=1)

        # Navigate to (0,0) again - should reuse child1
        tree.make_move_from(tree.root, action_p1=0, action_p2=0)

        # Should still be the same child
        assert tree.simulator_node == child1
        assert len(tree.root.children) == 2  # Only 2 children total


class TestNavigation:
    """Tests for tree navigation logic."""

    def test_navigate_to_deep_node(self, tree: MCTSTree) -> None:
        """Navigate through multiple levels of the tree."""
        # Create a path: root → a → b → c
        a, _ = tree.make_move_from(tree.root, 0, 0)
        b, _ = tree.make_move_from(a, 1, 1)
        c, _ = tree.make_move_from(b, 2, 2)

        # Simulator should be at c
        assert tree.simulator_node == c
        assert tree._sim_path == [tree.root, a, b, c]
        assert c.depth == 3

    def test_navigate_between_distant_siblings(self, tree: MCTSTree) -> None:
        """Navigate from deep node to sibling branch."""
        # Create path: root → a → a1
        a, _ = tree.make_move_from(tree.root, 0, 0)
        a1, _ = tree.make_move_from(a, 1, 1)

        # Now navigate to root → b (sibling of a)
        b, _ = tree.make_move_from(tree.root, 2, 2)

        # Should have unwound to root, then created b
        assert tree.simulator_node == b
        assert tree._sim_path == [tree.root, b]
        assert b.parent == tree.root

    def test_navigate_to_cousin_node(self, tree: MCTSTree) -> None:
        """Navigate between cousins in the tree."""
        # Create left branch: root → a → a1
        a, _ = tree.make_move_from(tree.root, 0, 0)
        a1, _ = tree.make_move_from(a, 1, 1)

        # Create right branch: root → b → b1
        # This requires navigating from a1 back to root, then down to b, then to b1
        b, _ = tree.make_move_from(tree.root, 2, 2)
        b1, _ = tree.make_move_from(b, 3, 3)

        # Should be at b1
        assert tree.simulator_node == b1
        assert tree._sim_path == [tree.root, b, b1]


class TestBackup:
    """Tests for backing up values through the tree."""

    def test_backup_single_step(self, tree: MCTSTree) -> None:
        """Backup a single value."""
        # Create a child
        child, _ = tree.make_move_from(tree.root, 0, 0)

        # Backup path with one step
        path = [(tree.root, 0, 0, 1.0)]  # (node, action_p1, action_p2, reward)
        tree.backup(path)

        # Root should have updated Q-value
        assert tree.root.action_visits[0, 0] == 1
        # value = reward + gamma * 0 = 1.0 + 1.0 * 0 = 1.0
        assert tree.root.payout_matrix[0, 0] == pytest.approx(1.0)

    def test_backup_with_discount_factor(self, tree: MCTSTree) -> None:
        """Backup should apply gamma discounting."""
        # Create tree with gamma < 1
        tree_discounted = MCTSTree(tree.game, tree.root, gamma=0.9)

        # Create path: root → child
        child, _ = tree_discounted.make_move_from(tree.root, 0, 0)

        # Backup path with two steps
        # At child: reward = 0.5, future value = 0
        # At root: reward = 1.0, future value = 0.5
        path = [
            (tree.root, 0, 0, 1.0),  # reward at root
            (child, 1, 1, 0.5),  # reward at child
        ]
        tree_discounted.backup(path)

        # Child: value = 0.5 + 0.9 * 0 = 0.5
        assert child.payout_matrix[1, 1] == pytest.approx(0.5)

        # Root: value = 1.0 + 0.9 * 0.5 = 1.45
        assert tree.root.payout_matrix[0, 0] == pytest.approx(1.45)

    def test_backup_multiple_visits(self, tree: MCTSTree) -> None:
        """Multiple backups should use incremental mean."""
        # First backup
        path1 = [(tree.root, 0, 0, 10.0)]
        tree.backup(path1)
        assert tree.root.payout_matrix[0, 0] == pytest.approx(10.0)

        # Second backup to same action
        path2 = [(tree.root, 0, 0, 6.0)]
        tree.backup(path2)
        # Q = 10.0 + (6.0 - 10.0) / 2 = 8.0
        assert tree.root.payout_matrix[0, 0] == pytest.approx(8.0)
        assert tree.root.action_visits[0, 0] == 2


class TestMultipleExpansions:
    """Tests for expanding multiple children."""

    def test_expand_multiple_children_from_root(self, tree: MCTSTree) -> None:
        """Expand several children and verify consistency."""
        children = []
        for i in range(3):
            child, _ = tree.make_move_from(tree.root, i, i)
            children.append(child)

        # Root should have 3 children
        assert len(tree.root.children) == 3

        # All children should have correct parent and depth
        for i, child in enumerate(children):
            assert child.parent == tree.root
            assert child.depth == 1
            assert (i, i) in tree.root.children

        # Simulator should be at last child
        assert tree.simulator_node == children[-1]

    def test_expand_from_different_nodes(self, tree: MCTSTree) -> None:
        """Expand from various nodes and verify tree structure."""
        # Create root → a
        a, _ = tree.make_move_from(tree.root, 0, 0)

        # Create root → b (requires navigation)
        b, _ = tree.make_move_from(tree.root, 1, 1)

        # Create a → a1 (requires navigation back to a)
        a1, _ = tree.make_move_from(a, 2, 2)

        # Verify tree structure
        assert len(tree.root.children) == 2
        assert len(a.children) == 1
        assert len(b.children) == 0

        # Verify simulator is at a1
        assert tree.simulator_node == a1
