"""Tests for MCTS Tree implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pyrat_engine.core.game import PyRat

from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeMoveUndo:
    """Undo token storing prior positions/scores/turn for FakeGame.

    Matches the public API of pyrat_engine.core.game.MoveUndo.
    """

    def __init__(
        self,
        p1_pos: tuple[int, int],
        p2_pos: tuple[int, int],
        p1_score: float,
        p2_score: float,
        turn: int,
        token: int,
        p1_mud: int = 0,
        p2_mud: int = 0,
    ) -> None:
        self.p1_pos = p1_pos
        self.p2_pos = p2_pos
        self.p1_score = p1_score
        self.p2_score = p2_score
        self.turn = turn
        self.token = token
        self.p1_mud = p1_mud
        self.p2_mud = p2_mud


class FakeGame:
    """Deterministic game stub to test tree navigation/state sync.

    Matches the public API of pyrat_engine.core.game.PyRat.
    """

    # Y-up coordinate system: UP increases Y, DOWN decreases Y
    _DELTAS = {
        0: (0, 1),  # UP
        1: (1, 0),  # RIGHT
        2: (0, -1),  # DOWN
        3: (-1, 0),  # LEFT
        4: (0, 0),  # STAY
    }

    def __init__(self) -> None:
        self.player1_position = (5, 5)
        self.player2_position = (5, 5)
        self.player1_score = 0.0
        self.player2_score = 0.0
        self.player1_mud_turns = 0
        self.player2_mud_turns = 0
        self.turn = 0
        self.max_turns = 300
        self._token_counter = 0
        self._cheese: list[tuple[int, int]] = [(3, 3), (7, 7), (5, 3)]

    def _apply(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = self._DELTAS[action]
        return pos[0] + dx, pos[1] + dy

    def make_move(self, p1_move: int, p2_move: int) -> FakeMoveUndo:
        undo = FakeMoveUndo(
            p1_pos=self.player1_position,
            p2_pos=self.player2_position,
            p1_score=self.player1_score,
            p2_score=self.player2_score,
            turn=self.turn,
            token=self._token_counter,
            p1_mud=self.player1_mud_turns,
            p2_mud=self.player2_mud_turns,
        )
        self._token_counter += 1
        self.player1_position = self._apply(self.player1_position, p1_move)
        self.player2_position = self._apply(self.player2_position, p2_move)
        self.turn += 1
        return undo

    def unmake_move(self, undo: FakeMoveUndo) -> None:
        self.player1_position = undo.p1_pos
        self.player2_position = undo.p2_pos
        self.player1_score = undo.p1_score
        self.player2_score = undo.p2_score
        self.turn = undo.turn

    def get_valid_moves(self, position: tuple[int, int]) -> list[int]:
        """Return all movement directions (no walls in FakeGame)."""
        # UP=0, RIGHT=1, DOWN=2, LEFT=3 are all valid (no walls)
        return [0, 1, 2, 3]

    def get_observation(self, is_player_one: bool = True) -> object:
        """Return a dummy observation for predict_fn tests."""
        return object()

    def cheese_positions(self) -> list[tuple[int, int]]:
        """Return remaining cheese positions."""
        return self._cheese.copy()


@pytest.fixture
def fake_game() -> FakeGame:
    """Provide a deterministic game stub for navigation tests."""
    return FakeGame()


@pytest.fixture
def fake_tree(fake_game: FakeGame) -> MCTSTree:
    """Tree backed by FakeGame."""
    prior_p1 = np.ones(5) / 5
    prior_p2 = np.ones(5) / 5
    nn_payout = np.zeros((2, 5, 5))  # Bimatrix format: [p1_payoffs, p2_payoffs]
    root = MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
    )
    return MCTSTree(game=fake_game, root=root, gamma=1.0)  # type: ignore[arg-type]


@pytest.fixture
def game() -> PyRat:
    """Create a small PyRat game for testing."""
    return PyRat(width=5, height=5, cheese_count=3, seed=42)


@pytest.fixture
def root_node() -> MCTSNode:
    """Create a root node with uniform priors."""
    prior_p1 = np.ones(5) / 5
    prior_p2 = np.ones(5) / 5
    nn_payout = np.zeros((2, 5, 5))  # Bimatrix format: [p1_payoffs, p2_payoffs]

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
        initial_p1_pos = tree.game.player1_position
        initial_p2_pos = tree.game.player2_position

        # Make move from root
        child, reward = tree.make_move_from(tree.root, action_p1=1, action_p2=1)

        # Child should be created under outcome index pair
        outcome_i = tree.root.action_to_outcome(1, 1)  # P1's action 1 -> outcome index
        outcome_j = tree.root.action_to_outcome(2, 1)  # P2's action 1 -> outcome index
        assert (outcome_i, outcome_j) in tree.root.children
        assert tree.root.children[(outcome_i, outcome_j)] == child

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
            tree.game.player1_position != initial_p1_pos
            or tree.game.player2_position != initial_p2_pos
            or tree.game.turn == 1
        )

    def test_reward_calculation_no_cheese(self, tree: MCTSTree) -> None:
        """Reward should be (0, 0) when no cheese is collected."""
        # Make a move that doesn't collect cheese
        child, reward = tree.make_move_from(tree.root, action_p1=4, action_p2=4)  # STAY

        # No cheese collected, reward should be (0, 0) tuple
        assert reward == (0.0, 0.0)

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

        # Should have two children of root (under outcome index pairs)
        outcome_00 = (tree.root.action_to_outcome(1, 0), tree.root.action_to_outcome(2, 0))
        outcome_11 = (tree.root.action_to_outcome(1, 1), tree.root.action_to_outcome(2, 1))
        assert outcome_00 in tree.root.children
        assert outcome_11 in tree.root.children

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


class TestNavigationStateSync:
    """Navigation should keep simulator aligned with the tree."""

    def test_replays_actions_when_switching_branches(self, fake_tree: MCTSTree) -> None:
        """After moving to a sibling branch, simulator should match replayed actions."""
        game = fake_tree.game

        # Path: root -> a (RIGHT, RIGHT) -> a1 (DOWN, LEFT)
        a, _ = fake_tree.make_move_from(fake_tree.root, 1, 1)
        fake_tree.make_move_from(a, 2, 3)

        # Now jump to sibling root -> b (UP, DOWN), which requires navigation
        b, _ = fake_tree.make_move_from(fake_tree.root, 0, 2)

        # Expected positions if we start from root and only apply (UP, DOWN)
        # Y-up coordinate system: UP increases Y, DOWN decreases Y
        assert game.player1_position == (5, 6)  # UP from (5, 5)
        assert game.player2_position == (5, 4)  # DOWN from (5, 5)
        assert game.turn == 1  # navigation unwinds to root then applies one move
        assert fake_tree.simulator_node == b

    def test_refreshes_move_undo_on_reuse(self, fake_tree: MCTSTree) -> None:
        """Reusing an existing child should refresh its undo token."""
        child, _ = fake_tree.make_move_from(fake_tree.root, 0, 0)
        first_token = child.move_undo.token

        # Move somewhere else to force navigation back
        fake_tree.make_move_from(fake_tree.root, 1, 1)

        # Revisit the same child; its undo should be refreshed
        fake_tree.make_move_from(fake_tree.root, 0, 0)
        assert child.move_undo.token != first_token

    def test_parent_action_required_for_navigation(self, fake_tree: MCTSTree) -> None:
        """Missing parent_action should raise when navigating."""
        child, _ = fake_tree.make_move_from(fake_tree.root, 0, 0)
        # Navigate away to a sibling
        sibling, _ = fake_tree.make_move_from(fake_tree.root, 1, 1)
        # Now remove parent_action from first child
        child.parent_action = None

        # Try to navigate back to the first child - should fail
        with pytest.raises(RuntimeError):
            fake_tree._navigate_to(child)

    def test_mud_counters_propagated(self, fake_tree: MCTSTree) -> None:
        """Child nodes should capture mud counters from the engine."""
        fake_tree.game.player1_mud_turns = 2  # type: ignore[misc]
        fake_tree.game.player2_mud_turns = 1  # type: ignore[misc]

        child, _ = fake_tree.make_move_from(fake_tree.root, 4, 4)  # STAY/STAY

        assert child.p1_mud_turns_remaining == 2
        assert child.p2_mud_turns_remaining == 1


class TestBackup:
    """Tests for backing up values through the tree."""

    def test_backup_single_step(self, tree: MCTSTree) -> None:
        """Backup a single value."""
        # Create a child
        child, _ = tree.make_move_from(tree.root, 0, 0)

        # Backup path with one step - tuple rewards (p1_reward, p2_reward)
        path = [(tree.root, 0, 0, (1.0, -0.5))]  # (node, action_p1, action_p2, reward)
        tree.backup(path)

        # Root should have updated Q-value for both players
        assert tree.root.action_visits[0, 0] == 1
        # value = reward + gamma * 0 = (1.0, -0.5) + 1.0 * (0, 0) = (1.0, -0.5)
        assert tree.root.payout_matrix[0, 0, 0] == pytest.approx(1.0)  # P1's payout
        assert tree.root.payout_matrix[1, 0, 0] == pytest.approx(-0.5)  # P2's payout

    def test_backup_with_discount_factor(self, tree: MCTSTree) -> None:
        """Backup should apply gamma discounting."""
        # Create tree with gamma < 1
        tree_discounted = MCTSTree(tree.game, tree.root, gamma=0.9)

        # Create path: root → child
        child, _ = tree_discounted.make_move_from(tree.root, 0, 0)

        # Backup path with two steps - tuple rewards (p1_reward, p2_reward)
        # At child: reward = (0.5, 0.0), future value = (0, 0)
        # At root: reward = (1.0, 0.0), future value = (0.5, 0.0)
        path = [
            (tree.root, 0, 0, (1.0, 0.0)),  # reward at root
            (child, 1, 1, (0.5, 0.0)),  # reward at child
        ]
        tree_discounted.backup(path)

        # Child: value = (0.5, 0.0) + 0.9 * (0, 0) = (0.5, 0.0)
        assert child.payout_matrix[0, 1, 1] == pytest.approx(0.5)  # P1's payout

        # Root: value = (1.0, 0.0) + 0.9 * (0.5, 0.0) = (1.45, 0.0)
        assert tree.root.payout_matrix[0, 0, 0] == pytest.approx(1.45)  # P1's payout

    def test_backup_multiple_visits(self, tree: MCTSTree) -> None:
        """Multiple backups should use incremental mean."""
        # First backup - tuple rewards (p1_reward, p2_reward)
        path1 = [(tree.root, 0, 0, (10.0, -10.0))]
        tree.backup(path1)
        assert tree.root.payout_matrix[0, 0, 0] == pytest.approx(10.0)  # P1
        assert tree.root.payout_matrix[1, 0, 0] == pytest.approx(-10.0)  # P2

        # Second backup to same action
        path2 = [(tree.root, 0, 0, (6.0, -6.0))]
        tree.backup(path2)
        # P1: Q = 10.0 + (6.0 - 10.0) / 2 = 8.0
        # P2: Q = -10.0 + (-6.0 - (-10.0)) / 2 = -8.0
        assert tree.root.payout_matrix[0, 0, 0] == pytest.approx(8.0)  # P1
        assert tree.root.payout_matrix[1, 0, 0] == pytest.approx(-8.0)  # P2
        assert tree.root.action_visits[0, 0] == 2


class TestMultipleExpansions:
    """Tests for expanding multiple children."""

    def test_expand_multiple_children_from_root(self, tree: MCTSTree) -> None:
        """Expand several children and verify consistency."""
        children = []
        for i in range(3):
            child, _ = tree.make_move_from(tree.root, i, i)
            children.append(child)

        # All children should have correct parent and depth
        for child in children:
            assert child.parent == tree.root
            assert child.depth == 1

        # Children are stored under effective action pairs
        # Number of unique children depends on how many actions are equivalent
        # (some may map to STAY if blocked by walls)
        assert len(tree.root.children) >= 1  # At least one child
        assert len(tree.root.children) <= 3  # At most 3 unique children

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


class TestAdvanceRoot:
    """Tests for advance_root method."""

    def test_advance_root_existing_child(self, fake_tree: MCTSTree) -> None:
        """Advance to a pre-existing child node."""
        old_root = fake_tree.root

        # Create a child first
        child, _ = fake_tree.make_move_from(fake_tree.root, 1, 2)

        # Navigate back to root for the advance
        fake_tree._sim_path = [old_root]

        # Advance root to that child
        fake_tree.advance_root(1, 2)

        # New root should be the child we created
        assert fake_tree.root == child
        assert fake_tree._sim_path == [child]

    def test_advance_root_creates_child(self, fake_tree: MCTSTree) -> None:
        """Advance to a child that doesn't exist yet (creates it)."""
        old_root = fake_tree.root

        # No children exist yet
        assert len(old_root.children) == 0

        # Advance to non-existent child - should create it
        fake_tree.advance_root(0, 1)

        # Child should now exist and be the new root
        assert fake_tree.root != old_root
        assert fake_tree.root.parent == old_root
        assert len(old_root.children) == 1

    def test_advance_root_keeps_parent_refs(self, fake_tree: MCTSTree) -> None:
        """Verify new root keeps its parent reference (history preserved)."""
        old_root = fake_tree.root

        # Advance root
        fake_tree.advance_root(2, 3)

        # New root should still have parent reference
        assert fake_tree.root.parent == old_root
        assert fake_tree.root.parent_action is not None

    def test_advance_root_resets_sim_path(self, fake_tree: MCTSTree) -> None:
        """Verify simulator path is reset to start at new root."""
        # Create some tree structure
        child1, _ = fake_tree.make_move_from(fake_tree.root, 0, 0)
        grandchild, _ = fake_tree.make_move_from(child1, 1, 1)

        # Simulator is at grandchild
        assert fake_tree.simulator_node == grandchild
        assert len(fake_tree._sim_path) == 3

        # Navigate back and advance root to child1
        fake_tree._navigate_to(fake_tree.root)
        fake_tree.advance_root(0, 0)

        # Simulator path should now start at new root
        assert fake_tree._sim_path == [child1]
        assert fake_tree.simulator_node == child1

    def test_advance_root_advances_game_when_at_root(
        self, fake_game: FakeGame, fake_tree: MCTSTree
    ) -> None:
        """When simulator is at root, advancing root should advance game state."""
        # Create a child first
        child, _ = fake_tree.make_move_from(fake_tree.root, 1, 0)

        # Navigate back to root
        fake_tree._navigate_to(fake_tree.root)
        initial_pos = fake_game.player1_position

        # Advance root - should also advance game since simulator is at root
        fake_tree.advance_root(1, 0)

        # Game should have advanced (player moved RIGHT)
        assert fake_game.player1_position != initial_pos


class TestPredictionCache:
    """Tests for NN prediction caching on transposed positions."""

    @staticmethod
    def _counting_predict_fn() -> tuple[
        list[int],
        Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray]],
    ]:
        """Create a predict_fn that counts how many times it's called."""
        call_count = [0]

        def predict_fn(
            observation: object,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            call_count[0] += 1
            return np.ones(5) / 5, np.ones(5) / 5, np.zeros((2, 5, 5))

        return call_count, predict_fn

    def test_cache_hit_same_state_via_transposition(self) -> None:
        """Two paths converging to the same state should call predict_fn once."""
        game = FakeGame()
        call_count, predict_fn = self._counting_predict_fn()

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=np.ones(5) / 5,
            prior_policy_p2=np.ones(5) / 5,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0, predict_fn=predict_fn)  # type: ignore[arg-type]

        # Root init calls predict_fn once (state: P1=(5,5), P2=(5,5), turn=0)
        assert call_count[0] == 1

        # Path 1: root → A (RIGHT, UP) → AA (UP, RIGHT)
        # A: P1=(6,5), P2=(5,6), turn=1
        a, _ = tree.make_move_from(tree.root, 1, 0)  # P1 RIGHT, P2 UP
        assert call_count[0] == 2
        # AA: P1=(6,6), P2=(6,6), turn=2
        _aa, _ = tree.make_move_from(a, 0, 1)  # P1 UP, P2 RIGHT
        assert call_count[0] == 3

        # Path 2: root → B (UP, RIGHT) → BB (RIGHT, UP)
        # B: P1=(5,6), P2=(6,5), turn=1 — different positions from A, so new call
        b, _ = tree.make_move_from(tree.root, 0, 1)  # P1 UP, P2 RIGHT
        assert call_count[0] == 4
        # BB: P1=(6,6), P2=(6,6), turn=2 — same state as AA → cache hit!
        _bb, _ = tree.make_move_from(b, 1, 0)  # P1 RIGHT, P2 UP
        assert call_count[0] == 4  # No new call — cache hit

    def test_cache_miss_different_states(self) -> None:
        """Different game states should each call predict_fn."""
        game = FakeGame()
        call_count, predict_fn = self._counting_predict_fn()

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=np.ones(5) / 5,
            prior_policy_p2=np.ones(5) / 5,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0, predict_fn=predict_fn)  # type: ignore[arg-type]
        assert call_count[0] == 1  # Root init

        # Each child has a unique position → all misses
        tree.make_move_from(tree.root, 0, 0)  # P1=(5,6), P2=(5,6)
        assert call_count[0] == 2
        tree.make_move_from(tree.root, 1, 1)  # P1=(6,5), P2=(6,5)
        assert call_count[0] == 3
        tree.make_move_from(tree.root, 2, 2)  # P1=(5,4), P2=(5,4)
        assert call_count[0] == 4

    def test_no_caching_without_predict_fn(self) -> None:
        """With no predict_fn, cache should remain empty (smart uniform is cheap)."""
        game = FakeGame()
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=np.ones(5) / 5,
            prior_policy_p2=np.ones(5) / 5,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Expand some children
        tree.make_move_from(tree.root, 0, 0)
        tree.make_move_from(tree.root, 1, 1)

        # Cache should be empty — uniform priors don't use it
        assert len(tree._prediction_cache) == 0

    def test_cached_arrays_are_independent_copies(self) -> None:
        """Mutating returned arrays should not corrupt the cache."""
        game = FakeGame()
        call_count, predict_fn = self._counting_predict_fn()

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=np.ones(5) / 5,
            prior_policy_p2=np.ones(5) / 5,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0, predict_fn=predict_fn)  # type: ignore[arg-type]

        # Build a transposition: two paths to P1=(6,6), P2=(6,6), turn=2
        a, _ = tree.make_move_from(tree.root, 1, 0)
        _aa, _ = tree.make_move_from(a, 0, 1)

        b, _ = tree.make_move_from(tree.root, 0, 1)
        count_before_bb = call_count[0]
        _bb, _ = tree.make_move_from(b, 1, 0)
        # BB was a cache hit — predict_fn was NOT called again
        assert call_count[0] == count_before_bb

        # The two nodes should have independent prior arrays
        # (node init uses np.add.at which mutates in place)
        assert _aa.prior_policy_p1 is not _bb.prior_policy_p1
