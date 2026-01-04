"""Tests for MCTSAgent tree reuse functionality."""

import warnings

import pytest
from pyrat_engine.core.game import PyRat

from alpharat.ai.mcts_agent import MCTSAgent
from alpharat.eval.game import play_game


@pytest.fixture
def small_game() -> PyRat:
    """Create a small game for testing."""
    return PyRat(width=5, height=5, cheese_count=3, max_turns=50, seed=42)


class TestObserveMoveBasics:
    """Tests for observe_move method behavior."""

    def test_observe_move_noop_when_reuse_disabled(self, small_game: PyRat) -> None:
        """observe_move should do nothing when reuse_tree=False."""
        agent = MCTSAgent(simulations=10, reuse_tree=False)

        # Get a move to set up state
        agent.get_move(small_game, player=1)

        # observe_move should not raise or warn
        agent.observe_move(0, 0)

        # Tree should remain None (not stored)
        assert agent._tree is None

    def test_observe_move_advances_tree_when_reuse_enabled(self, small_game: PyRat) -> None:
        """observe_move should advance tree root when reuse_tree=True."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # Get a move to create tree
        action = agent.get_move(small_game, player=1)

        # Get reference to current root
        assert agent._tree is not None
        old_root = agent._tree.root

        # Observe the move
        agent.observe_move(action, 4)  # action, STAY

        # Root should have changed
        assert agent._tree is not None
        assert agent._tree.root != old_root
        assert agent._tree.root.parent == old_root

    def test_observe_move_warns_without_prior_get_move(self, small_game: PyRat) -> None:
        """observe_move should warn if called without prior get_move."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # Call observe_move without get_move first
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.observe_move(0, 0)
            assert len(w) == 1
            assert "without prior get_move" in str(w[0].message)


class TestGetMoveWithReuse:
    """Tests for get_move with tree reuse enabled."""

    def test_get_move_creates_tree_on_first_call(self, small_game: PyRat) -> None:
        """First get_move should create a tree."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        assert agent._tree is None
        agent.get_move(small_game, player=1)
        assert agent._tree is not None

    def test_get_move_reuses_tree_after_observe(self, small_game: PyRat) -> None:
        """get_move should reuse tree after proper observe_move."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # First turn
        action = agent.get_move(small_game, player=1)
        assert agent._tree is not None
        tree_after_first = agent._tree
        root_after_first = tree_after_first.root

        # Make the actual move
        small_game.make_move(action, 4)  # Our action, opponent STAY
        agent.observe_move(action, 4)

        # Root should have advanced (it's now a child of original root)
        assert agent._tree is not None
        assert agent._tree.root != root_after_first
        assert agent._tree.root.parent == root_after_first

        # Second turn - same tree object should be reused
        agent.get_move(small_game, player=1)
        assert agent._tree is tree_after_first

    def test_get_move_warns_on_missing_observe(self, small_game: PyRat) -> None:
        """get_move should warn if observe_move was skipped."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # First get_move
        action = agent.get_move(small_game, player=1)

        # Make move but forget to call observe_move
        small_game.make_move(action, 4)

        # Second get_move without observe_move - should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent.get_move(small_game, player=1)
            assert len(w) == 1
            assert "without prior observe_move" in str(w[0].message)

    def test_is_tree_valid_after_observe_move(self, small_game: PyRat) -> None:
        """_is_tree_valid should validate after proper observe_move cycle."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # No tree yet - should be invalid
        assert not agent._is_tree_valid(small_game)

        # First turn: get_move + observe_move + game.make_move
        action = agent.get_move(small_game, player=1)
        small_game.make_move(action, 4)  # Our action, opponent STAY
        agent.observe_move(action, 4)

        # After proper cycle, tree should be valid for next game state
        assert agent._is_tree_valid(small_game)

        # Advance the game without telling the agent
        small_game.make_move(4, 4)  # STAY, STAY

        # Now tree is stale - should be invalid
        assert not agent._is_tree_valid(small_game)


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_tree_state(self, small_game: PyRat) -> None:
        """reset should clear all tree-related state."""
        agent = MCTSAgent(simulations=10, reuse_tree=True)

        # Build up some state
        agent.get_move(small_game, player=1)
        assert agent._tree is not None
        assert agent._player is not None
        assert agent._awaiting_observe is True

        # Reset
        agent.reset()

        # State should be cleared
        assert agent._tree is None
        assert agent._player is None
        assert agent._awaiting_observe is False


class TestFullGameWithReuse:
    """Integration tests for tree reuse in full games."""

    def test_full_game_completes_with_reuse(self) -> None:
        """A full game should complete successfully with tree reuse."""
        agent_p1 = MCTSAgent(simulations=10, reuse_tree=True)
        agent_p2 = MCTSAgent(simulations=10, reuse_tree=True)

        result = play_game(
            agent_p1,
            agent_p2,
            width=5,
            height=5,
            cheese_count=3,
            max_turns=30,
            seed=42,
        )

        # Game should complete without errors
        assert result.turns > 0
        assert result.p1_score >= 0
        assert result.p2_score >= 0

    def test_full_game_completes_without_reuse(self) -> None:
        """A full game should complete successfully without tree reuse (baseline)."""
        agent_p1 = MCTSAgent(simulations=10, reuse_tree=False)
        agent_p2 = MCTSAgent(simulations=10, reuse_tree=False)

        result = play_game(
            agent_p1,
            agent_p2,
            width=5,
            height=5,
            cheese_count=3,
            max_turns=30,
            seed=42,
        )

        # Game should complete without errors
        assert result.turns > 0
        assert result.p1_score >= 0
        assert result.p2_score >= 0

    def test_mixed_agents_reuse_and_fresh(self) -> None:
        """Game with one agent using reuse and one without should work."""
        agent_p1 = MCTSAgent(simulations=10, reuse_tree=True)
        agent_p2 = MCTSAgent(simulations=10, reuse_tree=False)

        result = play_game(
            agent_p1,
            agent_p2,
            width=5,
            height=5,
            cheese_count=3,
            max_turns=30,
            seed=42,
        )

        # Game should complete without errors
        assert result.turns > 0


class TestTreeStatePreservation:
    """Tests verifying that statistics are actually preserved."""

    def test_payout_matrix_preserved_after_advance(self, small_game: PyRat) -> None:
        """Payout matrix values should be preserved after advancing root."""
        agent = MCTSAgent(simulations=50, reuse_tree=True)

        # First turn
        action = agent.get_move(small_game, player=1)
        assert agent._tree is not None
        first_root = agent._tree.root

        # Store payout matrix before advancing
        payout_before = first_root.payout_matrix.copy()

        # Make move and observe
        small_game.make_move(action, 4)
        agent.observe_move(action, 4)

        # Old root (now parent of new root) should have unchanged payout matrix
        assert agent._tree is not None
        new_root = agent._tree.root
        assert new_root.parent is first_root

        # Payout matrix should be identical
        assert (first_root.payout_matrix == payout_before).all()

    def test_children_preserved_after_advance(self, small_game: PyRat) -> None:
        """Children of old root should be preserved after advancing."""
        agent = MCTSAgent(simulations=50, reuse_tree=True)

        # First turn
        action = agent.get_move(small_game, player=1)
        assert agent._tree is not None
        first_root = agent._tree.root

        # Should have some children from exploration
        num_children_before = len(first_root.children)
        assert num_children_before > 0

        # Make move and observe
        small_game.make_move(action, 4)
        agent.observe_move(action, 4)

        # Old root should still have all its children
        assert len(first_root.children) == num_children_before
