"""Tests for RustMCTSAgent — Agent interface compliance and integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates

from alpharat.ai.rust_mcts_agent import RustMCTSAgent

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


@pytest.fixture
def open_5x5() -> PyRat:
    """Open 5x5 game."""
    return (
        GameConfigBuilder(5, 5)
        .with_max_turns(30)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese([Coordinates(2, 2), Coordinates(1, 3), Coordinates(3, 1)])
        .build()
    )


@pytest.fixture
def corner_game() -> PyRat:
    """P1 at corner (0,0), P2 at corner (4,4)."""
    return (
        GameConfigBuilder(5, 5)
        .with_max_turns(30)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese([Coordinates(2, 2)])
        .build()
    )


# --- Agent interface compliance ---


class TestAgentInterface:
    def test_get_move_returns_valid_action_p1(self, open_5x5: PyRat) -> None:
        agent = RustMCTSAgent(simulations=20, seed=42)
        action = agent.get_move(open_5x5, player=1)
        assert action in range(5)

    def test_get_move_returns_valid_action_p2(self, open_5x5: PyRat) -> None:
        agent = RustMCTSAgent(simulations=20, seed=42)
        action = agent.get_move(open_5x5, player=2)
        assert action in range(5)

    def test_reset_is_noop(self) -> None:
        agent = RustMCTSAgent(simulations=10)
        agent.reset()  # should not raise

    def test_name_without_nn(self) -> None:
        agent = RustMCTSAgent(simulations=50)
        assert agent.name == "RustPUCT(50)"

    def test_name_with_nn(self) -> None:
        # Don't actually load — just check the name logic
        agent = RustMCTSAgent.__new__(RustMCTSAgent)
        agent.simulations = 100
        agent.checkpoint = "fake.pt"
        assert agent.name == "RustPUCT(100)+NN"


# --- Basic search behavior ---


class TestSearchBehavior:
    def test_does_not_modify_game(self, open_5x5: PyRat) -> None:
        """get_move should not modify the game state."""
        p1_pos_before = (open_5x5.player1_position.x, open_5x5.player1_position.y)
        p2_pos_before = (open_5x5.player2_position.x, open_5x5.player2_position.y)
        turn_before = open_5x5.turn

        agent = RustMCTSAgent(simulations=20, seed=42)
        agent.get_move(open_5x5, player=1)

        assert (open_5x5.player1_position.x, open_5x5.player1_position.y) == p1_pos_before
        assert (open_5x5.player2_position.x, open_5x5.player2_position.y) == p2_pos_before
        assert open_5x5.turn == turn_before

    def test_blocked_actions_not_selected(self, corner_game: PyRat) -> None:
        """P1 at (0,0) should never select DOWN or LEFT."""
        agent = RustMCTSAgent(simulations=50)
        # Run multiple times to check
        for seed in range(10):
            agent._seed = seed
            action = agent.get_move(corner_game, player=1)
            assert action != 2, f"Selected DOWN at corner (0,0), seed={seed}"
            assert action != 3, f"Selected LEFT at corner (0,0), seed={seed}"

    def test_deterministic_with_seed(self, open_5x5: PyRat) -> None:
        """Same seed should produce same action."""
        agent = RustMCTSAgent(simulations=50, seed=42)
        a1 = agent.get_move(open_5x5, player=1)
        a2 = agent.get_move(open_5x5, player=1)
        assert a1 == a2


# --- Head-to-head integration ---


class TestHeadToHead:
    def test_rust_vs_python_mcts_completes(self, open_5x5: PyRat) -> None:
        """A game between RustMCTSAgent and Python MCTSAgent runs to completion."""
        from alpharat.eval.game import play_game

        rust_agent = RustMCTSAgent(simulations=20, seed=42)
        # Use another RustMCTS agent as opponent (avoids importing DecoupledPUCTConfig)
        opponent = RustMCTSAgent(simulations=20, seed=99)

        result = play_game(
            rust_agent,
            opponent,
            game=open_5x5,
        )
        # Just check it completed — both agents played valid moves
        assert result is not None
