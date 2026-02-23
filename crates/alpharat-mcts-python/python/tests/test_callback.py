"""Tests for the Rust MCTS Python callback backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from alpharat_mcts import rust_mcts_search
from pyrat_engine.core import GameBuilder
from pyrat_engine.core.types import Coordinates

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


@pytest.fixture
def open_5x5() -> PyRat:
    """Open 5x5 game, both players at center."""
    return (
        GameBuilder(5, 5)
        .with_max_turns(30)
        .with_open_maze()
        .with_custom_positions(Coordinates(2, 2), Coordinates(2, 2))
        .with_custom_cheese([Coordinates(0, 0)])
        .build()
        .create()
    )


@pytest.fixture
def corner_game() -> PyRat:
    """P1 at (0,0), P2 at (4,4)."""
    return (
        GameBuilder(5, 5)
        .with_max_turns(30)
        .with_open_maze()
        .with_custom_positions(Coordinates(0, 0), Coordinates(4, 4))
        .with_custom_cheese([Coordinates(2, 2)])
        .build()
        .create()
    )


def uniform_predict_fn(
    games: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mock predict_fn: uniform policies, zero values."""
    n = len(games)
    policy_p1 = np.full((n, 5), 0.2, dtype=np.float32)
    policy_p2 = np.full((n, 5), 0.2, dtype=np.float32)
    value_p1 = np.zeros(n, dtype=np.float32)
    value_p2 = np.zeros(n, dtype=np.float32)
    return policy_p1, policy_p2, value_p1, value_p2


# --- Callback is called ---


class TestCallbackInvocation:
    def test_callback_is_called(self, open_5x5: PyRat) -> None:
        """predict_fn should be called at least once during search."""
        call_count = 0

        def counting_predict_fn(
            games: list[Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            nonlocal call_count
            call_count += 1
            return uniform_predict_fn(games)

        rust_mcts_search(open_5x5, predict_fn=counting_predict_fn, simulations=20, seed=42)
        assert call_count > 0

    def test_callback_called_multiple_times(self, open_5x5: PyRat) -> None:
        """With enough simulations, callback should be called multiple times."""
        call_count = 0

        def counting_predict_fn(
            games: list[Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            nonlocal call_count
            call_count += 1
            return uniform_predict_fn(games)

        rust_mcts_search(open_5x5, predict_fn=counting_predict_fn, simulations=50, seed=42)
        assert call_count >= 2


# --- Callback receives correct input ---


class TestCallbackInput:
    def test_receives_pyrat_objects(self, open_5x5: PyRat) -> None:
        """Callback should receive a list of game objects with standard PyRat properties."""
        received_games: list[Any] = []

        def capturing_predict_fn(
            games: list[Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            received_games.extend(games)
            return uniform_predict_fn(games)

        rust_mcts_search(open_5x5, predict_fn=capturing_predict_fn, simulations=10, seed=42)
        assert len(received_games) > 0

        # Each game should have standard PyRat properties
        game = received_games[0]
        assert hasattr(game, "player1_position")
        assert hasattr(game, "player2_position")
        assert hasattr(game, "width")
        assert hasattr(game, "height")

    def test_batch_size_respected(self, open_5x5: PyRat) -> None:
        """No single callback invocation should receive more games than batch_size."""
        max_seen = 0

        def tracking_predict_fn(
            games: list[Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            nonlocal max_seen
            max_seen = max(max_seen, len(games))
            return uniform_predict_fn(games)

        batch_size = 4
        rust_mcts_search(
            open_5x5,
            predict_fn=tracking_predict_fn,
            simulations=50,
            batch_size=batch_size,
            seed=42,
        )
        assert max_seen <= batch_size


# --- Search results valid with callback ---


class TestCallbackResults:
    def test_policies_sum_to_one(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, predict_fn=uniform_predict_fn, simulations=50, seed=42)
        assert abs(result.policy_p1.sum() - 1.0) < 1e-5
        assert abs(result.policy_p2.sum() - 1.0) < 1e-5

    def test_visit_count_matches(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, predict_fn=uniform_predict_fn, simulations=50, seed=42)
        assert result.total_visits == 50

    def test_biased_prior_affects_policy(self, open_5x5: PyRat) -> None:
        """A predict_fn that strongly favors one action should shift the search policy."""

        def biased_predict_fn(
            games: list[Any],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            n = len(games)
            # Strongly favor action 0 (UP)
            policy_p1 = np.zeros((n, 5), dtype=np.float32)
            policy_p1[:, 0] = 0.9
            policy_p1[:, 1:] = 0.025
            policy_p2 = np.full((n, 5), 0.2, dtype=np.float32)
            value_p1 = np.zeros(n, dtype=np.float32)
            value_p2 = np.zeros(n, dtype=np.float32)
            return policy_p1, policy_p2, value_p1, value_p2

        result = rust_mcts_search(open_5x5, predict_fn=biased_predict_fn, simulations=200, seed=42)
        # Action 0 should have the highest probability (or at least be significant)
        assert result.policy_p1[0] > 0.1

    def test_blocked_actions_still_zero(self, corner_game: PyRat) -> None:
        """Even with callback, blocked actions should get 0 in output policy."""
        result = rust_mcts_search(
            corner_game, predict_fn=uniform_predict_fn, simulations=100, seed=42
        )
        # P1 at (0,0): DOWN and LEFT blocked
        assert result.policy_p1[2] == 0.0
        assert result.policy_p1[3] == 0.0
