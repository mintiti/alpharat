"""Tests for the Rust MCTS search binding (SmartUniform mode)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from alpharat_mcts import SearchResult, rust_mcts_search
from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates, Wall

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


@pytest.fixture
def open_5x5() -> PyRat:
    """Open 5x5 game, both players at center — all 5 actions available."""
    return (
        GameConfigBuilder(5, 5)
        .with_max_turns(30)
        .with_player1_pos(Coordinates(2, 2))
        .with_player2_pos(Coordinates(2, 2))
        .with_cheese([Coordinates(0, 0)])
        .build()
    )


@pytest.fixture
def corner_game() -> PyRat:
    """P1 at (0,0), P2 at (4,4) — both in corners with blocked actions."""
    return (
        GameConfigBuilder(5, 5)
        .with_max_turns(30)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese([Coordinates(2, 2)])
        .build()
    )


@pytest.fixture
def wall_game() -> PyRat:
    """P1 at (2,2) with wall blocking UP — 4 unique outcomes."""
    return (
        GameConfigBuilder(5, 5)
        .with_max_turns(30)
        .with_player1_pos(Coordinates(2, 2))
        .with_player2_pos(Coordinates(0, 0))
        .with_cheese([Coordinates(4, 4)])
        .with_walls([Wall(Coordinates(2, 2), Coordinates(2, 3))])
        .build()
    )


# --- Policies sum to 1 ---


class TestPolicySumsToOne:
    def test_open_position(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=50, seed=42)
        assert abs(result.policy_p1.sum() - 1.0) < 1e-5
        assert abs(result.policy_p2.sum() - 1.0) < 1e-5

    def test_corner_position(self, corner_game: PyRat) -> None:
        result = rust_mcts_search(corner_game, simulations=50, seed=42)
        assert abs(result.policy_p1.sum() - 1.0) < 1e-5
        assert abs(result.policy_p2.sum() - 1.0) < 1e-5

    def test_wall_position(self, wall_game: PyRat) -> None:
        result = rust_mcts_search(wall_game, simulations=50, seed=42)
        assert abs(result.policy_p1.sum() - 1.0) < 1e-5
        assert abs(result.policy_p2.sum() - 1.0) < 1e-5


# --- Blocked actions get 0 probability ---


class TestBlockedActions:
    def test_corner_p1_blocked(self, corner_game: PyRat) -> None:
        """P1 at (0,0): DOWN (2) and LEFT (3) blocked by board edge."""
        result = rust_mcts_search(corner_game, simulations=100, seed=42)
        assert result.policy_p1[2] == 0.0  # DOWN
        assert result.policy_p1[3] == 0.0  # LEFT

    def test_corner_p1_valid(self, corner_game: PyRat) -> None:
        """P1 at (0,0): UP (0), RIGHT (1), STAY (4) should have nonzero probability."""
        result = rust_mcts_search(corner_game, simulations=100, seed=42)
        assert result.policy_p1[0] > 0.0  # UP
        assert result.policy_p1[1] > 0.0  # RIGHT
        assert result.policy_p1[4] > 0.0  # STAY

    def test_corner_p2_blocked(self, corner_game: PyRat) -> None:
        """P2 at (4,4): UP (0) and RIGHT (1) blocked by board edge."""
        result = rust_mcts_search(corner_game, simulations=100, seed=42)
        assert result.policy_p2[0] == 0.0  # UP
        assert result.policy_p2[1] == 0.0  # RIGHT

    def test_wall_blocks_action(self, wall_game: PyRat) -> None:
        """Wall between (2,2)-(2,3) blocks P1 UP (0)."""
        result = rust_mcts_search(wall_game, simulations=100, seed=42)
        assert result.policy_p1[0] == 0.0  # UP blocked by wall


# --- Determinism with seed ---


class TestDeterminism:
    def test_same_seed_same_result(self, open_5x5: PyRat) -> None:
        r1 = rust_mcts_search(open_5x5, simulations=50, seed=42)
        r2 = rust_mcts_search(open_5x5, simulations=50, seed=42)
        np.testing.assert_array_equal(r1.policy_p1, r2.policy_p1)
        np.testing.assert_array_equal(r1.policy_p2, r2.policy_p2)
        assert r1.value_p1 == r2.value_p1
        assert r1.value_p2 == r2.value_p2
        assert r1.total_visits == r2.total_visits

    def test_different_seed_both_valid(self, open_5x5: PyRat) -> None:
        r1 = rust_mcts_search(open_5x5, simulations=50, seed=42)
        r2 = rust_mcts_search(open_5x5, simulations=50, seed=99)
        assert abs(r1.policy_p1.sum() - 1.0) < 1e-5
        assert abs(r2.policy_p1.sum() - 1.0) < 1e-5


# --- Visit count matches simulations ---


class TestVisitCount:
    @pytest.mark.parametrize("n_sims", [10, 50, 100, 200])
    def test_total_visits(self, open_5x5: PyRat, n_sims: int) -> None:
        result = rust_mcts_search(open_5x5, simulations=n_sims, seed=42)
        assert result.total_visits == n_sims


# --- Return types ---


class TestReturnTypes:
    def test_policies_are_float32_numpy(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=10, seed=42)
        assert isinstance(result.policy_p1, np.ndarray)
        assert isinstance(result.policy_p2, np.ndarray)
        assert result.policy_p1.shape == (5,)
        assert result.policy_p2.shape == (5,)
        assert result.policy_p1.dtype == np.float32
        assert result.policy_p2.dtype == np.float32

    def test_values_are_floats(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=10, seed=42)
        assert isinstance(result.value_p1, float)
        assert isinstance(result.value_p2, float)

    def test_total_visits_is_int(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=10, seed=42)
        assert isinstance(result.total_visits, int)

    def test_result_type(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=10, seed=42)
        assert isinstance(result, SearchResult)

    def test_repr(self, open_5x5: PyRat) -> None:
        result = rust_mcts_search(open_5x5, simulations=10, seed=42)
        r = repr(result)
        assert "SearchResult" in r
        assert "value_p1" in r
