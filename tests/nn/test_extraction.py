"""Tests for ObservationInput extraction."""

from __future__ import annotations

import numpy as np
from pyrat_engine.core.types import Coordinates

from alpharat.data.types import GameData, PositionData
from alpharat.nn.extraction import (
    _cheese_positions_to_mask,
    from_game_arrays,
    from_pyrat_game,
)
from alpharat.nn.types import ObservationInput


class FakeGame:
    """Mock PyRat game for extraction tests."""

    def __init__(
        self,
        p1_pos: tuple[int, int] = (1, 2),
        p2_pos: tuple[int, int] = (3, 4),
        p1_score: float = 2.5,
        p2_score: float = 1.0,
        turn: int = 5,
        max_turns: int = 100,
        p1_mud: int = 0,
        p2_mud: int = 3,
        cheese: list[tuple[int, int]] | None = None,
    ) -> None:
        self.player1_position = Coordinates(p1_pos[0], p1_pos[1])
        self.player2_position = Coordinates(p2_pos[0], p2_pos[1])
        self.player1_score = p1_score
        self.player2_score = p2_score
        self.turn = turn
        self.max_turns = max_turns
        self.player1_mud_turns = p1_mud
        self.player2_mud_turns = p2_mud
        self._cheese = cheese or [(2, 2), (4, 1)]

    def cheese_positions(self) -> list[Coordinates]:
        return [Coordinates(x, y) for x, y in self._cheese]


def _make_game_data(
    *,
    width: int = 5,
    height: int = 5,
    max_turns: int = 100,
    final_p1_score: float = 3.0,
    final_p2_score: float = 2.0,
) -> GameData:
    """Create GameData with defaults for testing."""
    maze = np.ones((height, width, 4), dtype=np.int8)
    # Mark edges as walls
    maze[:, 0, 3] = -1  # LEFT
    maze[:, width - 1, 1] = -1  # RIGHT
    maze[0, :, 0] = -1  # UP
    maze[height - 1, :, 2] = -1  # DOWN

    initial_cheese = np.zeros((height, width), dtype=bool)
    initial_cheese[2, 2] = True

    return GameData(
        maze=maze,
        initial_cheese=initial_cheese,
        max_turns=max_turns,
        width=width,
        height=height,
        positions=[],
        result=1,
        final_p1_score=final_p1_score,
        final_p2_score=final_p2_score,
    )


def _make_position_data(
    *,
    p1_pos: tuple[int, int] = (1, 2),
    p2_pos: tuple[int, int] = (3, 3),
    p1_score: float = 1.0,
    p2_score: float = 0.5,
    p1_mud: int = 0,
    p2_mud: int = 2,
    cheese_positions: list[tuple[int, int]] | None = None,
    turn: int = 10,
) -> PositionData:
    """Create PositionData with defaults for testing."""
    if cheese_positions is None:
        cheese_positions = [(2, 2), (4, 1)]

    return PositionData(
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        p1_score=p1_score,
        p2_score=p2_score,
        p1_mud=p1_mud,
        p2_mud=p2_mud,
        cheese_positions=cheese_positions,
        turn=turn,
        payout_matrix=np.zeros((5, 5), dtype=np.float32),
        visit_counts=np.zeros((5, 5), dtype=np.int32),
        prior_p1=np.ones(5, dtype=np.float32) / 5,
        prior_p2=np.ones(5, dtype=np.float32) / 5,
        policy_p1=np.ones(5, dtype=np.float32) / 5,
        policy_p2=np.ones(5, dtype=np.float32) / 5,
    )


# =============================================================================
# from_game_arrays tests
# =============================================================================


class TestFromGameArrays:
    """Tests for from_game_arrays()."""

    def test_extracts_maze_from_game(self) -> None:
        """Maze should come from GameData."""
        game = _make_game_data()
        position = _make_position_data()

        result = from_game_arrays(game, position)

        np.testing.assert_array_equal(result.maze, game.maze)

    def test_extracts_dimensions_from_game(self) -> None:
        """Width and height should come from GameData."""
        game = _make_game_data(width=7, height=4)
        position = _make_position_data()

        result = from_game_arrays(game, position)

        assert result.width == 7
        assert result.height == 4

    def test_extracts_max_turns_from_game(self) -> None:
        """max_turns should come from GameData."""
        game = _make_game_data(max_turns=50)
        position = _make_position_data()

        result = from_game_arrays(game, position)

        assert result.max_turns == 50

    def test_extracts_positions_from_position_data(self) -> None:
        """Player positions should come from PositionData."""
        game = _make_game_data()
        position = _make_position_data(p1_pos=(2, 3), p2_pos=(4, 1))

        result = from_game_arrays(game, position)

        assert result.p1_pos == (2, 3)
        assert result.p2_pos == (4, 1)

    def test_extracts_scores_from_position_data(self) -> None:
        """Scores should come from PositionData."""
        game = _make_game_data()
        position = _make_position_data(p1_score=3.5, p2_score=1.5)

        result = from_game_arrays(game, position)

        assert result.p1_score == 3.5
        assert result.p2_score == 1.5

    def test_extracts_mud_from_position_data(self) -> None:
        """Mud turns should come from PositionData."""
        game = _make_game_data()
        position = _make_position_data(p1_mud=2, p2_mud=5)

        result = from_game_arrays(game, position)

        assert result.p1_mud == 2
        assert result.p2_mud == 5

    def test_extracts_turn_from_position_data(self) -> None:
        """Turn should come from PositionData."""
        game = _make_game_data()
        position = _make_position_data(turn=25)

        result = from_game_arrays(game, position)

        assert result.turn == 25

    def test_converts_cheese_positions_to_mask(self) -> None:
        """Cheese positions should be converted to boolean mask."""
        game = _make_game_data(width=5, height=5)
        position = _make_position_data(cheese_positions=[(1, 2), (3, 4)])

        result = from_game_arrays(game, position)

        assert result.cheese_mask.shape == (5, 5)
        assert result.cheese_mask.dtype == bool
        # mask[y, x] indexing
        assert result.cheese_mask[2, 1] is np.True_
        assert result.cheese_mask[4, 3] is np.True_
        assert result.cheese_mask.sum() == 2

    def test_returns_observation_input(self) -> None:
        """Should return ObservationInput dataclass."""
        game = _make_game_data()
        position = _make_position_data()

        result = from_game_arrays(game, position)

        assert isinstance(result, ObservationInput)


# =============================================================================
# from_pyrat_game tests
# =============================================================================


class TestFromPyratGame:
    """Tests for from_pyrat_game()."""

    def test_uses_provided_maze(self) -> None:
        """Should use the pre-built maze array."""
        game = FakeGame()
        maze = np.ones((5, 5, 4), dtype=np.int8) * 2  # Distinctive value

        result = from_pyrat_game(game, maze, max_turns=100)

        np.testing.assert_array_equal(result.maze, maze)

    def test_extracts_dimensions_from_maze(self) -> None:
        """Width and height should come from maze shape."""
        game = FakeGame()
        maze = np.ones((4, 7, 4), dtype=np.int8)  # (H, W, 4)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.height == 4
        assert result.width == 7

    def test_extracts_positions_from_coordinates(self) -> None:
        """Should convert Coordinates to (x, y) tuples."""
        game = FakeGame(p1_pos=(2, 3), p2_pos=(4, 1))
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.p1_pos == (2, 3)
        assert result.p2_pos == (4, 1)

    def test_extracts_scores(self) -> None:
        """Should extract scores from game."""
        game = FakeGame(p1_score=5.5, p2_score=2.0)
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.p1_score == 5.5
        assert result.p2_score == 2.0

    def test_extracts_mud_turns(self) -> None:
        """Should extract mud turns from game."""
        game = FakeGame(p1_mud=3, p2_mud=0)
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.p1_mud == 3
        assert result.p2_mud == 0

    def test_extracts_turn(self) -> None:
        """Should extract turn from game."""
        game = FakeGame(turn=15)
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.turn == 15

    def test_uses_provided_max_turns(self) -> None:
        """Should use the provided max_turns parameter."""
        game = FakeGame(max_turns=200)
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=50)

        assert result.max_turns == 50

    def test_builds_cheese_mask_from_coordinates(self) -> None:
        """Should convert cheese Coordinates to boolean mask."""
        game = FakeGame(cheese=[(1, 2), (4, 3)])
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert result.cheese_mask.shape == (5, 5)
        assert result.cheese_mask.dtype == bool
        # mask[y, x] indexing
        assert result.cheese_mask[2, 1] is np.True_
        assert result.cheese_mask[3, 4] is np.True_
        assert result.cheese_mask.sum() == 2

    def test_returns_observation_input(self) -> None:
        """Should return ObservationInput dataclass."""
        game = FakeGame()
        maze = np.ones((5, 5, 4), dtype=np.int8)

        result = from_pyrat_game(game, maze, max_turns=100)

        assert isinstance(result, ObservationInput)


# =============================================================================
# Helper function tests
# =============================================================================


class TestCheesePositionsToMask:
    """Tests for _cheese_positions_to_mask()."""

    def test_empty_positions(self) -> None:
        """Should return all-False mask for empty list."""
        mask = _cheese_positions_to_mask([], height=3, width=4)

        assert mask.shape == (3, 4)
        assert mask.sum() == 0

    def test_single_position(self) -> None:
        """Should mark single position correctly."""
        mask = _cheese_positions_to_mask([(2, 1)], height=3, width=4)

        # mask[y, x]
        assert mask[1, 2] is np.True_
        assert mask.sum() == 1

    def test_multiple_positions(self) -> None:
        """Should mark multiple positions correctly."""
        positions = [(0, 0), (3, 2), (1, 1)]
        mask = _cheese_positions_to_mask(positions, height=4, width=5)

        assert mask[0, 0] is np.True_
        assert mask[2, 3] is np.True_
        assert mask[1, 1] is np.True_
        assert mask.sum() == 3

    def test_dtype_is_bool(self) -> None:
        """Mask should have boolean dtype."""
        mask = _cheese_positions_to_mask([(0, 0)], height=2, width=2)

        assert mask.dtype == bool
