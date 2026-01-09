"""Tests for game data recording."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pyrat_engine.core.types import Coordinates, Direction, Mud, Wall

from alpharat.data.loader import load_game_data
from alpharat.data.maze import _coords_to_direction, _opposite_direction, build_maze_array
from alpharat.data.recorder import GameRecorder
from alpharat.mcts import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

    from alpharat.data.types import GameData


class FakeGame:
    """Mock PyRat game for recorder tests."""

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        walls: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        muds: list[tuple[tuple[int, int], tuple[int, int], int]] | None = None,
        cheese: list[tuple[int, int]] | None = None,
        max_turns: int = 100,
    ) -> None:
        self.width = width
        self.height = height
        self._walls = walls or []
        self._muds = muds or []
        self._cheese = cheese or [(2, 2), (3, 3)]
        self.max_turns = max_turns
        self.turn = 0
        self.player1_position = Coordinates(1, 1)
        self.player2_position = Coordinates(3, 3)
        self.player1_score = 0.0
        self.player2_score = 0.0
        self.player1_mud_turns = 0
        self.player2_mud_turns = 0

    def wall_entries(self) -> list[Wall]:
        return [Wall(p1, p2) for p1, p2 in self._walls]

    def mud_entries(self) -> list[Mud]:
        return [Mud(p1, p2, v) for p1, p2, v in self._muds]

    def cheese_positions(self) -> list[Coordinates]:
        return [Coordinates(x, y) for x, y in self._cheese]


# =============================================================================
# Maze building tests
# =============================================================================


class TestBuildMazeArray:
    """Tests for build_maze_array function."""

    def test_shape_and_dtype(self) -> None:
        """Maze array should have shape (H, W, 4) and dtype int8."""
        game = FakeGame(width=5, height=4)
        maze = build_maze_array(game, width=5, height=4)

        assert maze.shape == (4, 5, 4)
        assert maze.dtype == np.int8

    def test_edge_boundaries(self) -> None:
        """Edge boundaries should be marked as walls (-1)."""
        game = FakeGame(width=5, height=4)
        maze = build_maze_array(game, width=5, height=4)

        # Y-up coordinate system: y=0 is BOTTOM, y=height-1 is TOP
        # BOTTOM edge (y=0): can't move down (direction 2)
        assert np.all(maze[0, :, 2] == -1)

        # TOP edge (y=height-1): can't move up (direction 0)
        assert np.all(maze[3, :, 0] == -1)

        # LEFT edge (x=0): can't move left (direction 3)
        assert np.all(maze[:, 0, 3] == -1)

        # RIGHT edge (x=width-1): can't move right (direction 1)
        assert np.all(maze[:, 4, 1] == -1)

    def test_interior_default_cost(self) -> None:
        """Interior cells should have default cost 1."""
        game = FakeGame(width=5, height=5)
        maze = build_maze_array(game, width=5, height=5)

        # Interior cell (2, 2) should have cost 1 in all non-edge directions
        assert maze[2, 2, 0] == 1  # UP
        assert maze[2, 2, 1] == 1  # RIGHT
        assert maze[2, 2, 2] == 1  # DOWN
        assert maze[2, 2, 3] == 1  # LEFT

    def test_walls_marked(self) -> None:
        """Walls should be marked as -1 in both directions."""
        # Wall between (1,1) and (2,1) - horizontal wall
        walls = [((1, 1), (2, 1))]
        game = FakeGame(width=5, height=5, walls=walls)
        maze = build_maze_array(game, width=5, height=5)

        # From (1,1), direction RIGHT (1) should be blocked
        assert maze[1, 1, 1] == -1

        # From (2,1), direction LEFT (3) should be blocked
        assert maze[1, 2, 3] == -1

    def test_mud_costs(self) -> None:
        """Mud should have its cost value in both directions."""
        # Mud between (1,1) and (1,2) with cost 3 - vertical connection
        # Y-up: (1,1) to (1,2) is UP (Y increases)
        muds = [((1, 1), (1, 2), 3)]
        game = FakeGame(width=5, height=5, muds=muds)
        maze = build_maze_array(game, width=5, height=5)

        # From (1,1), direction UP (0) should cost 3
        assert maze[1, 1, 0] == 3

        # From (1,2), direction DOWN (2) should cost 3
        assert maze[2, 1, 2] == 3


class TestDirectionHelpers:
    """Tests for direction helper functions."""

    def test_coords_to_direction_up(self) -> None:
        # Y-up coordinate system: UP increases Y
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(2, 3)
        assert _coords_to_direction(from_pos, to_pos) == Direction.UP

    def test_coords_to_direction_right(self) -> None:
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(3, 2)
        assert _coords_to_direction(from_pos, to_pos) == Direction.RIGHT

    def test_coords_to_direction_down(self) -> None:
        # Y-up coordinate system: DOWN decreases Y
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(2, 1)
        assert _coords_to_direction(from_pos, to_pos) == Direction.DOWN

    def test_coords_to_direction_left(self) -> None:
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(1, 2)
        assert _coords_to_direction(from_pos, to_pos) == Direction.LEFT

    def test_coords_to_direction_non_adjacent(self) -> None:
        from_pos = Coordinates(0, 0)
        to_pos = Coordinates(2, 2)
        with pytest.raises(ValueError, match="Non-adjacent"):
            _coords_to_direction(from_pos, to_pos)

    def test_opposite_direction(self) -> None:
        assert _opposite_direction(Direction.UP) == Direction.DOWN
        assert _opposite_direction(Direction.RIGHT) == Direction.LEFT
        assert _opposite_direction(Direction.DOWN) == Direction.UP
        assert _opposite_direction(Direction.LEFT) == Direction.RIGHT


# =============================================================================
# GameRecorder tests
# =============================================================================


def make_mock_search_result() -> SearchResult:
    """Create a mock SearchResult for testing."""
    return SearchResult(
        payout_matrix=np.zeros((2, 5, 5), dtype=np.float64),
        policy_p1=np.ones(5, dtype=np.float64) / 5,
        policy_p2=np.ones(5, dtype=np.float64) / 5,
    )


class TestGameRecorder:
    """Tests for GameRecorder context manager."""

    def test_context_captures_maze(self) -> None:
        """Entering context should capture maze topology."""
        game = FakeGame(width=5, height=5)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            GameRecorder(game, tmpdir, width=5, height=5) as recorder,
        ):
            assert recorder.data is not None
            assert recorder.data.maze.shape == (5, 5, 4)
            assert recorder.data.maze.dtype == np.int8

    def test_context_captures_initial_cheese(self) -> None:
        """Entering context should capture initial cheese positions."""
        game = FakeGame(width=5, height=5, cheese=[(2, 2), (3, 3)])

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            GameRecorder(game, tmpdir, width=5, height=5) as recorder,
        ):
            assert recorder.data is not None
            assert recorder.data.initial_cheese[2, 2] is np.True_
            assert recorder.data.initial_cheese[3, 3] is np.True_
            assert recorder.data.initial_cheese[0, 0] is np.False_

    def test_context_wrong_turn_raises(self) -> None:
        """Entering context should raise if game is not at turn 0."""
        game = FakeGame()
        game.turn = 5

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(RuntimeError, match="turn 0"),
            GameRecorder(game, tmpdir, width=5, height=5),
        ):
            pass

    def test_context_nonexistent_dir_raises(self) -> None:
        """Entering context should raise if output directory doesn't exist."""
        game = FakeGame()

        with (
            pytest.raises(ValueError, match="does not exist"),
            GameRecorder(game, "/nonexistent/path", width=5, height=5),
        ):
            pass

    def test_record_position_without_context_raises(self) -> None:
        """record_position should raise if not inside context."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = GameRecorder(game, tmpdir, width=5, height=5)
            result = make_mock_search_result()

            with pytest.raises(RuntimeError, match="context manager"):
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

    def test_record_position_accumulates(self) -> None:
        """record_position should accumulate position data."""
        game = FakeGame()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            GameRecorder(game, tmpdir, width=5, height=5) as recorder,
        ):
            result = make_mock_search_result()
            recorder.record_position(
                game=game,
                search_result=result,
                prior_p1=np.ones(5) / 5,
                prior_p2=np.ones(5) / 5,
                visit_counts=np.zeros((5, 5), dtype=np.int32),
                action_p1=0,
                action_p2=0,
            )

            assert recorder.data is not None
            assert len(recorder.data.positions) == 1

            # Record another
            game.turn = 1
            recorder.record_position(
                game=game,
                search_result=result,
                prior_p1=np.ones(5) / 5,
                prior_p2=np.ones(5) / 5,
                visit_counts=np.zeros((5, 5), dtype=np.int32),
                action_p1=0,
                action_p2=0,
            )

            assert len(recorder.data.positions) == 2

    def test_context_exit_saves_file(self) -> None:
        """Exiting context should save npz file."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            assert recorder.saved_path.exists()
            assert recorder.saved_path.suffix == ".npz"

    def test_context_exit_sets_result(self) -> None:
        """Exiting context should set result based on scores."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )
                game.player1_score = 10.0
                game.player2_score = 5.0

            assert recorder.data is not None
            assert recorder.data.result == 1  # P1 wins
            assert recorder.data.final_p1_score == 10.0
            assert recorder.data.final_p2_score == 5.0

    def test_context_exit_draw(self) -> None:
        """Exiting context should set result=0 for draw."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )
                game.player1_score = 5.0
                game.player2_score = 5.0

            assert recorder.data is not None
            assert recorder.data.result == 0

    def test_context_exit_no_positions_no_save(self) -> None:
        """Exiting context with no positions should not save."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                pass  # No positions recorded

            assert recorder.saved_path is None

    def test_context_exit_on_exception_no_save(self) -> None:
        """Exiting context on exception should not save."""
        game = FakeGame()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                    result = make_mock_search_result()
                    recorder.record_position(
                        game=game,
                        search_result=result,
                        prior_p1=np.ones(5) / 5,
                        prior_p2=np.ones(5) / 5,
                        visit_counts=np.zeros((5, 5), dtype=np.int32),
                        action_p1=0,
                        action_p2=0,
                    )
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            assert recorder.saved_path is None


class TestSavedArrays:
    """Tests for saved npz array contents."""

    def test_saved_arrays_have_correct_keys(self) -> None:
        """Saved npz should contain all expected keys."""
        game = FakeGame(width=5, height=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            data = np.load(recorder.saved_path)

            expected_keys = {
                "maze",
                "initial_cheese",
                "cheese_outcomes",
                "max_turns",
                "result",
                "final_p1_score",
                "final_p2_score",
                "num_positions",
                "p1_pos",
                "p2_pos",
                "p1_score",
                "p2_score",
                "p1_mud",
                "p2_mud",
                "cheese_mask",
                "turn",
                "payout_matrix",
                "visit_counts",
                "prior_p1",
                "prior_p2",
                "policy_p1",
                "policy_p2",
                "action_p1",
                "action_p2",
            }

            assert set(data.keys()) == expected_keys

    def test_saved_arrays_shapes(self) -> None:
        """Saved arrays should have correct shapes per spec."""
        game = FakeGame(width=5, height=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=4) as recorder:
                result = make_mock_search_result()

                # Record 3 positions
                for i in range(3):
                    game.turn = i
                    recorder.record_position(
                        game=game,
                        search_result=result,
                        prior_p1=np.ones(5) / 5,
                        prior_p2=np.ones(5) / 5,
                        visit_counts=np.zeros((5, 5), dtype=np.int32),
                        action_p1=0,
                        action_p2=0,
                    )

            assert recorder.saved_path is not None
            data = np.load(recorder.saved_path)

            # Game-level
            assert data["maze"].shape == (4, 5, 4)
            assert data["initial_cheese"].shape == (4, 5)
            assert int(data["num_positions"]) == 3

            # Position-level (N=3)
            assert data["p1_pos"].shape == (3, 2)
            assert data["p2_pos"].shape == (3, 2)
            assert data["p1_score"].shape == (3,)
            assert data["p2_score"].shape == (3,)
            assert data["p1_mud"].shape == (3,)
            assert data["p2_mud"].shape == (3,)
            assert data["cheese_mask"].shape == (3, 4, 5)
            assert data["turn"].shape == (3,)
            assert data["payout_matrix"].shape == (3, 2, 5, 5)
            assert data["visit_counts"].shape == (3, 5, 5)
            assert data["prior_p1"].shape == (3, 5)
            assert data["prior_p2"].shape == (3, 5)
            assert data["policy_p1"].shape == (3, 5)
            assert data["policy_p2"].shape == (3, 5)

    def test_saved_arrays_dtypes(self) -> None:
        """Saved arrays should have correct dtypes per spec."""
        game = FakeGame(width=5, height=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            data = np.load(recorder.saved_path)

            # Check dtypes
            assert data["maze"].dtype == np.int8
            assert data["initial_cheese"].dtype == bool
            assert data["max_turns"].dtype == np.int16
            assert data["result"].dtype == np.int8
            assert data["final_p1_score"].dtype == np.float32
            assert data["final_p2_score"].dtype == np.float32
            assert data["num_positions"].dtype == np.int32

            assert data["p1_pos"].dtype == np.int8
            assert data["p2_pos"].dtype == np.int8
            assert data["p1_score"].dtype == np.float32
            assert data["p2_score"].dtype == np.float32
            assert data["p1_mud"].dtype == np.int8
            assert data["p2_mud"].dtype == np.int8
            assert data["cheese_mask"].dtype == bool
            assert data["turn"].dtype == np.int16
            assert data["payout_matrix"].dtype == np.float32
            assert data["visit_counts"].dtype == np.int32
            assert data["prior_p1"].dtype == np.float32
            assert data["prior_p2"].dtype == np.float32
            assert data["policy_p1"].dtype == np.float32
            assert data["policy_p2"].dtype == np.float32


# =============================================================================
# Roundtrip / Idempotence tests
# =============================================================================


class TestRoundtrip:
    """Tests for save/load roundtrip idempotence."""

    def test_roundtrip_game_level_data(self) -> None:
        """Game-level data should survive roundtrip."""
        walls = [((1, 1), (2, 1))]
        muds = [((2, 2), (2, 3), 3)]
        game = FakeGame(width=5, height=4, walls=walls, muds=muds, cheese=[(0, 0), (4, 3)])
        game.player1_score = 7.0
        game.player2_score = 3.0

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=4) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        assert loaded.width == 5
        assert loaded.height == 4
        assert loaded.max_turns == 100
        assert loaded.result == 1  # P1 wins
        assert loaded.final_p1_score == 7.0
        assert loaded.final_p2_score == 3.0
        assert loaded.maze.shape == (4, 5, 4)
        assert loaded.initial_cheese.shape == (4, 5)
        # Check cheese positions
        assert loaded.initial_cheese[0, 0] is np.True_
        assert loaded.initial_cheese[3, 4] is np.True_

    def test_roundtrip_maze_walls_and_mud(self) -> None:
        """Maze walls and mud should survive roundtrip."""
        walls = [((1, 1), (2, 1))]
        muds = [((2, 2), (2, 3), 5)]
        game = FakeGame(width=5, height=5, walls=walls, muds=muds)

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        # Wall at (1,1) -> (2,1): RIGHT from (1,1) blocked, LEFT from (2,1) blocked
        assert loaded.maze[1, 1, Direction.RIGHT] == -1
        assert loaded.maze[1, 2, Direction.LEFT] == -1

        # Mud at (2,2) -> (2,3): Y-up means (2,2) to (2,3) is UP
        # From (2,2), direction UP (0) costs 5
        assert loaded.maze[2, 2, Direction.UP] == 5
        # From (2,3), direction DOWN (2) costs 5
        assert loaded.maze[3, 2, Direction.DOWN] == 5

    def test_roundtrip_position_game_state(self) -> None:
        """Position game state should survive roundtrip."""
        game = FakeGame(width=5, height=5, cheese=[(1, 1), (3, 3)])
        game.player1_position = Coordinates(2, 3)
        game.player2_position = Coordinates(4, 1)
        game.player1_score = 2.5
        game.player2_score = 1.0
        game.player1_mud_turns = 2
        game.player2_mud_turns = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        assert len(loaded.positions) == 1
        pos = loaded.positions[0]
        assert pos.p1_pos == (2, 3)
        assert pos.p2_pos == (4, 1)
        assert pos.p1_score == 2.5
        assert pos.p2_score == 1.0
        assert pos.p1_mud == 2
        assert pos.p2_mud == 0
        assert pos.turn == 0
        assert set(pos.cheese_positions) == {(1, 1), (3, 3)}

    def test_roundtrip_mcts_outputs(self) -> None:
        """MCTS outputs should survive roundtrip with correct values."""
        game = FakeGame(width=5, height=5)

        # Create non-trivial MCTS data (bimatrix: P1 and P2 payoffs)
        payout = np.array(
            [
                # P1's payoffs
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [-0.1, -0.2, -0.3, -0.4, -0.5],
                    [1.0, 0.0, -1.0, 0.5, -0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.9, 0.8, 0.7, 0.6, 0.55],
                ],
                # P2's payoffs
                [
                    [-0.1, -0.2, -0.3, -0.4, -0.5],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [-1.0, 0.0, 1.0, -0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.9, -0.8, -0.7, -0.6, -0.55],
                ],
            ],
            dtype=np.float64,
        )
        policy_p1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        policy_p2 = np.array([0.1, 0.1, 0.1, 0.2, 0.5], dtype=np.float64)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
        prior_p2 = np.array([0.25, 0.25, 0.25, 0.25, 0.0], dtype=np.float64)
        visits = np.array(
            [
                [10, 5, 3, 2, 1],
                [8, 12, 4, 1, 0],
                [6, 7, 15, 3, 2],
                [4, 3, 2, 20, 5],
                [2, 1, 1, 4, 25],
            ],
            dtype=np.int32,
        )

        search_result = SearchResult(
            payout_matrix=payout,
            policy_p1=policy_p1,
            policy_p2=policy_p2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                recorder.record_position(
                    game=game,
                    search_result=search_result,
                    prior_p1=prior_p1,
                    prior_p2=prior_p2,
                    visit_counts=visits,
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        pos = loaded.positions[0]

        # Check values match (with float32 precision)
        np.testing.assert_allclose(pos.payout_matrix, payout, rtol=1e-6)
        np.testing.assert_allclose(pos.policy_p1, policy_p1, rtol=1e-6)
        np.testing.assert_allclose(pos.policy_p2, policy_p2, rtol=1e-6)
        np.testing.assert_allclose(pos.prior_p1, prior_p1, rtol=1e-6)
        np.testing.assert_allclose(pos.prior_p2, prior_p2, rtol=1e-6)
        np.testing.assert_array_equal(pos.visit_counts, visits)

    def test_roundtrip_multiple_positions(self) -> None:
        """Multiple positions should survive roundtrip in order."""
        game = FakeGame(width=5, height=5, cheese=[(2, 2)])

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                # Position 0
                game.turn = 0
                game.player1_position = Coordinates(0, 0)
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

                # Position 1
                game.turn = 1
                game.player1_position = Coordinates(1, 0)
                game._cheese = []  # Cheese collected
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32) * 2,
                    action_p1=0,
                    action_p2=0,
                )

                # Position 2
                game.turn = 2
                game.player1_position = Coordinates(2, 0)
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32) * 3,
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        assert len(loaded.positions) == 3
        assert loaded.positions[0].turn == 0
        assert loaded.positions[0].p1_pos == (0, 0)
        assert (2, 2) in loaded.positions[0].cheese_positions

        assert loaded.positions[1].turn == 1
        assert loaded.positions[1].p1_pos == (1, 0)
        assert len(loaded.positions[1].cheese_positions) == 0

        assert loaded.positions[2].turn == 2
        assert loaded.positions[2].p1_pos == (2, 0)

        # Verify visit counts are distinct per position
        assert loaded.positions[0].visit_counts[0, 0] == 1
        assert loaded.positions[1].visit_counts[0, 0] == 2
        assert loaded.positions[2].visit_counts[0, 0] == 3

    def test_roundtrip_p2_wins(self) -> None:
        """P2 win result should survive roundtrip."""
        game = FakeGame()
        game.player1_score = 2.0
        game.player2_score = 8.0

        with tempfile.TemporaryDirectory() as tmpdir:
            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.zeros((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            loaded = load_game_data(recorder.saved_path)

        assert loaded.result == 2  # P2 wins
        assert loaded.final_p1_score == 2.0
        assert loaded.final_p2_score == 8.0


# =============================================================================
# GameBundler tests
# =============================================================================


class TestGameBundler:
    """Tests for GameBundler class."""

    def _create_game_data(
        self,
        game: FakeGame,
        tmpdir: str,
        n_positions: int = 2,
        p1_score: float = 5.0,
        p2_score: float = 3.0,
    ) -> GameData:
        """Helper to create a finalized GameData."""
        game.player1_score = p1_score
        game.player2_score = p2_score

        with GameRecorder(game, tmpdir, width=5, height=5, auto_save=False) as recorder:
            for i in range(n_positions):
                game.turn = i
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32) * (i + 1),
                    action_p1=i % 5,
                    action_p2=(i + 1) % 5,
                )

        assert recorder.data is not None
        return recorder.data

    def test_add_game_buffers(self) -> None:
        """Adding games should buffer them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            bundler = GameBundler(tmpdir, width=5, height=5)
            assert bundler.buffered_games == 0

            game = FakeGame()
            game_data = self._create_game_data(game, tmpdir)
            bundler.add_game(game_data)

            assert bundler.buffered_games == 1
            assert len(bundler.saved_paths) == 0

    def test_flush_writes_bundle(self) -> None:
        """Flush should write bundle file and clear buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            bundler = GameBundler(tmpdir, width=5, height=5)

            # Add two games
            for _ in range(2):
                game = FakeGame()
                game_data = self._create_game_data(game, tmpdir)
                bundler.add_game(game_data)

            assert bundler.buffered_games == 2

            path = bundler.flush()

            assert path is not None
            assert path.exists()
            assert bundler.buffered_games == 0
            assert len(bundler.saved_paths) == 1

    def test_flush_empty_returns_none(self) -> None:
        """Flushing empty buffer returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            bundler = GameBundler(tmpdir, width=5, height=5)
            assert bundler.flush() is None

    def test_bundle_file_format(self) -> None:
        """Bundle file should have game_lengths and correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            bundler = GameBundler(tmpdir, width=5, height=5)

            # Add games with different position counts
            game1 = FakeGame()
            game_data1 = self._create_game_data(game1, tmpdir, n_positions=3)
            bundler.add_game(game_data1)

            game2 = FakeGame()
            game_data2 = self._create_game_data(game2, tmpdir, n_positions=2)
            bundler.add_game(game_data2)

            path = bundler.flush()
            assert path is not None

            # Check bundle format
            data = np.load(path)
            assert "game_lengths" in data.files

            # 2 games, lengths 3 and 2
            np.testing.assert_array_equal(data["game_lengths"], [3, 2])

            # Game-level arrays should have shape (2, ...)
            assert data["maze"].shape == (2, 5, 5, 4)
            assert data["initial_cheese"].shape == (2, 5, 5)
            assert data["result"].shape == (2,)

            # Position-level arrays should have shape (5, ...) total
            total_positions = 3 + 2
            assert data["p1_pos"].shape == (total_positions, 2)
            assert data["payout_matrix"].shape == (total_positions, 2, 5, 5)

    def test_auto_flush_on_threshold(self) -> None:
        """Should auto-flush when buffer exceeds threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            # Very small threshold to trigger auto-flush
            bundler = GameBundler(tmpdir, width=5, height=5, threshold_bytes=1)

            game = FakeGame()
            game_data = self._create_game_data(game, tmpdir)
            path = bundler.add_game(game_data)

            # Should have auto-flushed
            assert path is not None
            assert bundler.buffered_games == 0
            assert len(bundler.saved_paths) == 1

    def test_dimension_mismatch_raises(self) -> None:
        """Adding game with wrong dimensions should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from alpharat.data.recorder import GameBundler

            bundler = GameBundler(tmpdir, width=5, height=5)

            # Create game data with wrong dimensions
            game = FakeGame(width=10, height=10)
            game.player1_score = 5.0
            game.player2_score = 3.0

            with GameRecorder(game, tmpdir, width=10, height=10, auto_save=False) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.data is not None

            with pytest.raises(ValueError, match="dimensions"):
                bundler.add_game(recorder.data)


# =============================================================================
# Bundle loading tests
# =============================================================================


class TestBundleLoading:
    """Tests for loading bundled game files."""

    def _create_bundle(self, tmpdir: str, n_games: int = 3) -> Path:
        """Helper to create a bundle file with multiple games."""
        from alpharat.data.recorder import GameBundler

        bundler = GameBundler(tmpdir, width=5, height=5)

        for i in range(n_games):
            game = FakeGame()
            game.player1_score = float(i * 2)
            game.player2_score = float(i)

            with GameRecorder(game, tmpdir, width=5, height=5, auto_save=False) as recorder:
                for j in range(i + 1):  # i+1 positions per game
                    game.turn = j
                    game.player1_position = Coordinates(j % 5, j % 5)
                    result = make_mock_search_result()
                    recorder.record_position(
                        game=game,
                        search_result=result,
                        prior_p1=np.ones(5) / 5,
                        prior_p2=np.ones(5) / 5,
                        visit_counts=np.ones((5, 5), dtype=np.int32) * (j + 1),
                        action_p1=j % 5,
                        action_p2=0,
                    )

            assert recorder.data is not None
            bundler.add_game(recorder.data)

        path = bundler.flush()
        assert path is not None
        return path

    def test_is_bundle_file_detection(self) -> None:
        """is_bundle_file should correctly identify bundle vs single game files."""
        from alpharat.data.loader import is_bundle_file

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a bundle
            bundle_path = self._create_bundle(tmpdir, n_games=2)
            assert is_bundle_file(bundle_path) is True

            # Create a single game file
            game = FakeGame()
            game.player1_score = 5.0
            game.player2_score = 3.0

            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None
            assert is_bundle_file(recorder.saved_path) is False

    def test_load_game_bundle_returns_all_games(self) -> None:
        """load_game_bundle should return all games from bundle."""
        from alpharat.data.loader import load_game_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = self._create_bundle(tmpdir, n_games=3)

            games = load_game_bundle(bundle_path)

            assert len(games) == 3
            # Game 0 has 1 position, game 1 has 2, game 2 has 3
            assert len(games[0].positions) == 1
            assert len(games[1].positions) == 2
            assert len(games[2].positions) == 3

    def test_iter_games_from_bundle_yields_in_order(self) -> None:
        """iter_games_from_bundle should yield games in order."""
        from alpharat.data.loader import iter_games_from_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = self._create_bundle(tmpdir, n_games=3)

            games = list(iter_games_from_bundle(bundle_path))

            # Check final scores match creation order
            for i, game in enumerate(games):
                assert game.final_p1_score == float(i * 2)
                assert game.final_p2_score == float(i)

    def test_bundle_roundtrip_game_data(self) -> None:
        """Data should survive bundle roundtrip."""
        from alpharat.data.loader import load_game_bundle
        from alpharat.data.recorder import GameBundler

        with tempfile.TemporaryDirectory() as tmpdir:
            bundler = GameBundler(tmpdir, width=5, height=5)

            # Create game with specific data
            game = FakeGame(cheese=[(1, 1), (2, 2)])
            game.player1_score = 7.5
            game.player2_score = 2.5
            game.player1_position = Coordinates(3, 4)
            game.player2_position = Coordinates(1, 2)

            with GameRecorder(game, tmpdir, width=5, height=5, auto_save=False) as recorder:
                payout = np.random.rand(2, 5, 5).astype(np.float32)
                policy_p1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
                policy_p2 = np.array([0.1, 0.1, 0.3, 0.3, 0.2], dtype=np.float32)
                result = SearchResult(
                    payout_matrix=payout,
                    policy_p1=policy_p1,
                    policy_p2=policy_p2,
                )
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.arange(25).reshape(5, 5).astype(np.int32),
                    action_p1=2,
                    action_p2=3,
                )

            assert recorder.data is not None
            bundler.add_game(recorder.data)
            path = bundler.flush()
            assert path is not None

            # Load and verify
            games = load_game_bundle(path)
            assert len(games) == 1

            loaded = games[0]
            assert loaded.width == 5
            assert loaded.height == 5
            assert loaded.final_p1_score == 7.5
            assert loaded.final_p2_score == 2.5
            assert loaded.result == 1  # P1 wins

            pos = loaded.positions[0]
            assert pos.p1_pos == (3, 4)
            assert pos.p2_pos == (1, 2)
            assert pos.action_p1 == 2
            assert pos.action_p2 == 3
            np.testing.assert_allclose(pos.payout_matrix, payout, rtol=1e-6)
            np.testing.assert_allclose(pos.policy_p1, policy_p1, rtol=1e-6)
            np.testing.assert_allclose(pos.policy_p2, policy_p2, rtol=1e-6)

    def test_bundle_not_single_game_raises(self) -> None:
        """Loading single game file with bundle loader should raise."""
        from alpharat.data.loader import load_game_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            game = FakeGame()
            game.player1_score = 5.0
            game.player2_score = 3.0

            with GameRecorder(game, tmpdir, width=5, height=5) as recorder:
                result = make_mock_search_result()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=np.ones(5) / 5,
                    prior_p2=np.ones(5) / 5,
                    visit_counts=np.ones((5, 5), dtype=np.int32),
                    action_p1=0,
                    action_p2=0,
                )

            assert recorder.saved_path is not None

            with pytest.raises(ValueError, match="Not a bundle file"):
                load_game_bundle(recorder.saved_path)
