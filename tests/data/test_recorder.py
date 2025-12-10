"""Tests for game data recording."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from pyrat_engine.core.types import Coordinates, Direction, Mud, Wall

from alpharat.data.maze import _coords_to_direction, _opposite_direction, build_maze_array
from alpharat.data.recorder import GameRecorder
from alpharat.mcts.search import SearchResult


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

        # UP edge (y=0): can't move up (direction 0)
        assert np.all(maze[0, :, 0] == -1)

        # DOWN edge (y=height-1): can't move down (direction 2)
        assert np.all(maze[3, :, 2] == -1)

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
        muds = [((1, 1), (1, 2), 3)]
        game = FakeGame(width=5, height=5, muds=muds)
        maze = build_maze_array(game, width=5, height=5)

        # From (1,1), direction DOWN (2) should cost 3
        assert maze[1, 1, 2] == 3

        # From (1,2), direction UP (0) should cost 3
        assert maze[2, 1, 0] == 3


class TestDirectionHelpers:
    """Tests for direction helper functions."""

    def test_coords_to_direction_up(self) -> None:
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(2, 1)
        assert _coords_to_direction(from_pos, to_pos) == Direction.UP

    def test_coords_to_direction_right(self) -> None:
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(3, 2)
        assert _coords_to_direction(from_pos, to_pos) == Direction.RIGHT

    def test_coords_to_direction_down(self) -> None:
        from_pos = Coordinates(2, 2)
        to_pos = Coordinates(2, 3)
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
        payout_matrix=np.zeros((5, 5), dtype=np.float64),
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
                )

            assert recorder.saved_path is not None
            data = np.load(recorder.saved_path)

            expected_keys = {
                "maze",
                "initial_cheese",
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
            assert data["payout_matrix"].shape == (3, 5, 5)
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
