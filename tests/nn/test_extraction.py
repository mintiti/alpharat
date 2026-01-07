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
        self._cheese = cheese if cheese is not None else [(2, 2), (4, 1)]

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
    action_p1: int = 0,
    action_p2: int = 0,
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
        payout_matrix=np.zeros((2, 5, 5), dtype=np.float32),
        visit_counts=np.zeros((5, 5), dtype=np.int32),
        prior_p1=np.ones(5, dtype=np.float32) / 5,
        prior_p2=np.ones(5, dtype=np.float32) / 5,
        policy_p1=np.ones(5, dtype=np.float32) / 5,
        policy_p2=np.ones(5, dtype=np.float32) / 5,
        action_p1=action_p1,
        action_p2=action_p2,
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


# =============================================================================
# Train/inference consistency tests
# =============================================================================


class TestTrainInferenceConsistency:
    """Tests that both extraction paths produce identical ObservationInput.

    This is critical: the training path (from_game_arrays) and inference path
    (from_pyrat_game) must produce identical observations for the same game state.
    Any mismatch means the model sees different inputs at train vs inference time.

    Uses real PyRat games to test against the actual interface, not mocks.
    """

    def test_both_paths_produce_identical_observation_input(self) -> None:
        """from_game_arrays and from_pyrat_game should produce identical results."""
        from pyrat_engine.core import GameConfigBuilder

        from alpharat.data.maze import build_maze_array

        # Create a real game
        width, height = 5, 5
        max_turns = 100
        game = (
            GameConfigBuilder(width, height)
            .with_max_turns(max_turns)
            .with_player1_pos(Coordinates(1, 1))
            .with_player2_pos(Coordinates(3, 3))
            .with_cheese([Coordinates(2, 2), Coordinates(4, 4)])
            .build()
        )

        # Build maze from real game
        maze = build_maze_array(game, width, height)

        # Training path: simulate what would be stored in GameData/PositionData
        game_data = GameData(
            maze=maze,
            initial_cheese=np.zeros((height, width), dtype=bool),
            max_turns=max_turns,
            width=width,
            height=height,
            positions=[],
            result=0,
            final_p1_score=0.0,
            final_p2_score=0.0,
        )
        position_data = PositionData(
            p1_pos=(game.player1_position.x, game.player1_position.y),
            p2_pos=(game.player2_position.x, game.player2_position.y),
            p1_score=float(game.player1_score),
            p2_score=float(game.player2_score),
            p1_mud=int(game.player1_mud_turns),
            p2_mud=int(game.player2_mud_turns),
            cheese_positions=[(c.x, c.y) for c in game.cheese_positions()],
            turn=int(game.turn),
            payout_matrix=np.zeros((2, 5, 5), dtype=np.float32),
            visit_counts=np.zeros((5, 5), dtype=np.int32),
            prior_p1=np.ones(5, dtype=np.float32) / 5,
            prior_p2=np.ones(5, dtype=np.float32) / 5,
            policy_p1=np.ones(5, dtype=np.float32) / 5,
            policy_p2=np.ones(5, dtype=np.float32) / 5,
            action_p1=0,
            action_p2=0,
        )

        # Extract via both paths
        obs_training = from_game_arrays(game_data, position_data)
        obs_inference = from_pyrat_game(game, maze, max_turns)

        # All fields must match exactly
        np.testing.assert_array_equal(obs_training.maze, obs_inference.maze)
        assert obs_training.p1_pos == obs_inference.p1_pos
        assert obs_training.p2_pos == obs_inference.p2_pos
        np.testing.assert_array_equal(obs_training.cheese_mask, obs_inference.cheese_mask)
        assert obs_training.p1_score == obs_inference.p1_score
        assert obs_training.p2_score == obs_inference.p2_score
        assert obs_training.turn == obs_inference.turn
        assert obs_training.max_turns == obs_inference.max_turns
        assert obs_training.p1_mud == obs_inference.p1_mud
        assert obs_training.p2_mud == obs_inference.p2_mud
        assert obs_training.width == obs_inference.width
        assert obs_training.height == obs_inference.height

    def test_consistency_after_moves(self) -> None:
        """Both paths should produce identical results after game state changes."""
        from pyrat_engine.core import GameConfigBuilder

        from alpharat.data.maze import build_maze_array

        width, height = 5, 5
        max_turns = 100
        game = (
            GameConfigBuilder(width, height)
            .with_max_turns(max_turns)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(1, 0), Coordinates(3, 4)])
            .build()
        )

        maze = build_maze_array(game, width, height)

        # Make some moves - P1 moves right to collect cheese, P2 moves left
        game.make_move(1, 3)  # P1: RIGHT, P2: LEFT

        # Build training data from current state
        game_data = GameData(
            maze=maze,
            initial_cheese=np.zeros((height, width), dtype=bool),
            max_turns=max_turns,
            width=width,
            height=height,
            positions=[],
            result=0,
            final_p1_score=0.0,
            final_p2_score=0.0,
        )
        position_data = PositionData(
            p1_pos=(game.player1_position.x, game.player1_position.y),
            p2_pos=(game.player2_position.x, game.player2_position.y),
            p1_score=float(game.player1_score),
            p2_score=float(game.player2_score),
            p1_mud=int(game.player1_mud_turns),
            p2_mud=int(game.player2_mud_turns),
            cheese_positions=[(c.x, c.y) for c in game.cheese_positions()],
            turn=int(game.turn),
            payout_matrix=np.zeros((2, 5, 5), dtype=np.float32),
            visit_counts=np.zeros((5, 5), dtype=np.int32),
            prior_p1=np.ones(5, dtype=np.float32) / 5,
            prior_p2=np.ones(5, dtype=np.float32) / 5,
            policy_p1=np.ones(5, dtype=np.float32) / 5,
            policy_p2=np.ones(5, dtype=np.float32) / 5,
            action_p1=0,
            action_p2=0,
        )

        obs_training = from_game_arrays(game_data, position_data)
        obs_inference = from_pyrat_game(game, maze, max_turns)

        # State should reflect the move
        assert obs_training.turn == obs_inference.turn == 1
        assert obs_training.p1_pos == obs_inference.p1_pos
        assert obs_training.p2_pos == obs_inference.p2_pos
        np.testing.assert_array_equal(obs_training.cheese_mask, obs_inference.cheese_mask)
        assert obs_training.p1_score == obs_inference.p1_score
        assert obs_training.p2_score == obs_inference.p2_score

    def test_consistency_with_corner_positions(self) -> None:
        """Both paths should handle corner positions identically."""
        from pyrat_engine.core import GameConfigBuilder

        from alpharat.data.maze import build_maze_array

        width, height = 5, 5
        game = (
            GameConfigBuilder(width, height)
            .with_max_turns(50)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(0, 4), Coordinates(4, 0)])
            .build()
        )

        maze = build_maze_array(game, width, height)

        game_data = GameData(
            maze=maze,
            initial_cheese=np.zeros((height, width), dtype=bool),
            max_turns=50,
            width=width,
            height=height,
            positions=[],
            result=0,
            final_p1_score=0.0,
            final_p2_score=0.0,
        )
        position_data = PositionData(
            p1_pos=(game.player1_position.x, game.player1_position.y),
            p2_pos=(game.player2_position.x, game.player2_position.y),
            p1_score=float(game.player1_score),
            p2_score=float(game.player2_score),
            p1_mud=int(game.player1_mud_turns),
            p2_mud=int(game.player2_mud_turns),
            cheese_positions=[(c.x, c.y) for c in game.cheese_positions()],
            turn=int(game.turn),
            payout_matrix=np.zeros((2, 5, 5), dtype=np.float32),
            visit_counts=np.zeros((5, 5), dtype=np.int32),
            prior_p1=np.ones(5, dtype=np.float32) / 5,
            prior_p2=np.ones(5, dtype=np.float32) / 5,
            policy_p1=np.ones(5, dtype=np.float32) / 5,
            policy_p2=np.ones(5, dtype=np.float32) / 5,
            action_p1=0,
            action_p2=0,
        )

        obs_training = from_game_arrays(game_data, position_data)
        obs_inference = from_pyrat_game(game, maze, max_turns=50)

        # Verify positions in correct corners
        assert obs_training.p1_pos == obs_inference.p1_pos == (0, 0)
        assert obs_training.p2_pos == obs_inference.p2_pos == (4, 4)
        np.testing.assert_array_equal(obs_training.cheese_mask, obs_inference.cheese_mask)

    def test_consistency_after_cheese_collected(self) -> None:
        """Both paths should handle state after all cheese collected identically."""
        from pyrat_engine.core import GameConfigBuilder

        from alpharat.data.maze import build_maze_array

        width, height = 5, 5
        # Place cheese where P1 will collect it on first move
        game = (
            GameConfigBuilder(width, height)
            .with_max_turns(50)
            .with_player1_pos(Coordinates(1, 1))
            .with_player2_pos(Coordinates(3, 3))
            .with_cheese([Coordinates(2, 1)])  # P1 can reach by moving RIGHT
            .build()
        )

        maze = build_maze_array(game, width, height)

        # P1 moves right to collect cheese, P2 stays
        game.make_move(1, 4)  # P1: RIGHT, P2: STAY

        # Now cheese list should be empty
        assert len(game.cheese_positions()) == 0

        game_data = GameData(
            maze=maze,
            initial_cheese=np.zeros((height, width), dtype=bool),
            max_turns=50,
            width=width,
            height=height,
            positions=[],
            result=0,
            final_p1_score=0.0,
            final_p2_score=0.0,
        )
        position_data = PositionData(
            p1_pos=(game.player1_position.x, game.player1_position.y),
            p2_pos=(game.player2_position.x, game.player2_position.y),
            p1_score=float(game.player1_score),
            p2_score=float(game.player2_score),
            p1_mud=int(game.player1_mud_turns),
            p2_mud=int(game.player2_mud_turns),
            cheese_positions=[(c.x, c.y) for c in game.cheese_positions()],
            turn=int(game.turn),
            payout_matrix=np.zeros((2, 5, 5), dtype=np.float32),
            visit_counts=np.zeros((5, 5), dtype=np.int32),
            prior_p1=np.ones(5, dtype=np.float32) / 5,
            prior_p2=np.ones(5, dtype=np.float32) / 5,
            policy_p1=np.ones(5, dtype=np.float32) / 5,
            policy_p2=np.ones(5, dtype=np.float32) / 5,
            action_p1=0,
            action_p2=0,
        )

        obs_training = from_game_arrays(game_data, position_data)
        obs_inference = from_pyrat_game(game, maze, max_turns=50)

        np.testing.assert_array_equal(obs_training.cheese_mask, obs_inference.cheese_mask)
        assert obs_training.cheese_mask.sum() == 0
        assert obs_inference.cheese_mask.sum() == 0
        # Verify P1 collected the cheese
        assert obs_training.p1_score == obs_inference.p1_score == 1.0
