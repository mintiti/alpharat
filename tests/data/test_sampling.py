"""Tests for alpharat.data.sampling."""

from __future__ import annotations

from alpharat.config.game import CheeseConfig, ClassicMaze, GameConfig, OpenMaze
from alpharat.data.sampling import create_game


class TestCreateGame:
    """Tests for create_game function."""

    def test_open_maze_no_walls(self) -> None:
        """Verify open maze creates game with no walls."""
        params = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=OpenMaze(),
            cheese=CheeseConfig(count=3),
        )
        game = create_game(params)

        assert len(game.wall_entries()) == 0

    def test_open_maze_no_mud(self) -> None:
        """Verify open maze creates game with no mud."""
        params = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=OpenMaze(),
            cheese=CheeseConfig(count=3),
        )
        game = create_game(params)

        assert len(game.mud_entries()) == 0

    def test_classic_maze_has_walls_and_mud(self) -> None:
        """Verify classic maze uses PyRat defaults (non-zero walls/mud).

        Uses a larger grid because small grids can have 0 mud entries
        depending on seed.
        """
        params = GameConfig(
            width=15,
            height=11,
            max_turns=100,
            maze=ClassicMaze(),
            cheese=CheeseConfig(count=10),
        )
        game = create_game(params)

        # PyRat defaults produce walls and mud on a large enough grid
        assert len(game.wall_entries()) > 0
        assert len(game.mud_entries()) > 0

    def test_basic_game_config_applied(self) -> None:
        """Verify basic config (width, height, etc.) is applied."""
        params = GameConfig(
            width=7,
            height=9,
            max_turns=50,
            cheese=CheeseConfig(count=5),
        )
        game = create_game(params)

        assert game.width == 7
        assert game.height == 9
        assert game.max_turns == 50
        assert len(game.cheese_positions()) == 5
