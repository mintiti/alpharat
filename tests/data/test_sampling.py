"""Tests for alpharat.data.sampling."""

from __future__ import annotations

from alpharat.config.game import GameConfig
from alpharat.data.sampling import create_game


class TestCreateGame:
    """Tests for create_game function."""

    def test_respects_wall_density_zero(self) -> None:
        """Verify wall_density=0.0 creates game with no walls."""
        params = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=3,
            wall_density=0.0,
        )
        game = create_game(params)

        assert len(game.wall_entries()) == 0

    def test_respects_mud_density_zero(self) -> None:
        """Verify mud_density=0.0 creates game with no mud."""
        params = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=3,
            mud_density=0.0,
        )
        game = create_game(params)

        assert len(game.mud_entries()) == 0

    def test_none_density_uses_pyrat_defaults(self) -> None:
        """Verify None density uses PyRat defaults (non-zero walls/mud).

        Uses a larger grid because 5x5 with 0.1 mud density can produce
        0 mud entries depending on seed (~15% of the time).
        """
        params = GameConfig(
            width=15,
            height=11,
            max_turns=100,
            cheese_count=10,
            wall_density=None,
            mud_density=None,
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
            cheese_count=5,
        )
        game = create_game(params)

        assert game.width == 7
        assert game.height == 9
        assert game.max_turns == 50
        assert len(game.cheese_positions()) == 5
