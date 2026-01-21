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
        game = create_game(params, seed=42)

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
        game = create_game(params, seed=42)

        assert len(game.mud_entries()) == 0

    def test_none_density_uses_pyrat_defaults(self) -> None:
        """Verify None density uses PyRat defaults (non-zero walls/mud)."""
        params = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=3,
            wall_density=None,
            mud_density=None,
        )
        game = create_game(params, seed=42)

        # PyRat defaults produce walls and mud
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
        game = create_game(params, seed=123)

        assert game.width == 7
        assert game.height == 9
        assert game.max_turns == 50
        assert len(game.cheese_positions()) == 5
