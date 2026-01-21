"""Tests for GameConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alpharat.config.game import GameConfig


class TestGameConfigValidation:
    """Tests for GameConfig field validation."""

    def test_accepts_valid_config(self) -> None:
        """GameConfig accepts valid configuration."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
        )
        assert config.width == 5
        assert config.height == 5
        assert config.max_turns == 30
        assert config.cheese_count == 5
        assert config.symmetric is True  # default

    def test_accepts_all_fields(self) -> None:
        """GameConfig accepts all fields including optional."""
        config = GameConfig(
            width=7,
            height=9,
            max_turns=50,
            cheese_count=10,
            wall_density=0.3,
            mud_density=0.1,
            symmetric=False,
        )
        assert config.wall_density == 0.3
        assert config.mud_density == 0.1
        assert config.symmetric is False

    @pytest.mark.parametrize(
        ("field", "value", "valid_config"),
        [
            ("width", 0, {"height": 5, "max_turns": 30, "cheese_count": 5}),
            ("height", -1, {"width": 5, "max_turns": 30, "cheese_count": 5}),
            ("width", 51, {"height": 5, "max_turns": 30, "cheese_count": 5}),
            ("cheese_count", 0, {"width": 5, "height": 5, "max_turns": 30}),
            ("wall_density", 1.5, {"width": 5, "height": 5, "max_turns": 30, "cheese_count": 5}),
            ("mud_density", -0.1, {"width": 5, "height": 5, "max_turns": 30, "cheese_count": 5}),
        ],
    )
    def test_rejects_invalid_field_value(
        self, field: str, value: int | float, valid_config: dict
    ) -> None:
        """GameConfig rejects invalid field values."""
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(**{field: value, **valid_config})
        assert field in str(exc_info.value)

    def test_rejects_unknown_fields(self) -> None:
        """GameConfig rejects unknown fields (inherits from StrictBaseModel)."""
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(
                width=5,
                height=5,
                max_turns=30,
                cheese_count=5,
                unknown_field="oops",  # type: ignore[call-arg]
            )
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


class TestGameConfigSemanticValidation:
    """Tests for GameConfig semantic validators."""

    def test_cheese_count_fits_in_grid(self) -> None:
        """GameConfig allows cheese_count that fits in grid."""
        # 5x5 = 25 cells, minus 2 for players = 23 max cheese
        config = GameConfig(width=5, height=5, max_turns=30, cheese_count=23)
        assert config.cheese_count == 23

    def test_rejects_too_many_cheese(self) -> None:
        """GameConfig rejects cheese_count that exceeds available cells."""
        # 5x5 = 25 cells, minus 2 for players = 23 max cheese
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(width=5, height=5, max_turns=30, cheese_count=24)
        assert "cheese_count" in str(exc_info.value)
        assert "exceeds" in str(exc_info.value).lower()

    def test_rejects_way_too_many_cheese(self) -> None:
        """GameConfig rejects obviously impossible cheese counts."""
        # 3x3 = 9 cells, minus 2 for players = 7 max cheese
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(width=3, height=3, max_turns=30, cheese_count=100)
        assert "cheese_count" in str(exc_info.value)

    def test_small_grid_max_cheese(self) -> None:
        """GameConfig correctly validates small grids."""
        # 2x2 = 4 cells, minus 2 = 2 max cheese
        config = GameConfig(width=2, height=2, max_turns=10, cheese_count=2)
        assert config.cheese_count == 2

        with pytest.raises(ValidationError):
            GameConfig(width=2, height=2, max_turns=10, cheese_count=3)


class TestGameConfigBuild:
    """Tests for GameConfig.build() method."""

    def test_build_creates_game(self) -> None:
        """GameConfig.build() creates a PyRat game instance."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
            wall_density=0.0,
            mud_density=0.0,
        )
        game = config.build(seed=42)

        assert game.width == 5
        assert game.height == 5
        assert game.max_turns == 30
        assert len(game.cheese_positions()) == 5

    def test_build_no_walls_when_zero_density(self) -> None:
        """GameConfig.build() creates game with no walls when density=0."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
            wall_density=0.0,
            mud_density=0.0,
        )
        game = config.build(seed=42)
        assert len(game.wall_entries()) == 0

    def test_build_no_mud_when_zero_density(self) -> None:
        """GameConfig.build() creates game with no mud when density=0."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
            wall_density=0.0,
            mud_density=0.0,
        )
        game = config.build(seed=42)
        assert len(game.mud_entries()) == 0

    def test_build_uses_pyrat_defaults_when_none(self) -> None:
        """GameConfig.build() uses pyrat defaults when density is None."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
            wall_density=None,
            mud_density=None,
        )
        game = config.build(seed=42)
        # PyRat defaults produce walls and mud
        assert len(game.wall_entries()) > 0
        assert len(game.mud_entries()) > 0

    def test_build_deterministic_with_same_seed(self) -> None:
        """GameConfig.build() produces same game with same seed."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese_count=5,
            wall_density=None,
            mud_density=None,
        )
        game1 = config.build(seed=12345)
        game2 = config.build(seed=12345)

        # Check cheese positions are the same
        cheese1 = sorted((c.x, c.y) for c in game1.cheese_positions())
        cheese2 = sorted((c.x, c.y) for c in game2.cheese_positions())
        assert cheese1 == cheese2

    def test_build_different_with_different_seed(self) -> None:
        """GameConfig.build() produces different games with different seeds."""
        config = GameConfig(
            width=10,
            height=10,
            max_turns=50,
            cheese_count=20,
            wall_density=None,
            mud_density=None,
        )
        game1 = config.build(seed=1)
        game2 = config.build(seed=2)

        # With enough cheese and randomness, positions should differ
        cheese1 = sorted((c.x, c.y) for c in game1.cheese_positions())
        cheese2 = sorted((c.x, c.y) for c in game2.cheese_positions())
        # Very unlikely to be equal with different seeds
        assert cheese1 != cheese2


class TestGameConfigFromDict:
    """Tests for loading GameConfig from dict (common YAML pattern)."""

    def test_load_from_dict(self) -> None:
        """GameConfig loads from dict."""
        data = {
            "width": 5,
            "height": 5,
            "max_turns": 30,
            "cheese_count": 5,
        }
        config = GameConfig.model_validate(data)
        assert config.width == 5
        assert config.cheese_count == 5

    def test_load_from_dict_with_null_densities(self) -> None:
        """GameConfig handles null densities from YAML."""
        data = {
            "width": 5,
            "height": 5,
            "max_turns": 30,
            "cheese_count": 5,
            "wall_density": None,
            "mud_density": None,
        }
        config = GameConfig.model_validate(data)
        assert config.wall_density is None
        assert config.mud_density is None
