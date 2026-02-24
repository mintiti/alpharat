"""Tests for GameConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alpharat.config.game import (
    CheeseConfig,
    ClassicMaze,
    GameConfig,
    OpenMaze,
    RandomMaze,
)


class TestGameConfigValidation:
    """Tests for GameConfig field validation."""

    def test_accepts_valid_config(self) -> None:
        """GameConfig accepts valid configuration."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese=CheeseConfig(count=5),
        )
        assert config.width == 5
        assert config.height == 5
        assert config.max_turns == 30
        assert config.cheese.count == 5
        assert config.cheese.symmetric is True  # default
        assert isinstance(config.maze, OpenMaze)  # default
        assert config.positions == "corners"  # default

    def test_accepts_all_fields(self) -> None:
        """GameConfig accepts all fields including maze and positions."""
        config = GameConfig(
            width=7,
            height=9,
            max_turns=50,
            maze=RandomMaze(wall_density=0.3, mud_density=0.1, symmetric=False),
            positions="random",
            cheese=CheeseConfig(count=10, symmetric=False),
        )
        assert isinstance(config.maze, RandomMaze)
        assert config.maze.wall_density == 0.3
        assert config.maze.mud_density == 0.1
        assert config.maze.symmetric is False
        assert config.positions == "random"
        assert config.cheese.symmetric is False

    def test_classic_maze(self) -> None:
        """GameConfig accepts classic maze type."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=ClassicMaze(),
            cheese=CheeseConfig(count=5),
        )
        assert isinstance(config.maze, ClassicMaze)

    @pytest.mark.parametrize(
        ("field", "value", "valid_config"),
        [
            ("width", 0, {"height": 5, "max_turns": 30, "cheese": {"count": 5}}),
            ("height", -1, {"width": 5, "max_turns": 30, "cheese": {"count": 5}}),
            ("width", 51, {"height": 5, "max_turns": 30, "cheese": {"count": 5}}),
        ],
    )
    def test_rejects_invalid_field_value(
        self, field: str, value: int | float, valid_config: dict
    ) -> None:
        """GameConfig rejects invalid field values."""
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(**{field: value, **valid_config})
        assert field in str(exc_info.value)

    def test_rejects_invalid_cheese_count(self) -> None:
        """GameConfig rejects cheese count of 0."""
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(width=5, height=5, max_turns=30, cheese=CheeseConfig(count=0))
        assert "count" in str(exc_info.value)

    def test_rejects_invalid_maze_density(self) -> None:
        """GameConfig rejects out-of-range maze densities."""
        with pytest.raises(ValidationError):
            RandomMaze(wall_density=1.5)
        with pytest.raises(ValidationError):
            RandomMaze(mud_density=-0.1)

    def test_rejects_unknown_fields(self) -> None:
        """GameConfig rejects unknown fields (inherits from StrictBaseModel)."""
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(
                width=5,
                height=5,
                max_turns=30,
                cheese=CheeseConfig(count=5),
                unknown_field="oops",  # type: ignore[call-arg]
            )
        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)

    def test_maze_discriminated_union_from_dict(self) -> None:
        """Maze config dispatches correctly from dict (YAML pattern)."""
        config = GameConfig.model_validate(
            {
                "width": 5,
                "height": 5,
                "max_turns": 30,
                "maze": {"type": "random", "wall_density": 0.5},
                "cheese": {"count": 5},
            }
        )
        assert isinstance(config.maze, RandomMaze)
        assert config.maze.wall_density == 0.5


class TestGameConfigSemanticValidation:
    """Tests for GameConfig semantic validators."""

    def test_cheese_count_fits_in_grid(self) -> None:
        """GameConfig allows cheese count that fits in grid."""
        # 5x5 = 25 cells, minus 2 for players = 23 max cheese
        config = GameConfig(width=5, height=5, max_turns=30, cheese=CheeseConfig(count=23))
        assert config.cheese.count == 23

    def test_rejects_too_many_cheese(self) -> None:
        """GameConfig rejects cheese count that exceeds available cells."""
        # 5x5 = 25 cells, minus 2 for players = 23 max cheese
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(width=5, height=5, max_turns=30, cheese=CheeseConfig(count=24))
        assert "cheese.count" in str(exc_info.value)
        assert "exceeds" in str(exc_info.value).lower()

    def test_rejects_way_too_many_cheese(self) -> None:
        """GameConfig rejects obviously impossible cheese counts."""
        # 3x3 = 9 cells, minus 2 for players = 7 max cheese
        with pytest.raises(ValidationError) as exc_info:
            GameConfig(width=3, height=3, max_turns=30, cheese=CheeseConfig(count=100))
        assert "cheese.count" in str(exc_info.value)

    def test_small_grid_max_cheese(self) -> None:
        """GameConfig correctly validates small grids."""
        # 2x2 = 4 cells, minus 2 = 2 max cheese
        config = GameConfig(width=2, height=2, max_turns=10, cheese=CheeseConfig(count=2))
        assert config.cheese.count == 2

        with pytest.raises(ValidationError):
            GameConfig(width=2, height=2, max_turns=10, cheese=CheeseConfig(count=3))


class TestGameConfigBuild:
    """Tests for GameConfig.build() method."""

    def test_build_creates_game(self) -> None:
        """GameConfig.build() creates a PyRat game instance."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            cheese=CheeseConfig(count=5),
        )
        game = config.build(seed=42)

        assert game.width == 5
        assert game.height == 5
        assert game.max_turns == 30
        assert len(game.cheese_positions()) == 5

    def test_build_open_maze_no_walls(self) -> None:
        """GameConfig.build() with open maze creates game with no walls."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=OpenMaze(),
            cheese=CheeseConfig(count=5),
        )
        game = config.build(seed=42)
        assert len(game.wall_entries()) == 0
        assert len(game.mud_entries()) == 0

    def test_build_classic_maze_has_walls(self) -> None:
        """GameConfig.build() with classic maze creates game with walls."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=ClassicMaze(),
            cheese=CheeseConfig(count=5),
        )
        game = config.build(seed=42)
        # PyRat classic maze produces walls and mud
        assert len(game.wall_entries()) > 0
        assert len(game.mud_entries()) > 0

    def test_build_deterministic_with_same_seed(self) -> None:
        """GameConfig.build() produces same game with same seed."""
        config = GameConfig(
            width=5,
            height=5,
            max_turns=30,
            maze=ClassicMaze(),
            cheese=CheeseConfig(count=5),
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
            maze=ClassicMaze(),
            cheese=CheeseConfig(count=20),
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
            "cheese": {"count": 5},
        }
        config = GameConfig.model_validate(data)
        assert config.width == 5
        assert config.cheese.count == 5

    def test_load_from_dict_with_maze(self) -> None:
        """GameConfig handles maze config from dict."""
        data = {
            "width": 5,
            "height": 5,
            "max_turns": 30,
            "maze": {"type": "classic"},
            "cheese": {"count": 5, "symmetric": False},
        }
        config = GameConfig.model_validate(data)
        assert isinstance(config.maze, ClassicMaze)
        assert config.cheese.symmetric is False
