"""Game configuration with three independent axes: maze, positions, cheese.

GameConfig uses discriminated unions for maze type and nested CheeseConfig,
matching the pyrat-engine two-phase builder API (GameBuilder → GameConfig → .create(seed)).

Example YAML:
    width: 5
    height: 5
    max_turns: 30
    maze:
      type: open
    positions: corners
    cheese:
      count: 5
      symmetric: true
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, model_validator

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    from pyrat_engine.core import GameConfig as EngineGameConfig
    from pyrat_engine.core.game import PyRat


# --- Maze configs (discriminated union) ---


class OpenMaze(StrictBaseModel):
    """Open maze — no walls, no mud."""

    type: Literal["open"] = "open"


class ClassicMaze(StrictBaseModel):
    """Classic maze — pyrat-engine default wall/mud generation."""

    type: Literal["classic"] = "classic"


class RandomMaze(StrictBaseModel):
    """Random maze with configurable densities."""

    type: Literal["random"] = "random"
    wall_density: float = Field(default=0.7, ge=0.0, le=1.0)
    mud_density: float = Field(default=0.1, ge=0.0, le=1.0)
    symmetric: bool = Field(default=True)


MazeConfig = Annotated[OpenMaze | ClassicMaze | RandomMaze, Field(discriminator="type")]


# --- Cheese config ---


class CheeseConfig(StrictBaseModel):
    """Cheese placement configuration."""

    count: int = Field(gt=0)
    symmetric: bool = Field(default=True)


# --- Main GameConfig ---


class GameConfig(StrictBaseModel):
    """Game/environment configuration with three independent axes.

    Axes:
    - **maze**: open | classic | random (discriminated union)
    - **positions**: corners | random
    - **cheese**: count + symmetric

    Example YAML:
        width: 5
        height: 5
        max_turns: 30
        maze:
          type: open
        positions: corners
        cheese:
          count: 5
          symmetric: true
    """

    width: int = Field(gt=0, le=50)
    height: int = Field(gt=0, le=50)
    max_turns: int = Field(gt=0)
    maze: MazeConfig = Field(default_factory=OpenMaze)
    positions: Literal["corners", "random"] = "corners"
    cheese: CheeseConfig

    @model_validator(mode="after")
    def check_cheese_fits(self) -> Self:
        """Ensure cheese count doesn't exceed available cells.

        Grid has width*height cells, minus 2 for player starting positions.
        """
        max_cheese = self.width * self.height - 2
        if self.cheese.count > max_cheese:
            raise ValueError(
                f"cheese.count ({self.cheese.count}) exceeds available cells "
                f"({max_cheese} = {self.width}x{self.height} - 2 player positions)"
            )
        return self

    def to_engine_config(self) -> EngineGameConfig:
        """Build a reusable pyrat_engine GameConfig template.

        The returned config can stamp out game instances cheaply via .create(seed=N).
        """
        from pyrat_engine.core import GameBuilder

        builder = GameBuilder(self.width, self.height)
        builder = builder.with_max_turns(self.max_turns)

        # Maze axis
        match self.maze:
            case OpenMaze():
                builder = builder.with_open_maze()
            case ClassicMaze():
                builder = builder.with_classic_maze()
            case RandomMaze(wall_density=wd, mud_density=md, symmetric=sym):
                builder = builder.with_random_maze(
                    wall_density=wd,
                    mud_density=md,
                    symmetric=sym,
                )

        # Positions axis
        if self.positions == "corners":
            builder = builder.with_corner_positions()
        else:
            builder = builder.with_random_positions()

        # Cheese axis
        builder = builder.with_random_cheese(self.cheese.count, symmetric=self.cheese.symmetric)

        return builder.build()

    def build(self, seed: int) -> PyRat:
        """Create a PyRat game instance from this config.

        Args:
            seed: Random seed for maze generation.

        Returns:
            Configured PyRat game instance.
        """
        return self.to_engine_config().create(seed=seed)
