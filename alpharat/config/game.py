"""Game configuration with semantic validation.

GameConfig replaces GameParams with:
- Strict field validation (ranges, constraints)
- Semantic validators (cheese fits in grid, etc.)
- build() method for creating PyRat instances
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import Field, model_validator

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    from pyrat_engine.core import GameConfig as EngineGameConfig
    from pyrat_engine.core.game import PyRat


class GameConfig(StrictBaseModel):
    """Game/environment configuration with validation.

    Validates:
    - Field ranges (width/height 1-50, densities 0-1, etc.)
    - Semantic constraints (cheese_count fits in grid)

    Example YAML:
        width: 5
        height: 5
        max_turns: 30
        cheese_count: 5
        wall_density: 0.0
        mud_density: 0.0
        symmetric: true
    """

    width: int = Field(gt=0, le=50, description="Maze width (1-50)")
    height: int = Field(gt=0, le=50, description="Maze height (1-50)")
    max_turns: int = Field(gt=0, description="Maximum game turns")
    cheese_count: int = Field(gt=0, description="Number of cheese pieces")
    wall_density: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Wall density (0-1). None uses pyrat default (0.7)",
    )
    mud_density: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Mud density (0-1). None uses pyrat default (0.1)",
    )
    symmetric: bool = Field(
        default=True,
        description="Symmetric maze/cheese generation (recommended for fair games)",
    )

    @model_validator(mode="after")
    def check_cheese_fits(self) -> Self:
        """Ensure cheese count doesn't exceed available cells.

        Grid has width*height cells, minus 2 for player starting positions.
        """
        max_cheese = self.width * self.height - 2
        if self.cheese_count > max_cheese:
            raise ValueError(
                f"cheese_count ({self.cheese_count}) exceeds available cells "
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
        if self.wall_density is not None or self.mud_density is not None or not self.symmetric:
            builder = builder.with_random_maze(
                wall_density=self.wall_density if self.wall_density is not None else 0.7,
                mud_density=self.mud_density if self.mud_density is not None else 0.1,
                symmetric=self.symmetric,
            )
        else:
            builder = builder.with_classic_maze()
        builder = builder.with_corner_positions()
        builder = builder.with_random_cheese(self.cheese_count, symmetric=self.symmetric)
        return builder.build()

    def build(self, seed: int) -> PyRat:
        """Create a PyRat game instance from this config.

        Args:
            seed: Random seed for maze generation.

        Returns:
            Configured PyRat game instance.
        """
        return self.to_engine_config().create(seed=seed)
