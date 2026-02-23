"""Type stubs for the game builder API."""

from pyrat_engine.core.game import PyRat
from pyrat_engine.core.types import Coordinates, Mud, Wall

class GameBuilder:
    """Two-phase builder for creating PyRat game instances.

    GameBuilder(w, h) → configure → .build() → GameConfig → .create(seed) → PyRat

    Example:
        >>> game = (GameBuilder(5, 5)
        ...         .with_max_turns(100)
        ...         .with_open_maze()
        ...         .with_custom_positions(Coordinates(0, 0), Coordinates(4, 4))
        ...         .with_custom_cheese([Coordinates(2, 2)])
        ...         .build()
        ...         .create(seed=42))
    """

    def __init__(self, width: int, height: int) -> None: ...
    def with_max_turns(self, max_turns: int) -> GameBuilder: ...

    # Maze configuration (pick one)
    def with_open_maze(self) -> GameBuilder: ...
    def with_classic_maze(self) -> GameBuilder: ...
    def with_random_maze(
        self,
        *,
        wall_density: float = 0.7,
        mud_density: float = 0.1,
        symmetric: bool = True,
    ) -> GameBuilder: ...
    def with_custom_maze(
        self,
        walls: list[Wall] | list[tuple[tuple[int, int], tuple[int, int]]],
        mud: list[Mud] | list[tuple[tuple[int, int], tuple[int, int], int]],
    ) -> GameBuilder: ...

    # Player positions (pick one)
    def with_corner_positions(self) -> GameBuilder: ...
    def with_random_positions(self) -> GameBuilder: ...
    def with_custom_positions(
        self,
        p1: Coordinates | tuple[int, int],
        p2: Coordinates | tuple[int, int],
    ) -> GameBuilder: ...

    # Cheese configuration (pick one)
    def with_random_cheese(self, count: int, symmetric: bool = True) -> GameBuilder: ...
    def with_custom_cheese(
        self, positions: list[Coordinates] | list[tuple[int, int]]
    ) -> GameBuilder: ...
    def build(self) -> GameConfig: ...

class GameConfig:
    """Reusable game configuration. Call .create(seed) to instantiate games."""

    @staticmethod
    def classic(width: int, height: int, cheese_count: int) -> GameConfig: ...
    def create(self, seed: int | None = None) -> PyRat: ...
