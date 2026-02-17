"""Game configuration builder for custom games.

This module contains the builder pattern for creating custom game configurations.
"""

from pyrat_engine.core.game import PyRat
from pyrat_engine.core.types import Coordinates, Mud, Wall

class GameConfigBuilder:
    """Builder for creating custom PyRat game configurations.

    This class provides a fluent interface for constructing custom game states
    with specific maze layouts, including walls, mud patches, cheese positions,
    and player starting positions.

    Example:
        >>> game = (GameConfigBuilder(width=4, height=4)
        ...         .with_walls([((1, 1), (1, 2))])  # Horizontal wall
        ...         .with_mud([((1, 1), (2, 1), 3)])  # 3-turn mud
        ...         .with_cheese([(1, 2), (3, 1)])
        ...         .with_player1_pos((0, 0))
        ...         .with_player2_pos((3, 3))
        ...         .with_max_turns(100)
        ...         .build())

    Args:
        width: Width of the game board
        height: Height of the game board
    """

    def __init__(self, width: int, height: int) -> None: ...
    def with_walls(
        self, walls: list[Wall] | list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> GameConfigBuilder:
        """Add walls to the maze configuration.

        Args:
            walls: List of wall definitions, where each wall is defined by two adjacent
                  cell positions it blocks movement between.
                  Format: [((x1,y1), (x2,y2)), ...]

        Returns:
            Self for method chaining
        """
        ...

    def with_mud(
        self, mud: list[Mud] | list[tuple[tuple[int, int], tuple[int, int], int]]
    ) -> GameConfigBuilder:
        """Add mud patches to the maze configuration.

        Args:
            mud: List of mud definitions, where each mud is defined by two adjacent
                 cell positions and the number of turns it takes to cross.
                 Format: [((x1,y1), (x2,y2), turns), ...]

        Returns:
            Self for method chaining
        """
        ...

    def with_cheese(self, cheese: list[Coordinates] | list[tuple[int, int]]) -> GameConfigBuilder:
        """Set cheese positions in the maze.

        Args:
            cheese: List of coordinates where cheese should be placed.
                   Format: [(x1,y1), (x2,y2), ...]

        Returns:
            Self for method chaining
        """
        ...

    def with_player1_pos(self, pos: Coordinates | tuple[int, int]) -> GameConfigBuilder:
        """Set the starting position for player 1.

        Args:
            pos: Tuple of (x,y) coordinates for player 1's starting position

        Returns:
            Self for method chaining
        """
        ...

    def with_player2_pos(self, pos: Coordinates | tuple[int, int]) -> GameConfigBuilder:
        """Set the starting position for player 2.

        Args:
            pos: Tuple of (x,y) coordinates for player 2's starting position

        Returns:
            Self for method chaining
        """
        ...

    def with_max_turns(self, max_turns: int) -> GameConfigBuilder:
        """Set the maximum number of turns for the game.

        Args:
            max_turns: Maximum number of turns before the game is truncated.
                      Must be greater than 0.

        Returns:
            Self for method chaining
        """
        ...

    def build(self) -> PyRat:
        """Construct and return the final game state.

        Returns:
            A new PyRat instance with the configured parameters

        Raises:
            ValueError: If the configuration is invalid (e.g., invalid positions,
                       overlapping walls/mud, no cheese placed)
        """
        ...

# Rename the class to match the Rust name
PyGameConfigBuilder = GameConfigBuilder
