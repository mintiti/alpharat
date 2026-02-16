"""Core types for the PyRat game engine.

This module contains the fundamental types used throughout the engine:
- Coordinates: Position on the game board
- Direction: Movement directions (IntEnum)
- Wall: Barriers between cells
- Mud: Passages that slow movement
"""

from collections.abc import Iterator
from enum import IntEnum

class Coordinates:
    """A position on the game board with x and y coordinates.

    This is the core position type used throughout the engine.
    Coordinates are 0-indexed, with (0,0) at the bottom-left corner.

    Args:
        x: X coordinate (0-255)
        y: Y coordinate (0-255)

    Attributes:
        x: The x coordinate (read-only)
        y: The y coordinate (read-only)
    """

    x: int
    y: int

    def __init__(self, x: int, y: int) -> None: ...
    def get_neighbor(self, direction: int) -> Coordinates:
        """Get the coordinates of an adjacent cell in the given direction.

        Args:
            direction: Direction value (0-4: UP, RIGHT, DOWN, LEFT, STAY)

        Returns:
            New Coordinates object for the neighboring position

        Raises:
            ValueError: If direction is not in range 0-4
        """
        ...

    def is_adjacent_to(self, other: Coordinates) -> bool:
        """Check if this position is adjacent to another (not diagonally).

        Args:
            other: Another Coordinates object

        Returns:
            True if positions are orthogonally adjacent, False otherwise
        """
        ...

    def manhattan_distance(self, other: Coordinates) -> int:
        """Calculate Manhattan distance to another position.

        Args:
            other: Another Coordinates object

        Returns:
            Manhattan distance (sum of absolute differences in x and y)
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __getitem__(self, index: int) -> int: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __add__(self, other: tuple[int, int] | int) -> Coordinates:
        """Add a delta tuple or Direction to this position.

        Args:
            other: Either:
                - tuple (dx, dy): Delta to add (can be negative)
                - int (0-4): Direction value to move in

        Returns:
            New Coordinates after addition (saturates at 0 and 255)

        Raises:
            ValueError: If direction value is invalid (not 0-4)
        """
        ...

    def __sub__(self, other: Coordinates | tuple[int, int]) -> tuple[int, int]:
        """Calculate delta between this position and another.

        Args:
            other: Position to subtract (Coordinates or tuple)

        Returns:
            Signed tuple (dx, dy) representing the delta
        """
        ...

class Direction(IntEnum):
    """Movement directions in the game.

    This is an IntEnum with the following values:
    - UP = 0: Move up (increase y)
    - RIGHT = 1: Move right (increase x)
    - DOWN = 2: Move down (decrease y)
    - LEFT = 3: Move left (decrease x)
    - STAY = 4: Don't move

    As an IntEnum, Direction members are also integers:
    - Direction.UP == 0 is True
    - isinstance(Direction.UP, int) is True
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4

class Wall:
    """A wall between two adjacent cells that blocks movement.

    Walls are always between orthogonally adjacent cells.
    The order of positions is normalized (smaller position first).

    Args:
        pos1: First position
        pos2: Second position (must be adjacent to pos1)

    Raises:
        ValueError: If positions are not adjacent

    Attributes:
        pos1: First position (read-only)
        pos2: Second position (read-only)
    """

    pos1: Coordinates
    pos2: Coordinates

    def __init__(
        self,
        pos1: Coordinates | tuple[int, int],
        pos2: Coordinates | tuple[int, int],
    ) -> None: ...
    def blocks_movement(self, from_pos: Coordinates, to_pos: Coordinates) -> bool:
        """Check if this wall blocks movement between two positions.

        Args:
            from_pos: Starting position
            to_pos: Target position

        Returns:
            True if the wall blocks this movement, False otherwise
        """
        ...

    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __iter__(self) -> Iterator[Coordinates]:
        """Iterate over pos1 and pos2 for unpacking."""
        ...

    def __len__(self) -> int:
        """Return 2 (number of positions)."""
        ...

class Mud:
    """A muddy passage between two cells that slows movement.

    Mud makes movement take multiple turns. A mud value of N means
    it takes N turns total to traverse (including the initial turn).

    Args:
        pos1: First position
        pos2: Second position (must be adjacent to pos1)
        value: Number of turns to traverse (must be >= 2)

    Raises:
        ValueError: If positions are not adjacent or value < 2

    Attributes:
        pos1: First position (read-only)
        pos2: Second position (read-only)
        value: Number of turns to traverse (read-only)
    """

    pos1: Coordinates
    pos2: Coordinates
    value: int

    def __init__(
        self,
        pos1: Coordinates | tuple[int, int],
        pos2: Coordinates | tuple[int, int],
        value: int,
    ) -> None: ...
    def affects_movement(self, from_pos: Coordinates, to_pos: Coordinates) -> bool:
        """Check if this mud affects movement between two positions.

        Args:
            from_pos: Starting position
            to_pos: Target position

        Returns:
            True if this mud affects the movement, False otherwise
        """
        ...

    def blocks_movement(self, from_pos: Coordinates, to_pos: Coordinates) -> bool:
        """Alias for affects_movement for consistency with Wall API."""
        ...

    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __iter__(self) -> Iterator[Coordinates | int]:
        """Iterate over pos1, pos2, and value for unpacking."""
        ...

    def __len__(self) -> int:
        """Return 3 (number of elements)."""
        ...
