"""Core game state and management classes.

This module contains the main game state management:
- PyRat: The core game engine
- MoveUndo: Undo information for game tree search
"""

from pyrat_engine.core.observation import GameObservation
from pyrat_engine.core.types import Coordinates, Mud, Wall

class MoveUndo:
    """Information needed to undo a move in the game.

    This class contains all state information required to reverse a move,
    including player positions, scores, and collected cheese.
    """
    @property
    def p1_pos(self) -> Coordinates:
        """Player 1's position before the move."""
        ...

    @property
    def p2_pos(self) -> Coordinates:
        """Player 2's position before the move."""
        ...

    @property
    def p1_target(self) -> Coordinates:
        """Position player 1 was attempting to move to."""
        ...

    @property
    def p2_target(self) -> Coordinates:
        """Position player 2 was attempting to move to."""
        ...

    @property
    def p1_mud(self) -> int:
        """Number of mud turns remaining for player 1."""
        ...

    @property
    def p2_mud(self) -> int:
        """Number of mud turns remaining for player 2."""
        ...

    @property
    def p1_score(self) -> float:
        """Player 1's score before the move."""
        ...

    @property
    def p2_score(self) -> float:
        """Player 2's score before the move."""
        ...

    @property
    def p1_misses(self) -> int:
        """Number of failed moves for player 1."""
        ...

    @property
    def p2_misses(self) -> int:
        """Number of failed moves for player 2."""
        ...

    @property
    def collected_cheese(self) -> list[Coordinates]:
        """List of positions where cheese was collected during this move."""
        ...

    @property
    def turn(self) -> int:
        """Turn number before the move was made."""
        ...

class PyRat:
    """Core PyRat game engine implementation in Rust.

    This class provides the main interface to the Rust game engine.
    It manages all game state including:
    - Player positions and scores
    - Cheese placement and collection
    - Mud effects and movement delays
    - Turn counting and game termination

    Args:
        width: Board width (default: 21)
        height: Board height (default: 15)
        cheese_count: Number of cheese pieces (default: 41)
        symmetric: Whether to generate symmetric mazes (default: True)
        seed: Random seed for reproducible games (default: None)
        max_turns: Maximum number of turns before game ends (default: 300)
        wall_density: Proportion of walls in the maze, 0.0 to 1.0 (default: 0.7)
        mud_density: Proportion of passages with mud, 0.0 to 1.0 (default: 0.1)
    """
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        cheese_count: int | None = None,
        symmetric: bool = True,
        seed: int | None = None,
        max_turns: int | None = None,
        wall_density: float | None = None,
        mud_density: float | None = None,
    ) -> None: ...
    @staticmethod
    def create_preset(
        preset: str = "default",
        *,
        seed: int | None = None,
    ) -> PyRat:
        """Create a game from a preset configuration.

        Available presets:
        - "tiny": 11x9 board, 13 cheese, 150 turns
        - "small": 15x11 board, 21 cheese, 200 turns
        - "default": 21x15 board, 41 cheese, 300 turns (standard)
        - "large": 31x21 board, 85 cheese, 400 turns
        - "huge": 41x31 board, 165 cheese, 500 turns
        - "empty": No walls or mud, good for testing
        - "asymmetric": Standard size but asymmetric generation

        Args:
            preset: Name of the preset configuration
            seed: Random seed for reproducible games

        Returns:
            A new PyRat instance with the preset configuration
        """
        ...

    @staticmethod
    def create_custom(
        width: int,
        height: int,
        walls: list[Wall] | list[tuple[tuple[int, int], tuple[int, int]]] = [],
        mud: list[Mud] | list[tuple[tuple[int, int], tuple[int, int], int]] = [],
        cheese: list[Coordinates] | list[tuple[int, int]] = [],
        player1_pos: Coordinates | tuple[int, int] | None = None,
        player2_pos: Coordinates | tuple[int, int] | None = None,
        max_turns: int = 300,
        symmetric: bool = True,
    ) -> PyRat:
        """Create a game with a fully specified maze configuration.

        Args:
            width: Width of the game board
            height: Height of the game board
            walls: List of wall pairs, each defined by two (x,y) positions
            mud: List of mud patches, each defined by two positions and mud value
            cheese: List of cheese positions as (x,y) coordinates
            player1_pos: Starting position for player 1 (default: (0,0))
            player2_pos: Starting position for player 2 (default: (width-1, height-1))
            max_turns: Maximum number of turns before game ends
            symmetric: If True (default), validate that walls, mud, cheese, and
                player positions are 180 degrees rotationally symmetric.

        Returns:
            A new PyRat instance with the specified configuration
        """
        ...

    @staticmethod
    def create_from_maze(
        width: int,
        height: int,
        walls: list[Wall] | list[tuple[tuple[int, int], tuple[int, int]]],
        *,
        seed: int | None = None,
        max_turns: int = 300,
        symmetric: bool = True,
    ) -> PyRat:
        """Create a game with a specific maze layout and random cheese placement.

        Args:
            width: Width of the game board
            height: Height of the game board
            walls: List of wall pairs, each defined by two (x,y) positions
            seed: Random seed for reproducible cheese placement
            max_turns: Maximum number of turns before game ends
            symmetric: If True (default), validate walls are symmetric and
                generate symmetric cheese/player positions.

        Returns:
            A new PyRat instance with the specified maze and random cheese
        """
        ...

    @staticmethod
    def create_from_walls(
        width: int,
        height: int,
        walls: list[Wall],
        *,
        seed: int | None = None,
        max_turns: int = 300,
        symmetric: bool = True,
    ) -> PyRat:
        """Create a game from a list of validated Wall objects.

        Args:
            width: Width of the game board
            height: Height of the game board
            walls: List of Wall objects defining the maze structure
            seed: Random seed for reproducible cheese placement
            max_turns: Maximum number of turns before game ends
            symmetric: If True (default), validate walls are symmetric and
                generate symmetric cheese/player positions.

        Returns:
            A new PyRat instance with the specified walls and random cheese
        """
        ...

    @staticmethod
    def create_with_starts(
        width: int,
        height: int,
        player1_start: Coordinates | tuple[int, int],
        player2_start: Coordinates | tuple[int, int],
        *,
        preset: str = "default",
        seed: int | None = None,
    ) -> PyRat:
        """Create a game with custom starting positions.

        Args:
            width: Width of the game board
            height: Height of the game board
            player1_start: Starting (x,y) position for player 1
            player2_start: Starting (x,y) position for player 2
            preset: Preset configuration to use for maze generation
            seed: Random seed for reproducible maze generation

        Returns:
            A new PyRat instance with custom starting positions
        """
        ...

    @property
    def width(self) -> int:
        """Width of the game board."""
        ...

    @property
    def height(self) -> int:
        """Height of the game board."""
        ...

    @property
    def turn(self) -> int:
        """Current turn number (starts at 0)."""
        ...

    @property
    def max_turns(self) -> int:
        """Maximum number of turns before the game ends."""
        ...

    @property
    def player1_position(self) -> Coordinates:
        """Current position of player 1."""
        ...

    @property
    def player2_position(self) -> Coordinates:
        """Current position of player 2."""
        ...

    @property
    def player1_score(self) -> float:
        """Current score of player 1."""
        ...

    @property
    def player2_score(self) -> float:
        """Current score of player 2."""
        ...

    @property
    def player1_mud_turns(self) -> int:
        """Number of turns player 1 remains stuck in mud (0 if not in mud)."""
        ...

    @property
    def player2_mud_turns(self) -> int:
        """Number of turns player 2 remains stuck in mud (0 if not in mud)."""
        ...

    def cheese_positions(self) -> list[Coordinates]:
        """Get positions of all remaining cheese pieces.

        Returns:
            List of Coordinates where cheese pieces are located
        """
        ...

    def mud_entries(self) -> list[Mud]:
        """Get all mud patches in the maze.

        Returns:
            List of Mud objects, each with pos1, pos2, and value attributes
        """
        ...

    def wall_entries(self) -> list[Wall]:
        """Get all walls in the maze.

        Returns:
            List of Wall objects, each with pos1 and pos2 attributes
        """
        ...

    def get_valid_moves(self, pos: Coordinates | tuple[int, int]) -> list[int]:
        """Get valid movement directions from a position.

        Returns direction values where movement is possible (not blocked by
        walls or board boundaries). Does not include STAY. Does not account
        for mud state.

        Direction values: UP=0, RIGHT=1, DOWN=2, LEFT=3.
        Convert to Direction enums with: ``[Direction(v) for v in valid]``

        Args:
            pos: Position to check, as Coordinates or (x, y) tuple

        Returns:
            List of direction integers for valid moves

        Raises:
            ValueError: If position is outside board bounds
        """
        ...

    def effective_actions(self, pos: Coordinates | tuple[int, int]) -> list[int]:
        """Get effective action mapping for a position (ignores mud state).

        Returns a list of 5 integers where ``result[action] = effective_action``.
        Blocked actions (walls, boundaries) map to STAY (4).
        Valid actions map to themselves.

        Direction values: UP=0, RIGHT=1, DOWN=2, LEFT=3, STAY=4.

        Args:
            pos: Position to check, as Coordinates or (x, y) tuple

        Returns:
            List of 5 integers mapping each action to its effective action

        Raises:
            ValueError: If position is outside board bounds
        """
        ...

    def effective_actions_p1(self) -> list[int]:
        """Get effective action mapping for player 1, accounting for mud.

        If player 1 is in mud, all actions map to STAY: ``[4, 4, 4, 4, 4]``.
        Otherwise, returns the same as ``effective_actions(player1_position)``.

        Returns:
            List of 5 integers mapping each action to its effective action
        """
        ...

    def effective_actions_p2(self) -> list[int]:
        """Get effective action mapping for player 2, accounting for mud.

        If player 2 is in mud, all actions map to STAY: ``[4, 4, 4, 4, 4]``.
        Otherwise, returns the same as ``effective_actions(player2_position)``.

        Returns:
            List of 5 integers mapping each action to its effective action
        """
        ...

    def step(self, p1_move: int, p2_move: int) -> tuple[bool, list[Coordinates]]:
        """Execute one game step with the given moves.

        Args:
            p1_move: Direction for player 1 (0-4: UP, RIGHT, DOWN, LEFT, STAY)
            p2_move: Direction for player 2 (0-4: UP, RIGHT, DOWN, LEFT, STAY)

        Returns:
            Tuple of (game_over, collected_cheese) where game_over is True
            if the game ended this turn and collected_cheese lists positions
            where cheese was picked up.
        """
        ...

    def make_move(self, p1_move: int, p2_move: int) -> MoveUndo:
        """Execute a move and return undo information for backtracking.

        Args:
            p1_move: Direction for player 1 (0-4: UP, RIGHT, DOWN, LEFT, STAY)
            p2_move: Direction for player 2 (0-4: UP, RIGHT, DOWN, LEFT, STAY)

        Returns:
            MoveUndo object that must be passed to ``unmake_move()`` to revert
            this move.
        """
        ...

    def unmake_move(self, undo: MoveUndo) -> None:
        """Revert a move using saved undo information.

        Args:
            undo: MoveUndo object from a previous ``make_move()`` call
        """
        ...

    def reset(self, seed: int | None = None) -> None:
        """Reset the game to initial state.

        Args:
            seed: Optional random seed for reproducible maze generation
        """
        ...

    def get_observation(self, is_player_one: bool) -> GameObservation:
        """Get the current game observation for a player.

        Args:
            is_player_one: True to get player 1's perspective, False for player 2

        Returns:
            GameObservation containing the game state from the player's perspective
        """
        ...

# Alias for the Rust MoveUndo type
PyMoveUndo = MoveUndo
