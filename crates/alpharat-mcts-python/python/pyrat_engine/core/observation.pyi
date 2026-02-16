"""Game observation and state tracking classes.

This module contains classes for observing and tracking game state:
- GameObservation: Player's view of the game state
- ObservationHandler: Efficient observation updates
"""

import numpy as np

from pyrat_engine.core.game import PyRat
from pyrat_engine.core.types import Coordinates

class GameObservation:
    """Game state observation from a player's perspective.

    This class provides a complete view of the game state from either player's perspective,
    including positions, scores, mud status, and the current game progression.
    All coordinates are provided as Coordinates objects where (0,0) is the bottom-left corner.

    Note:
        When is_player_one=True in get_observation(), this represents player 1's view.
        When False, player/opponent properties are swapped to represent player 2's view.
    """

    @property
    def player_position(self) -> Coordinates:
        """Current position of the observing player.

        Returns:
            Coordinates of the player's position
        """
        ...

    @property
    def player_mud_turns(self) -> int:
        """Remaining turns the observing player is stuck in mud.

        Returns:
            Number of turns remaining in mud (0 if not in mud)
        """
        ...

    @property
    def player_score(self) -> float:
        """Current score of the observing player.

        Returns:
            Player's score (number of cheese pieces collected)
        """
        ...

    @property
    def opponent_position(self) -> Coordinates:
        """Current position of the opponent player.

        Returns:
            Coordinates of the opponent's position
        """
        ...

    @property
    def opponent_mud_turns(self) -> int:
        """Remaining turns the opponent is stuck in mud.

        Returns:
            Number of turns remaining in mud (0 if not in mud)
        """
        ...

    @property
    def opponent_score(self) -> float:
        """Current score of the opponent player.

        Returns:
            Opponent's score (number of cheese pieces collected)
        """
        ...

    @property
    def current_turn(self) -> int:
        """Current game turn number.

        Returns:
            Current turn (starts at 0)
        """
        ...

    @property
    def max_turns(self) -> int:
        """Maximum number of turns before game truncation.

        Returns:
            Maximum allowed turns for this game
        """
        ...

    @property
    def cheese_matrix(self) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
        """Binary matrix indicating cheese positions.

        Returns:
            2D numpy array of shape (width, height) where 1 indicates
            cheese presence and 0 indicates no cheese
        """
        ...

    @property
    def movement_matrix(self) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        """Matrix encoding valid moves and their costs.

        Returns:
            3D numpy array of shape (width, height, 4) where:
            - The first two dimensions correspond to board positions
            - The third dimension corresponds to moves [UP, RIGHT, DOWN, LEFT]
            - Values:
                -1: Invalid move (wall or out of bounds)
                0: Valid immediate move
                N>0: Valid move with N turns of mud delay
        """
        ...

class ObservationHandler:
    """Handles efficient updates and access to game observations.

    This class manages the observation state for the game, including cheese positions
    and movement constraints, providing efficient updates during gameplay.
    """

    def __init__(self, game: PyRat) -> None:
        """Creates a new observation handler for tracking game state.

        Args:
            game: The game state to create observations for
        """
        ...

    def update_collected_cheese(self, collected: list[Coordinates]) -> None:
        """Updates the observation state after cheese collection.

        Args:
            collected: List of Coordinates where cheese was collected
        """
        ...

    def update_restored_cheese(self, restored: list[Coordinates]) -> None:
        """Updates the observation state when cheese is restored during move undo.

        Args:
            restored: List of Coordinates where cheese should be restored
        """
        ...

    def get_observation(self, game: PyRat, is_player_one: bool) -> GameObservation:
        """Gets the current game observation from a player's perspective.

        Args:
            game: Current game state
            is_player_one: True to get player 1's perspective, False for player 2

        Returns:
            Complete game state observation from the specified player's perspective
        """
        ...

# Rename the classes to match the Rust names
PyGameObservation = GameObservation
PyObservationHandler = ObservationHandler
