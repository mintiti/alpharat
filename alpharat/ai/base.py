"""Base class for PyRat AI agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


class Agent(ABC):
    """Base class for PyRat AI agents.

    Agents receive the game state and return an action (0-4).
    They can maintain state between turns (e.g., for tree reuse).

    Player convention:
        player=1 means Rat (player 1)
        player=2 means Python (player 2)
    """

    @abstractmethod
    def get_move(self, game: PyRat, player: int) -> int:
        """Select an action given the current game state.

        Args:
            game: The PyRat game instance. DO NOT modify this.
            player: Which player we are (1 = Rat, 2 = Python).

        Returns:
            Action index (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY).
        """
        ...

    def reset(self) -> None:
        """Reset agent state for a new game.

        Override this if your agent maintains state between turns.
        Default implementation does nothing.
        """
        return  # noqa: B027

    def observe_move(self, action_p1: int, action_p2: int) -> None:
        """Called after both players' actions are known.

        Enables tree reuse: agents can advance internal state based on actual moves.
        Override this if your agent maintains state (e.g., MCTS tree) between turns.

        Args:
            action_p1: Player 1's action (0-4).
            action_p2: Player 2's action (0-4).
        """
        return  # noqa: B027

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        return self.__class__.__name__
