"""Greedy agent that moves toward the closest cheese."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpharat.ai.base import Agent

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=STAY
UP, RIGHT, DOWN, LEFT, STAY = 0, 1, 2, 3, 4


class GreedyAgent(Agent):
    """Agent that always moves toward the closest cheese by Manhattan distance.

    Tie-breaking: if multiple cheeses are equidistant, picks the first one
    (arbitrary but deterministic). If multiple directions reduce distance
    equally, prefers vertical (UP/DOWN) over horizontal.
    """

    def get_move(self, game: PyRat, player: int) -> int:
        """Move toward closest cheese."""
        pos = game.player1_position if player == 1 else game.player2_position

        cheeses = game.cheese_positions()
        if not cheeses:
            return STAY

        # Find closest cheese by Manhattan distance
        target = min(cheeses, key=lambda c: abs(c.x - pos.x) + abs(c.y - pos.y))

        # Coordinates subtraction returns (dx, dy) tuple
        dx, dy = target - pos

        # Prefer vertical movement, then horizontal
        if dy > 0:
            return UP
        if dy < 0:
            return DOWN
        if dx > 0:
            return RIGHT
        if dx < 0:
            return LEFT

        return STAY

    @property
    def name(self) -> str:
        return "Greedy"
