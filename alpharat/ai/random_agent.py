"""Random agent for PyRat games."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from alpharat.ai.base import Agent

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


class RandomAgent(Agent):
    """Agent that selects uniformly random actions."""

    def get_move(self, game: PyRat, player: int) -> int:
        """Select a random action (0-4)."""
        return random.randint(0, 4)

    @property
    def name(self) -> str:
        return "Random"
