"""Generic agent wrapping any Searcher implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpharat.ai.base import Agent
from alpharat.ai.utils import select_action_from_strategy

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat

    from alpharat.mcts.searcher import Searcher


class SearcherAgent(Agent):
    """Agent that delegates to a Searcher for move selection.

    Works with both Python and Rust MCTS backends transparently.

    Args:
        searcher: Any Searcher implementation.
        temperature: Sampling temperature (0 = argmax, 1.0 = proportional).
        simulations: Number of simulations (for display name only).
        checkpoint: Checkpoint path (for display name only).
    """

    def __init__(
        self,
        searcher: Searcher,
        temperature: float = 1.0,
        simulations: int = 0,
        checkpoint: str | None = None,
    ) -> None:
        self._searcher = searcher
        self._temperature = temperature
        self._simulations = simulations
        self._checkpoint = checkpoint

    def get_move(self, game: PyRat, player: int) -> int:
        """Select action using the wrapped Searcher.

        Args:
            game: Current game state (not modified).
            player: Which player we are (1 = Rat, 2 = Python).

        Returns:
            Action index (0-4).
        """
        result = self._searcher.search(game)
        policy = result.policy_p1 if player == 1 else result.policy_p2

        if self._simulations > 0:
            return select_action_from_strategy(policy, temperature=1.0)
        else:
            return select_action_from_strategy(policy, temperature=self._temperature)

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        if self._simulations == 0:
            temp_str = "argmax" if self._temperature == 0 else f"t={self._temperature}"
            return f"NN({temp_str})"

        base = f"MCTS({self._simulations})"
        if self._checkpoint:
            return f"{base}+NN"
        return base
