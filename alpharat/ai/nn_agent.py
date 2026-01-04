"""Neural network agent for PyRat games.

Thin wrapper around MCTSAgent with simulations=0 (pure NN mode).
Prefer using MCTSAgentConfig with simulations=0 directly for new code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpharat.ai.base import Agent
from alpharat.ai.mcts_agent import MCTSAgent
from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig

if TYPE_CHECKING:
    from pathlib import Path

    from pyrat_engine.core.game import PyRat


class NNAgent(Agent):
    """Agent that uses a trained neural network to select actions.

    This is a convenience wrapper around MCTSAgent with simulations=0.
    For new code, prefer using MCTSAgent directly or NNAgentConfig.

    Attributes:
        temperature: Sampling temperature (0 = argmax, >0 = sample).
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        temperature: float = 0.0,
        device: str = "cpu",
    ) -> None:
        """Initialize NN agent from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file).
            temperature: Sampling temperature. 0 = argmax, >0 = sample from policy.
            device: Device to run inference on ("cpu", "cuda", "mps").
        """
        mcts_config = DecoupledPUCTConfig(simulations=0)
        self._agent = MCTSAgent(
            mcts_config=mcts_config,
            checkpoint=str(checkpoint_path),
            temperature=temperature,
            device=device,
        )
        self.temperature = temperature

    def get_move(self, game: PyRat, player: int) -> int:
        """Select action using neural network policy."""
        return self._agent.get_move(game, player)

    def reset(self) -> None:
        """Reset for new game."""
        self._agent.reset()

    def observe_move(self, action_p1: int, action_p2: int) -> None:
        """Forward move observation to wrapped agent."""
        self._agent.observe_move(action_p1, action_p2)

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        temp_str = "argmax" if self.temperature == 0 else f"t={self.temperature}"
        return f"NN({temp_str})"
