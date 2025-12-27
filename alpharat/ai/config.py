"""Agent configuration with discriminated union pattern.

Each config type inherits from AgentConfigBase and implements `build()`.
Use Pydantic's discriminator on the `variant` field for automatic dispatch.

Example YAML:
    agents:
      random:
        variant: random
      greedy:
        variant: greedy
      pure_nn:
        variant: nn
        checkpoint: checkpoints/best_model.pt
      mcts_baseline:
        variant: mcts
        simulations: 200
      mcts_with_nn:
        variant: mcts
        checkpoint: checkpoints/best_model.pt
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from alpharat.ai.base import Agent


class AgentConfigBase(BaseModel):
    """Base class for agent configurations.

    All agent configs must implement the `build()` method that constructs
    the corresponding Agent instance.
    """

    @abstractmethod
    def build(self, device: str = "cpu") -> Agent:
        """Build the agent from this configuration.

        Args:
            device: Device for NN inference ("cpu", "cuda", "mps").

        Returns:
            Configured Agent instance.
        """
        ...


class RandomAgentConfig(AgentConfigBase):
    """Configuration for random agent."""

    variant: Literal["random"] = "random"

    def build(self, device: str = "cpu") -> Agent:
        """Build a RandomAgent."""
        from alpharat.ai.random_agent import RandomAgent

        return RandomAgent()


class GreedyAgentConfig(AgentConfigBase):
    """Configuration for greedy agent."""

    variant: Literal["greedy"] = "greedy"

    def build(self, device: str = "cpu") -> Agent:
        """Build a GreedyAgent."""
        from alpharat.ai.greedy_agent import GreedyAgent

        return GreedyAgent()


class NNAgentConfig(AgentConfigBase):
    """Configuration for pure neural network agent.

    Uses the NN policy head directly, no MCTS search.
    Syntactic sugar for MCTSAgentConfig with simulations=0.
    """

    variant: Literal["nn"] = "nn"
    checkpoint: str
    temperature: float = 0.0  # 0 = argmax

    def build(self, device: str = "cpu") -> Agent:
        """Build an MCTSAgent with simulations=0 (pure NN mode)."""
        from alpharat.ai.mcts_agent import MCTSAgent

        return MCTSAgent(
            simulations=0,
            checkpoint=self.checkpoint,
            temperature=self.temperature,
            device=device,
        )


class MCTSAgentConfig(AgentConfigBase):
    """Configuration for MCTS agent with optional NN priors.

    When checkpoint is set, uses NN predictions as priors during search.
    When simulations=0, skips MCTS and returns raw NN policy.
    """

    variant: Literal["mcts"] = "mcts"
    simulations: int = 200
    c_puct: float = 4.73  # Optimized default from sampling
    gamma: float = 1.0
    search_variant: Literal["prior_sampling", "decoupled_puct"] = "decoupled_puct"
    checkpoint: str | None = None
    temperature: float = 1.0  # For action sampling (Nash uses 1.0)

    def build(self, device: str = "cpu") -> Agent:
        """Build an MCTSAgent."""
        from alpharat.ai.mcts_agent import MCTSAgent

        return MCTSAgent(
            simulations=self.simulations,
            c_puct=self.c_puct,
            gamma=self.gamma,
            search_variant=self.search_variant,
            checkpoint=self.checkpoint,
            temperature=self.temperature,
            device=device,
        )


# Discriminated union using Pydantic's Annotated + Field(discriminator=...)
AgentConfig = Annotated[
    RandomAgentConfig | GreedyAgentConfig | NNAgentConfig | MCTSAgentConfig,
    Field(discriminator="variant"),
]
