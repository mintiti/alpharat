"""MCTSConfig — Rust MCTS backend configuration.

Provides MCTSConfigBase (abstract) and RustMCTSConfig (concrete).
MCTSConfig is an alias for RustMCTSConfig.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Self

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    from alpharat.ai.base import Agent
    from alpharat.mcts.searcher import Searcher


class MCTSConfigBase(StrictBaseModel):
    """Base class for MCTS backend configurations."""

    def for_evaluation(self) -> Self:
        """Return a copy suitable for evaluation (no exploration noise).

        Sampling uses Dirichlet noise at the root to encourage exploration.
        Evaluation should measure true playing strength, so noise is disabled.

        Default: returns self (no noise to strip).
        """
        return self

    @abstractmethod
    def build_searcher(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> Searcher:
        """Build a Searcher from this configuration.

        Args:
            checkpoint: Path to NN checkpoint for guided search.
            device: Device for NN inference.

        Returns:
            Configured Searcher instance.
        """
        ...

    @abstractmethod
    def build_agent(
        self,
        checkpoint: str | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> Agent:
        """Build an Agent wrapping a Searcher from this configuration.

        Args:
            checkpoint: Path to NN checkpoint for guided search.
            temperature: Sampling temperature for action selection.
            device: Device for NN inference.

        Returns:
            Configured Agent instance.
        """
        ...


class RustMCTSConfig(MCTSConfigBase):
    """Configuration for the Rust MCTS backend."""

    backend: Literal["rust"] = "rust"
    simulations: int = 100
    c_puct: float = 1.5
    force_k: float = 2.0
    fpu_reduction: float = 0.2
    batch_size: int = 8
    noise_epsilon: float = 0.0
    noise_concentration: float = 10.83
    collision_limit_min: int = 1
    collision_limit_max: int = 256
    collision_scaling_start: int = 800
    collision_scaling_end: int = 50_000
    collision_scaling_power: float = 1.0

    def for_evaluation(self) -> Self:
        """Return a copy with Dirichlet noise disabled."""
        if self.noise_epsilon == 0.0:
            return self
        return self.model_copy(update={"noise_epsilon": 0.0})

    def build_searcher(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> Searcher:
        from alpharat.mcts.searcher import RustSearcher

        predict_fn = None
        if checkpoint is not None:
            from alpharat.ai.predict_batch import make_batched_predict_fn

            predict_fn = make_batched_predict_fn(checkpoint, device=device)

        return RustSearcher(
            simulations=self.simulations,
            c_puct=self.c_puct,
            force_k=self.force_k,
            fpu_reduction=self.fpu_reduction,
            batch_size=self.batch_size,
            noise_epsilon=self.noise_epsilon,
            noise_concentration=self.noise_concentration,
            collision_limit_min=self.collision_limit_min,
            collision_limit_max=self.collision_limit_max,
            collision_scaling_start=self.collision_scaling_start,
            collision_scaling_end=self.collision_scaling_end,
            collision_scaling_power=self.collision_scaling_power,
            predict_fn=predict_fn,
        )

    def build_agent(
        self,
        checkpoint: str | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> Agent:
        from alpharat.ai.searcher_agent import SearcherAgent

        searcher = self.build_searcher(checkpoint=checkpoint, device=device)
        return SearcherAgent(
            searcher=searcher,
            temperature=temperature,
            simulations=self.simulations,
            checkpoint=checkpoint,
        )


MCTSConfig = RustMCTSConfig
