"""MCTSConfig — discriminated union over Python and Rust MCTS backends.

Follows the same pattern as AgentConfig in alpharat/ai/config.py.
Each variant implements build_searcher() and build_agent() so consumers
never need to know which backend they're using.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field

from alpharat.config.base import StrictBaseModel

if TYPE_CHECKING:
    from alpharat.ai.base import Agent
    from alpharat.mcts.searcher import Searcher


class MCTSConfigBase(StrictBaseModel):
    """Base class for MCTS backend configurations."""

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


class PythonMCTSConfig(MCTSConfigBase):
    """Configuration for the Python MCTS backend."""

    backend: Literal["python"] = "python"
    simulations: int = 100
    gamma: float = 1.0
    c_puct: float = 1.5
    force_k: float = 2.0
    fpu_reduction: float = 0.2

    def build_searcher(
        self,
        checkpoint: str | None = None,
        device: str = "cpu",
    ) -> Searcher:
        from alpharat.data.sampling import NNContext, load_nn_context
        from alpharat.mcts.searcher import PythonSearcher

        nn_ctx: NNContext | None = None
        if checkpoint is not None:
            nn_ctx = load_nn_context(checkpoint, device=device)

        return PythonSearcher(
            simulations=self.simulations,
            gamma=self.gamma,
            c_puct=self.c_puct,
            force_k=self.force_k,
            fpu_reduction=self.fpu_reduction,
            nn_ctx=nn_ctx,
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

    def to_decoupled_puct_config(self) -> Any:
        """Convert to DecoupledPUCTConfig for backward compatibility."""
        from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig

        return DecoupledPUCTConfig(
            simulations=self.simulations,
            gamma=self.gamma,
            c_puct=self.c_puct,
            force_k=self.force_k,
            fpu_reduction=self.fpu_reduction,
        )


class RustMCTSConfig(MCTSConfigBase):
    """Configuration for the Rust MCTS backend."""

    backend: Literal["rust"] = "rust"
    simulations: int = 100
    c_puct: float = 1.5
    force_k: float = 2.0
    fpu_reduction: float = 0.2
    batch_size: int = 8

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


# Discriminated union using Pydantic's Annotated + Field(discriminator=...)
MCTSConfig = Annotated[
    PythonMCTSConfig | RustMCTSConfig,
    Field(discriminator="backend"),
]
