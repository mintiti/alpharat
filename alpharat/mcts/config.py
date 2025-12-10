"""MCTS algorithm configuration variants."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PriorSamplingConfig(BaseModel):
    """MCTS config using prior policy sampling for action selection."""

    variant: Literal["prior_sampling"] = "prior_sampling"
    simulations: int
    gamma: float = 0.99


class DecoupledPUCTConfig(BaseModel):
    """MCTS config using decoupled PUCT for action selection."""

    variant: Literal["decoupled_puct"] = "decoupled_puct"
    simulations: int
    gamma: float = 0.99
    c_puct: float = 1.5


MCTSConfig = PriorSamplingConfig | DecoupledPUCTConfig
