"""Rust MCTS agent for PyRat games with optional NN priors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from alpharat.ai.base import Agent
from alpharat.ai.utils import select_action_from_strategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pyrat_engine.core.game import PyRat


class RustMCTSAgent(Agent):
    """Agent using the Rust MCTS search with optional NN priors.

    Each get_move call creates a fresh search tree — no tree reuse.
    When checkpoint is provided, uses batched NN inference for priors.

    Attributes:
        simulations: Number of MCTS simulations per move.
        checkpoint: Path to NN checkpoint, or None for uniform priors.
        temperature: Sampling temperature for action selection.
    """

    def __init__(
        self,
        simulations: int = 100,
        c_puct: float = 1.5,
        fpu_reduction: float = 0.2,
        force_k: float = 2.0,
        batch_size: int = 8,
        checkpoint: str | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.simulations = simulations
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.force_k = force_k
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.temperature = temperature
        self._device = device
        self._seed = seed

        # Lazy-loaded predict_fn
        self._predict_fn: (
            Callable[[list[Any]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None
        ) = None
        self._predict_fn_loaded = False

        if checkpoint is not None:
            self._load_predict_fn(checkpoint)

    def _load_predict_fn(self, checkpoint_path: str) -> None:
        """Load NN model and create batched predict_fn."""
        from alpharat.ai.predict_batch import make_batched_predict_fn

        self._predict_fn = make_batched_predict_fn(checkpoint_path, device=self._device)
        self._predict_fn_loaded = True

    def get_move(self, game: PyRat, player: int) -> int:
        """Select action using Rust MCTS search.

        Args:
            game: Current game state (not modified).
            player: Which player we are (1 = Rat, 2 = Python).

        Returns:
            Action index (0-4).
        """
        from alpharat_mcts import rust_mcts_search

        result = rust_mcts_search(
            game,
            predict_fn=self._predict_fn,
            simulations=self.simulations,
            batch_size=self.batch_size,
            c_puct=self.c_puct,
            fpu_reduction=self.fpu_reduction,
            force_k=self.force_k,
            seed=self._seed,
        )

        policy = result.policy_p1 if player == 1 else result.policy_p2
        rng = np.random.default_rng(self._seed) if self._seed is not None else None
        return select_action_from_strategy(policy, temperature=self.temperature, rng=rng)

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        base = f"RustPUCT({self.simulations})"
        if self.checkpoint:
            return f"{base}+NN"
        return base
