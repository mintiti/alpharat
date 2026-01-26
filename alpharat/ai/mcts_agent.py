"""MCTS agent for PyRat games with optional NN priors."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from alpharat.ai.base import Agent
from alpharat.config.checkpoint import make_predict_fn
from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat

    from alpharat.mcts import MCTSConfig
    from alpharat.nn.builders.flat import FlatObservationBuilder
    from alpharat.nn.models import LocalValueMLP, PyRatMLP, SymmetricMLP


class MCTSAgent(Agent):
    """Agent that uses MCTS to select actions, with optional NN priors.

    When checkpoint is provided, uses NN predictions as priors during MCTS.
    When simulations=0 and checkpoint is set, returns raw NN policy (pure NN mode).

    Attributes:
        mcts_config: MCTS search configuration (DecoupledPUCTConfig).
        checkpoint: Path to NN checkpoint, or None for uniform priors.
        temperature: Sampling temperature for action selection.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        checkpoint: str | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initialize MCTS agent.

        Args:
            mcts_config: MCTS search configuration containing simulations, c_puct, etc.
            checkpoint: Path to NN checkpoint, or None for uniform priors.
            temperature: Sampling temperature. 0 = argmax, 1.0 = proportional.
            device: Device for NN inference ("cpu", "cuda", "mps").
        """
        self.mcts_config = mcts_config
        self.checkpoint = checkpoint
        self.temperature = temperature
        self._device = device

        # NN components (lazily loaded)
        self._model: PyRatMLP | LocalValueMLP | SymmetricMLP | None = None
        self._builder: FlatObservationBuilder | None = None
        self._width: int = 0
        self._height: int = 0
        self._model_loaded = False

        if checkpoint is not None:
            self._load_model(checkpoint)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load NN model from checkpoint using ModelConfig.build_model()."""
        from alpharat.config.checkpoint import load_model_from_checkpoint

        model, builder, width, height = load_model_from_checkpoint(
            checkpoint_path,
            device=self._device,
            compile_model=True,
        )

        self._model = model  # type: ignore[assignment]
        self._builder = builder  # type: ignore[assignment]
        self._width = width
        self._height = height
        self._model_loaded = True

    def _validate_dimensions(self, game: PyRat) -> None:
        """Check game dimensions match checkpoint training dimensions."""
        if not self._model_loaded:
            return

        game_width = game.width
        game_height = game.height

        if game_width != self._width or game_height != self._height:
            msg = (
                f"Game dimensions ({game_width}x{game_height}) don't match "
                f"checkpoint training dimensions ({self._width}x{self._height}). "
                f"Cannot use NN priors."
            )
            raise ValueError(msg)

    def _build_search(self, tree: MCTSTree) -> Any:
        """Build the appropriate search object from config."""
        return self.mcts_config.build(tree)

    def _sample_action(self, policy: np.ndarray) -> int:
        """Sample action from policy using temperature."""
        if self.temperature == 0:
            return int(np.argmax(policy))

        logits = np.log(policy + 1e-8) / self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        return int(np.random.choice(len(probs), p=probs))

    def get_move(self, game: PyRat, player: int) -> int:
        """Select action using MCTS search (or pure NN if simulations=0).

        Args:
            game: Current game state (will be deep-copied, not modified).
            player: Which player we are (1 = Rat, 2 = Python).

        Returns:
            Action index (0-4).
        """
        if self._model_loaded:
            self._validate_dimensions(game)

        simulator = copy.deepcopy(game)

        predict_fn = None
        if self._model_loaded:
            assert self._model is not None
            assert self._builder is not None
            predict_fn = make_predict_fn(
                self._model, self._builder, simulator, self._width, self._height, self._device
            )

        p1_mud = simulator.player1_mud_turns
        p2_mud = simulator.player2_mud_turns

        dummy = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=dummy,
            prior_policy_p2=dummy,
            nn_payout_prediction=np.zeros((2, 5, 5)),
            parent=None,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
        )

        tree = MCTSTree(
            game=simulator,
            root=root,
            gamma=self.mcts_config.gamma,
            predict_fn=predict_fn,
        )

        search = self._build_search(tree)
        result = search.search()

        policy = result.policy_p1 if player == 1 else result.policy_p2

        # Use temperature=1.0 for MCTS Nash (proportional), custom temp for pure NN
        if self.mcts_config.simulations > 0:
            return select_action_from_strategy(policy, temperature=1.0)
        else:
            return self._sample_action(policy)

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        sims = self.mcts_config.simulations
        if sims == 0:
            # Pure NN mode
            temp_str = "argmax" if self.temperature == 0 else f"t={self.temperature}"
            return f"NN({temp_str})"

        # MCTS mode
        if isinstance(self.mcts_config, DecoupledPUCTConfig):
            base = f"PUCT({sims})"
        else:
            base = f"PS({sims})"
        if self.checkpoint:
            return f"{base}+NN"
        return base
