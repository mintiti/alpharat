"""MCTS agent for PyRat games with optional NN priors."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from alpharat.ai.base import Agent
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrat_engine.core.game import PyRat

    from alpharat.nn.builders.flat import FlatObservationBuilder
    from alpharat.nn.models import LocalValueMLP, PyRatMLP


class MCTSAgent(Agent):
    """Agent that uses MCTS to select actions, with optional NN priors.

    When checkpoint is provided, uses NN predictions as priors during MCTS.
    When simulations=0 and checkpoint is set, returns raw NN policy (pure NN mode).

    Attributes:
        simulations: Number of MCTS simulations.
        c_puct: Exploration constant for PUCT.
        gamma: Discount factor.
        search_variant: "prior_sampling" or "decoupled_puct".
        checkpoint: Path to NN checkpoint, or None for uniform priors.
        temperature: Sampling temperature for action selection.
    """

    def __init__(
        self,
        simulations: int = 200,
        c_puct: float = 4.73,
        gamma: float = 1.0,
        search_variant: Literal["prior_sampling", "decoupled_puct"] = "decoupled_puct",
        checkpoint: str | None = None,
        temperature: float = 1.0,
        force_k: float = 2.0,
        device: str = "cpu",
    ) -> None:
        """Initialize MCTS agent.

        Args:
            simulations: Number of MCTS simulations. 0 = pure NN mode.
            c_puct: Exploration constant for decoupled PUCT.
            gamma: Discount factor for future rewards.
            search_variant: Which search algorithm to use.
            checkpoint: Path to NN checkpoint, or None for uniform priors.
            temperature: Sampling temperature. 0 = argmax, 1.0 = proportional.
            force_k: Forced playout scaling (0 disables, 2.0 is KataGo default).
            device: Device for NN inference ("cpu", "cuda", "mps").
        """
        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.search_variant = search_variant
        self.checkpoint = checkpoint
        self.temperature = temperature
        self.force_k = force_k
        self._device = device

        # NN components (lazily loaded)
        self._model: PyRatMLP | LocalValueMLP | None = None
        self._builder: FlatObservationBuilder | None = None
        self._width: int = 0
        self._height: int = 0
        self._model_loaded = False

        if checkpoint is not None:
            self._load_model(checkpoint)

    def _load_model(self, checkpoint_path: str) -> None:
        """Load NN model from checkpoint."""
        import torch

        from alpharat.nn.builders.flat import FlatObservationBuilder

        device = torch.device(self._device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self._width = ckpt.get("width", 5)
        self._height = ckpt.get("height", 5)
        self._builder = FlatObservationBuilder(width=self._width, height=self._height)

        config = ckpt.get("config", {})
        model_config = config.get("model", {})
        hidden_dim = model_config.get("hidden_dim", 256)
        dropout = model_config.get("dropout", 0.0)

        obs_dim = self._width * self._height * 7 + 6

        optim_config = config.get("optim", {})
        if "ownership_weight" in optim_config:
            from alpharat.nn.models import LocalValueMLP

            self._model = LocalValueMLP(
                obs_dim=obs_dim,
                width=self._width,
                height=self._height,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            from alpharat.nn.models import PyRatMLP

            self._model = PyRatMLP(
                obs_dim=obs_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.to(device)
        self._model.eval()
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

    def _make_predict_fn(
        self, simulator: PyRat
    ) -> Callable[[Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Create predict_fn closure that captures the simulator.

        The closure reads from the simulator (which the tree mutates during search)
        and runs NN inference to produce priors.
        """
        import torch

        from alpharat.data.maze import build_maze_array
        from alpharat.nn.extraction import from_pyrat_game

        # Capture references (guaranteed non-None when this is called)
        builder = self._builder
        model = self._model
        device = self._device
        assert builder is not None
        assert model is not None

        maze = build_maze_array(simulator, self._width, self._height)
        max_turns = simulator.max_turns

        def predict_fn(_observation: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Run NN inference on current simulator state."""
            obs_input = from_pyrat_game(simulator, maze, max_turns)
            obs = builder.build(obs_input)

            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                result = model.predict(obs_tensor)
                policy_p1 = result[0].squeeze(0).cpu().numpy()
                policy_p2 = result[1].squeeze(0).cpu().numpy()
                payout = result[2].squeeze(0).cpu().numpy()

            return policy_p1, policy_p2, payout

        return predict_fn

    def _build_search(self, tree: MCTSTree) -> Any:
        """Build the appropriate search object based on variant."""
        if self.search_variant == "prior_sampling":
            from alpharat.mcts.search import MCTSSearch

            return MCTSSearch(tree, self.simulations)
        else:
            from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig, DecoupledPUCTSearch

            config = DecoupledPUCTConfig(
                simulations=self.simulations,
                gamma=self.gamma,
                c_puct=self.c_puct,
                force_k=self.force_k,
            )
            return DecoupledPUCTSearch(tree, config)

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

        predict_fn = self._make_predict_fn(simulator) if self._model_loaded else None

        p1_mud = simulator.player1_mud_turns
        p2_mud = simulator.player2_mud_turns

        dummy = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=dummy,
            prior_policy_p2=dummy,
            nn_payout_prediction=np.zeros((5, 5)),
            parent=None,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
        )

        tree = MCTSTree(
            game=simulator,
            root=root,
            gamma=self.gamma,
            predict_fn=predict_fn,
        )

        search = self._build_search(tree)
        result = search.search()

        policy = result.policy_p1 if player == 1 else result.policy_p2

        # Use temperature=1.0 for MCTS Nash (proportional), custom temp for pure NN
        if self.simulations > 0:
            return select_action_from_strategy(policy, temperature=1.0)
        else:
            return self._sample_action(policy)

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        if self.simulations == 0:
            # Pure NN mode
            temp_str = "argmax" if self.temperature == 0 else f"t={self.temperature}"
            return f"NN({temp_str})"

        # MCTS mode
        if self.search_variant == "decoupled_puct":
            base = f"PUCT({self.simulations})"
        else:
            base = f"PS({self.simulations})"
        if self.checkpoint:
            return f"{base}+NN"
        return base
