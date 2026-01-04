"""MCTS agent for PyRat games with optional NN priors."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from alpharat.ai.base import Agent
from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrat_engine.core.game import PyRat

    from alpharat.mcts import MCTSConfig
    from alpharat.nn.builders.flat import FlatObservationBuilder
    from alpharat.nn.models import LocalValueMLP, PyRatMLP, SymmetricMLP


class MCTSAgent(Agent):
    """Agent that uses MCTS to select actions, with optional NN priors.

    When checkpoint is provided, uses NN predictions as priors during MCTS.
    When simulations=0 and checkpoint is set, returns raw NN policy (pure NN mode).

    Attributes:
        mcts_config: MCTS search configuration (DecoupledPUCTConfig or PriorSamplingConfig).
        checkpoint: Path to NN checkpoint, or None for uniform priors.
        temperature: Sampling temperature for action selection.
        reuse_tree: Whether to preserve tree between turns.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        checkpoint: str | None = None,
        temperature: float = 1.0,
        device: str = "cpu",
        reuse_tree: bool = False,
    ) -> None:
        """Initialize MCTS agent.

        Args:
            mcts_config: MCTS search configuration containing simulations, c_puct, etc.
            checkpoint: Path to NN checkpoint, or None for uniform priors.
            temperature: Sampling temperature. 0 = argmax, 1.0 = proportional.
            device: Device for NN inference ("cpu", "cuda", "mps").
            reuse_tree: If True, preserve tree between turns and advance root.
        """
        self.mcts_config = mcts_config
        self.checkpoint = checkpoint
        self.temperature = temperature
        self._device = device
        self.reuse_tree = reuse_tree

        # NN components (lazily loaded)
        self._model: PyRatMLP | LocalValueMLP | SymmetricMLP | None = None
        self._builder: FlatObservationBuilder | None = None
        self._width: int = 0
        self._height: int = 0
        self._model_loaded = False

        # Tree reuse state
        self._tree: MCTSTree | None = None
        self._player: int | None = None
        self._awaiting_observe: bool = False  # State machine for call ordering

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

        # Detect model type from checkpoint
        model_type = ckpt.get("model_type", None)
        optim_config = config.get("optim", {})

        if model_type == "symmetric":
            from alpharat.nn.models import SymmetricMLP

            self._model = SymmetricMLP(
                width=self._width,
                height=self._height,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif "ownership_weight" in optim_config:
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

    def _create_tree(self, game: PyRat) -> MCTSTree:
        """Create a fresh MCTS tree from the given game state.

        Args:
            game: Current game state (will be deep-copied).

        Returns:
            New MCTSTree ready for search.
        """
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

        return MCTSTree(
            game=simulator,
            root=root,
            gamma=self.mcts_config.gamma,
            predict_fn=predict_fn,
        )

    def _is_tree_valid(self, game: PyRat) -> bool:
        """Check if the current tree's state matches the game.

        Args:
            game: Current game state to validate against.

        Returns:
            True if tree can be reused, False if fresh tree needed.
        """
        if self._tree is None:
            return False

        # Check turn number matches
        if self._tree.game.turn != game.turn:
            return False

        # Check player positions match
        if self._tree.game.player1_position != game.player1_position:
            return False
        return self._tree.game.player2_position == game.player2_position

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

        self._player = player  # Remember which player we are

        # Check call ordering when tree reuse is enabled
        if self.reuse_tree and self._awaiting_observe:
            warnings.warn(
                "MCTSAgent.get_move called without prior observe_move, "
                "invalidating tree for fresh start",
                stacklevel=2,
            )
            self._tree = None

        # Try to reuse tree if enabled and available
        if self.reuse_tree and self._is_tree_valid(game):
            assert self._tree is not None  # _is_tree_valid ensures this
            tree = self._tree
        else:
            # Create fresh tree
            tree = self._create_tree(game)
            if self.reuse_tree:
                self._tree = tree

        search = self._build_search(tree)
        result = search.search()

        policy = result.policy_p1 if player == 1 else result.policy_p2

        # Mark that we're now awaiting observe_move
        if self.reuse_tree:
            self._awaiting_observe = True

        # Use temperature=1.0 for MCTS Nash (proportional), custom temp for pure NN
        if self.mcts_config.simulations > 0:
            return select_action_from_strategy(policy, temperature=1.0)
        else:
            return self._sample_action(policy)

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self._tree = None
        self._player = None
        self._awaiting_observe = False

    def observe_move(self, action_p1: int, action_p2: int) -> None:
        """Called after both players' actions are known.

        Advances the internal tree to the child reached by the given actions,
        preserving accumulated statistics for future searches.

        Args:
            action_p1: Player 1's action (0-4).
            action_p2: Player 2's action (0-4).
        """
        if not self.reuse_tree:
            return

        if not self._awaiting_observe:
            warnings.warn(
                "MCTSAgent.observe_move called without prior get_move, ignoring",
                stacklevel=2,
            )
            return

        if self._tree is not None:
            self._tree.advance_root(action_p1, action_p2)

        self._awaiting_observe = False

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
