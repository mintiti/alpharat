"""Integration tests for config validation.

Beyond "does it load", verify configs actually work end-to-end:
- Game configs build playable games
- Tournament configs build agents that can play
- Sample configs can run MCTS simulations
- Train configs build models that forward
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat

from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config
from alpharat.data.sampling import SamplingConfig
from alpharat.eval.tournament import TournamentConfig
from alpharat.mcts import DecoupledPUCTConfig
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree
from alpharat.nn.config import TrainConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"


def get_tracked_configs(subdir: str) -> list[str]:
    """Get git-tracked config names from configs/{subdir}/."""
    result = subprocess.run(
        ["git", "ls-files", f"configs/{subdir}/*.yaml"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if not result.stdout.strip():
        return []
    return [Path(p).stem for p in result.stdout.strip().split("\n") if p]


# --- Game Config Integration ---


class TestGameConfigIntegration:
    """Game configs build playable PyRat instances."""

    @pytest.fixture(params=get_tracked_configs("game"))
    def config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_builds_playable_game(self, config_name: str) -> None:
        """Game config builds a PyRat instance that can be played."""
        config = load_config(GameConfig, CONFIGS / "game", config_name)
        game = config.build(seed=42)

        # Verify game properties match config
        assert game.width == config.width
        assert game.height == config.height

        # Verify game is playable (can make moves)
        valid_moves = game.get_valid_moves(game.player1_position)  # type: ignore[attr-defined]
        assert len(valid_moves) > 0

        # Can actually make a move without error
        undo = game.make_move(0, 0)
        game.unmake_move(undo)


# --- Tournament Config Integration ---


class TestTournamentConfigIntegration:
    """Tournament config agents can be built and make moves."""

    def test_all_agents_build_and_play(self) -> None:
        """All agents in tournament.yaml can be built and make moves."""
        config = load_config(TournamentConfig, CONFIGS, "tournament")
        game = config.game.build(seed=42)

        for name, agent_config in config.agents.items():
            # Skip NN agents (need checkpoint)
            if hasattr(agent_config, "checkpoint") and agent_config.checkpoint:
                continue

            agent = agent_config.build(device="cpu")
            # Agent can make a move (player=1 for P1)
            move = agent.get_move(game, player=1)
            assert move in range(5), f"Agent {name} returned invalid move {move}"


# --- Sampling Config Integration ---


def _create_mcts_tree(game: PyRat, mcts_config: DecoupledPUCTConfig) -> MCTSTree:
    """Helper to create MCTS tree from game and config."""
    # Create root with dummy priors (tree will reinitialize with smart uniform)
    dummy = np.ones(5) / 5
    root = MCTSNode(
        game_state=None,
        prior_policy_p1=dummy,
        prior_policy_p2=dummy,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=game.player1_mud_turns,
        p2_mud_turns_remaining=game.player2_mud_turns,
    )
    return MCTSTree(game=game, root=root, gamma=mcts_config.gamma)


class TestSamplingConfigIntegration:
    """Sampling configs can initialize and run MCTS."""

    def test_can_run_mcts_simulation(self) -> None:
        """Sampling config can initialize MCTS and run simulations."""
        config = load_config(SamplingConfig, CONFIGS, "sample")
        game = config.game.build(seed=42)

        # Build MCTS tree and run search
        tree = _create_mcts_tree(game, config.mcts)
        search = config.mcts.build(tree)

        # Run search (returns SearchResult with policies)
        result = search.search()

        # Policies should sum to ~1
        assert abs(result.policy_p1.sum() - 1.0) < 0.01
        assert abs(result.policy_p2.sum() - 1.0) < 0.01

    @pytest.fixture(params=get_tracked_configs("sample"))
    def sample_config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_sample_subconfigs_run_mcts(self, sample_config_name: str) -> None:
        """All sample sub-configs can run MCTS simulations."""
        # Load from configs root with sample/ prefix (Hydra resolves defaults)
        config = load_config(SamplingConfig, CONFIGS, f"sample/{sample_config_name}")
        game = config.game.build(seed=42)

        # Build MCTS tree and search
        tree = _create_mcts_tree(game, config.mcts)
        search = config.mcts.build(tree)

        # Run search (returns SearchResult with policies)
        result = search.search()

        # Policies should sum to ~1
        assert result.policy_p1.sum() > 0.99
        assert result.policy_p2.sum() > 0.99


# --- Train Config Integration ---


class TestTrainConfigIntegration:
    """Train config builds models that can forward."""

    def test_builds_model_that_forwards(self) -> None:
        """Train config builds a model that can do forward pass."""
        config = load_config(TrainConfig, CONFIGS, "train")

        # Set dimensions and build model
        config.model.set_data_dimensions(width=5, height=5)
        model = config.model.build_model()
        model.eval()  # type: ignore[attr-defined]  # Required for BatchNorm with batch_size=1

        # Build observation builder
        builder = config.model.build_observation_builder(width=5, height=5)

        # Create dummy observation and forward (obs_shape returns tuple)
        obs_dim = builder.obs_shape[0]  # type: ignore[attr-defined]
        obs = torch.randn(1, obs_dim)
        output = model.predict(obs)

        # Verify output structure
        assert "policy_p1" in output
        assert "policy_p2" in output
        assert "value_p1" in output
        assert "value_p2" in output

        # Verify output shapes
        assert output["policy_p1"].shape == (1, 5)
        assert output["policy_p2"].shape == (1, 5)
        # Value shapes are scalar per sample
        assert output["value_p1"].shape == (1,)
        assert output["value_p2"].shape == (1,)


# --- Composition Semantics ---


class TestCompositionSemantics:
    """Verify Hydra composition works correctly."""

    def test_sample_yaml_uses_default_game(self) -> None:
        """Main sample.yaml uses 5x5_open game by default."""
        config = load_config(SamplingConfig, CONFIGS, "sample")
        # 5x5_open has width=5, height=5
        assert config.game.width == 5
        assert config.game.height == 5

    def test_sample_yaml_uses_default_mcts(self) -> None:
        """Main sample.yaml uses 5x5_tuned MCTS by default."""
        config = load_config(SamplingConfig, CONFIGS, "sample")
        mcts_config = load_config(DecoupledPUCTConfig, CONFIGS / "mcts", "5x5_tuned")
        assert config.mcts.simulations == mcts_config.simulations
        assert config.mcts.c_puct == mcts_config.c_puct

    def test_tournament_yaml_uses_default_game(self) -> None:
        """Main tournament.yaml uses 5x5_open game by default."""
        config = load_config(TournamentConfig, CONFIGS, "tournament")
        assert config.game.width == 5
        assert config.game.height == 5
