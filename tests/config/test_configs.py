"""Tests that all git-tracked configs resolve correctly.

Uses git ls-files to discover configs dynamically, so tests auto-update
when configs are added/renamed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config, load_raw_config
from alpharat.data.sampling import SamplingConfig
from alpharat.eval.tournament import TournamentConfig
from alpharat.mcts import MCTSConfig
from alpharat.nn.config import TrainConfig

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS = PROJECT_ROOT / "configs"


def get_tracked_configs(subdir: str) -> list[str]:
    """Get git-tracked config names from configs/{subdir}/.

    Returns config names (stems) without .yaml extension.
    """
    result = subprocess.run(
        ["git", "ls-files", f"configs/{subdir}/*.yaml"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if not result.stdout.strip():
        return []
    return [Path(p).stem for p in result.stdout.strip().split("\n") if p]


# --- Game Configs ---


class TestGameConfigs:
    """All game sub-configs resolve to valid GameConfig."""

    @pytest.fixture(params=get_tracked_configs("game"))
    def config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_resolves_to_game_config(self, config_name: str) -> None:
        """Game config loads and validates."""
        config = load_config(GameConfig, CONFIGS / "game", config_name)
        assert config.width > 0
        assert config.height > 0
        assert config.cheese_count > 0


# --- MCTS Configs ---


class TestMCTSConfigs:
    """All MCTS sub-configs resolve to valid MCTSConfig."""

    @pytest.fixture(params=get_tracked_configs("mcts"))
    def config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_resolves_to_mcts_config(self, config_name: str) -> None:
        """MCTS config loads and validates."""
        config = load_config(MCTSConfig, CONFIGS / "mcts", config_name)
        assert config.simulations > 0
        assert config.c_puct > 0


# --- Sample Sub-Configs ---


class TestSampleConfigs:
    """All sample sub-configs resolve to valid SamplingConfig.

    Sample sub-configs use @package _global_ and reference /mcts, /game defaults,
    so they must be loaded from the configs root (not configs/sample).
    """

    @pytest.fixture(params=get_tracked_configs("sample"))
    def config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_resolves_to_sampling_config(self, config_name: str) -> None:
        """Sample sub-config loads with composed game/mcts."""
        # Load from configs root with sample/ prefix (Hydra resolves defaults)
        config = load_config(SamplingConfig, CONFIGS, f"sample/{config_name}")
        assert isinstance(config.game, GameConfig)
        assert config.mcts.variant == "decoupled_puct"
        assert config.sampling.num_games > 0


# --- Model Configs (Special Case) ---


class TestModelConfigs:
    """Model configs use @package _global_ so they're not standalone.

    We test they contain expected structure without full Pydantic validation.
    """

    @pytest.fixture(params=get_tracked_configs("model"))
    def config_name(self, request: pytest.FixtureRequest) -> str:
        return str(request.param)

    def test_model_config_valid_yaml(self, config_name: str) -> None:
        """Model configs have expected structure."""
        raw = load_raw_config(CONFIGS / "model", config_name)
        assert "model" in raw
        assert "optim" in raw
        assert raw["model"]["architecture"] == raw["optim"]["architecture"]


# --- Entry Point Configs ---


class TestEntryPointConfigs:
    """Main entry point configs (sample.yaml, train.yaml, tournament.yaml)."""

    def test_sample_yaml(self) -> None:
        """Main sample.yaml composes into valid SamplingConfig."""
        config = load_config(SamplingConfig, CONFIGS, "sample")
        assert isinstance(config.game, GameConfig)
        assert config.mcts.variant == "decoupled_puct"

    def test_train_yaml(self) -> None:
        """Main train.yaml composes into valid TrainConfig."""
        config = load_config(TrainConfig, CONFIGS, "train")
        assert config.model.architecture == config.optim.architecture

    def test_tournament_yaml(self) -> None:
        """Main tournament.yaml composes into valid TournamentConfig."""
        config = load_config(TournamentConfig, CONFIGS, "tournament")
        assert len(config.agents) > 0
        assert isinstance(config.game, GameConfig)
