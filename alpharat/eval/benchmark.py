"""Benchmark utilities for evaluating trained models against baselines.

This module provides reusable utilities for building benchmark tournaments.
Uses proper Pydantic configs rather than ad-hoc construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpharat.ai.config import (
    AgentConfig,
    GreedyAgentConfig,
    MCTSAgentConfig,
    NNAgentConfig,
    RandomAgentConfig,
)
from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.eval.elo import compute_elo, from_tournament_result
from alpharat.eval.tournament import TournamentConfig, TournamentResult
from alpharat.mcts.config import MCTSConfig  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


class BenchmarkConfig(StrictBaseModel):
    """Configuration for benchmarking a trained model against baselines.

    This is a convenience config that generates a standard tournament setup.
    For full control over agents, use TournamentConfig directly.

    Example YAML:
        benchmark:
          games_per_matchup: 50
          workers: 4
          device: cuda
          mcts:
            simulations: 629
            c_puct: 0.531
            force_k: 0.067
    """

    games_per_matchup: int = 50
    workers: int = 4
    device: str = "cpu"
    mcts: MCTSConfig


def get_game_config_from_checkpoint(checkpoint_path: Path) -> GameConfig:
    """Extract game dimensions from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.

    Returns:
        GameConfig with dimensions extracted from checkpoint.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    width = checkpoint.get("width", 5)
    height = checkpoint.get("height", 5)

    # Use defaults for other params, matching training data
    return GameConfig(
        width=width,
        height=height,
        max_turns=30,
        cheese_count=5,
        wall_density=0.0,
        mud_density=0.0,
    )


def build_standard_agents(
    checkpoint_path: Path,
    mcts_config: MCTSConfig,
    baseline_checkpoint: Path | None = None,
) -> dict[str, AgentConfig]:
    """Build standard benchmark agent set.

    Creates agents for comparing a trained model:
    - random: Random baseline
    - greedy: Greedy baseline (moves toward closest cheese)
    - mcts: Pure MCTS (no NN)
    - nn: Pure NN (no search)
    - mcts+nn: MCTS with NN priors

    If baseline_checkpoint is provided, also adds:
    - nn-prev: Previous iteration's NN
    - mcts+nn-prev: Previous iteration's MCTS+NN

    Dirichlet noise is stripped from the MCTS config — benchmarks measure
    true playing strength, not noisy exploration.

    Args:
        checkpoint_path: Path to the model checkpoint to evaluate.
        mcts_config: MCTS configuration for search-based agents.
        baseline_checkpoint: Optional path to previous iteration's checkpoint.

    Returns:
        Dict of agent name -> AgentConfig.
    """
    checkpoint_str = str(checkpoint_path)

    # Strip exploration noise — benchmarks should measure true strength.
    eval_mcts = mcts_config.for_evaluation()

    agents: dict[str, AgentConfig] = {
        "random": RandomAgentConfig(),
        "greedy": GreedyAgentConfig(),
        "mcts": MCTSAgentConfig(mcts=eval_mcts),
        "nn": NNAgentConfig(checkpoint=checkpoint_str, temperature=1.0),
        "mcts+nn": MCTSAgentConfig(mcts=eval_mcts, checkpoint=checkpoint_str),
    }

    if baseline_checkpoint is not None:
        baseline_str = str(baseline_checkpoint)
        agents["nn-prev"] = NNAgentConfig(checkpoint=baseline_str, temperature=1.0)
        agents["mcts+nn-prev"] = MCTSAgentConfig(mcts=eval_mcts, checkpoint=baseline_str)

    return agents


def build_benchmark_tournament(
    benchmark_name: str,
    checkpoint_path: Path,
    config: BenchmarkConfig,
    game_config: GameConfig | None = None,
    baseline_checkpoint: Path | None = None,
) -> TournamentConfig:
    """Build tournament config for benchmarking a trained model.

    Args:
        benchmark_name: Name for this benchmark.
        checkpoint_path: Path to the model checkpoint to evaluate.
        config: Benchmark configuration.
        game_config: Game configuration. If None, extracts from checkpoint.
        baseline_checkpoint: Optional path to previous iteration's checkpoint.

    Returns:
        TournamentConfig ready to pass to run_tournament().
    """
    if game_config is None:
        game_config = get_game_config_from_checkpoint(checkpoint_path)

    agents = build_standard_agents(
        checkpoint_path=checkpoint_path,
        mcts_config=config.mcts,
        baseline_checkpoint=baseline_checkpoint,
    )

    return TournamentConfig(
        name=benchmark_name,
        agents=agents,
        games_per_matchup=config.games_per_matchup,
        game=game_config,
        workers=config.workers,
        device=config.device,
    )


def print_benchmark_results(
    result: TournamentResult,
    anchor: str = "greedy",
) -> None:
    """Print benchmark tables: standings, WDL, cheese, and Elo ratings.

    Args:
        result: Tournament result to display.
        anchor: Agent name to anchor Elo ratings at 1000.
    """
    print()
    print(result.standings_table())
    print()
    print(result.wdl_table())
    print()
    print(result.cheese_table())
    print()

    records = from_tournament_result(result)
    elo_result = compute_elo(records, anchor=anchor, anchor_elo=1000, compute_uncertainty=True)
    print(elo_result.format_table())
