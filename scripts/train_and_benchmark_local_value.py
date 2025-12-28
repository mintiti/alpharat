#!/usr/bin/env python3
"""Train a LocalValueMLP and benchmark it against baselines.

This convenience script chains training with evaluation:
1. Trains a LocalValueMLP using the provided config
2. Benchmarks the trained model against Random, Greedy, and pure MCTS

Usage:
    uv run python scripts/train_and_benchmark_local_value.py configs/train_local_value.yaml
    uv run python scripts/train_and_benchmark_local_value.py configs/train_local_value.yaml \\
        --games 100 --device mps
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from alpharat.ai.config import (
    AgentConfigBase,
    GreedyAgentConfig,
    MCTSAgentConfig,
    NNAgentConfig,
    RandomAgentConfig,
)
from alpharat.data.batch import GameParams
from alpharat.eval.elo import compute_elo, from_tournament_result
from alpharat.eval.tournament import TournamentConfig, run_tournament
from alpharat.nn.local_value_training import LocalValueTrainConfig, run_local_value_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSettings:
    """Settings for the benchmark phase."""

    games_per_matchup: int
    workers: int
    device: str
    mcts_simulations: int


def get_game_params_from_checkpoint(checkpoint_path: Path) -> GameParams:
    """Extract game dimensions from checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    width = checkpoint.get("width", 5)
    height = checkpoint.get("height", 5)

    return GameParams(
        width=width,
        height=height,
        max_turns=30,
        cheese_count=5,
        wall_density=0.0,
        mud_density=0.0,
    )


def build_benchmark_config(
    checkpoint_path: Path,
    settings: BenchmarkSettings,
) -> TournamentConfig:
    """Build tournament config for benchmarking a trained model."""
    game_params = get_game_params_from_checkpoint(checkpoint_path)

    checkpoint_str = str(checkpoint_path)

    agents: dict[str, AgentConfigBase] = {
        "random": RandomAgentConfig(),
        "greedy": GreedyAgentConfig(),
        "mcts": MCTSAgentConfig(
            simulations=settings.mcts_simulations,
        ),
        "nn": NNAgentConfig(
            checkpoint=checkpoint_str,
            temperature=1.0,
        ),
        "mcts+nn": MCTSAgentConfig(
            simulations=settings.mcts_simulations,
            checkpoint=checkpoint_str,
        ),
    }

    return TournamentConfig(
        agents=agents,  # type: ignore[arg-type]
        games_per_matchup=settings.games_per_matchup,
        game=game_params,
        workers=settings.workers,
        device=settings.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a LocalValueMLP and benchmark it against baselines."
    )

    # Training args
    parser.add_argument("config", type=Path, help="Path to training YAML config")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output dir")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for tensorboard")

    # Benchmark args
    parser.add_argument("--games", type=int, default=50, help="Games per matchup")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for games")
    parser.add_argument("--device", type=str, default="cpu", help="Device for NN inference")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations for baseline")

    # Skip flags
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training, use existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Existing checkpoint (with --skip-training)"
    )

    args = parser.parse_args()

    # Phase 1: Training
    if args.skip_training:
        if args.checkpoint is None:
            parser.error("--skip-training requires --checkpoint")
        checkpoint_path = args.checkpoint
        logger.info(f"Skipping training, using checkpoint: {checkpoint_path}")
    else:
        logger.info("=" * 60)
        logger.info("Phase 1: Training (LocalValueMLP)")
        logger.info("=" * 60)

        config_data = yaml.safe_load(args.config.read_text())
        config = LocalValueTrainConfig.model_validate(config_data)

        checkpoint_path = run_local_value_training(
            config,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )

        logger.info(f"Training complete. Best checkpoint: {checkpoint_path}")

    # Phase 2: Benchmark
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Benchmark")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("  → Used by: nn, mcts+nn")
    logger.info("")

    settings = BenchmarkSettings(
        games_per_matchup=args.games,
        workers=args.workers,
        device=args.device,
        mcts_simulations=args.mcts_sims,
    )

    tournament_config = build_benchmark_config(checkpoint_path, settings)
    game_params = tournament_config.game
    logger.info(
        "Game settings: %dx%d, %d cheese, %d max turns",
        game_params.width,
        game_params.height,
        game_params.cheese_count,
        game_params.max_turns,
    )
    logger.info("")

    result = run_tournament(tournament_config)

    # Print results
    print()
    print(result.standings_table())
    print()
    print(result.wdl_table())
    print()
    print(result.cheese_table())

    # Compute and print Elo ratings
    print()
    records = from_tournament_result(result)
    elo_result = compute_elo(records, anchor="greedy", anchor_elo=1000, compute_uncertainty=True)
    print(elo_result.format_table())


if __name__ == "__main__":
    main()
