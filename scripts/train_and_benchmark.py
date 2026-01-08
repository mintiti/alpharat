#!/usr/bin/env python3
"""Train a neural network and benchmark it against baselines.

This convenience script chains training with evaluation:
1. Trains a model using the provided config
2. Benchmarks the trained model against Random, Greedy, and pure MCTS

Usage:
    uv run python scripts/train_and_benchmark.py configs/train.yaml
    uv run python scripts/train_and_benchmark.py configs/train.yaml --games 100 --device mps
    uv run python scripts/train_and_benchmark.py configs/train.yaml --epochs 50 --workers 8
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
from alpharat.nn.training import TrainConfig, run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSettings:
    """Settings for the benchmark phase."""

    games_per_matchup: int
    workers: int
    device: str
    mcts_simulations: int
    baseline_checkpoint: Path | None = None


def get_game_params_from_checkpoint(checkpoint_path: Path) -> GameParams:
    """Extract game dimensions from checkpoint."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    width = checkpoint.get("width", 5)
    height = checkpoint.get("height", 5)

    # Use defaults for other params, matching training data
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
    from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig

    game_params = get_game_params_from_checkpoint(checkpoint_path)

    checkpoint_str = str(checkpoint_path)

    # Default MCTS config for benchmarking (matches sampling defaults)
    mcts_config = DecoupledPUCTConfig(
        simulations=settings.mcts_simulations,
        c_puct=8.34,
        force_k=0.88,
    )

    agents: dict[str, AgentConfigBase] = {
        "random": RandomAgentConfig(),
        "greedy": GreedyAgentConfig(),
        "mcts": MCTSAgentConfig(
            mcts=mcts_config,
        ),
        "nn": NNAgentConfig(
            checkpoint=checkpoint_str,
            temperature=1.0,
        ),
        "mcts+nn": MCTSAgentConfig(
            mcts=mcts_config,
            checkpoint=checkpoint_str,
        ),
    }

    if settings.baseline_checkpoint:
        baseline_str = str(settings.baseline_checkpoint)
        agents["nn-prev"] = NNAgentConfig(checkpoint=baseline_str, temperature=1.0)
        agents["mcts+nn-prev"] = MCTSAgentConfig(
            mcts=mcts_config,
            checkpoint=baseline_str,
        )

    return TournamentConfig(
        agents=agents,  # type: ignore[arg-type]  # AgentConfigBase is compatible
        games_per_matchup=settings.games_per_matchup,
        game=game_params,
        workers=settings.workers,
        device=settings.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a neural network and benchmark it against baselines."
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
    parser.add_argument("--mcts-sims", type=int, default=554, help="MCTS simulations for baseline")

    # Skip flags
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training, use existing checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Existing checkpoint (with --skip-training)"
    )

    # AMP flags (mutually exclusive: auto-detect by default)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", help="Force enable AMP")
    amp_group.add_argument("--no-amp", action="store_true", help="Force disable AMP")

    args = parser.parse_args()

    # Parse config (needed for resume_from even when skipping training)
    config_data = yaml.safe_load(args.config.read_text())
    config = TrainConfig.model_validate(config_data)

    # Phase 1: Training
    if args.skip_training:
        if args.checkpoint is None:
            parser.error("--skip-training requires --checkpoint")
        checkpoint_path = args.checkpoint
        logger.info(f"Skipping training, using checkpoint: {checkpoint_path}")
    else:
        logger.info("=" * 60)
        logger.info("Phase 1: Training")
        logger.info("=" * 60)

        # Tri-state AMP: True (force on), False (force off), None (auto-detect)
        use_amp = True if args.amp else (False if args.no_amp else None)

        checkpoint_path = run_training(
            config,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            output_dir=args.output_dir,
            run_name=args.run_name,
            use_amp=use_amp,
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

    baseline = Path(config.resume_from) if config.resume_from else None
    if baseline:
        logger.info("Baseline: %s", baseline)
        logger.info("  → Used by: nn-prev, mcts+nn-prev")
    logger.info("")

    settings = BenchmarkSettings(
        games_per_matchup=args.games,
        workers=args.workers,
        device=args.device,
        mcts_simulations=args.mcts_sims,
        baseline_checkpoint=baseline,
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
