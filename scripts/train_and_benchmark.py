#!/usr/bin/env python3
"""Train a neural network and benchmark it against baselines.

This convenience script chains training with evaluation:
1. Trains a model using the provided config (via ExperimentManager)
2. Benchmarks the trained model against Random, Greedy, and pure MCTS
3. Saves benchmark results as {run_name}_benchmark

Usage:
    alpharat-train-and-benchmark configs/train.yaml --name mlp_v1 --shards grp/uuid
    alpharat-train-and-benchmark configs/train.yaml --games 100 --device mps
    alpharat-train-and-benchmark configs/train.yaml --epochs 50 --workers 8

CLI overrides (--name, --shards) are passed as Hydra overrides,
so the frozen config in experiments/runs/{name}/config.yaml has actual values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from alpharat.config.display import format_config_summary
from alpharat.config.game import GameConfig
from alpharat.config.loader import load_config, split_config_path
from alpharat.eval.benchmark import (
    BenchmarkConfig,
    build_benchmark_tournament,
    print_benchmark_results,
)
from alpharat.eval.tournament import run_tournament
from alpharat.experiments import ExperimentManager
from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig
from alpharat.nn.config import TrainConfig
from alpharat.nn.training import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a neural network and benchmark it against baselines."
    )

    # Training args
    parser.add_argument(
        "config",
        type=str,
        help="Config path: 'configs/train.yaml' or 'configs/train/mlp' (with or without .yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override run name (merged into frozen config)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Shard ID as GROUP/UUID (e.g., 'mygroup/abc123'). Overrides data paths in config.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )

    # Benchmark args
    parser.add_argument("--games", type=int, default=50, help="Games per matchup")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for games")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    parser.add_argument(
        "--mcts",
        type=str,
        default="5x5_tuned",
        help="MCTS sub-config name from configs/mcts/ (default: 5x5_tuned)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="5x5_open",
        help="Game sub-config name from configs/game/ (default: 5x5_open)",
    )

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

    exp = ExperimentManager(args.experiments_dir)

    config_dir, config_name = split_config_path(args.config)

    # Build Hydra overrides from CLI args
    overrides: list[str] = []
    if args.shards:
        if "/" not in args.shards:
            parser.error("--shards requires 'GROUP/UUID' format (e.g., 'mygroup/abc123')")
        shard_path = exp.get_shard_path(args.shards)
        if not shard_path.exists():
            parser.error(f"Shard not found: {args.shards} (looked in {shard_path})")
        overrides.append(f"data.train_dir={shard_path / 'train'}")
        overrides.append(f"data.val_dir={shard_path / 'val'}")
    if args.name:
        overrides.append(f"name={args.name}")

    # Load config using Hydra (resolves defaults) + Pydantic (validates)
    config = load_config(TrainConfig, config_dir, config_name, overrides=overrides or None)

    # Tri-state AMP: True (force on), False (force off), None (auto-detect)
    use_amp = True if args.amp else (False if args.no_amp else None)

    # Derive source_shards for manifest (from --shards or from config paths)
    if args.shards:
        source_shards = args.shards
    else:
        source_shards = exp.shard_id_from_data_path(config.data.train_dir)
        if source_shards is None:
            logger.error(
                f"Cannot auto-detect source shards from '{config.data.train_dir}'. "
                f"Expected path like 'experiments/shards/GROUP/UUID/train'. "
                f"Use --shards GROUP/UUID to specify explicitly."
            )
            sys.exit(1)

    # Phase 1: Training
    # Note: config.name has the merged value (from --name if provided, else from YAML)
    if args.skip_training:
        if args.checkpoint is None:
            parser.error("--skip-training requires --checkpoint")
        checkpoint_path = args.checkpoint
        actual_run_name = config.name  # Use merged name when skipping
        logger.info(f"Skipping training, using checkpoint: {checkpoint_path}")
    else:
        # Create run via ExperimentManager
        run_dir = exp.create_run(
            name=config.name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )
        # Name might have been auto-incremented if same config exists
        actual_run_name = run_dir.name
        if actual_run_name != config.name:
            logger.info(f"Run '{config.name}' exists with same config, using '{actual_run_name}'")

        logger.info("=" * 60)
        logger.info("Phase 1: Training")
        logger.info("=" * 60)
        logger.info(f"Created run: {actual_run_name}")
        logger.info(f"  Run directory: {run_dir}")
        logger.info(
            "\n%s",
            format_config_summary(
                ("Model", config.model),
                ("Optimizer", config.optim),
                ("Data", config.data),
            ),
        )

        checkpoint_path = run_training(
            config,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
            device=args.device,
            output_dir=run_dir.parent,  # experiments/runs/
            run_name=actual_run_name,
            use_amp=use_amp,
            checkpoints_subdir="checkpoints",
        )

        logger.info(f"Training complete. Best checkpoint: {checkpoint_path}")

    # Phase 2: Benchmark
    benchmark_name = f"{actual_run_name}_benchmark"

    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Benchmark")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Benchmark: %s", benchmark_name)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("  → Used by: nn, mcts+nn")

    baseline = Path(config.resume_from) if config.resume_from else None
    if baseline:
        logger.info("Baseline: %s", baseline)
        logger.info("  → Used by: nn-prev, mcts+nn-prev")
    logger.info("")

    # Load MCTS and game configs from Hydra sub-configs
    mcts_config = load_config(DecoupledPUCTConfig, "configs/mcts", args.mcts)
    game_config = load_config(GameConfig, "configs/game", args.game)

    benchmark_config = BenchmarkConfig(
        games_per_matchup=args.games,
        workers=args.workers,
        device=args.device,
        mcts=mcts_config,
    )

    tournament_config = build_benchmark_tournament(
        benchmark_name,
        checkpoint_path,
        benchmark_config,
        game_config=game_config,
        baseline_checkpoint=baseline,
    )
    logger.info(
        "\n%s",
        format_config_summary(
            ("Game", game_config),
            ("MCTS", mcts_config),
            ("Benchmark", benchmark_config),
        ),
    )
    logger.info("")

    # Create benchmark via ExperimentManager
    bench_dir = exp.create_benchmark(
        name=benchmark_name,
        config=tournament_config.model_dump(),
        checkpoints=[str(checkpoint_path)],
    )
    logger.info(f"Created benchmark: {benchmark_name}")
    logger.info(f"  Benchmark directory: {bench_dir}")

    result = run_tournament(tournament_config)

    # Save and print results
    exp.save_benchmark_results(benchmark_name, result.to_dict())
    logger.info(f"Results saved to {bench_dir / 'results.json'}")

    print_benchmark_results(result, anchor="greedy")


if __name__ == "__main__":
    main()
