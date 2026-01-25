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

CLI overrides (--name, --shards) are merged into the config before saving,
so the frozen config in experiments/runs/{name}/config.yaml has actual values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from alpharat.eval.benchmark import BenchmarkConfig, build_benchmark_tournament
from alpharat.eval.elo import compute_elo, from_tournament_result
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
    parser.add_argument("config", type=Path, help="Path to training YAML config")
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

    exp = ExperimentManager(args.experiments_dir)

    # Load config as dict first (for merging CLI overrides)
    config_data = yaml.safe_load(args.config.read_text())

    # Merge CLI overrides BEFORE validation (so frozen config has actual values)
    if args.shards:
        # Validate format
        if "/" not in args.shards:
            parser.error("--shards requires 'GROUP/UUID' format (e.g., 'mygroup/abc123')")
        shard_path = exp.get_shard_path(args.shards)
        if not shard_path.exists():
            parser.error(f"Shard not found: {args.shards} (looked in {shard_path})")
        # Merge into config
        config_data.setdefault("data", {})
        config_data["data"]["train_dir"] = str(shard_path / "train")
        config_data["data"]["val_dir"] = str(shard_path / "val")

    if args.name:
        config_data["name"] = args.name

    # Now validate the merged config
    config = TrainConfig.model_validate(config_data)

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

    # Build MCTS config for benchmark agents
    mcts_config = DecoupledPUCTConfig(
        simulations=args.mcts_sims,
        c_puct=8.34,
        force_k=0.88,
    )

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
        baseline_checkpoint=baseline,
    )
    game_config = tournament_config.game
    logger.info(
        "Game settings: %dx%d, %d cheese, %d max turns",
        game_config.width,
        game_config.height,
        game_config.cheese_count,
        game_config.max_turns,
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

    # Save results via ExperimentManager
    results_dict = {
        "standings": result.standings(),
        "wdl_matrix": {
            agent: {
                opp: {"wins": wdl[0], "draws": wdl[1], "losses": wdl[2]}
                for opp, wdl in opps.items()
            }
            for agent, opps in result.wdl_matrix().items()
        },
        "cheese_stats": {
            agent: {
                opp: {"scored": cheese[0], "conceded": cheese[1]} for opp, cheese in opps.items()
            }
            for agent, opps in result.cheese_matrix().items()
        },
    }
    exp.save_benchmark_results(benchmark_name, results_dict)
    logger.info(f"Results saved to {bench_dir / 'results.json'}")

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
