#!/usr/bin/env python3
"""Rust self-play sampling CLI.

Runs the Rust self-play pipeline (ONNX NN + multi-threaded MCTS) and writes
game bundles to the experiments directory with lineage tracking.

Usage:
    # Sample with pure MCTS (no NN)
    alpharat-rust-sample configs/iterate.yaml --group iter0 --num-games 1000

    # Sample with NN checkpoint
    alpharat-rust-sample configs/iterate.yaml --group iter1 --num-games 1000 \
        --checkpoint experiments/runs/mlp_v1/checkpoints/best_model.pt

    # Override threads and bundle size
    alpharat-rust-sample configs/iterate.yaml --group iter0 --num-games 500 \
        --threads 8 --max-bundle 64
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pydantic import Field

from alpharat.config.base import StrictBaseModel
from alpharat.config.display import format_config_summary
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.config.loader import load_config, split_config_path
from alpharat.mcts.config import RustMCTSConfig  # noqa: TC001

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RustSampleConfig(StrictBaseModel):
    """Configuration for Rust self-play sampling."""

    game: GameConfig
    mcts: RustMCTSConfig = Field(default_factory=RustMCTSConfig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rust self-play sampling")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("--group", type=str, required=True, help="Batch group name")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of games")
    parser.add_argument("--threads", type=int, default=4, help="Worker threads")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--max-bundle", type=int, default=32, help="Max games per NPZ bundle")
    parser.add_argument("--mux-batch", type=int, default=256, help="Max mux batch size for ONNX")
    parser.add_argument(
        "--experiments-dir", type=Path, default=Path("experiments"), help="Experiments directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference backend (auto, cpu, cuda, coreml, mps, tensorrt)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="ONNX execution provider (auto, cpu, cuda, coreml, tensorrt)",
    )
    args = parser.parse_args()

    config_dir, config_name = split_config_path(args.config)
    config = load_config(RustSampleConfig, config_dir, config_name)

    logger.info("Rust Self-Play Sampling")
    logger.info("=" * 60)
    summary = format_config_summary(
        ("Game", config.game),
        ("MCTS", config.mcts),
    )
    logger.info("\n%s", summary)
    logger.info(
        "Games: %d, Threads: %d, Checkpoint: %s",
        args.num_games,
        args.threads,
        args.checkpoint or "(none)",
    )

    from alpharat.data.rust_sampling import resolve_sampling_device, run_rust_sampling

    resolved_device = resolve_sampling_device(args.device)
    logger.info("Device: %s (resolved: %s)", args.device, resolved_device)

    batch_dir, metrics = run_rust_sampling(
        game=config.game,
        mcts=config.mcts,
        num_games=args.num_games,
        group=args.group,
        num_threads=args.threads,
        max_games_per_bundle=args.max_bundle,
        mux_max_batch_size=args.mux_batch,
        checkpoint=args.checkpoint,
        experiments_dir=args.experiments_dir,
        verbose=not args.quiet,
        device=resolved_device,
    )

    logger.info("")
    logger.info("Results:")
    logger.info("  Batch dir: %s", batch_dir)
    logger.info("  Games: %d", metrics.total_games)
    logger.info("  Positions: %d", metrics.total_positions)
    logger.info("  Simulations: %d", metrics.total_simulations)
    logger.info("  Throughput: %.0f sims/s", metrics.simulations_per_second)
    logger.info("  Elapsed: %.1fs", metrics.elapsed_seconds)
    logger.info("  Outcomes: P1=%d P2=%d Draw=%d", metrics.p1_wins, metrics.p2_wins, metrics.draws)
    logger.info("  Cheese utilization: %.1f%%", metrics.cheese_utilization * 100)


if __name__ == "__main__":
    main()
