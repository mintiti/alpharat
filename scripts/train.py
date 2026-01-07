#!/usr/bin/env python3
"""Training script for PyRat neural network.

Usage:
    uv run python scripts/train.py configs/train.yaml --name mlp_baseline_v1
    uv run python scripts/train.py configs/train.yaml --name mlp_v2 --epochs 200
    uv run python scripts/train.py configs/train.yaml --name mlp_v2 --amp  # Force AMP
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from alpharat.experiments import ExperimentManager
from alpharat.nn.config import TrainConfig
from alpharat.nn.training import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyRat neural network.")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Human-readable run name (e.g., 'mlp_baseline_v1', 'localvalue_v2')",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--metrics-every", type=int, default=10, help="Detailed metrics every N")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )
    parser.add_argument(
        "--source-shards",
        type=str,
        default=None,
        help="Shard ID used for training (auto-detected from config if not specified)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")

    # AMP flags (mutually exclusive: auto-detect by default)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", action="store_true", help="Force enable AMP")
    amp_group.add_argument("--no-amp", action="store_true", help="Force disable AMP")

    args = parser.parse_args()

    # Load config
    config_data = yaml.safe_load(args.config.read_text())
    config = TrainConfig.model_validate(config_data)

    # Tri-state AMP: True (force on), False (force off), None (auto-detect)
    use_amp = True if args.amp else (False if args.no_amp else None)

    exp = ExperimentManager(args.experiments_dir)

    # Auto-detect source shards from data path if not specified
    source_shards = args.source_shards
    if source_shards is None:
        # Try to extract shard ID from train_dir path
        train_dir = Path(config.data.train_dir)
        # Expected format: experiments/shards/{uuid}/train or shards/{uuid}/train
        if train_dir.name == "train" and train_dir.parent.exists():
            source_shards = train_dir.parent.name
        else:
            source_shards = "unknown"

    # Create run via ExperimentManager (unless resuming)
    if args.resume is None:
        run_dir = exp.create_run(
            name=args.name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )
        logger.info(f"Created run: {args.name}")
        logger.info(f"  Run directory: {run_dir}")
    else:
        # Resuming - use existing run directory and set resume_from in config
        run_dir = exp.get_run_path(args.name)
        if not run_dir.exists():
            logger.error(f"Run '{args.name}' not found at {run_dir}")
            return
        # Update config with resume path (TrainConfig handles resume via resume_from field)
        config = config.model_copy(update={"resume_from": str(args.resume)})
        logger.info(f"Resuming run: {args.name}")
        logger.info(f"  Run directory: {run_dir}")

    # Run training
    # Use experiments/runs as output_dir and run_name as the run name
    # Checkpoints go to experiments/runs/{name}/checkpoints/
    best_model = run_training(
        config,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        device=args.device,
        output_dir=run_dir.parent,  # experiments/runs/
        run_name=args.name,  # creates experiments/runs/{name}/
        use_amp=use_amp,
        checkpoints_subdir="checkpoints",  # checkpoints go to runs/{name}/checkpoints/
    )

    logger.info(f"Training complete. Best model: {best_model}")


if __name__ == "__main__":
    main()
