#!/usr/bin/env python3
"""Training script for PyRat neural network.

Usage:
    uv run python scripts/train.py configs/train.yaml
    uv run python scripts/train.py configs/train.yaml --epochs 200
    uv run python scripts/train.py configs/train.yaml --amp  # Force AMP
    uv run python scripts/train.py configs/train.yaml --name override_name  # Override config name
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from alpharat.experiments import ExperimentManager
from alpharat.experiments.paths import shard_id_from_path
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
        default=None,
        help="Override run name from config (default: use config.name)",
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

    # Run name: CLI override or config.name
    run_name = args.name if args.name else config.name

    exp = ExperimentManager(args.experiments_dir)

    # Auto-detect source shards from data path if not specified
    source_shards = args.source_shards
    if source_shards is None:
        # Try to extract shard ID from train_dir path
        # Expected format: experiments/shards/{group}/{uuid}/train
        train_dir = Path(config.data.train_dir)
        if train_dir.name == "train":
            shard_dir = train_dir.parent  # {uuid}
            group_dir = shard_dir.parent  # {group}
            shards_dir = group_dir.parent  # shards/
            if shards_dir.name == "shards":
                source_shards = shard_id_from_path(shard_dir)
            else:
                raise ValueError(
                    f"Cannot auto-detect source shards from '{config.data.train_dir}'. "
                    f"Expected path like 'experiments/shards/{{group}}/{{uuid}}/train'. "
                    f"Use --source-shards group/uuid to specify explicitly."
                )
        else:
            raise ValueError(
                f"Cannot auto-detect source shards from '{config.data.train_dir}'. "
                f"Expected path ending in '/train'. "
                f"Use --source-shards group/uuid to specify explicitly."
            )

    # Create run via ExperimentManager (unless resuming)
    if args.resume is None:
        run_dir = exp.create_run(
            name=run_name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )
        # Name might have been auto-incremented if same config exists
        actual_name = run_dir.name
        if actual_name != run_name:
            logger.info(f"Run '{run_name}' exists with same config, using '{actual_name}'")
        logger.info(f"Created run: {actual_name}")
        logger.info(f"  Run directory: {run_dir}")
    else:
        # Resuming - use existing run directory and set resume_from in config
        run_dir = exp.get_run_path(run_name)
        if not run_dir.exists():
            logger.error(f"Run '{run_name}' not found at {run_dir}")
            return
        actual_name = run_name
        # Update config with resume path (TrainConfig handles resume via resume_from field)
        config = config.model_copy(update={"resume_from": str(args.resume)})
        logger.info(f"Resuming run: {actual_name}")
        logger.info(f"  Run directory: {run_dir}")

    # Run training
    # Checkpoints go to experiments/runs/{name}/checkpoints/
    best_model = run_training(
        config,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        device=args.device,
        output_dir=run_dir.parent,  # experiments/runs/
        run_name=actual_name,
        use_amp=use_amp,
        checkpoints_subdir="checkpoints",
    )

    logger.info(f"Training complete. Best model: {best_model}")


if __name__ == "__main__":
    main()
