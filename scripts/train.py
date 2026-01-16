#!/usr/bin/env python3
"""Training script for PyRat neural network.

Usage:
    uv run python scripts/train.py configs/train.yaml --name mlp_v1 --shards mygroup/abc123
    uv run python scripts/train.py configs/train.yaml --epochs 200
    uv run python scripts/train.py configs/train.yaml --amp  # Force AMP

CLI overrides (--name, --shards) are merged into the config before saving,
so the frozen config in experiments/runs/{name}/config.yaml has actual values.
"""

from __future__ import annotations

import argparse
import logging
import sys
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
        help="Override run name (merged into frozen config)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Shard ID as GROUP/UUID (e.g., 'mygroup/abc123'). Overrides data paths in config.",
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
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")

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
        # Auto-detect from data path
        train_dir = Path(config.data.train_dir)
        if train_dir.name == "train":
            shard_dir = train_dir.parent  # {uuid}
            group_dir = shard_dir.parent  # {group}
            shards_dir = group_dir.parent  # shards/
            if shards_dir.name == "shards":
                source_shards = shard_id_from_path(shard_dir)
            else:
                logger.error(
                    f"Cannot auto-detect source shards from '{config.data.train_dir}'. "
                    f"Expected path like 'experiments/shards/GROUP/UUID/train'. "
                    f"Use --shards GROUP/UUID to specify explicitly."
                )
                sys.exit(1)
        else:
            logger.error(
                f"Cannot auto-detect source shards from '{config.data.train_dir}'. "
                f"Expected path ending in '/train'. "
                f"Use --shards GROUP/UUID to specify explicitly."
            )
            sys.exit(1)

    # Create run via ExperimentManager (unless resuming)
    # Note: config.name has the merged value (from --name if provided, else from YAML)
    if args.resume is None:
        run_dir = exp.create_run(
            name=config.name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )
        # Name might have been auto-incremented if same config exists
        actual_name = run_dir.name
        if actual_name != config.name:
            logger.info(f"Run '{config.name}' exists with same config, using '{actual_name}'")
        logger.info(f"Created run: {actual_name}")
        logger.info(f"  Run directory: {run_dir}")
    else:
        # Resuming - use existing run directory and set resume_from in config
        run_dir = exp.get_run_path(config.name)
        if not run_dir.exists():
            logger.error(f"Run '{config.name}' not found at {run_dir}")
            return
        actual_name = config.name
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
