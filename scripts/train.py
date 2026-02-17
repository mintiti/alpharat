#!/usr/bin/env python3
"""Training script for PyRat neural network.

Usage:
    alpharat-train configs/train.yaml --name mlp_v1 --shards mygroup/abc123
    alpharat-train configs/train/mlp --epochs 200
    alpharat-train configs/train.yaml --amp  # Force AMP

CLI overrides (--name, --shards) are merged into the config using Hydra overrides,
so the frozen config in experiments/runs/{name}/config.yaml has actual values.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from alpharat.config.loader import load_config, split_config_path
from alpharat.experiments import ExperimentManager
from alpharat.nn.config import TrainConfig
from alpharat.nn.training import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyRat neural network.")
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

    config_dir, config_name = split_config_path(args.config)

    # Build Hydra overrides from CLI args
    overrides: list[str] = []

    # Handle --shards: needs special handling to resolve path before Hydra
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

    # Create run via ExperimentManager (unless resuming)
    # Note: config.name has the merged value (from --name if provided, else from YAML)
    if args.resume is None:
        run_dir, actual_name = exp.prepare_run(
            name=config.name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )
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

    # Register run in manifest now that training succeeded
    if args.resume is None:
        exp.register_run(
            name=actual_name,
            config=config.model_dump(),
            source_shards=source_shards,
            parent_checkpoint=None,
        )

    logger.info(f"Training complete. Best model: {best_model}")


if __name__ == "__main__":
    main()
