#!/usr/bin/env python3
"""Prepare training shards from sampled game batches.

Converts raw game batches into shuffled, sharded training data with train/val split.
Games (not positions) are split to prevent data leakage.

The observation builder is determined by the architecture — this ensures the shard
format matches what the model expects during training.

Usage:
    # Using --architecture directly (simpler)
    alpharat-prepare-shards --architecture mlp --group myshards --batches mybatches

    # Using config file (resolves Hydra defaults)
    alpharat-prepare-shards configs/train.yaml --group myshards --batches mybatches
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
from pathlib import Path

from pydantic import TypeAdapter

from alpharat.config.loader import load_raw_config, split_config_path
from alpharat.data.sharding import prepare_training_set_with_split
from alpharat.experiments import ExperimentManager
from alpharat.experiments.paths import METADATA_FILE, TRAIN_DIR, VAL_DIR, get_shards_dir
from alpharat.nn.config import ARCHITECTURES, ModelConfig

# TypeAdapter for validating union types
_model_config_adapter: TypeAdapter[ModelConfig] = TypeAdapter(ModelConfig)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training shards from game batches.")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        help="Config path, e.g. 'configs/train.yaml' (optional if --architecture)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=ARCHITECTURES,
        help=f"Architecture for observation builder. One of: {', '.join(ARCHITECTURES)}",
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Shard group name (e.g., 'mygroup', 'iteration2')",
    )
    parser.add_argument(
        "--batches",
        type=str,
        required=True,
        help=(
            "Batch IDs to process. Can be: "
            "'group/*' (all in group), "
            "'group/uuid,group/uuid2' (specific batches), "
            "or 'group' (shorthand for 'group/*')"
        ),
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of games for validation (default: 0.1)",
    )
    parser.add_argument(
        "--positions-per-shard",
        type=int,
        default=3_000_000,
        help="Max positions per shard file (default: 3M, ~1-2GB compressed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    # Get architecture from --architecture or from config file
    if args.architecture:
        model_config = _model_config_adapter.validate_python({"architecture": args.architecture})
        logger.info(f"Using architecture: {args.architecture}")
    elif args.config:
        config_dir, config_name = split_config_path(args.config)
        config_data = load_raw_config(config_dir, config_name)
        if "model" not in config_data:
            logger.error(f"Config file {args.config} must have a 'model' section")
            return
        model_config = _model_config_adapter.validate_python(config_data["model"])
        logger.info(f"Using architecture: {model_config.architecture}")
    else:
        parser.error("Either config file or --architecture is required")

    exp = ExperimentManager(args.experiments_dir)

    # Resolve batch IDs from pattern
    batch_ids = _resolve_batch_pattern(exp, args.batches)
    if not batch_ids:
        logger.error(f"No batches found matching '{args.batches}'")
        return

    logger.info(f"Processing {len(batch_ids)} batches: {batch_ids}")

    # Get batch directories
    batch_dirs = [exp.get_batch_path(bid) for bid in batch_ids]

    # Get dimensions from first batch metadata
    width, height = _get_dimensions_from_batch(batch_dirs[0])
    logger.info(f"Grid dimensions: {width}x{height}")

    # Count games
    total_games = sum(
        len(list((d / "games").glob("*.npz"))) for d in batch_dirs if (d / "games").exists()
    )
    logger.info(f"Total games to process: {total_games}")

    # Create builder from model config
    builder = model_config.build_observation_builder(width, height)
    logger.info(f"Using observation builder: {builder.version}")

    # Prepare shards (output to experiments/shards/{group}/)
    # Policy targets come from recorded games (chosen at MCTS time via policy strategy)
    logger.info("Processing games into shards...")
    shards_group_dir = get_shards_dir(exp.root) / args.group
    shards_group_dir.mkdir(parents=True, exist_ok=True)
    result = prepare_training_set_with_split(
        batch_dirs=batch_dirs,
        output_dir=shards_group_dir,
        builder=builder,
        val_ratio=args.val_ratio,
        positions_per_shard=args.positions_per_shard,
        seed=args.seed,
    )

    # Register in manifest
    exp.register_shards(
        group=args.group,
        shard_uuid=result.shard_id,
        source_batches=batch_ids,
        total_positions=result.total_positions,
        train_positions=result.train_positions,
        val_positions=result.val_positions,
        shuffle_seed=args.seed,
    )

    shard_id = f"{args.group}/{result.shard_id}"
    output_path = Path(result.shard_dir)
    logger.info(f"Training set created: {output_path}")
    logger.info(f"  Shard ID: {shard_id}")
    logger.info(f"  Train dir: {output_path / 'train'}")
    logger.info(f"  Val dir: {output_path / 'val'}")
    logger.info(f"  Total positions: {result.total_positions}")
    logger.info(f"  Train positions: {result.train_positions}")
    logger.info(f"  Val positions: {result.val_positions}")

    # Print summary
    _print_summary(output_path)


def _resolve_batch_pattern(exp: ExperimentManager, pattern: str) -> list[str]:
    """Resolve batch pattern to list of batch IDs.

    Patterns:
        'group/*' -> all batches in group
        'group' -> shorthand for 'group/*'
        'group/uuid,group/uuid2' -> specific batches
    """
    all_batches = exp.list_batches()

    # Handle comma-separated list
    if "," in pattern:
        return [p.strip() for p in pattern.split(",")]

    # Handle 'group' (no /) as 'group/*'
    if "/" not in pattern:
        pattern = f"{pattern}/*"

    # Handle wildcard patterns
    if "*" in pattern:
        return [b for b in all_batches if fnmatch.fnmatch(b, pattern)]

    # Single batch ID
    return [pattern] if pattern in all_batches else []


def _get_dimensions_from_batch(batch_dir: Path) -> tuple[int, int]:
    """Extract width/height from batch metadata."""
    metadata_path = batch_dir / METADATA_FILE
    if not metadata_path.exists():
        raise ValueError(f"No metadata.json in {batch_dir}")

    metadata = json.loads(metadata_path.read_text())
    game_config = metadata["game"]
    return game_config["width"], game_config["height"]


def _print_summary(output_path: Path) -> None:
    """Print summary of created training set."""
    for split in [TRAIN_DIR, VAL_DIR]:
        split_dir = output_path / split
        if not split_dir.exists():
            continue

        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            logger.info(
                f"  {split}: {manifest['total_positions']} positions, "
                f"{manifest['shard_count']} shards"
            )


if __name__ == "__main__":
    main()
