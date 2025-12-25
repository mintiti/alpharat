#!/usr/bin/env python3
"""Prepare training shards from sampled game batches.

Converts raw game batches into shuffled, sharded training data with train/val split.
Games (not positions) are split to prevent data leakage.

Usage:
    uv run python scripts/prepare_shards.py --batches-dir batches --output-dir data
    uv run python scripts/prepare_shards.py --batches-dir batches --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alpharat.data.sharding import prepare_training_set_with_split
from alpharat.nn.builders.flat import FlatObservationBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training shards from game batches.")
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=Path("batches"),
        help="Directory containing batch subdirectories (default: batches)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for training set (default: data)",
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
        default=10000,
        help="Max positions per shard file (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    # Find batch directories
    batch_dirs = [
        d for d in args.batches_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    if not batch_dirs:
        logger.error(f"No batch directories found in {args.batches_dir}")
        return

    logger.info(f"Found {len(batch_dirs)} batch directories")

    # Get dimensions from first batch metadata
    width, height = _get_dimensions_from_batch(batch_dirs[0])
    logger.info(f"Grid dimensions: {width}x{height}")

    # Count games
    total_games = sum(
        len(list((d / "games").glob("*.npz"))) for d in batch_dirs if (d / "games").exists()
    )
    logger.info(f"Total games to process: {total_games}")

    # Create builder
    builder = FlatObservationBuilder(width=width, height=height)

    # Prepare shards
    logger.info("Processing games into shards...")
    output_path = prepare_training_set_with_split(
        batch_dirs=batch_dirs,
        output_dir=args.output_dir,
        builder=builder,
        val_ratio=args.val_ratio,
        positions_per_shard=args.positions_per_shard,
        seed=args.seed,
    )

    logger.info(f"Training set created: {output_path}")
    logger.info(f"  Train dir: {output_path / 'train'}")
    logger.info(f"  Val dir: {output_path / 'val'}")

    # Print summary
    _print_summary(output_path)


def _get_dimensions_from_batch(batch_dir: Path) -> tuple[int, int]:
    """Extract width/height from batch metadata."""
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata.json in {batch_dir}")

    metadata = json.loads(metadata_path.read_text())
    game_params = metadata["game_params"]
    return game_params["width"], game_params["height"]


def _print_summary(output_path: Path) -> None:
    """Print summary of created training set."""
    for split in ["train", "val"]:
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
