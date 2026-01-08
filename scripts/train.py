#!/usr/bin/env python3
"""Training script for PyRat neural network.

Usage:
    uv run python scripts/train.py configs/train.yaml
    uv run python scripts/train.py configs/train.yaml --epochs 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from alpharat.nn.config import TrainConfig
from alpharat.nn.training import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyRat neural network.")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--metrics-every", type=int, default=10, help="Detailed metrics every N")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output dir")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for tensorboard")

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

    # Run training
    run_training(
        config,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
