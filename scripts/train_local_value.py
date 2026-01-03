#!/usr/bin/env python3
"""Training script for LocalValueMLP with ownership prediction.

Usage:
    uv run python scripts/train_local_value.py configs/train_local_value.yaml
    uv run python scripts/train_local_value.py configs/train_local_value.yaml --epochs 200
    uv run python scripts/train_local_value.py configs/train_local_value.yaml \\
        --resume checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from alpharat.nn.local_value_training import LocalValueTrainConfig, run_local_value_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LocalValueMLP with ownership prediction.")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save every N epochs")
    parser.add_argument("--metrics-every", type=int, default=10, help="Detailed metrics every N")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"), help="Output dir")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for tensorboard")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config_data = yaml.safe_load(args.config.read_text())
    config = LocalValueTrainConfig.model_validate(config_data)

    # Run training
    run_local_value_training(
        config,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
