#!/usr/bin/env python3
"""Self-play sampling script for training data generation.

Usage:
    uv run python scripts/sample.py configs/sample.yaml
    uv run python scripts/sample.py configs/sample.yaml --workers 8
    uv run python scripts/sample.py configs/sample.yaml --group override_name  # Override config
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from alpharat.data.sampling import SamplingConfig, run_sampling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-play training data with MCTS agents."
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Override batch group name from config (default: use config.group)",
    )
    parser.add_argument("--workers", type=int, help="Override number of parallel workers")
    args = parser.parse_args()

    # Load config
    config_data = yaml.safe_load(args.config.read_text())

    # CLI --group overrides config
    if args.group is not None:
        config_data["group"] = args.group

    config = SamplingConfig.model_validate(config_data)

    # Apply overrides
    if args.workers is not None:
        config.sampling.workers = args.workers

    # Run sampling
    run_sampling(config)


if __name__ == "__main__":
    main()
