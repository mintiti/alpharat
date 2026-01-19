#!/usr/bin/env python3
"""Self-play sampling script for training data generation.

Usage:
    alpharat-sample configs/sample.yaml --group mygroup
    alpharat-sample configs/sample.yaml --checkpoint path/to/model.pt
    alpharat-sample configs/sample.yaml --workers 8

CLI overrides (--group, --checkpoint) are merged into the config before saving,
so the batch metadata has actual values.
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
        help="Override batch group name (merged into saved metadata)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path for NN-guided MCTS (merged into saved metadata)",
    )
    parser.add_argument("--workers", type=int, help="Override number of parallel workers")
    args = parser.parse_args()

    # Load config as dict first (for merging CLI overrides)
    config_data = yaml.safe_load(args.config.read_text())

    # Merge CLI overrides BEFORE validation (so saved metadata has actual values)
    if args.group is not None:
        config_data["group"] = args.group

    if args.checkpoint is not None:
        config_data["checkpoint"] = args.checkpoint

    config = SamplingConfig.model_validate(config_data)

    # Apply runtime overrides (not saved to metadata)
    if args.workers is not None:
        config.sampling.workers = args.workers

    # Run sampling
    run_sampling(config)


if __name__ == "__main__":
    main()
