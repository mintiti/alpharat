#!/usr/bin/env python3
"""Self-play sampling script for training data generation.

Usage:
    alpharat-sample configs/sample.yaml --group mygroup
    alpharat-sample configs/sample/5x5 --checkpoint path/to/model.pt
    alpharat-sample configs/sample.yaml --workers 8

CLI overrides (--group, --checkpoint) are merged into the config using Hydra overrides,
so the batch metadata has actual values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpharat.config.loader import load_config
from alpharat.data.sampling import SamplingConfig, run_sampling


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-play training data with MCTS agents."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Config path: 'configs/sample.yaml' or 'configs/sample/5x5' (with or without .yaml)",
    )
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

    # Parse config path: extract configs directory and config name
    config_path = Path(args.config)
    config_name = str(config_path.with_suffix(""))  # Remove .yaml if present
    if config_name.startswith("configs/"):
        config_dir = "configs"
        config_name = config_name[len("configs/") :]
    else:
        # Config is in current directory or explicit path
        config_dir = str(config_path.parent) if config_path.parent.name else "."
        config_name = config_path.stem

    # Build Hydra overrides from CLI args
    overrides: list[str] = []
    if args.group is not None:
        overrides.append(f"group={args.group}")
    if args.checkpoint is not None:
        overrides.append(f"checkpoint={args.checkpoint}")
    if args.workers is not None:
        overrides.append(f"sampling.workers={args.workers}")

    # Load config using Hydra (resolves defaults) + Pydantic (validates)
    config = load_config(SamplingConfig, config_dir, config_name, overrides=overrides or None)

    # Run sampling
    run_sampling(config)


if __name__ == "__main__":
    main()
