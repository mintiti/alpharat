#!/usr/bin/env python3
"""Run a round-robin tournament from a YAML config.

Usage:
    uv run python scripts/run_tournament.py configs/tree_reuse_tournament.yaml
    uv run python scripts/run_tournament.py configs/tree_reuse_tournament.yaml --workers 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from alpharat.eval.tournament import TournamentConfig, run_tournament


def main() -> None:
    """Run tournament from config file."""
    parser = argparse.ArgumentParser(description="Run agent tournament")
    parser.add_argument("config", type=Path, help="Path to tournament config YAML")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of workers (default: from config)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Override games per matchup (default: from config)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    # Apply overrides
    if args.workers is not None:
        config_dict["workers"] = args.workers
    if args.games is not None:
        config_dict["games_per_matchup"] = args.games

    config = TournamentConfig(**config_dict)

    print(f"Tournament: {args.config.name}")
    print(f"Agents: {list(config.agents.keys())}")
    print(f"Games per matchup: {config.games_per_matchup}")
    print(f"Workers: {config.workers}")
    print()

    # Run tournament
    result = run_tournament(config, verbose=True)

    # Print results
    print()
    print(result.standings_table())
    print()
    print(result.wdl_table())
    print()
    print(result.cheese_table())


if __name__ == "__main__":
    main()
