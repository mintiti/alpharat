#!/usr/bin/env python3
"""Run round-robin tournament from YAML config.

Usage:
    uv run python scripts/benchmark.py configs/tournament.yaml
    uv run python scripts/benchmark.py configs/tournament.yaml --workers 8

Example YAML config:
    agents:
      ps_100:
        variant: prior_sampling
        simulations: 100
      puct_100:
        variant: decoupled_puct
        simulations: 100
        c_puct: 1.5

    games_per_matchup: 20

    game:
      width: 7
      height: 7
      max_turns: 100
      cheese_count: 10

    workers: 4
"""

import argparse
from pathlib import Path

import yaml

from alpharat.eval.elo import compute_elo, from_tournament_result
from alpharat.eval.tournament import TournamentConfig, run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Run round-robin MCTS tournament")
    parser.add_argument("config", type=Path, help="Tournament YAML config file")
    parser.add_argument("--workers", type=int, help="Override worker count from config")
    args = parser.parse_args()

    # Load and parse config
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    data = yaml.safe_load(args.config.read_text())
    config = TournamentConfig.model_validate(data)

    if args.workers:
        config = config.model_copy(update={"workers": args.workers})

    # Run tournament
    result = run_tournament(config)

    # Print results
    print(result.standings_table())
    print()
    print(result.wdl_table())
    print()
    print(result.cheese_table())

    # Compute and print Elo ratings
    print()
    records = from_tournament_result(result)
    # Use first agent as anchor if no greedy, otherwise greedy
    anchor = "greedy" if "greedy" in config.agents else list(config.agents.keys())[0]
    elo_result = compute_elo(records, anchor=anchor, anchor_elo=1000, compute_uncertainty=True)
    print(elo_result.format_table())


if __name__ == "__main__":
    main()
