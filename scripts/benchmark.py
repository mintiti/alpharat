#!/usr/bin/env python3
"""Run round-robin tournament from YAML config.

Usage:
    alpharat-benchmark configs/tournament.yaml
    alpharat-benchmark configs/tournament.yaml --name override  # Override config
    alpharat-benchmark configs/tournament.yaml --workers 8

Example YAML config:
    name: tournament_v1  # Required: benchmark name

    agents:
      puct_100:
        variant: mcts
        mcts:
          simulations: 100
          c_puct: 1.5
      puct_500:
        variant: mcts
        mcts:
          simulations: 500
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
import logging
from pathlib import Path

from alpharat.config.loader import load_config, split_config_path
from alpharat.eval.benchmark import print_benchmark_results
from alpharat.eval.tournament import TournamentConfig, run_tournament
from alpharat.experiments import ExperimentManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run round-robin MCTS tournament")
    parser.add_argument(
        "config",
        type=str,
        help="Config path: 'configs/tournament.yaml' (with or without .yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override benchmark name from config (default: use config.name)",
    )
    parser.add_argument("--workers", type=int, help="Override worker count from config")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )
    args = parser.parse_args()

    config_dir, config_name = split_config_path(args.config)

    # Build Hydra overrides from CLI args
    overrides: list[str] = []
    if args.name:
        overrides.append(f"name={args.name}")
    if args.workers:
        overrides.append(f"workers={args.workers}")

    # Load config using Hydra (resolves defaults) + Pydantic (validates)
    config = load_config(TournamentConfig, config_dir, config_name, overrides=overrides or None)

    # CLI --name overrides config
    benchmark_name = args.name if args.name else config.name

    exp = ExperimentManager(args.experiments_dir)

    # Extract checkpoint names from agent configs (for lineage tracking)
    checkpoints = []
    for agent_name, agent_config in config.agents.items():
        if hasattr(agent_config, "checkpoint") and agent_config.checkpoint:
            checkpoints.append(agent_config.checkpoint)
        else:
            checkpoints.append(agent_name)  # Use agent name if no checkpoint

    # Run tournament first, then create benchmark with results
    logger.info("Running tournament...")
    result = run_tournament(config)

    bench_dir = exp.create_benchmark(
        name=benchmark_name,
        config=config.model_dump(),
        checkpoints=checkpoints,
        results=result.to_dict(),
    )
    logger.info(f"Created benchmark: {benchmark_name}")
    logger.info(f"  Results saved to {bench_dir / 'results.json'}")

    anchor = "greedy" if "greedy" in config.agents else list(config.agents.keys())[0]
    print_benchmark_results(result, anchor=anchor)


if __name__ == "__main__":
    main()
