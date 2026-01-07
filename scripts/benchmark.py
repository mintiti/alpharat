#!/usr/bin/env python3
"""Run round-robin tournament from YAML config.

Usage:
    uv run python scripts/benchmark.py configs/tournament.yaml --name tournament_v1
    uv run python scripts/benchmark.py configs/tournament.yaml --name baseline_test --workers 8

Example YAML config:
    agents:
      puct_100:
        variant: decoupled_puct
        simulations: 100
        c_puct: 1.5
      puct_500:
        variant: decoupled_puct
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

import yaml

from alpharat.eval.elo import compute_elo, from_tournament_result
from alpharat.eval.tournament import TournamentConfig, run_tournament
from alpharat.experiments import ExperimentManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run round-robin MCTS tournament")
    parser.add_argument("config", type=Path, help="Tournament YAML config file")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Human-readable benchmark name (e.g., 'tournament_v1', 'baseline_comparison')",
    )
    parser.add_argument("--workers", type=int, help="Override worker count from config")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )
    args = parser.parse_args()

    # Load and parse config
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    data = yaml.safe_load(args.config.read_text())
    config = TournamentConfig.model_validate(data)

    if args.workers:
        config = config.model_copy(update={"workers": args.workers})

    exp = ExperimentManager(args.experiments_dir)

    # Extract checkpoint names from agent configs (for lineage tracking)
    checkpoints = []
    for agent_name, agent_config in config.agents.items():
        if hasattr(agent_config, "checkpoint") and agent_config.checkpoint:
            checkpoints.append(agent_config.checkpoint)
        else:
            checkpoints.append(agent_name)  # Use agent name if no checkpoint

    # Create benchmark via ExperimentManager
    bench_dir = exp.create_benchmark(
        name=args.name,
        config=data,  # Save original config
        checkpoints=checkpoints,
    )
    logger.info(f"Created benchmark: {args.name}")
    logger.info(f"  Benchmark directory: {bench_dir}")

    # Run tournament
    logger.info("Running tournament...")
    result = run_tournament(config)

    # Save results
    results_dict = {
        "standings": result.standings(),
        "wdl_matrix": {
            agent: {
                opp: {"wins": wdl[0], "draws": wdl[1], "losses": wdl[2]}
                for opp, wdl in opps.items()
            }
            for agent, opps in result.wdl_matrix().items()
        },
        "cheese_stats": {
            agent: {
                opp: {"scored": cheese[0], "conceded": cheese[1]} for opp, cheese in opps.items()
            }
            for agent, opps in result.cheese_matrix().items()
        },
    }
    exp.save_benchmark_results(args.name, results_dict)
    logger.info(f"Results saved to {bench_dir / 'results.json'}")

    # Print results
    print()
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
