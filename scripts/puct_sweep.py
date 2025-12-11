#!/usr/bin/env python3
"""PUCT parameter sweep: measure cheese collected across n_sims × c_puct grid.

Usage:
    uv run python scripts/puct_sweep.py configs/puct_sweep.yaml
    uv run python scripts/puct_sweep.py configs/puct_sweep.yaml --workers 4

Results are saved incrementally to CSV, so you can kill and resume.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from alpharat.ai import MCTSAgent, RandomAgent
from alpharat.data.batch import GameParams  # noqa: TC001
from alpharat.eval.game import play_game
from alpharat.mcts import DecoupledPUCTConfig


class SweepConfig(BaseModel):
    """Configuration for PUCT parameter sweep."""

    n_sims_values: list[int]
    c_puct_values: list[float]
    games_per_config: int
    game: GameParams
    output_file: str


@dataclass
class GameTask:
    """Single game to run."""

    n_sims: int
    c_puct: float
    game_idx: int
    seed: int


@dataclass
class GameResultRow:
    """Result of a single game, ready for CSV."""

    n_sims: int
    c_puct: float
    game_idx: int
    seed: int
    cheese_collected: float
    opponent_cheese: float
    turns: int
    winner: int  # 1=PUCT won, 2=Random won, 0=draw
    elapsed_seconds: float


def run_single_game(task: GameTask, game_params: GameParams) -> GameResultRow:
    """Run a single game and return result."""
    config = DecoupledPUCTConfig(
        simulations=task.n_sims,
        c_puct=task.c_puct,
    )
    agent = MCTSAgent(config)
    opponent = RandomAgent()

    start = time.perf_counter()
    result = play_game(
        agent,
        opponent,
        seed=task.seed,
        width=game_params.width,
        height=game_params.height,
        cheese_count=game_params.cheese_count,
        max_turns=game_params.max_turns,
    )
    elapsed = time.perf_counter() - start

    return GameResultRow(
        n_sims=task.n_sims,
        c_puct=task.c_puct,
        game_idx=task.game_idx,
        seed=task.seed,
        cheese_collected=result.p1_score,
        opponent_cheese=result.p2_score,
        turns=result.turns,
        winner=result.winner,
        elapsed_seconds=elapsed,
    )


def load_existing_results(output_path: Path) -> set[tuple[int, float, int]]:
    """Load already-completed (n_sims, c_puct, game_idx) from CSV."""
    completed: set[tuple[int, float, int]] = set()
    if output_path.exists():
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row["n_sims"]), float(row["c_puct"]), int(row["game_idx"]))
                completed.add(key)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUCT parameter sweep")
    parser.add_argument("config", type=Path, help="Sweep YAML config file")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        parser.error(f"Config file not found: {args.config}")

    data = yaml.safe_load(args.config.read_text())
    config = SweepConfig.model_validate(data)

    # Setup output
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check what's already done
    completed = load_existing_results(output_path)

    # Build task list
    tasks: list[GameTask] = []
    for n_sims in config.n_sims_values:
        for c_puct in config.c_puct_values:
            for game_idx in range(config.games_per_config):
                if (n_sims, c_puct, game_idx) in completed:
                    continue
                seed = hash((n_sims, c_puct, game_idx)) % (2**31)
                tasks.append(GameTask(n_sims=n_sims, c_puct=c_puct, game_idx=game_idx, seed=seed))

    total_configs = len(config.n_sims_values) * len(config.c_puct_values)
    total_games = total_configs * config.games_per_config
    already_done = len(completed)

    print(f"PUCT Sweep: {len(config.n_sims_values)} n_sims × {len(config.c_puct_values)} c_puct")
    print(f"  n_sims: {config.n_sims_values}")
    print(f"  c_puct: {config.c_puct_values}")
    print(f"  {config.games_per_config} games per config = {total_games} total games")
    print(f"  Already completed: {already_done}, remaining: {len(tasks)}")
    print(f"  Output: {output_path}")
    print()

    if not tasks:
        print("All games already completed!")
        return

    # Track progress
    done = 0
    start_time = time.perf_counter()

    # Aggregate stats for live reporting
    stats: dict[tuple[int, float], list[float]] = {}
    completed_configs: set[tuple[int, float]] = set()

    # Open CSV for appending
    write_header = not output_path.exists()
    fieldnames = [
        "n_sims",
        "c_puct",
        "game_idx",
        "seed",
        "cheese_collected",
        "opponent_cheese",
        "turns",
        "winner",
        "elapsed_seconds",
    ]

    with open(output_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        def record_result(result: GameResultRow) -> None:
            nonlocal done
            done += 1

            # Write to CSV
            writer.writerow(
                {
                    "n_sims": result.n_sims,
                    "c_puct": result.c_puct,
                    "game_idx": result.game_idx,
                    "seed": result.seed,
                    "cheese_collected": result.cheese_collected,
                    "opponent_cheese": result.opponent_cheese,
                    "turns": result.turns,
                    "winner": result.winner,
                    "elapsed_seconds": f"{result.elapsed_seconds:.2f}",
                }
            )
            csv_file.flush()

            # Update stats
            key = (result.n_sims, result.c_puct)
            if key not in stats:
                stats[key] = []
            stats[key].append(result.cheese_collected)

            # Print summary when a config finishes all its games
            if len(stats[key]) == config.games_per_config and key not in completed_configs:
                completed_configs.add(key)
                values = stats[key]
                avg = sum(values) / len(values)
                std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                print(f"  ✓ n_sims={key[0]:>4}, c_puct={key[1]:>4.1f} → {avg:.2f} ± {std:.2f}")

            # Progress report every 10 games
            if done % 10 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"[{done}/{len(tasks)}] "
                    f"{elapsed:.0f}s elapsed, {rate:.2f} games/s, ETA {eta:.0f}s"
                )

        # Run games
        if args.workers == 1:
            for task in tasks:
                result = run_single_game(task, config.game)
                record_result(result)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_single_game, task, config.game): task for task in tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    record_result(result)

    # Print summary
    print()
    print("=" * 70)
    print("Summary: Cheese collected per config (mean ± std)")
    print("=" * 70)
    print(f"{'n_sims':>8} {'c_puct':>8} {'mean':>10} {'std':>10} {'games':>8}")
    print("-" * 50)

    for n_sims in config.n_sims_values:
        for c_puct in config.c_puct_values:
            key = (n_sims, c_puct)
            if key in stats:
                values = stats[key]
                avg = sum(values) / len(values)
                std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
                print(f"{n_sims:>8} {c_puct:>8.1f} {avg:>10.2f} {std:>10.2f} {len(values):>8}")

    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
