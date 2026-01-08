#!/usr/bin/env python3
"""Benchmark MCTS simulations per second.

Usage:
    uv run python scripts/benchmark_mcts.py
    uv run python scripts/benchmark_mcts.py --profile
"""

from __future__ import annotations

import argparse
import cProfile
import statistics
import time

from alpharat.data.batch import GameParams
from alpharat.data.sampling import build_tree, create_game
from alpharat.mcts import DecoupledPUCTConfig, DecoupledPUCTSearch


def benchmark_sims_per_second(
    n_sims: int = 1000,
    width: int = 5,
    height: int = 5,
    cheese_count: int = 5,
) -> tuple[float, float, int]:
    """Run benchmark and return simulations per second."""
    params = GameParams(
        width=width,
        height=height,
        cheese_count=cheese_count,
        max_turns=100,
    )
    game = create_game(params, seed=42)
    tree = build_tree(game, gamma=1.0)
    config = DecoupledPUCTConfig(simulations=n_sims)
    search = DecoupledPUCTSearch(tree, config)

    start = time.perf_counter()
    search.search()
    elapsed = time.perf_counter() - start

    sims_per_sec = n_sims / elapsed
    return sims_per_sec, elapsed, n_sims


def run_for_profile(n_sims: int = 5000) -> None:
    """Run search in a way that's easy to profile."""
    params = GameParams(width=5, height=5, cheese_count=5, max_turns=100)
    game = create_game(params, seed=42)
    tree = build_tree(game, gamma=1.0)
    config = DecoupledPUCTConfig(simulations=n_sims)
    search = DecoupledPUCTSearch(tree, config)
    search.search()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Run with cProfile")
    parser.add_argument("--sims", type=int, default=1000, help="Number of simulations")
    args = parser.parse_args()

    if args.profile:
        print(f"Profiling {args.sims} simulations...")

        profiler = cProfile.Profile()
        profiler.enable()
        run_for_profile(n_sims=args.sims)
        profiler.disable()

        # Save to file for snakeviz
        profile_path = "mcts_profile.prof"
        profiler.dump_stats(profile_path)
        print(f"\nProfile saved to {profile_path}")
        print(f"Run: snakeviz {profile_path}")
    else:
        print("MCTS Benchmark")
        print("=" * 50)

        # Warm-up runs
        print("\nWarm-up (3 runs)...")
        for _ in range(3):
            benchmark_sims_per_second(n_sims=args.sims)

        # Actual benchmark (5 trials)
        n_sims = args.sims
        print(f"\nBenchmarking {n_sims} simulations (5 trials)...")

        rates = []
        for i in range(5):
            sims_per_sec, elapsed, _ = benchmark_sims_per_second(n_sims=n_sims)
            rates.append(sims_per_sec)
            print(f"  Trial {i + 1}: {sims_per_sec:.1f} sims/s")

        mean_rate = statistics.mean(rates)
        stdev = statistics.stdev(rates)

        print("\nResults:")
        print(f"  Mean: {mean_rate:.1f} sims/s (±{stdev:.1f})")
        print(f"  Time per sim: {1000 / mean_rate:.3f}ms")


if __name__ == "__main__":
    main()
