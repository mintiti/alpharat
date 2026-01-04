#!/usr/bin/env python3
"""Benchmark MCTS tree reuse between turns.

Tests whether tree reuse leads to stronger play by measuring win rate
against a greedy baseline. Compares MCTS with tree reuse vs MCTS without.

Usage:
    uv run python scripts/benchmark_tree_reuse.py
    uv run python scripts/benchmark_tree_reuse.py --games 50 --simulations 50,100,200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from alpharat.ai.base import Agent
from alpharat.ai.greedy_agent import GreedyAgent
from alpharat.ai.mcts_agent import MCTSAgent
from alpharat.eval.game import play_game


@dataclass
class BenchmarkResult:
    """Result of MCTS vs Greedy benchmark."""

    simulations: int
    reuse_tree: bool
    games: int
    mcts_wins: int
    greedy_wins: int
    draws: int
    total_time: float

    @property
    def win_rate(self) -> float:
        """MCTS win rate (wins / decisive games)."""
        decisive = self.mcts_wins + self.greedy_wins
        return self.mcts_wins / decisive if decisive > 0 else 0.5

    @property
    def time_per_game(self) -> float:
        """Average time per game in seconds."""
        return self.total_time / self.games


def run_vs_greedy(
    simulations: int,
    reuse_tree: bool,
    n_games: int,
    width: int = 5,
    height: int = 5,
    cheese_count: int = 5,
    max_turns: int = 100,
) -> BenchmarkResult:
    """Run MCTS agent vs Greedy agent.

    Alternates which agent plays as P1/P2 for fairness.

    Args:
        simulations: Number of MCTS simulations per move.
        reuse_tree: Whether to enable tree reuse.
        n_games: Number of games to play.
        width: Maze width.
        height: Maze height.
        cheese_count: Number of cheese pieces.
        max_turns: Maximum turns per game.

    Returns:
        BenchmarkResult with statistics.
    """
    mcts_wins = 0
    greedy_wins = 0
    draws = 0

    start_time = time.perf_counter()

    for game_idx in range(n_games):
        # Alternate sides for fairness
        mcts_is_p1 = game_idx % 2 == 0

        agent_mcts = MCTSAgent(
            simulations=simulations,
            reuse_tree=reuse_tree,
            search_variant="prior_sampling",
        )
        agent_greedy = GreedyAgent()

        agent_p1: Agent
        agent_p2: Agent
        if mcts_is_p1:
            agent_p1, agent_p2 = agent_mcts, agent_greedy
        else:
            agent_p1, agent_p2 = agent_greedy, agent_mcts

        result = play_game(
            agent_p1,
            agent_p2,
            width=width,
            height=height,
            cheese_count=cheese_count,
            max_turns=max_turns,
            seed=game_idx,
        )

        # Determine winner from MCTS agent's perspective
        if result.winner == 0:
            draws += 1
        elif (result.winner == 1 and mcts_is_p1) or (result.winner == 2 and not mcts_is_p1):
            mcts_wins += 1
        else:
            greedy_wins += 1

    total_time = time.perf_counter() - start_time

    return BenchmarkResult(
        simulations=simulations,
        reuse_tree=reuse_tree,
        games=n_games,
        mcts_wins=mcts_wins,
        greedy_wins=greedy_wins,
        draws=draws,
        total_time=total_time,
    )


def format_table(results: list[tuple[BenchmarkResult, BenchmarkResult]]) -> str:
    """Format results as a table.

    Args:
        results: List of (fresh, reuse) result pairs.

    Returns:
        Formatted table string.
    """
    lines = [
        "MCTS vs Greedy: Tree Reuse Comparison",
        "=" * 70,
        "",
        f"{'Sims':>6} | {'Mode':<8} | {'MCTS W':>7} | {'Greedy W':>8} | {'Draw':>5} | {'Win%':>7}",
        "-" * 70,
    ]

    for fresh, reuse in results:
        # Fresh tree row
        lines.append(
            f"{fresh.simulations:>6} | {'Fresh':<8} | {fresh.mcts_wins:>7} | "
            f"{fresh.greedy_wins:>8} | {fresh.draws:>5} | {fresh.win_rate:>6.1%}"
        )

        # Reuse tree row
        delta = reuse.win_rate - fresh.win_rate
        delta_str = f"({delta:+.1%})"
        lines.append(
            f"{'':>6} | {'Reuse':<8} | {reuse.mcts_wins:>7} | "
            f"{reuse.greedy_wins:>8} | {reuse.draws:>5} | {reuse.win_rate:>6.1%} {delta_str}"
        )
        lines.append("-" * 70)

    lines.append("")
    lines.append("Delta shows change in win rate: positive = reuse helps")

    return "\n".join(lines)


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS tree reuse: measure win rate vs greedy"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Number of games per configuration (default: 20)",
    )
    parser.add_argument(
        "--simulations",
        type=str,
        default="50,100,200",
        help="Comma-separated list of simulation counts (default: 50,100,200)",
    )
    parser.add_argument("--width", type=int, default=5, help="Maze width (default: 5)")
    parser.add_argument("--height", type=int, default=5, help="Maze height (default: 5)")
    parser.add_argument("--cheese", type=int, default=5, help="Cheese count (default: 5)")
    parser.add_argument("--max-turns", type=int, default=100, help="Max turns (default: 100)")

    args = parser.parse_args()

    sim_counts = [int(s.strip()) for s in args.simulations.split(",")]

    print("MCTS Tree Reuse Benchmark")
    print("=" * 40)
    print("Testing MCTS (fresh vs reuse) against Greedy baseline")
    print(f"Games per config: {args.games}")
    print(f"Simulation counts: {sim_counts}")
    print(f"Maze: {args.width}x{args.height}, {args.cheese} cheese, max {args.max_turns} turns")
    print()

    results = []
    for sims in sim_counts:
        print(f"Running {args.games} games with {sims} simulations...")

        print("  Fresh tree...")
        fresh = run_vs_greedy(
            simulations=sims,
            reuse_tree=False,
            n_games=args.games,
            width=args.width,
            height=args.height,
            cheese_count=args.cheese,
            max_turns=args.max_turns,
        )
        print(f"    Win rate: {fresh.win_rate:.1%} ({fresh.time_per_game:.2f}s/game)")

        print("  Tree reuse...")
        reuse = run_vs_greedy(
            simulations=sims,
            reuse_tree=True,
            n_games=args.games,
            width=args.width,
            height=args.height,
            cheese_count=args.cheese,
            max_turns=args.max_turns,
        )
        print(f"    Win rate: {reuse.win_rate:.1%} ({reuse.time_per_game:.2f}s/game)")

        results.append((fresh, reuse))
        print()

    print()
    print(format_table(results))


if __name__ == "__main__":
    main()
