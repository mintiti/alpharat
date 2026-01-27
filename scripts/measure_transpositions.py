#!/usr/bin/env python3
"""Measure transposition frequency in MCTS to estimate MCGS benefit.

Transpositions occur when the same game state is reached via different move sequences.
MCGS (Monte Carlo Graph Search) shares nodes for transpositions; MCTS duplicates them.

This script measures:
1. Total nodes vs unique states (with/without turn in hash)
2. Depth distribution of transpositions
3. Compression potential if we used MCGS

Usage:
    # Default sweep across sim counts
    uv run python scripts/measure_transpositions.py

    # Single sim count
    uv run python scripts/measure_transpositions.py --sims 200

    # Custom sweep
    uv run python scripts/measure_transpositions.py --sims 50 200 1000

    # Quick sanity check
    uv run python scripts/measure_transpositions.py --sims 50 --games 2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyrat_engine.core.types import Direction

from alpharat.config.game import GameConfig

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat
from alpharat.data.sampling import build_tree, create_game
from alpharat.eval.game import is_terminal
from alpharat.mcts import DecoupledPUCTConfig, DecoupledPUCTSearch, MCTSNode, MCTSTree
from alpharat.mcts.nash import select_action_from_strategy


@dataclass
class TranspositionStats:
    """Statistics from a single MCTS search."""

    total_nodes: int = 0
    unique_states_with_turn: int = 0
    unique_states_no_turn: int = 0
    depth_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    depth_unique_with_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    depth_unique_no_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AggregateStats:
    """Aggregated statistics across all searches."""

    total_nodes: int = 0
    unique_states_with_turn: int = 0
    unique_states_no_turn: int = 0
    num_searches: int = 0
    depth_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    depth_unique_with_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    depth_unique_no_turn: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, stats: TranspositionStats) -> None:
        self.total_nodes += stats.total_nodes
        self.unique_states_with_turn += stats.unique_states_with_turn
        self.unique_states_no_turn += stats.unique_states_no_turn
        self.num_searches += 1
        for depth, count in stats.depth_counts.items():
            self.depth_counts[depth] += count
        for depth, count in stats.depth_unique_with_turn.items():
            self.depth_unique_with_turn[depth] += count
        for depth, count in stats.depth_unique_no_turn.items():
            self.depth_unique_no_turn[depth] += count

    @property
    def duplicate_ratio_with_turn(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return 1 - (self.unique_states_with_turn / self.total_nodes)

    @property
    def duplicate_ratio_no_turn(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return 1 - (self.unique_states_no_turn / self.total_nodes)

    @property
    def compression_with_turn(self) -> float:
        """How much smaller the graph would be vs tree (with turn)."""
        if self.total_nodes == 0:
            return 0.0
        return self.unique_states_with_turn / self.total_nodes

    @property
    def compression_no_turn(self) -> float:
        """How much smaller the graph would be vs tree (without turn)."""
        if self.total_nodes == 0:
            return 0.0
        return self.unique_states_no_turn / self.total_nodes


@dataclass
class GameTranspositionStats:
    """Per-game raw data for JSON output."""

    seed: int
    num_searches: int
    total_nodes: int
    unique_states_with_turn: int
    unique_states_no_turn: int
    depth_breakdown: dict[int, dict[str, int]] = field(default_factory=dict)


@dataclass
class SimCountResult:
    """One row in the running summary table."""

    sims: int
    duplicate_ratio_with_turn: float
    duplicate_ratio_no_turn: float
    avg_nodes_per_search: float


def hash_state_with_turn(game: PyRat) -> tuple:
    """Hash game state including turn number."""
    return (
        (game.player1_position.x, game.player1_position.y),
        (game.player2_position.x, game.player2_position.y),
        frozenset((c.x, c.y) for c in game.cheese_positions()),
        game.player1_mud_turns,
        game.player2_mud_turns,
        game.turn,
    )


def hash_state_no_turn(game: PyRat) -> tuple:
    """Hash game state without turn number (position-only transpositions)."""
    return (
        (game.player1_position.x, game.player1_position.y),
        (game.player2_position.x, game.player2_position.y),
        frozenset((c.x, c.y) for c in game.cheese_positions()),
        game.player1_mud_turns,
        game.player2_mud_turns,
    )


def count_transpositions(tree: MCTSTree) -> TranspositionStats:
    """Traverse tree, hash each state, count transpositions by depth."""
    stats = TranspositionStats()
    seen_with_turn: dict[tuple, set[int]] = defaultdict(set)  # hash -> set of depths
    seen_no_turn: dict[tuple, set[int]] = defaultdict(set)

    def traverse(node: MCTSNode) -> None:
        # Navigate simulator to this node
        tree._navigate_to(node)

        # Hash state both ways
        h_with = hash_state_with_turn(tree.game)
        h_no = hash_state_no_turn(tree.game)
        depth = node.depth

        stats.total_nodes += 1
        stats.depth_counts[depth] += 1

        # Track unique states at each depth
        seen_with_turn[h_with].add(depth)
        seen_no_turn[h_no].add(depth)

        # Recurse to children
        for child in node.children.values():
            traverse(child)

    traverse(tree.root)

    # Count unique states
    stats.unique_states_with_turn = len(seen_with_turn)
    stats.unique_states_no_turn = len(seen_no_turn)

    # Count unique states per depth (a state seen at multiple depths counts once per depth)
    for depths in seen_with_turn.values():
        for d in depths:
            stats.depth_unique_with_turn[d] += 1
    for depths in seen_no_turn.values():
        for d in depths:
            stats.depth_unique_no_turn[d] += 1

    return stats


def run_game_with_measurement(
    game_config: GameConfig,
    mcts_config: DecoupledPUCTConfig,
    seed: int,
) -> GameTranspositionStats:
    """Play one game, measuring transpositions at each move."""
    game = create_game(game_config, seed)
    aggregate = AggregateStats()

    while not is_terminal(game):
        tree = build_tree(game, gamma=mcts_config.gamma)
        search = DecoupledPUCTSearch(tree, mcts_config)
        result = search.search()

        # Measure transpositions in this search tree
        stats = count_transpositions(tree)
        aggregate.add(stats)

        # Play the move
        a1 = select_action_from_strategy(result.policy_p1)
        a2 = select_action_from_strategy(result.policy_p2)
        game.make_move(Direction(a1), Direction(a2))

    # Build per-depth breakdown
    all_depths = sorted(
        set(aggregate.depth_counts)
        | set(aggregate.depth_unique_with_turn)
        | set(aggregate.depth_unique_no_turn)
    )
    depth_breakdown: dict[int, dict[str, int]] = {}
    for d in all_depths:
        depth_breakdown[d] = {
            "nodes": aggregate.depth_counts.get(d, 0),
            "unique_with_turn": aggregate.depth_unique_with_turn.get(d, 0),
            "unique_no_turn": aggregate.depth_unique_no_turn.get(d, 0),
        }

    return GameTranspositionStats(
        seed=seed,
        num_searches=aggregate.num_searches,
        total_nodes=aggregate.total_nodes,
        unique_states_with_turn=aggregate.unique_states_with_turn,
        unique_states_no_turn=aggregate.unique_states_no_turn,
        depth_breakdown=depth_breakdown,
    )


def print_report(stats: AggregateStats) -> None:
    """Print detailed transposition report."""
    print("\n" + "=" * 70)
    print("TRANSPOSITION ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nSearches analyzed: {stats.num_searches}")
    print(f"Total tree nodes:  {stats.total_nodes:,}")

    print("\n--- With Turn in Hash (time-aware transpositions) ---")
    print(f"Unique states:     {stats.unique_states_with_turn:,}")
    print(f"Duplicate nodes:   {stats.total_nodes - stats.unique_states_with_turn:,}")
    print(f"Duplicate ratio:   {stats.duplicate_ratio_with_turn:.2%}")
    print(f"MCGS compression:  {stats.compression_with_turn:.2%} of tree size")

    print("\n--- Without Turn in Hash (position-only transpositions) ---")
    print(f"Unique states:     {stats.unique_states_no_turn:,}")
    print(f"Duplicate nodes:   {stats.total_nodes - stats.unique_states_no_turn:,}")
    print(f"Duplicate ratio:   {stats.duplicate_ratio_no_turn:.2%}")
    print(f"MCGS compression:  {stats.compression_no_turn:.2%} of tree size")

    # Depth distribution
    max_depth = max(stats.depth_counts.keys()) if stats.depth_counts else 0
    if max_depth > 0:
        print("\n--- Depth Distribution ---")
        header = (
            f"{'Depth':<6} {'Nodes':<10} {'Unique(+t)':<12} "
            f"{'Dup%(+t)':<10} {'Unique(-t)':<12} {'Dup%(-t)':<10}"
        )
        print(header)
        print("-" * 70)

        for depth in range(max_depth + 1):
            nodes = stats.depth_counts.get(depth, 0)
            unique_with = stats.depth_unique_with_turn.get(depth, 0)
            unique_no = stats.depth_unique_no_turn.get(depth, 0)

            dup_pct_with = (1 - unique_with / nodes) * 100 if nodes > 0 else 0
            dup_pct_no = (1 - unique_no / nodes) * 100 if nodes > 0 else 0

            row = (
                f"{depth:<6} {nodes:<10,} {unique_with:<12,} "
                f"{dup_pct_with:<10.1f} {unique_no:<12,} {dup_pct_no:<10.1f}"
            )
            print(row)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    dup_with = stats.duplicate_ratio_with_turn
    dup_no = stats.duplicate_ratio_no_turn

    if dup_with < 0.05:
        print("With turn:    Very few transpositions (<5%). MCGS would provide minimal benefit.")
    elif dup_with < 0.15:
        print("With turn:    Moderate transpositions (5-15%). MCGS could provide some speedup.")
    else:
        print(
            f"With turn:    Significant transpositions ({dup_with:.0%}). "
            "MCGS could provide meaningful benefit."
        )

    if dup_no < 0.05:
        print("Without turn: Very few position-only transpositions. Paths rarely converge.")
    elif dup_no < 0.15:
        print("Without turn: Some position convergence. Game has moderate path diversity.")
    else:
        print(
            f"Without turn: High position convergence ({dup_no:.0%}). "
            "Many paths lead to same positions."
        )


def run_sim_count(
    game_config: GameConfig,
    sims: int,
    c_puct: float,
    num_games: int,
) -> tuple[AggregateStats, list[GameTranspositionStats]]:
    """Run all games for one sim count and return aggregate + per-game stats."""
    mcts_config = DecoupledPUCTConfig(simulations=sims, c_puct=c_puct)
    total_stats = AggregateStats()
    game_stats_list: list[GameTranspositionStats] = []

    # Print ~4 progress updates per sim count
    report_interval = max(1, num_games // 4)

    for i in range(num_games):
        game_stats = run_game_with_measurement(game_config, mcts_config, seed=i)
        game_stats_list.append(game_stats)

        # Accumulate into aggregate
        total_stats.total_nodes += game_stats.total_nodes
        total_stats.unique_states_with_turn += game_stats.unique_states_with_turn
        total_stats.unique_states_no_turn += game_stats.unique_states_no_turn
        total_stats.num_searches += game_stats.num_searches

        for depth, breakdown in game_stats.depth_breakdown.items():
            total_stats.depth_counts[depth] += breakdown["nodes"]
            total_stats.depth_unique_with_turn[depth] += breakdown["unique_with_turn"]
            total_stats.depth_unique_no_turn[depth] += breakdown["unique_no_turn"]

        if (i + 1) % report_interval == 0 or i == num_games - 1:
            dup_with = total_stats.duplicate_ratio_with_turn
            dup_no = total_stats.duplicate_ratio_no_turn
            print(
                f"  [{i + 1}/{num_games}] "
                f"nodes={total_stats.total_nodes:,} | "
                f"dup%(+turn)={dup_with:.1%} | "
                f"dup%(-turn)={dup_no:.1%}",
                flush=True,
            )

    return total_stats, game_stats_list


def save_json_results(
    output_dir: Path,
    game_config: GameConfig,
    sims: int,
    c_puct: float,
    aggregate: AggregateStats,
    game_stats: list[GameTranspositionStats],
) -> Path:
    """Write one JSON file per sim count with config, per-game data, and aggregate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sims_{sims}.json"

    data: dict[str, Any] = {
        "config": {
            "sims": sims,
            "c_puct": c_puct,
            "width": game_config.width,
            "height": game_config.height,
            "cheese_count": game_config.cheese_count,
            "max_turns": game_config.max_turns,
            "wall_density": game_config.wall_density,
            "mud_density": game_config.mud_density,
            "num_games": len(game_stats),
        },
        "aggregate": {
            "total_nodes": aggregate.total_nodes,
            "unique_states_with_turn": aggregate.unique_states_with_turn,
            "unique_states_no_turn": aggregate.unique_states_no_turn,
            "num_searches": aggregate.num_searches,
            "duplicate_ratio_with_turn": aggregate.duplicate_ratio_with_turn,
            "duplicate_ratio_no_turn": aggregate.duplicate_ratio_no_turn,
        },
        "games": [
            {
                "seed": gs.seed,
                "num_searches": gs.num_searches,
                "total_nodes": gs.total_nodes,
                "unique_states_with_turn": gs.unique_states_with_turn,
                "unique_states_no_turn": gs.unique_states_no_turn,
                "depth_breakdown": {str(d): v for d, v in sorted(gs.depth_breakdown.items())},
            }
            for gs in game_stats
        ],
    }

    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def print_running_summary(results: list[SimCountResult]) -> None:
    """Print compact table of all completed sim counts so far."""
    print("\n--- Progress ---")
    print(f"{'Sims':<8} {'Dup% (+turn)':<16} {'Dup% (-turn)':<16} {'Avg Nodes/Search':<18}")
    for r in results:
        print(
            f"{r.sims:<8} {r.duplicate_ratio_with_turn:<16.1%}"
            f" {r.duplicate_ratio_no_turn:<16.1%}"
            f" {r.avg_nodes_per_search:<18.1f}"
        )
    print()


def print_final_summary(results: list[SimCountResult]) -> None:
    """Print final summary table."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Sims':<8} {'Dup% (+turn)':<16} {'Dup% (-turn)':<16} {'Avg Nodes/Search':<18}")
    print("-" * 58)
    for r in results:
        print(
            f"{r.sims:<8} {r.duplicate_ratio_with_turn:<16.1%}"
            f" {r.duplicate_ratio_no_turn:<16.1%}"
            f" {r.avg_nodes_per_search:<18.1f}"
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure transposition frequency in MCTS")
    parser.add_argument("--games", type=int, default=20, help="Number of games to analyze")
    parser.add_argument(
        "--sims",
        type=int,
        nargs="*",
        help="Sim counts to sweep (default: 50 200 554 1000 2000)",
    )
    parser.add_argument("--width", type=int, default=7, help="Game width")
    parser.add_argument("--height", type=int, default=7, help="Game height")
    parser.add_argument("--cheese", type=int, default=10, help="Number of cheese")
    parser.add_argument("--max-turns", type=int, default=50, help="Max turns per game")
    parser.add_argument("--c-puct", type=float, default=8.34, help="PUCT exploration constant")
    parser.add_argument("--wall-density", type=float, default=0.0, help="Wall density (0.0–1.0)")
    parser.add_argument("--mud-density", type=float, default=0.0, help="Mud density (0.0–1.0)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/transposition_measurement",
        help="Directory for JSON output",
    )
    args = parser.parse_args()

    sim_counts: list[int] = args.sims if args.sims else [50, 200, 554, 1000, 2000]

    game_config = GameConfig(
        width=args.width,
        height=args.height,
        cheese_count=args.cheese,
        max_turns=args.max_turns,
        wall_density=args.wall_density,
        mud_density=args.mud_density,
    )

    output_dir = Path(args.output_dir)

    terrain = []
    if args.wall_density > 0:
        terrain.append(f"walls={args.wall_density}")
    if args.mud_density > 0:
        terrain.append(f"mud={args.mud_density}")
    terrain_str = ", ".join(terrain) if terrain else "open"

    print(f"Transposition sweep: sims={sim_counts}")
    print(
        f"  Game: {args.width}x{args.height}, {args.cheese} cheese, "
        f"{args.max_turns} max turns, terrain={terrain_str}"
    )
    print(f"  Games per sim count: {args.games}, c_puct={args.c_puct}")
    print(f"  Output: {output_dir}")
    print()

    results: list[SimCountResult] = []

    for sims in sim_counts:
        print(f"=== sims={sims} ===")
        aggregate, game_stats = run_sim_count(game_config, sims, args.c_puct, args.games)

        print_report(aggregate)

        path = save_json_results(output_dir, game_config, sims, args.c_puct, aggregate, game_stats)
        print(f"Saved: {path}")

        avg_nodes = aggregate.total_nodes / aggregate.num_searches if aggregate.num_searches else 0
        results.append(
            SimCountResult(
                sims=sims,
                duplicate_ratio_with_turn=aggregate.duplicate_ratio_with_turn,
                duplicate_ratio_no_turn=aggregate.duplicate_ratio_no_turn,
                avg_nodes_per_search=avg_nodes,
            )
        )

        if len(results) < len(sim_counts):
            print_running_summary(results)

    print_final_summary(results)


if __name__ == "__main__":
    main()
