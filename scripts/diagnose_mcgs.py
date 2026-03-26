"""MCGS vs MCTS diagnostic: compare policies, Q-values, and visit distributions.

Sets up controlled positions (open maze, known cheese locations) and runs both
searches at multiple sim budgets. Reveals whether MCGS's transposition sharing
produces sharper policies or more accurate value estimates.

Run with:
    uv run python scripts/diagnose_mcgs.py
    uv run python scripts/diagnose_mcgs.py --sims 100 500 2000 --grid 21
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyrat_engine.core import GameBuilder
from pyrat_engine.core.types import Coordinates

ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]


@dataclass
class Scenario:
    """A controlled position to diagnose search behavior."""

    name: str
    grid_size: int
    p1_pos: tuple[int, int]
    p2_pos: tuple[int, int]
    cheese: list[tuple[int, int]]
    max_turns: int


@dataclass
class SearchOutput:
    """Captured search output for one searcher on one scenario."""

    searcher_name: str
    policy_p1: np.ndarray
    q_values_p1: np.ndarray
    value_p1: float
    visit_counts_p1: np.ndarray
    total_visits: int
    nn_evals: int
    collisions: int
    terminals: int


def build_scenarios(grid: int) -> list[Scenario]:
    """Build diagnostic scenarios on a grid x grid open maze."""
    cx, cy = grid // 2, grid // 2
    max_turns = grid * 3

    scenarios = []

    # 1-cheese symmetric: single cheese at increasing distance to the right
    for d in [1, 2, 3, 5, 8]:
        if cx + d < grid:
            scenarios.append(
                Scenario(
                    name=f"1cheese_right_d{d}",
                    grid_size=grid,
                    p1_pos=(cx, cy),
                    p2_pos=(cx, cy),
                    cheese=[(cx + d, cy)],
                    max_turns=max_turns,
                )
            )

    # 2-cheese symmetric: opposite directions at equal distance
    for d in [2, 5]:
        if cx - d >= 0 and cx + d < grid:
            scenarios.append(
                Scenario(
                    name=f"2cheese_symmetric_d{d}",
                    grid_size=grid,
                    p1_pos=(cx, cy),
                    p2_pos=(cx, cy),
                    cheese=[(cx + d, cy), (cx - d, cy)],
                    max_turns=max_turns,
                )
            )

    # 2-cheese asymmetric: one close, one far
    for d_close, d_far in [(2, 8), (1, 5)]:
        if cx + d_far < grid:
            scenarios.append(
                Scenario(
                    name=f"2cheese_asym_d{d_close}_d{d_far}",
                    grid_size=grid,
                    p1_pos=(cx, cy),
                    p2_pos=(cx, cy),
                    cheese=[(cx + d_close, cy), (cx + d_far, cy)],
                    max_turns=max_turns,
                )
            )

    return scenarios


def make_game(scenario: Scenario) -> Any:
    """Build a PyRat game from a scenario."""
    g = scenario.grid_size
    return (
        GameBuilder(g, g)
        .with_max_turns(scenario.max_turns)
        .with_open_maze()
        .with_custom_positions(
            Coordinates(*scenario.p1_pos),
            Coordinates(*scenario.p2_pos),
        )
        .with_custom_cheese([Coordinates(*c) for c in scenario.cheese])
        .build()
        .create(seed=42)
    )


def run_search(game: Any, searcher_name: str, searcher: Any) -> SearchOutput:
    """Run a searcher and capture the output."""
    result = searcher.search(game)
    # Pull raw Rust stats from the underlying rust_result if available
    return SearchOutput(
        searcher_name=searcher_name,
        policy_p1=result.policy_p1,
        q_values_p1=result.q_values_p1,
        value_p1=result.value_p1,
        visit_counts_p1=result.visit_counts_p1,
        total_visits=result.total_visits,
        nn_evals=0,
        collisions=0,
        terminals=0,
    )


def format_array(arr: np.ndarray, fmt: str = ".3f") -> str:
    """Format a 5-element array with action labels."""
    parts = [f"{ACTION_NAMES[i]}={arr[i]:{fmt}}" for i in range(5)]
    return "  ".join(parts)


def print_comparison(scenario: Scenario, outputs: list[SearchOutput]) -> None:
    """Print side-by-side comparison for a scenario."""
    cheese_str = ", ".join(f"({c[0]},{c[1]})" for c in scenario.cheese)
    print(f"\n{'=' * 80}")
    print(f"Scenario: {scenario.name}")
    print(f"  Grid: {scenario.grid_size}x{scenario.grid_size} open")
    print(
        f"  P1: ({scenario.p1_pos[0]},{scenario.p1_pos[1]})  "
        f"P2: ({scenario.p2_pos[0]},{scenario.p2_pos[1]})"
    )
    print(f"  Cheese: {cheese_str}")
    print(f"{'=' * 80}")

    for out in outputs:
        best_action = ACTION_NAMES[int(np.argmax(out.policy_p1))]
        policy_max = float(np.max(out.policy_p1))
        print(
            f"\n  [{out.searcher_name}]  visits={out.total_visits}  "
            f"value_p1={out.value_p1:.4f}  best={best_action} ({policy_max:.3f})"
        )
        print(f"    policy:  {format_array(out.policy_p1)}")
        print(f"    Q-vals:  {format_array(out.q_values_p1, '.4f')}")
        print(f"    visits:  {format_array(out.visit_counts_p1, '.0f')}")

    # Diff summary
    if len(outputs) == 2:
        a, b = outputs[0], outputs[1]
        q_diff = b.q_values_p1 - a.q_values_p1
        policy_diff = b.policy_p1 - a.policy_p1
        print(f"\n  [delta: {b.searcher_name} - {a.searcher_name}]")
        print(f"    value:   {b.value_p1 - a.value_p1:+.4f}")
        print(f"    Q-vals:  {format_array(q_diff, '+.4f')}")
        print(f"    policy:  {format_array(policy_diff, '+.3f')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCGS vs MCTS diagnostic")
    parser.add_argument("--sims", type=int, nargs="+", default=[100, 500, 2000, 5000])
    parser.add_argument("--grid", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from alpharat.mcts.config import RustMCGSConfig, RustMCTSConfig

    scenarios = build_scenarios(args.grid)

    for sims in args.sims:
        mcts_cfg = RustMCTSConfig(simulations=sims).for_evaluation()
        mcgs_cfg = RustMCGSConfig(simulations=sims).for_evaluation()
        mcts_searcher = mcts_cfg.build_searcher()
        mcgs_searcher = mcgs_cfg.build_searcher()

        print(f"\n{'#' * 80}")
        print(f"# Simulations: {sims}")
        print(f"{'#' * 80}")

        for scenario in scenarios:
            game = make_game(scenario)
            mcts_out = run_search(game, f"MCTS({sims})", mcts_searcher)
            mcgs_out = run_search(game, f"MCGS({sims})", mcgs_searcher)
            print_comparison(scenario, [mcts_out, mcgs_out])


if __name__ == "__main__":
    main()
