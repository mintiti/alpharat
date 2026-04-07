"""Measure search horizon: sim budget needed to detect cheese at distance d.

Places cheese randomly at manhattan distance d on a large open maze.
Runs search at increasing sim budgets. Tracks Q-value of the greedy action.
Reports the sim budget where Q crosses detection thresholds.

    uv run python scripts/search_horizon.py
    uv run python scripts/search_horizon.py --algorithms mcts
    uv run python scripts/search_horizon.py --distances 1 2 3 5 8 --trials 50 --max-sims 20000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from pyrat_engine.core import GameBuilder
from pyrat_engine.core.types import Coordinates


def cells_at_distance(cx: int, cy: int, d: int, grid: int) -> list[tuple[int, int]]:
    """All grid cells at exactly manhattan distance d from (cx, cy), within bounds."""
    cells = set()
    for dx in range(-d, d + 1):
        dy_abs = d - abs(dx)
        for dy in (dy_abs, -dy_abs):
            x, y = cx + dx, cy + dy
            if 0 <= x < grid and 0 <= y < grid:
                cells.add((x, y))
    return list(cells)


def greedy_actions(cx: int, cy: int, tx: int, ty: int) -> list[int]:
    """Actions that reduce manhattan distance from (cx,cy) toward (tx,ty)."""
    actions = []
    dx, dy = tx - cx, ty - cy
    if dy > 0:
        actions.append(0)  # UP
    if dx > 0:
        actions.append(1)  # RIGHT
    if dy < 0:
        actions.append(2)  # DOWN
    if dx < 0:
        actions.append(3)  # LEFT
    return actions


def make_game(grid: int, cx: int, cy: int, cheese_x: int, cheese_y: int) -> Any:
    return (
        GameBuilder(grid, grid)
        .with_max_turns(grid * 3)
        .with_open_maze()
        .with_custom_positions(Coordinates(cx, cy), Coordinates(cx, cy))
        .with_custom_cheese([Coordinates(cheese_x, cheese_y)])
        .build()
        .create(seed=42)
    )


def sim_budgets(max_sims: int) -> list[int]:
    """~20 sim budgets on a log scale up to max_sims."""
    raw = np.geomspace(10, max_sims, num=22)
    budgets = sorted({max(10, int(round(x / 5) * 5)) for x in raw})
    if budgets[-1] != max_sims:
        budgets.append(max_sims)
    return budgets


def run_horizon(
    distances: list[int],
    n_trials: int,
    max_sims: int,
    grid: int,
    seed: int,
    algorithms: list[str],
) -> list[dict[str, Any]]:
    """Run the full measurement. Returns list of result rows."""
    from alpharat_mcts import rust_mcts_search

    search_fns: list[tuple[str, Callable[..., Any]]] = []
    if "mcts" in algorithms:
        search_fns.append(("MCTS", rust_mcts_search))
    if "mcgs" in algorithms:
        try:
            from alpharat_mcgs import rust_mcgs_search
        except ImportError:
            print("  WARNING: alpharat_mcgs not available, skipping MCGS", file=sys.stderr)
        else:
            search_fns.append(("MCGS", rust_mcgs_search))

    if not search_fns:
        print("  No algorithms available.", file=sys.stderr)
        return []

    cx = cy = grid // 2
    budgets = sim_budgets(max_sims)
    rng = random.Random(seed)

    # Pre-generate cheese placements
    trials = []
    for d in distances:
        candidates = cells_at_distance(cx, cy, d, grid)
        for t in range(n_trials):
            cheese_x, cheese_y = rng.choice(candidates)
            trials.append((d, t, cheese_x, cheese_y, greedy_actions(cx, cy, cheese_x, cheese_y)))

    total = len(trials) * len(budgets) * len(search_fns)
    results: list[dict[str, Any]] = []
    done = 0
    total_sims_by_algo: dict[str, int] = defaultdict(int)
    wall_time_by_algo: dict[str, float] = defaultdict(float)
    t0 = time.time()

    for d, trial_idx, cheese_x, cheese_y, correct in trials:
        game = make_game(grid, cx, cy, cheese_x, cheese_y)

        for sims in budgets:
            for algo_name, search_fn in search_fns:
                t_search = time.perf_counter()
                r = search_fn(
                    game,
                    simulations=sims,
                    batch_size=8,
                    c_puct=1.5,
                    fpu_reduction=0.2,
                    force_k=2.0,
                    seed=trial_idx,
                )
                dt = time.perf_counter() - t_search
                wall_time_by_algo[algo_name] += dt
                total_sims_by_algo[algo_name] += r.total_visits

                q = np.asarray(r.q_values_p1)
                q_correct = max(float(q[a]) for a in correct)

                results.append(
                    {
                        "algorithm": algo_name,
                        "distance": d,
                        "trial": trial_idx,
                        "sims": sims,
                        "q_correct": round(q_correct, 6),
                    }
                )

                done += 1
                if done % max(1, total // 20) == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(
                        f"\r  {done / total * 100:5.1f}%  ({done}/{total})  "
                        f"{elapsed:.0f}s elapsed  ~{eta:.0f}s left",
                        end="",
                        flush=True,
                    )

    wall = time.time() - t0
    print(f"\r  Done. {total} searches in {wall:.1f}s" + " " * 30)

    print("\n  Throughput:")
    for algo in sorted(total_sims_by_algo):
        sims_total = total_sims_by_algo[algo]
        dt = wall_time_by_algo[algo]
        print(f"    {algo}: {sims_total:,} sims in {dt:.2f}s = {sims_total / dt:,.0f} sims/s")

    return results


def compute_horizons(
    results: list[dict[str, Any]],
    thresholds: list[float],
    percentiles: list[int],
    max_sims: int,
) -> dict[str, Any]:
    """For each (algo, distance, threshold, percentile), find the sim budget needed."""
    # Group by (algo, distance, trial) -> sorted [(sims, q)]
    grouped: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in results:
        grouped[(r["algorithm"], r["distance"], r["trial"])].append((r["sims"], r["q_correct"]))

    # For each trial, find first sim count crossing each threshold
    crossing: dict[tuple, list[int]] = defaultdict(list)  # (algo, d, thresh) -> [sims...]

    algos = sorted({r["algorithm"] for r in results})
    distances = sorted({r["distance"] for r in results})
    n_trials = max(r["trial"] for r in results) + 1
    overflow = max_sims * 2

    for algo in algos:
        for d in distances:
            for t in range(n_trials):
                traj = sorted(grouped[(algo, d, t)])
                for thresh in thresholds:
                    first = overflow
                    for sims, q in traj:
                        if q >= thresh:
                            first = sims
                            break
                    crossing[(algo, d, thresh)].append(first)

    # Compute percentiles
    horizons: dict[str, Any] = {}
    for algo in algos:
        horizons[algo] = {}
        for d in distances:
            horizons[algo][d] = {}
            for thresh in thresholds:
                vals = sorted(crossing[(algo, d, thresh)])
                horizons[algo][d][thresh] = {}
                for p in percentiles:
                    idx = min(int(len(vals) * p / 100), len(vals) - 1)
                    horizons[algo][d][thresh][p] = vals[idx]

    return horizons


def print_tables(
    horizons: dict[str, Any], thresholds: list[float], percentiles: list[int], max_sims: int
) -> None:
    algos = sorted(horizons.keys())
    distances = sorted(next(iter(horizons.values())).keys())
    cw = 8  # column width

    for p in percentiles:
        print(f"\n{'=' * 78}")
        print(f"  p{p}: sim budget where {p}% of trials have Q_correct >= threshold")
        print(f"{'=' * 78}")

        for algo in algos:
            hdr = " | ".join(f"{'Q>' + str(t):>{cw}}" for t in thresholds)
            print(f"\n  {algo}")
            print(f"  {'d':>4} | {hdr}")
            print(f"  {'----':>4}-+-" + "-+-".join("-" * cw for _ in thresholds))

            for d in distances:
                cells = []
                for t in thresholds:
                    v = horizons[algo][d][t][p]
                    cells.append(f"{'>' + str(max_sims):>{cw}}" if v > max_sims else f"{v:>{cw}}")
                print(f"  {d:>4} | " + " | ".join(cells))

        # Ratio table
        if len(algos) == 2:
            a, b = algos
            print(f"\n  Ratio {a}/{b} (p{p}) — below 1.0 means {a} detects sooner")
            hdr = " | ".join(f"{'Q>' + str(t):>{cw}}" for t in thresholds)
            print(f"  {'d':>4} | {hdr}")
            print(f"  {'----':>4}-+-" + "-+-".join("-" * cw for _ in thresholds))

            for d in distances:
                cells = []
                for t in thresholds:
                    va = horizons[a][d][t][p]
                    vb = horizons[b][d][t][p]
                    if va > max_sims or vb > max_sims or vb == 0:
                        cells.append(f"{'---':>{cw}}")
                    else:
                        cells.append(f"{va / vb:>{cw}.2f}")
                print(f"  {d:>4} | " + " | ".join(cells))


def save_csv(results: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  Raw data: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure search horizon")
    parser.add_argument("--distances", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-sims", type=int, default=10000)
    parser.add_argument("--grid", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default="search_horizon.csv")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["mcts", "mcgs"],
        choices=["mcts", "mcgs"],
        help="Which search algorithms to compare (default: both)",
    )
    args = parser.parse_args()

    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
    percentiles = [50, 95]

    print("Search Horizon")
    print(
        f"  {args.grid}x{args.grid} open maze, {args.trials} trials/distance,"
        f" up to {args.max_sims} sims"
    )
    print(f"  Distances: {args.distances}")
    print(f"  Algorithms: {args.algorithms}")
    print()

    results = run_horizon(
        distances=args.distances,
        n_trials=args.trials,
        max_sims=args.max_sims,
        grid=args.grid,
        seed=args.seed,
        algorithms=args.algorithms,
    )

    horizons = compute_horizons(results, thresholds, percentiles, args.max_sims)
    print_tables(horizons, thresholds, percentiles, args.max_sims)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
