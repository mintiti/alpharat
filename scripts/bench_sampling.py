"""Benchmark: full Rust sampling vs Python/Rust boundary sampling.

Runs both pipelines with equivalent configs, compares throughput and output.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from alpharat.config.game import GameConfig
from alpharat.data.loader import load_game_bundle
from alpharat.mcts.config import RustMCTSConfig

if TYPE_CHECKING:
    from alpharat.data.types import GameData

# --- Config ---

GAME = GameConfig(
    width=7,
    height=7,
    max_turns=50,
    cheese_count=10,
    wall_density=0.0,
    mud_density=0.0,
    symmetric=True,
)

MCTS = RustMCTSConfig(
    simulations=1897,
    c_puct=0.512,
    force_k=0.103,
    fpu_reduction=0.459,
    batch_size=16,
    noise_epsilon=0.25,
    noise_concentration=10.83,
)

CHECKPOINT = "experiments/runs/scalar_baseline_7x7_iter0/checkpoints/best_model.pt"
NUM_GAMES = 200


# --- Helpers ---


def load_all_games(batch_dir: Path) -> list[GameData]:
    games_dir = batch_dir / "games"
    all_games: list[GameData] = []
    for bundle_path in sorted(games_dir.glob("*.npz")):
        all_games.extend(load_game_bundle(bundle_path))
    return all_games


def summarize_games(games: list[GameData], label: str) -> dict[str, float]:
    """Print and return summary stats from loaded game data."""
    n_games = len(games)
    n_positions = sum(len(g.positions) for g in games)
    results = [g.result for g in games]
    p1_wins = results.count(1)
    p2_wins = results.count(2)
    draws = results.count(0)
    avg_turns = n_positions / n_games if n_games else 0

    all_p1_sums = []
    all_p2_sums = []
    all_actions = []
    for g in games:
        for p in g.positions:
            all_p1_sums.append(float(p.policy_p1.sum()))
            all_p2_sums.append(float(p.policy_p2.sum()))
            all_actions.append(p.action_p1)
            all_actions.append(p.action_p2)

    p1_sums = np.array(all_p1_sums)
    p2_sums = np.array(all_p2_sums)

    cheese_collected = sum(g.final_p1_score + g.final_p2_score for g in games)
    cheese_available = n_games * GAME.cheese_count

    print(f"\n  [{label}] Loaded game data:")
    print(f"    Games: {n_games}, Positions: {n_positions}, Avg turns: {avg_turns:.1f}")
    print(f"    W/D/L: {p1_wins}/{draws}/{p2_wins}")
    pct = cheese_collected / cheese_available
    print(f"    Cheese: {cheese_collected:.0f}/{cheese_available} ({pct:.0%})")
    p1_lo, p1_hi = p1_sums.min(), p1_sums.max()
    p2_lo, p2_hi = p2_sums.min(), p2_sums.max()
    print(f"    Policy sums — P1: [{p1_lo:.4f}, {p1_hi:.4f}]  P2: [{p2_lo:.4f}, {p2_hi:.4f}]")
    print(f"    Actions range: [{min(all_actions)}, {max(all_actions)}]")

    shapes_ok = all(g.maze.shape == (GAME.height, GAME.width, 4) for g in games)
    outcomes_ok = all(
        g.cheese_outcomes is not None and g.cheese_outcomes.shape == (GAME.height, GAME.width)
        for g in games
    )
    print(f"    Shapes OK: maze={shapes_ok}, outcomes={outcomes_ok}")

    return {
        "n_games": n_games,
        "n_positions": n_positions,
        "avg_turns": avg_turns,
        "cheese_util": cheese_collected / cheese_available if cheese_available else 0,
        "draw_rate": draws / n_games if n_games else 0,
    }


# --- Benchmarks ---


def bench_rust(tmp_dir: Path) -> dict[str, float]:
    """Full Rust pipeline: run_rust_sampling()."""
    from alpharat.data.rust_sampling import run_rust_sampling

    print(f"\n{'=' * 60}")
    print("FULL RUST PIPELINE (run_rust_sampling)")
    print(f"{'=' * 60}")

    exp_dir = tmp_dir / "exp_rust"
    t0 = time.perf_counter()
    batch_dir, metrics = run_rust_sampling(
        game=GAME,
        mcts=MCTS,
        num_games=NUM_GAMES,
        group="bench_rust",
        num_threads=8,
        checkpoint=CHECKPOINT,
        experiments_dir=exp_dir,
        verbose=False,
    )
    wall_time = time.perf_counter() - t0

    print(f"\n  Wall time:  {wall_time:.2f}s")
    print(f"  Sims/s:     {metrics.simulations_per_second:,.0f}")
    print(f"  Pos/s:      {metrics.positions_per_second:,.0f}")
    print(f"  Games/s:    {metrics.games_per_second:,.1f}")

    games = load_all_games(batch_dir)
    summarize_games(games, "Rust")

    return {
        "wall_time": wall_time,
        "sims_per_s": metrics.simulations_per_second,
        "pos_per_s": metrics.positions_per_second,
        "games_per_s": metrics.games_per_second,
    }


def bench_python_rust(tmp_dir: Path) -> dict[str, float]:
    """Python/Rust boundary: run_sampling() with RustMCTSConfig."""
    from alpharat.data.sampling import SamplingConfig, SamplingParams, run_sampling

    print(f"\n{'=' * 60}")
    print("PYTHON/RUST BOUNDARY (run_sampling + RustMCTSConfig)")
    print(f"{'=' * 60}")

    exp_dir = tmp_dir / "exp_python"
    config = SamplingConfig(
        mcts=MCTS,
        game=GAME,
        sampling=SamplingParams(num_games=NUM_GAMES, workers=8),
        group="bench_python_rust",
        experiments_dir=str(exp_dir),
        checkpoint=CHECKPOINT,
    )

    t0 = time.perf_counter()
    batch_dir, metrics = run_sampling(config, verbose=False)
    wall_time = time.perf_counter() - t0

    print(f"\n  Wall time:  {wall_time:.2f}s")
    print(f"  Sims/s:     {metrics.simulations_per_second:,.0f}")
    print(f"  Pos/s:      {metrics.positions_per_second:,.0f}")
    print(f"  Games/s:    {metrics.games_per_second:,.1f}")

    games = load_all_games(batch_dir)
    summarize_games(games, "Python/Rust")

    return {
        "wall_time": wall_time,
        "sims_per_s": metrics.simulations_per_second,
        "pos_per_s": metrics.positions_per_second,
        "games_per_s": metrics.games_per_second,
    }


def main() -> None:
    print("Sampling Pipeline Benchmark")
    g = GAME
    print(f"  Game: {g.width}x{g.height}, {g.cheese_count} cheese, {g.max_turns} max turns")
    print(f"  MCTS: {MCTS.simulations} sims, batch={MCTS.batch_size}")
    print(f"  Games: {NUM_GAMES}")
    print(f"  Checkpoint: {CHECKPOINT}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rust = bench_rust(tmp_dir)
        python = bench_python_rust(tmp_dir)

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<15} {'Rust':>12} {'Python/Rust':>12} {'Speedup':>10}")
    print(f"  {'-' * 49}")
    for key, label in [
        ("wall_time", "Wall time (s)"),
        ("sims_per_s", "Sims/s"),
        ("pos_per_s", "Pos/s"),
        ("games_per_s", "Games/s"),
    ]:
        r, p = rust[key], python[key]
        if key == "wall_time":
            speedup = p / r if r > 0 else 0
            print(f"  {label:<15} {r:>12.2f} {p:>12.2f} {speedup:>9.1f}x")
        else:
            speedup = r / p if p > 0 else 0
            print(f"  {label:<15} {r:>12,.0f} {p:>12,.0f} {speedup:>9.1f}x")


if __name__ == "__main__":
    main()
