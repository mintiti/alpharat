#!/usr/bin/env python3
"""PUCT parameter sweep using Optuna.

Usage:
    uv run python scripts/optuna_sweep.py
    uv run python scripts/optuna_sweep.py --n-jobs 4
    uv run python scripts/optuna_sweep.py --study-name my_sweep
    uv run python scripts/optuna_sweep.py --seed-from results/puct_vs_greedy_5x5.csv --seed-top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd

from alpharat.ai import GreedyAgent, MCTSAgent
from alpharat.eval.game import play_game
from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig

# Game parameters - 7x7 open maze
WIDTH, HEIGHT = 7, 7
CHEESE_COUNT, MAX_TURNS = 10, 50
WALL_DENSITY, MUD_DENSITY = 0.0, 0.0
GAMES_PER_CONFIG = 100

# Known-good configs to seed the search
SEED_CONFIGS = [
    # Previous Optuna best (pre-pruning)
    {"n_sims": 554, "c_puct": 8.34, "force_k": 0.88},
    # KataGo default force_k
    {"n_sims": 200, "c_puct": 4.73, "force_k": 2.0},
    # Higher forcing variants (pruning should handle these better now)
    {"n_sims": 554, "c_puct": 8.34, "force_k": 4.0},
    {"n_sims": 554, "c_puct": 8.34, "force_k": 8.0},
]


def objective(trial: optuna.Trial) -> tuple[float, int]:
    """Run games vs Greedy, return win rate."""
    n_sims = trial.suggest_int("n_sims", 200, 1200, log=True)
    c_puct = trial.suggest_float("c_puct", 0.5, 16.0, log=True)
    # force_k is under sqrt, so log scale. 0.01 ≈ disabled, 64 = aggressive forcing
    force_k = trial.suggest_float("force_k", 0.01, 64.0, log=True)

    wins = 0.0
    for game_idx in range(GAMES_PER_CONFIG):
        seed = game_idx  # Same mazes for all configs
        mcts_config = DecoupledPUCTConfig(simulations=n_sims, c_puct=c_puct, force_k=force_k)
        agent = MCTSAgent(mcts_config=mcts_config)
        opponent = GreedyAgent()

        result = play_game(
            agent,
            opponent,
            seed=seed,
            width=WIDTH,
            height=HEIGHT,
            cheese_count=CHEESE_COUNT,
            max_turns=MAX_TURNS,
            wall_density=WALL_DENSITY,
            mud_density=MUD_DENSITY,
        )
        if result.winner == 1:
            wins += 1
        elif result.winner == 0:
            wins += 0.5  # Draw counts as half win

    return wins / GAMES_PER_CONFIG, n_sims


def enqueue_seed_trials(study: optuna.Study, csv_path: str, top_n: int) -> None:
    """Enqueue top N configs from a previous sweep as starting points.

    Sorts by: best score first, then fewest simulations (lexicographic).
    """
    df = pd.read_csv(csv_path)
    df = df[df["State"] == "COMPLETE"]
    df = df.sort_values(["Value", "Param n_sims"], ascending=[False, True]).head(top_n)

    for _, row in df.iterrows():
        study.enqueue_trial(
            {
                "n_sims": int(row["Param n_sims"]),
                "c_puct": float(row["Param c_puct"]),
                "force_k": float(row["Param force_k"]),
            }
        )
    print(f"Enqueued {len(df)} seed trials from {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PUCT parameter sweep with Optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="centroid_7x7_mo", help="Study name")
    parser.add_argument("--seed-from", type=str, help="CSV file to seed trials from")
    parser.add_argument("--seed-top", type=int, default=20, help="Number of top configs to seed")
    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    storage = "sqlite:///results/centroid_7x7_mo.db"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "minimize"],  # win_rate up, n_sims down
        load_if_exists=True,
    )

    # Seed with known-good configs
    for cfg in SEED_CONFIGS:
        study.enqueue_trial(cfg)
    print(f"Enqueued {len(SEED_CONFIGS)} known-good configs")

    # Optionally seed from CSV results
    if args.seed_from:
        enqueue_seed_trials(study, args.seed_from, args.seed_top)

    n_trials = 20000  # TPE sampler explores the space
    study.optimize(objective, n_trials=n_trials, n_jobs=args.n_jobs)

    # Visualizations
    fig = optuna.visualization.plot_contour(study, params=["n_sims", "c_puct"])
    fig.write_image("results/puct_contour.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("results/puct_importance.png")

    # Summary — Pareto front
    print("\nPareto front:")
    for trial in study.best_trials:
        win_rate, n_sims = trial.values
        print(f"  win_rate={win_rate:.2%}, n_sims={n_sims} — {trial.params}")
    print("\nVisualizations saved to results/")


if __name__ == "__main__":
    main()
