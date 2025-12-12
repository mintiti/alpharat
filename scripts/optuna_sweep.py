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
from alpharat.mcts import DecoupledPUCTConfig

# Game parameters - 5x5 open maze
WIDTH, HEIGHT = 5, 5
CHEESE_COUNT, MAX_TURNS = 5, 30
WALL_DENSITY, MUD_DENSITY = 0.0, 0.0
GAMES_PER_CONFIG = 100


def objective(trial: optuna.Trial) -> float:
    """Run games vs Greedy, return win rate."""
    # Continuous ranges for TPE compatibility (GridSampler ignores these)
    n_sims = trial.suggest_int("n_sims", 10, 2000, log=True)
    c_puct = trial.suggest_float("c_puct", 0.5, 16.0, log=True)

    config = DecoupledPUCTConfig(simulations=n_sims, c_puct=c_puct)

    wins = 0.0
    for game_idx in range(GAMES_PER_CONFIG):
        seed = game_idx  # Same mazes for all configs
        agent = MCTSAgent(config)
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

        # Report intermediate result for pruning
        trial.report(wins / (game_idx + 1), step=game_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return wins / GAMES_PER_CONFIG


def enqueue_seed_trials(study: optuna.Study, csv_path: str, top_n: int) -> None:
    """Enqueue top N configs from a previous sweep as starting points.

    Sorts by: best score first, then fewest simulations (lexicographic).
    """
    df = pd.read_csv(csv_path)
    df = df[df["State"] == "COMPLETE"]
    df = df.sort_values(["Value", "Param n_sims"], ascending=[False, True]).head(top_n)

    for _, row in df.iterrows():
        study.enqueue_trial(
            {"n_sims": int(row["Param n_sims"]), "c_puct": float(row["Param c_puct"])}
        )
    print(f"Enqueued {len(df)} seed trials from {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PUCT parameter sweep with Optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="puct_vs_greedy_5x5", help="Study name")
    parser.add_argument("--seed-from", type=str, help="CSV file to seed trials from")
    parser.add_argument("--seed-top", type=int, default=20, help="Number of top configs to seed")
    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    storage = "sqlite:///results/puct_vs_greedy_5x5.db"
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=40,
        max_resource=GAMES_PER_CONFIG,
        reduction_factor=2,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    # Seed with good configs from previous sweep
    if args.seed_from:
        enqueue_seed_trials(study, args.seed_from, args.seed_top)

    n_trials = 20000  # TPE sampler explores the space
    study.optimize(objective, n_trials=n_trials, n_jobs=args.n_jobs)

    # Visualizations
    fig = optuna.visualization.plot_contour(study, params=["n_sims", "c_puct"])
    fig.write_image("results/puct_contour.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image("results/puct_importance.png")

    # Summary
    print(f"\nBest params: {study.best_params}")
    print(f"Best score: {study.best_value:.2f}")
    print("\nVisualizations saved to results/")


if __name__ == "__main__":
    main()
