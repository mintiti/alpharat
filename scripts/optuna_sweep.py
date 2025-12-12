#!/usr/bin/env python3
"""PUCT parameter sweep using Optuna.

Usage:
    uv run python scripts/optuna_sweep.py
    uv run python scripts/optuna_sweep.py --n-jobs 4
    uv run python scripts/optuna_sweep.py --study-name my_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna

from alpharat.ai import GreedyAgent, MCTSAgent
from alpharat.eval.game import play_game
from alpharat.mcts import DecoupledPUCTConfig

# Grid values (GridSampler uses these)
N_SIMS_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000]
C_PUCT_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# Game parameters - 5x5 open maze
WIDTH, HEIGHT = 5, 5
CHEESE_COUNT, MAX_TURNS = 5, 30
WALL_DENSITY, MUD_DENSITY = 0.0, 0.0
GAMES_PER_CONFIG = 20


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


def main() -> None:
    parser = argparse.ArgumentParser(description="PUCT parameter sweep with Optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="puct_vs_greedy_5x5", help="Study name")
    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    storage = "sqlite:///results/puct_vs_greedy_5x5.db"
    # sampler = optuna.samplers.GridSampler({"n_sims": N_SIMS_GRID, "c_puct": C_PUCT_GRID})
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        # sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

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
