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

from alpharat.ai import MCTSAgent, RandomAgent
from alpharat.eval.game import play_game
from alpharat.mcts import DecoupledPUCTConfig

# Grid values (GridSampler uses these)
N_SIMS_GRID = [300, 600, 900, 1200, 1500, 1800, 2100]
C_PUCT_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# Game parameters (hardcoded)
WIDTH, HEIGHT = 15, 11
CHEESE_COUNT, MAX_TURNS = 21, 100
GAMES_PER_CONFIG = 10


def objective(trial: optuna.Trial) -> float:
    """Run 10 games, return mean cheese collected."""
    # Continuous ranges for TPE compatibility (GridSampler ignores these)
    n_sims = trial.suggest_int("n_sims", 300, 2100, step=50)
    c_puct = trial.suggest_float("c_puct", 0.5, 16.0, log=True)

    config = DecoupledPUCTConfig(simulations=n_sims, c_puct=c_puct)

    scores = []
    for game_idx in range(GAMES_PER_CONFIG):
        seed = game_idx  # Same 10 mazes for all configs
        agent = MCTSAgent(config)
        opponent = RandomAgent()

        result = play_game(
            agent,
            opponent,
            seed=seed,
            width=WIDTH,
            height=HEIGHT,
            cheese_count=CHEESE_COUNT,
            max_turns=MAX_TURNS,
        )
        scores.append(result.p1_score)

        # Report intermediate result for pruning
        trial.report(sum(scores) / len(scores), step=game_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="PUCT parameter sweep with Optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="puct_sweep", help="Study name")
    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    storage = "sqlite:///results/puct_sweep_2.db"
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

    n_trials = len(N_SIMS_GRID) * len(C_PUCT_GRID)  # 42
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
