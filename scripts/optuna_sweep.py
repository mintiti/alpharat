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

# Seed configs for normalized Q-values (Q in [0, 1])
# Pareto front from per_node_puct_7x7 sweep (847 trials, no FPU reduction).
# Each config seeded at both KataGo (0.2) and LC0 (0.33) FPU values.
_PARETO_FRONT = [
    {"n_sims": 200, "c_puct": 0.548, "force_k": 1.184},
    {"n_sims": 206, "c_puct": 0.507, "force_k": 0.064},
    {"n_sims": 213, "c_puct": 0.507, "force_k": 0.064},
    {"n_sims": 224, "c_puct": 0.531, "force_k": 0.178},
    {"n_sims": 272, "c_puct": 0.531, "force_k": 2.201},
    {"n_sims": 381, "c_puct": 0.612, "force_k": 0.018},
    {"n_sims": 400, "c_puct": 0.643, "force_k": 0.432},
    {"n_sims": 409, "c_puct": 0.507, "force_k": 0.011},
    {"n_sims": 420, "c_puct": 0.507, "force_k": 0.011},
    {"n_sims": 545, "c_puct": 0.507, "force_k": 0.145},
    {"n_sims": 591, "c_puct": 0.686, "force_k": 0.067},
    {"n_sims": 629, "c_puct": 0.531, "force_k": 0.067},
    {"n_sims": 711, "c_puct": 0.612, "force_k": 0.307},
    {"n_sims": 855, "c_puct": 0.576, "force_k": 0.099},
    {"n_sims": 1007, "c_puct": 0.505, "force_k": 0.265},
]
SEED_CONFIGS = [{**cfg, "fpu_reduction": fpu} for cfg in _PARETO_FRONT for fpu in (0.2, 0.33)]


def objective(trial: optuna.Trial) -> tuple[float, int]:
    """Run games vs Greedy, return win rate."""
    n_sims = trial.suggest_int("n_sims", 200, 1200, log=True)
    c_puct = trial.suggest_float("c_puct", 0.5, 2.0, log=True)
    force_k = trial.suggest_float("force_k", 0.01, 5.0, log=True)
    fpu_reduction = trial.suggest_float("fpu_reduction", 0.1, 0.5)

    wins = 0.0
    for game_idx in range(GAMES_PER_CONFIG):
        seed = game_idx  # Same mazes for all configs
        mcts_config = DecoupledPUCTConfig(
            simulations=n_sims, c_puct=c_puct, force_k=force_k, fpu_reduction=fpu_reduction
        )
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
        params: dict[str, int | float] = {
            "n_sims": int(row["Param n_sims"]),
            "c_puct": float(row["Param c_puct"]),
            "force_k": float(row["Param force_k"]),
        }
        if "Param fpu_reduction" in row.index:
            params["fpu_reduction"] = float(row["Param fpu_reduction"])
        study.enqueue_trial(params)
    print(f"Enqueued {len(df)} seed trials from {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PUCT parameter sweep with Optuna")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers")
    parser.add_argument("--study-name", default="per_node_puct_7x7", help="Study name")
    parser.add_argument("--seed-from", type=str, help="CSV file to seed trials from")
    parser.add_argument("--seed-top", type=int, default=20, help="Number of top configs to seed")
    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    journal_path = f"results/{args.study_name}.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path),
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["maximize", "minimize"],  # win_rate up, n_sims down
        load_if_exists=True,
    )
    print(f"Study: {args.study_name}, journal: {journal_path}")

    # Seed with known-good configs
    for cfg in SEED_CONFIGS:
        study.enqueue_trial(cfg)
    print(f"Enqueued {len(SEED_CONFIGS)} known-good configs")

    # Optionally seed from CSV results
    if args.seed_from:
        enqueue_seed_trials(study, args.seed_from, args.seed_top)

    n_trials = 20000  # TPE sampler explores the space
    study.optimize(objective, n_trials=n_trials, n_jobs=args.n_jobs)

    # Visualizations (MO-compatible)
    fig = optuna.visualization.plot_pareto_front(study, target_names=["win_rate", "n_sims"])
    fig.write_image(f"results/{args.study_name}_pareto.png")

    fig = optuna.visualization.plot_param_importances(
        study,
        target=lambda t: t.values[0],  # importance for win_rate
    )
    fig.write_image(f"results/{args.study_name}_importance.png")

    # Summary — Pareto front
    print("\nPareto front:")
    for trial in study.best_trials:
        win_rate, n_sims = trial.values
        print(f"  win_rate={win_rate:.2%}, n_sims={n_sims} — {trial.params}")
    print(f"\nVisualizations saved to results/{args.study_name}_*.png")


if __name__ == "__main__":
    main()
