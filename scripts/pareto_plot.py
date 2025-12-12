#!/usr/bin/env python3
"""Generate Pareto front plot from Optuna sweep results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Find Pareto-efficient points (minimizing all objectives).

    Args:
        costs: (n_points, n_objectives) array where lower is better

    Returns:
        Boolean mask of Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep points that are not dominated by point i
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1) | np.all(
                costs[is_efficient] == c, axis=1
            )
            is_efficient[i] = True  # Keep self
    return is_efficient


def main() -> None:
    df = pd.read_csv("results/puct_vs_greedy_5x5.csv")

    # Filter to completed trials only
    df = df[df["State"] == "COMPLETE"]

    scores: np.ndarray = df["Value"].to_numpy()
    n_sims: np.ndarray = df["Param n_sims"].to_numpy()
    c_puct: np.ndarray = df["Param c_puct"].to_numpy()

    # For Pareto: minimize n_sims, maximize score (so minimize -score)
    costs: np.ndarray = np.column_stack([-scores, n_sims])
    pareto_mask = is_pareto_efficient(costs)

    # Sort Pareto points by n_sims for line plot
    pareto_idx = np.where(pareto_mask)[0]
    sort_order = np.argsort(n_sims[pareto_mask])
    pareto_idx_sorted = pareto_idx[sort_order]

    fig, ax = plt.subplots(figsize=(10, 6))

    # All points, colored by c_puct
    scatter = ax.scatter(
        n_sims[~pareto_mask],
        scores[~pareto_mask],
        c=c_puct[~pareto_mask],
        cmap="viridis",
        alpha=0.4,
        s=30,
        label="Dominated",
    )

    # Pareto front points
    ax.scatter(
        n_sims[pareto_mask],
        scores[pareto_mask],
        c=c_puct[pareto_mask],
        cmap="viridis",
        edgecolors="red",
        linewidths=2,
        s=100,
        label="Pareto front",
        zorder=5,
    )

    # Connect Pareto front with line
    ax.plot(
        n_sims[pareto_idx_sorted],
        scores[pareto_idx_sorted],
        "r--",
        alpha=0.7,
        linewidth=1.5,
        zorder=4,
    )

    # Annotate Pareto points
    for idx in pareto_idx:
        ax.annotate(
            f"c={c_puct[idx]:.1f}",
            (n_sims[idx], scores[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.colorbar(scatter, label="c_puct")
    ax.set_xlabel("Number of MCTS simulations")
    ax.set_ylabel("Win rate vs Greedy")
    ax.set_title("Pareto Front: Score vs Computation (5x5 maze, 100 games/config)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale for x-axis since n_sims spans wide range
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig("results/pareto_front.png", dpi=150)
    print("Saved to results/pareto_front.png")

    # Print Pareto front details
    print("\nPareto front points:")
    print(f"{'n_sims':>8} {'c_puct':>8} {'score':>8}")
    print("-" * 26)
    for idx in pareto_idx_sorted:
        print(f"{n_sims[idx]:>8} {c_puct[idx]:>8.2f} {scores[idx]:>8.3f}")


if __name__ == "__main__":
    main()
