"""Plot search horizon data: MCGS vs MCTS detection curves.

uv run python scripts/plot_horizon.py search_horizon_2m.csv
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_q_vs_sims(df: pd.DataFrame, out: str) -> None:
    """Q-value vs sim budget, both algorithms overlaid, one panel per distance."""
    distances = sorted(df["distance"].unique())
    n = len(distances)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    colors = {"MCTS": "#4878CF", "MCGS": "#D65F5F"}

    for idx, d in enumerate(distances):
        ax = axes[idx // cols][idx % cols]
        sub = df[df["distance"] == d]

        for algo in ["MCTS", "MCGS"]:
            asub = sub[sub["algorithm"] == algo]
            color = colors[algo]

            # Individual trials (faint)
            for trial in asub["trial"].unique():
                t = asub[asub["trial"] == trial].sort_values("sims")
                ax.plot(
                    t["sims"], t["q_correct"].clip(lower=0), color=color, alpha=0.12, linewidth=0.7
                )

            # Median across trials (bold)
            median = asub.groupby("sims")["q_correct"].median().clip(lower=0)
            ax.plot(median.index, median.values, color=color, linewidth=2.2, label=algo)

        ax.set_xscale("log")
        ax.set_title(f"d = {d}", fontsize=12)
        ax.set_xlabel("simulations")
        ax.set_ylabel("Q (greedy action)")
        ax.set_ylim(-0.02, 0.55)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Q-value vs sim budget (median bold, individual trials faint)", fontsize=13, y=1.01
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def plot_horizon(df: pd.DataFrame, out: str, threshold: float = 0.001) -> None:
    """Horizon (sims to detect) vs distance, both algorithms."""
    distances = sorted(df["distance"].unique())
    max_sims = df["sims"].max()

    colors = {"MCTS": "#4878CF", "MCGS": "#D65F5F"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for algo in ["MCTS", "MCGS"]:
        p50s = []
        p95s = []
        for d in distances:
            sub = df[(df["algorithm"] == algo) & (df["distance"] == d)]
            horizons = []
            for trial in sub["trial"].unique():
                t = sub[sub["trial"] == trial].sort_values("sims")
                crossed = t[t["q_correct"] >= threshold]["sims"]
                horizons.append(crossed.iloc[0] if len(crossed) > 0 else max_sims * 2)
            horizons.sort()
            n = len(horizons)
            p50s.append(horizons[n // 2])
            p95s.append(horizons[min(int(n * 0.95), n - 1)])

        color = colors[algo]
        valid_50 = [(d, s) for d, s in zip(distances, p50s, strict=True) if s <= max_sims]
        valid_95 = [(d, s) for d, s in zip(distances, p95s, strict=True) if s <= max_sims]

        if valid_50:
            ds, ss = zip(*valid_50, strict=True)
            ax.plot(ds, ss, "o-", color=color, linewidth=2.2, markersize=6, label=f"{algo} p50")
        if valid_95:
            ds, ss = zip(*valid_95, strict=True)
            ax.plot(
                ds,
                ss,
                "s--",
                color=color,
                linewidth=1.5,
                markersize=5,
                alpha=0.6,
                label=f"{algo} p95",
            )

    ax.set_yscale("log")
    ax.set_xlabel("distance (manhattan)", fontsize=11)
    ax.set_ylabel(f"sims to detect (Q > {threshold})", fontsize=11)
    ax.set_title("Search horizon: MCGS vs MCTS", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(distances)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to search_horizon CSV")
    parser.add_argument("--prefix", default="docs/horizon", help="Output file prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plot_q_vs_sims(df, f"{args.prefix}_q_vs_sims.png")
    plot_horizon(df, f"{args.prefix}_detection.png")


if __name__ == "__main__":
    main()
