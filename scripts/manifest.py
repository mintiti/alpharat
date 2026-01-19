#!/usr/bin/env python3
"""Query experiment artifacts.

Usage:
    uv run python scripts/manifest.py batches
    uv run python scripts/manifest.py shards
    uv run python scripts/manifest.py runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpharat.experiments import ExperimentManager


def print_batches(exp: ExperimentManager) -> None:
    """Print batches table."""
    batches = exp.list_batches_with_info()
    if not batches:
        print("No batches found.")
        return

    # Header
    print(f"{'GROUP':<18} {'UUID':<10} {'CREATED':<17} {'SIZE':<5} {'SIMS':<5} {'PARENT'}")
    print("-" * 80)

    for b in batches:
        group, uuid = b["id"].split("/", 1)
        short_uuid = uuid[:8]
        sims = b["simulations"]
        row = f"{group:<18} {short_uuid:<10} {b['created']:<17} {b['size']:<5} {sims:<5}"
        print(f"{row} {b['parent']}")


def print_shards(exp: ExperimentManager) -> None:
    """Print shards table with lineage."""
    shards = exp.list_shards_with_info()
    if not shards:
        print("No shards found.")
        return

    # Header
    print(f"{'GROUP':<18} {'UUID':<10} {'CREATED':<17} {'TRAIN':>8} {'VAL':>8}  FROM")
    print("-" * 85)

    for s in shards:
        group, uuid = s["id"].split("/", 1)
        short_uuid = uuid[:8]
        # Summarize source batches: show group names only
        if s["source_batches"]:
            source_groups = {b.split("/")[0] for b in s["source_batches"]}
            sources = ", ".join(sorted(source_groups))
        else:
            sources = "-"
        row = f"{group:<18} {short_uuid:<10} {s['created']:<17} {s['train']:>8} {s['val']:>8}"
        print(f"{row}  {sources}")


def print_runs(exp: ExperimentManager) -> None:
    """Print runs table."""
    runs = exp.list_runs_with_info()
    if not runs:
        print("No runs found.")
        return

    # Header
    print(f"{'NAME':<25} {'CREATED':<17} {'FROM SHARDS':<20} {'EPOCH':>6} {'VAL LOSS':>10}")
    print("-" * 85)

    for r in runs:
        # Show shard group only
        shard_group = r["source_shards"].split("/")[0] if r["source_shards"] else "-"
        epoch = str(r["final_epoch"]) if r["final_epoch"] is not None else "-"
        loss = f"{r['best_val_loss']:.4f}" if r["best_val_loss"] is not None else "-"
        print(f"{r['name']:<25} {r['created']:<17} {shard_group:<20} {epoch:>6} {loss:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query experiment artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/manifest.py batches
    uv run python scripts/manifest.py shards
    uv run python scripts/manifest.py runs
""",
    )
    parser.add_argument(
        "artifact",
        choices=["batches", "shards", "runs"],
        help="Artifact type to list",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory (default: experiments)",
    )
    args = parser.parse_args()

    exp = ExperimentManager(args.experiments_dir)

    if args.artifact == "batches":
        print_batches(exp)
    elif args.artifact == "shards":
        print_shards(exp)
    elif args.artifact == "runs":
        print_runs(exp)


if __name__ == "__main__":
    main()
