#!/usr/bin/env python3
"""A/B benchmark: Rust self-play with and without NN evaluation cache.

Runs rust_self_play twice (cache_size=0 vs cache_size=2048) with the same
game configs and reports throughput, cache hit rate, and speedup.

Usage:
    uv run python scripts/bench_rust_nn_cache.py
    uv run python scripts/bench_rust_nn_cache.py --games 100 --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

# --- Defaults ---
DEFAULT_CHECKPOINT = "experiments/runs/7x7_iter1/checkpoints/best_model.pt"

GAME_CONFIG: dict[str, Any] = {
    "width": 7,
    "height": 7,
    "max_turns": 50,
    "cheese_count": 10,
    "maze_type": "open",
    "positions": "corners",
    "cheese_symmetric": True,
}

MCTS_CONFIG: dict[str, Any] = {
    "simulations": 1897,
    "batch_size": 16,
    "c_puct": 0.512,
    "fpu_reduction": 0.459,
    "force_k": 0.103,
    "noise_epsilon": 0.25,
    "noise_concentration": 10.83,
    "max_collisions": 0,
}


def _run(
    onnx_path: str,
    num_games: int,
    cache_size: int,
    num_threads: int,
    output_dir: Path,
) -> Any:
    from alpharat_sampling import rust_self_play

    output_dir.mkdir(parents=True, exist_ok=True)
    return rust_self_play(
        **GAME_CONFIG,
        **MCTS_CONFIG,
        num_games=num_games,
        num_threads=num_threads,
        output_dir=str(output_dir),
        max_games_per_bundle=32,
        onnx_model_path=onnx_path,
        mux_max_batch_size=256,
        device="cpu",
        cache_size=cache_size,
    )


def _ensure_onnx(checkpoint: str) -> str:
    pt_path = Path(checkpoint)
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        return str(onnx_path)
    print(f"Exporting ONNX from {pt_path} ...")
    from scripts.export_onnx import export_onnx

    return str(export_onnx(pt_path, onnx_path))


def _print_stats(label: str, stats: Any) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Games:          {stats.total_games}")
    print(f"  Positions:      {stats.total_positions}")
    print(f"  Elapsed:        {stats.elapsed_secs:.2f}s")
    print(f"  Sims/s:         {stats.simulations_per_second:.0f}")
    print(f"  NN evals/s:     {stats.nn_evals_per_second:.0f}")
    print(f"  NN eval frac:   {stats.nn_eval_fraction * 100:.1f}%")
    print(f"  Cache hits:     {stats.cache_hits}")
    print(f"  Cache misses:   {stats.cache_misses}")
    print(f"  Cache hit rate: {stats.cache_hit_rate * 100:.1f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rust NN cache A/B benchmark")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=2048)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    onnx_path = _ensure_onnx(args.checkpoint)

    print("Rust NN Cache A/B Benchmark")
    print(
        f"Grid: {GAME_CONFIG['width']}x{GAME_CONFIG['height']}, "
        f"sims: {MCTS_CONFIG['simulations']}, games: {args.games}, "
        f"threads: {args.threads}"
    )
    print(f"ONNX: {onnx_path}")
    print(f"Cache size: {args.cache_size}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- No cache ---
        print("Running WITHOUT cache ...")
        stats_off = _run(onnx_path, args.games, 0, args.threads, tmp / "off")

        # --- With cache ---
        print("Running WITH cache ...")
        stats_on = _run(onnx_path, args.games, args.cache_size, args.threads, tmp / "on")

    _print_stats("NO CACHE", stats_off)
    _print_stats(f"WITH CACHE (size={args.cache_size})", stats_on)

    # --- Comparison ---
    print(f"{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")

    speedup = stats_on.simulations_per_second / stats_off.simulations_per_second
    nn_saved = stats_off.total_nn_evals - stats_on.total_nn_evals

    print(f"  Sims/s (no cache):   {stats_off.simulations_per_second:.0f}")
    print(f"  Sims/s (with cache): {stats_on.simulations_per_second:.0f}")
    print(f"  Speedup:             {speedup:.2f}x ({(speedup - 1) * 100:+.1f}%)")
    print(f"  NN evals saved:      {nn_saved}")
    print(f"  Cache hit rate:      {stats_on.cache_hit_rate * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()
