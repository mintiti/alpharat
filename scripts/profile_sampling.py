"""Profile single-process game sampling with NN.

Replicates what _worker_loop does without multiprocessing overhead,
so cProfile captures where time actually goes in the game loop.

Usage:
    uv run python scripts/profile_sampling.py
    uv run snakeviz profile_sampling.prof  # visualize
"""

from __future__ import annotations

import cProfile
import pstats
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from alpharat.config.game import GameConfig
from alpharat.data.sampling import (
    SamplingConfig,
    SamplingParams,
    play_and_record_game,
)
from alpharat.mcts import PythonMCTSConfig

if TYPE_CHECKING:
    from alpharat.mcts.searcher import Searcher


def run_games(
    config: SamplingConfig,
    games_dir: Path,
    num_games: int,
    searcher: Searcher,
) -> None:
    """Run N games sequentially (the part we're profiling)."""
    for seed in range(num_games):
        play_and_record_game(config, games_dir, seed, searcher)
        print(f"  Game {seed + 1}/{num_games} done")


def main() -> None:
    # Config matching 7x7_open + 5x5_tuned MCTS
    checkpoint = "experiments/runs/cnn_7x7_visits/checkpoints/checkpoint_epoch_100.pt"

    mcts_config = PythonMCTSConfig(
        simulations=554,
        c_puct=1.5,
        force_k=2.0,
    )

    config = SamplingConfig(
        mcts=mcts_config,
        game=GameConfig(
            width=7,
            height=7,
            max_turns=50,
            cheese_count=10,
            wall_density=0.0,
            mud_density=0.0,
            symmetric=True,
        ),
        sampling=SamplingParams(num_games=10, workers=1),
        group="profiling",
        checkpoint=checkpoint,
    )

    # Build searcher with NN (like worker does at startup)
    print(f"Loading NN from {checkpoint}...")
    searcher = mcts_config.build_searcher(checkpoint=checkpoint, device="cpu")
    print("NN loaded.\n")

    # Temp dir for game files (we don't care about saving them)
    games_dir = Path(tempfile.mkdtemp(prefix="profiling_"))
    num_games = 50

    print(f"Profiling {num_games} games...")

    profiler = cProfile.Profile()
    profiler.enable()

    run_games(config, games_dir, num_games, searcher)

    profiler.disable()

    # Print top functions by cumulative time
    print("\n" + "=" * 60)
    print("PROFILE RESULTS (top 50 by cumulative time)")
    print("=" * 60 + "\n")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(50)

    # Save for snakeviz
    prof_path = "profile_sampling.prof"
    stats.dump_stats(prof_path)
    print(f"\nProfile saved to {prof_path}")
    print(f"View with: uv run snakeviz {prof_path}")


if __name__ == "__main__":
    main()
