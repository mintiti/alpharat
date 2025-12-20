"""Self-play sampling for training data generation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from alpharat.data.batch import GameParams, create_batch
from alpharat.data.recorder import GameRecorder
from alpharat.eval.game import is_terminal
from alpharat.mcts import MCTSConfig  # noqa: TC001
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from pathlib import Path

    from pyrat_engine.core.game import PyRat


# --- Config Models ---


class SamplingParams(BaseModel):
    """Parameters for the sampling process."""

    num_games: int
    workers: int = 4


class SamplingConfig(BaseModel):
    """Full configuration for a sampling run."""

    mcts: MCTSConfig = Field(discriminator="variant")
    game: GameParams
    sampling: SamplingParams
    output_dir: str
    checkpoint: str | None = None


@dataclass
class GameStats:
    """Statistics from a single game."""

    positions: int
    simulations: int


@dataclass
class WorkerStats:
    """Aggregated statistics from a worker process."""

    games_completed: int = 0
    total_positions: int = 0
    total_simulations: int = 0

    def add_game(self, game_stats: GameStats) -> None:
        """Add stats from a completed game."""
        self.games_completed += 1
        self.total_positions += game_stats.positions
        self.total_simulations += game_stats.simulations


@dataclass
class SamplingMetrics:
    """Throughput metrics from a sampling run."""

    total_games: int
    total_positions: int
    total_simulations: int
    elapsed_seconds: float
    workers: int

    @property
    def games_per_second(self) -> float:
        """Games completed per second."""
        return self.total_games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def positions_per_second(self) -> float:
        """Positions (steps) processed per second."""
        return self.total_positions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def simulations_per_second(self) -> float:
        """MCTS simulations per second."""
        return self.total_simulations / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"SamplingMetrics(\n"
            f"  workers={self.workers},\n"
            f"  elapsed={self.elapsed_seconds:.2f}s,\n"
            f"  games={self.total_games} ({self.games_per_second:.2f}/s),\n"
            f"  positions={self.total_positions} ({self.positions_per_second:.2f}/s),\n"
            f"  simulations={self.total_simulations} ({self.simulations_per_second:.2f}/s)\n"
            f")"
        )


# --- Helper Functions ---


def build_tree(game: PyRat, gamma: float) -> MCTSTree:
    """Build fresh MCTS tree for search.

    The tree will initialize the root with smart uniform priors that only
    assign probability to distinct effective actions (blocked moves get 0).

    Args:
        game: Current game state (will be deep-copied for simulation).
        gamma: Discount factor for value backup.

    Returns:
        MCTSTree ready for search.
    """
    simulator = copy.deepcopy(game)

    # Dummy priors - tree will overwrite with smart uniform via _init_root_priors()
    dummy = np.ones(5) / 5

    root = MCTSNode(
        game_state=None,
        prior_policy_p1=dummy,
        prior_policy_p2=dummy,
        nn_payout_prediction=np.zeros((5, 5)),
        parent=None,
        p1_mud_turns_remaining=simulator.player1_mud_turns,
        p2_mud_turns_remaining=simulator.player2_mud_turns,
    )

    return MCTSTree(game=simulator, root=root, gamma=gamma, predict_fn=None)


def create_game(params: GameParams, seed: int) -> PyRat:
    """Create a new game from parameters.

    Args:
        params: Game configuration.
        seed: Random seed for maze generation.

    Returns:
        Configured PyRat game instance.
    """
    from pyrat_engine.core.game import PyRat

    return PyRat(
        width=params.width,
        height=params.height,
        cheese_count=params.cheese_count,
        max_turns=params.max_turns,
        seed=seed,
    )


# --- Core Sampling ---


def play_and_record_game(
    config: SamplingConfig,
    games_dir: Path,
    seed: int,
) -> GameStats:
    """Play one game with MCTS and record all positions.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save the game file.
        seed: Random seed for game generation.

    Returns:
        GameStats with position and simulation counts.
    """
    game = create_game(config.game, seed)
    positions = 0
    total_simulations = 0

    with GameRecorder(game, games_dir, config.game.width, config.game.height) as recorder:
        while not is_terminal(game):
            # Build tree and run search
            tree = build_tree(game, config.mcts.gamma)
            search = config.mcts.build(tree)
            result = search.search()

            positions += 1
            total_simulations += config.mcts.simulations

            # Record position before making move
            recorder.record_position(
                game=game,
                search_result=result,
                prior_p1=tree.root.prior_policy_p1,
                prior_p2=tree.root.prior_policy_p2,
                visit_counts=tree.root.action_visits,
            )

            # Sample actions from Nash equilibrium policies
            a1 = select_action_from_strategy(result.policy_p1)
            a2 = select_action_from_strategy(result.policy_p2)

            game.make_move(a1, a2)

    return GameStats(positions=positions, simulations=total_simulations)


def _worker_loop(
    config: SamplingConfig,
    games_dir: Path,
    work_queue: Queue[int | None],
    results_queue: Queue[GameStats],
) -> None:
    """Worker process that continuously processes games from the queue.

    Runs until it receives None (sentinel) from the work queue.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save game files.
        work_queue: Queue of seeds to process (None = stop).
        results_queue: Queue to send completed game stats.
    """
    while True:
        seed = work_queue.get()
        if seed is None:
            # Sentinel received, exit
            break

        game_stats = play_and_record_game(config, games_dir, seed)
        results_queue.put(game_stats)


def run_sampling(config: SamplingConfig, *, verbose: bool = True) -> tuple[Path, SamplingMetrics]:
    """Run full sampling session.

    Creates a batch directory, plays games in parallel, and records
    all positions to npz files.

    Args:
        config: Sampling configuration.
        verbose: Print progress if True.

    Returns:
        Tuple of (batch_dir, metrics) with path and throughput statistics.
    """
    batch_dir = create_batch(
        parent_dir=config.output_dir,
        checkpoint_path=config.checkpoint,
        mcts_config=config.mcts,
        game_params=config.game,
    )
    games_dir = batch_dir / "games"

    if verbose:
        print(f"Sampling {config.sampling.num_games} games")
        print(f"  MCTS: {config.mcts.variant}, {config.mcts.simulations} sims")
        print(
            f"  Game: {config.game.width}x{config.game.height}, {config.game.cheese_count} cheese"
        )
        print(f"  Workers: {config.sampling.workers}")
        print(f"  Output: {batch_dir}")
        print()

    total_positions = 0
    total_simulations = 0
    num_workers = config.sampling.workers
    num_games = config.sampling.num_games

    # Create queues for work distribution and results collection
    work_queue: Queue[int | None] = Queue()
    results_queue: Queue[GameStats] = Queue()

    # Pre-fill work queue with all seeds
    for seed in range(num_games):
        work_queue.put(seed)

    # Add sentinel values to stop workers (one per worker)
    for _ in range(num_workers):
        work_queue.put(None)

    # Start worker processes
    workers = [
        Process(target=_worker_loop, args=(config, games_dir, work_queue, results_queue))
        for _ in range(num_workers)
    ]

    start_time = time.perf_counter()

    for worker in workers:
        worker.start()

    # Collect results as they come in
    for completed in range(1, num_games + 1):
        game_stats = results_queue.get()
        total_positions += game_stats.positions
        total_simulations += game_stats.simulations

        if verbose and completed % 10 == 0:
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed
            print(f"Completed {completed}/{num_games} games ({rate:.2f} games/s)")

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    elapsed_seconds = time.perf_counter() - start_time

    metrics = SamplingMetrics(
        total_games=config.sampling.num_games,
        total_positions=total_positions,
        total_simulations=total_simulations,
        elapsed_seconds=elapsed_seconds,
        workers=config.sampling.workers,
    )

    if verbose:
        print()
        print(metrics)

    return batch_dir, metrics
