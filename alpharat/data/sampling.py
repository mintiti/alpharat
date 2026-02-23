"""Self-play sampling for training data generation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

from alpharat.ai.utils import select_action_from_strategy
from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.data.recorder import GameBundler, GameRecorder
from alpharat.eval.game import is_terminal
from alpharat.mcts.config import MCTSConfig  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

    from pyrat_engine.core.game import PyRat

    from alpharat.data.types import GameData
    from alpharat.mcts.searcher import Searcher


# --- Config Models ---


@dataclass
class NNContext:
    """Worker-local NN model context for NN-guided sampling.

    Loaded once per worker process and reused across all games.
    """

    model: Any  # TrainableModel (nn.Module with predict method)
    builder: Any  # ObservationBuilder
    width: int
    height: int
    device: str


class SamplingParams(StrictBaseModel):
    """Parameters for the sampling process."""

    num_games: int
    workers: int = 4
    device: str = "cpu"


class SamplingConfig(StrictBaseModel):
    """Full configuration for a sampling run."""

    mcts: MCTSConfig
    game: GameConfig
    sampling: SamplingParams
    group: str  # Required: human-readable grouping name (e.g., "uniform_5x5")
    experiments_dir: str = "experiments"  # Root directory for experiments
    checkpoint: str | None = None


@dataclass
class GameStats:
    """Statistics from a single game."""

    positions: int
    simulations: int
    winner: int  # 0=draw, 1=p1, 2=p2
    score_p1: float
    score_p2: float
    turns: int
    cheese_available: int
    nn_calls: int = 0
    nn_time_s: float = 0.0


@dataclass
class SamplingMetrics:
    """Metrics from a sampling run."""

    # Throughput
    total_games: int
    total_positions: int
    total_simulations: int
    elapsed_seconds: float
    workers: int

    # Game outcomes (from P1 perspective)
    p1_wins: int
    p2_wins: int
    draws: int

    # Cheese stats
    total_cheese_collected: float
    total_cheese_available: int

    # Game length
    total_turns: int
    min_turns: int
    max_turns: int

    # NN timing
    total_nn_calls: int = 0
    total_nn_time_s: float = 0.0

    @property
    def games_per_second(self) -> float:
        return self.total_games / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def positions_per_second(self) -> float:
        return self.total_positions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def simulations_per_second(self) -> float:
        return self.total_simulations / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def avg_turns(self) -> float:
        return self.total_turns / self.total_games if self.total_games > 0 else 0.0

    @property
    def cheese_utilization(self) -> float:
        """Fraction of available cheese that was collected."""
        if self.total_cheese_available == 0:
            return 0.0
        return self.total_cheese_collected / self.total_cheese_available

    @property
    def avg_nn_latency_ms(self) -> float:
        if self.total_nn_calls == 0:
            return 0.0
        return self.total_nn_time_s / self.total_nn_calls * 1000

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0

    def __str__(self) -> str:
        nn_line = ""
        if self.total_nn_calls > 0:
            nn_line = (
                f"    nn_latency={self.avg_nn_latency_ms:.1f}ms avg ({self.total_nn_calls} calls)\n"
            )
        return (
            f"SamplingMetrics(\n"
            f"  Throughput:\n"
            f"    workers={self.workers}, elapsed={self.elapsed_seconds:.2f}s\n"
            f"    simulations={self.total_simulations} ({self.simulations_per_second:.0f}/s)\n"
            f"    positions={self.total_positions} ({self.positions_per_second:.0f}/s)\n"
            f"    games={self.total_games} ({self.games_per_second:.1f}/s)\n"
            f"{nn_line}"
            f"  Outcomes (P1 perspective):\n"
            f"    W/D/L = {self.p1_wins}/{self.draws}/{self.p2_wins}\n"
            f"    draw_rate = {self.draw_rate:.1%}\n"
            f"  Cheese:\n"
            f"    collected = {int(self.total_cheese_collected)}/{self.total_cheese_available} "
            f"({self.cheese_utilization:.1%})\n"
            f"  Game length:\n"
            f"    avg={self.avg_turns:.1f}, min={self.min_turns}, max={self.max_turns}\n"
            f")"
        )


# --- NN Loading ---


def load_nn_context(checkpoint_path: str, device: str = "cpu") -> NNContext:
    """Load NN model for use in sampling using ModelConfig.build_model().

    Called once per worker process at startup. Uses Pydantic config from checkpoint
    to build the correct model and observation builder automatically.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to run inference on.

    Returns:
        NNContext with loaded model and builder.
    """
    from alpharat.config.checkpoint import load_model_from_checkpoint

    model, builder, width, height = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
        compile_model=True,
    )

    return NNContext(
        model=model,
        builder=builder,
        width=width,
        height=height,
        device=device,
    )


# --- Helper Functions ---


def _build_searcher(config: SamplingConfig, device: str) -> Searcher:
    """Build a Searcher from SamplingConfig.

    Args:
        config: Sampling configuration with mcts and checkpoint.
        device: Device for NN inference.

    Returns:
        Configured Searcher instance.
    """
    return config.mcts.build_searcher(checkpoint=config.checkpoint, device=device)


def create_game(config: GameConfig) -> PyRat:
    """Create a new game from configuration.

    Generates a random seed internally — each game gets independent randomness.

    Args:
        config: Game configuration.

    Returns:
        Configured PyRat game instance.
    """
    seed = random.randrange(2**32)
    return config.to_engine_config().create(seed=seed)


# --- Core Sampling ---


def play_and_record_game(
    config: SamplingConfig,
    games_dir: Path,
    searcher: Searcher,
    *,
    auto_save: bool = True,
) -> tuple[GameStats, GameData | None]:
    """Play one game with MCTS and record all positions.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save the game file.
        searcher: Searcher instance for MCTS search.
        auto_save: If True (default), save to disk. If False, return GameData
            for external handling (e.g., bundling).

    Returns:
        Tuple of (GameStats, GameData or None). GameData is only returned if
        auto_save=False.
    """
    game = create_game(config.game)
    cheese_available = config.game.cheese.count
    positions = 0
    simulations = config.mcts.simulations

    with GameRecorder(
        game, games_dir, config.game.width, config.game.height, auto_save=auto_save
    ) as recorder:
        while not is_terminal(game):
            result = searcher.search(game)

            positions += 1

            # Sample actions from search policies
            a1 = select_action_from_strategy(result.policy_p1)
            a2 = select_action_from_strategy(result.policy_p2)

            # Record position before making move
            recorder.record_position(game, result, a1, a2)

            game.make_move(a1, a2)

    # Determine winner
    score_p1, score_p2 = game.player1_score, game.player2_score
    if score_p1 > score_p2:
        winner = 1
    elif score_p2 > score_p1:
        winner = 2
    else:
        winner = 0

    game_stats = GameStats(
        positions=positions,
        simulations=positions * simulations,
        winner=winner,
        score_p1=score_p1,
        score_p2=score_p2,
        turns=game.turn,
        cheese_available=cheese_available,
        nn_calls=0,
        nn_time_s=0.0,
    )

    # Return game data if not auto-saving (for bundling)
    game_data: GameData | None = recorder.data if not auto_save else None

    return game_stats, game_data


def _worker_loop(
    config: SamplingConfig,
    games_dir: Path,
    work_queue: Queue[int | None],
    results_queue: Queue[GameStats],
    device: str = "cpu",
    *,
    use_bundler: bool = True,
) -> None:
    """Worker process that continuously processes games from the queue.

    Runs until it receives None (sentinel) from the work queue. Builds a Searcher
    once at startup and reuses it for all games.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save game files.
        work_queue: Queue of game indices to process (None = stop).
        results_queue: Queue to send completed game stats.
        device: Device for NN inference (if checkpoint configured).
        use_bundler: If True (default), buffer games and write bundles. If False,
            write individual game files (legacy behavior).
    """
    # Build searcher once per worker (loads NN if checkpoint configured)
    searcher = _build_searcher(config, device)

    # Create bundler for this worker if enabled
    bundler: GameBundler | None = None
    if use_bundler:
        bundler = GameBundler(
            output_dir=games_dir,
            width=config.game.width,
            height=config.game.height,
        )

    while True:
        game_index = work_queue.get()
        if game_index is None:
            # Sentinel received, flush bundler and exit
            if bundler is not None:
                bundler.flush()
            break

        game_stats, game_data = play_and_record_game(
            config, games_dir, searcher, auto_save=not use_bundler
        )

        if bundler is not None and game_data is not None:
            bundler.add_game(game_data)

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
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager(config.experiments_dir)
    batch_dir, batch_uuid = exp.prepare_batch(
        group=config.group,
        mcts_config=config.mcts,
        game=config.game,
        checkpoint_path=config.checkpoint,
    )
    games_dir = batch_dir / "games"

    if verbose:
        from alpharat.config.display import format_config_summary

        summary = format_config_summary(
            ("Game", config.game),
            ("MCTS", config.mcts),
            ("Sampling", config.sampling),
        )
        print(summary)
        if config.checkpoint:
            print(f"Checkpoint: {config.checkpoint}")
        print(f"Output: {batch_dir}")
        print()

    num_workers = config.sampling.workers
    num_games = config.sampling.num_games

    # Aggregation accumulators
    total_positions = 0
    total_simulations = 0
    total_nn_calls = 0
    total_nn_time_s = 0.0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_cheese_collected = 0.0
    total_cheese_available = 0
    total_turns = 0
    min_turns: int | float = float("inf")
    max_turns = 0

    # Create queues for work distribution and results collection
    # Bounded queue prevents blocking when adding many items
    queue_size = num_workers * 2
    work_queue: Queue[int | None] = Queue(maxsize=queue_size)
    results_queue: Queue[GameStats] = Queue()

    # Start worker processes
    device = config.sampling.device
    workers = [
        Process(target=_worker_loop, args=(config, games_dir, work_queue, results_queue, device))
        for _ in range(num_workers)
    ]

    start_time = time.perf_counter()

    for worker in workers:
        worker.start()

    # Track how many game indices we've submitted and results collected
    next_index = 0
    completed = 0

    # Initially fill the queue (non-blocking since queue_size > 0)
    while next_index < num_games and next_index < queue_size:
        work_queue.put(next_index)
        next_index += 1

    # Collect results and submit more work as slots free up
    while completed < num_games:
        game_stats = results_queue.get()
        completed += 1

        # Submit next game index if any remain
        if next_index < num_games:
            work_queue.put(next_index)
            next_index += 1

        # Throughput stats
        total_positions += game_stats.positions
        total_simulations += game_stats.simulations
        total_nn_calls += game_stats.nn_calls
        total_nn_time_s += game_stats.nn_time_s

        # Outcome stats
        if game_stats.winner == 1:
            p1_wins += 1
        elif game_stats.winner == 2:
            p2_wins += 1
        else:
            draws += 1

        # Cheese stats
        total_cheese_collected += game_stats.score_p1 + game_stats.score_p2
        total_cheese_available += game_stats.cheese_available

        # Game length stats
        total_turns += game_stats.turns
        min_turns = min(min_turns, game_stats.turns)
        max_turns = max(max_turns, game_stats.turns)

        if verbose and completed % 100 == 0:
            elapsed = time.perf_counter() - start_time
            games_rate = completed / elapsed
            sims_rate = total_simulations / elapsed
            pos_rate = total_positions / elapsed
            cheese_util = (
                total_cheese_collected / total_cheese_available
                if total_cheese_available > 0
                else 0.0
            )
            draw_rate = draws / completed

            sims_str = f"{sims_rate / 1000:.1f}k" if sims_rate >= 1000 else f"{sims_rate:.0f}"
            nn_str = ""
            if total_nn_calls > 0:
                nn_latency = total_nn_time_s / total_nn_calls * 1000
                nn_str = f" | nn {nn_latency:.1f}ms"
            print(
                f"[{completed}/{num_games}] "
                f"{sims_str} sims/s | {pos_rate:.0f} pos/s{nn_str} | "
                f"{games_rate:.1f} games/s | "
                f"W/D/L {p1_wins}/{draws}/{p2_wins} ({draw_rate:.0%} draws) | "
                f"cheese {cheese_util:.0%}"
            )

    # Send sentinel values to stop workers
    for _ in range(num_workers):
        work_queue.put(None)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    elapsed_seconds = time.perf_counter() - start_time

    # Register batch in manifest now that sampling succeeded
    exp.register_batch(
        group=config.group,
        batch_uuid=batch_uuid,
        mcts_config=config.mcts,
        game=config.game,
        checkpoint_path=config.checkpoint,
    )

    metrics = SamplingMetrics(
        total_games=num_games,
        total_positions=total_positions,
        total_simulations=total_simulations,
        elapsed_seconds=elapsed_seconds,
        workers=num_workers,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        total_cheese_collected=total_cheese_collected,
        total_cheese_available=total_cheese_available,
        total_turns=total_turns,
        min_turns=int(min_turns) if min_turns != float("inf") else 0,
        max_turns=max_turns,
        total_nn_calls=total_nn_calls,
        total_nn_time_s=total_nn_time_s,
    )

    if verbose:
        print()
        print(metrics)

    return batch_dir, metrics
