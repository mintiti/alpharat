"""Self-play sampling for training data generation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any

import numpy as np

from alpharat.config.base import StrictBaseModel
from alpharat.config.checkpoint import make_predict_fn
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.data.recorder import GameBundler, GameRecorder
from alpharat.eval.game import is_terminal
from alpharat.mcts import DecoupledPUCTConfig  # noqa: TC001
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyrat_engine.core.game import PyRat

    from alpharat.data.types import GameData


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

    mcts: DecoupledPUCTConfig
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
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"SamplingMetrics(\n"
            f"  Throughput:\n"
            f"    workers={self.workers}, elapsed={self.elapsed_seconds:.2f}s\n"
            f"    games={self.total_games} ({self.games_per_second:.2f}/s)\n"
            f"    positions={self.total_positions} ({self.positions_per_second:.2f}/s)\n"
            f"    simulations={self.total_simulations} ({self.simulations_per_second:.2f}/s)\n"
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


def build_tree(
    game: PyRat,
    gamma: float,
    predict_fn: Callable[[Any], tuple[np.ndarray, np.ndarray, float, float]] | None = None,
) -> MCTSTree:
    """Build fresh MCTS tree for search.

    The tree will initialize the root with smart uniform priors that only
    assign probability to distinct effective actions (blocked moves get 0),
    or use NN predictions if predict_fn is provided.

    Args:
        game: Current game state (will be deep-copied for simulation).
        gamma: Discount factor for value backup.
        predict_fn: Optional NN prediction function for priors.

    Returns:
        MCTSTree ready for search.
    """
    simulator = copy.deepcopy(game)

    # Dummy priors - tree will overwrite with smart uniform via _init_root_priors()
    # or use NN priors if predict_fn is provided
    dummy = np.ones(5) / 5

    root = MCTSNode(
        game_state=None,
        prior_policy_p1=dummy,
        prior_policy_p2=dummy,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=simulator.player1_mud_turns,
        p2_mud_turns_remaining=simulator.player2_mud_turns,
    )

    return MCTSTree(game=simulator, root=root, gamma=gamma, predict_fn=predict_fn)


def create_game(config: GameConfig, seed: int) -> PyRat:
    """Create a new game from configuration.

    Args:
        config: Game configuration.
        seed: Random seed for maze generation.

    Returns:
        Configured PyRat game instance.
    """
    return config.build(seed)


# --- Core Sampling ---


def play_and_record_game(
    config: SamplingConfig,
    games_dir: Path,
    seed: int,
    nn_ctx: NNContext | None = None,
    *,
    auto_save: bool = True,
) -> tuple[GameStats, GameData | None]:
    """Play one game with MCTS and record all positions.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save the game file (used even if auto_save=False
            since GameRecorder requires it for output_dir validation).
        seed: Random seed for game generation.
        nn_ctx: Optional NN context for guided search.
        auto_save: If True (default), save to disk. If False, return GameData
            for external handling (e.g., bundling).

    Returns:
        Tuple of (GameStats, GameData or None). GameData is only returned if
        auto_save=False.
    """
    game = create_game(config.game, seed)
    cheese_available = config.game.cheese_count
    positions = 0
    total_simulations = 0

    with GameRecorder(
        game, games_dir, config.game.width, config.game.height, auto_save=auto_save
    ) as recorder:
        while not is_terminal(game):
            # Create predict_fn if NN context available
            # Note: build_tree deep-copies the game, so we need to create predict_fn
            # for the copy, not the original. We pass None here and let build_tree
            # handle it, or we restructure to pass nn_ctx.
            # Actually, we need to create predict_fn AFTER build_tree gets the simulator.
            # Let's restructure build_tree to handle this.
            predict_fn = None
            if nn_ctx is not None:
                # We need the simulator from inside build_tree, so we'll inline
                # the tree creation here to capture the simulator reference
                simulator = copy.deepcopy(game)
                predict_fn = make_predict_fn(
                    nn_ctx.model,
                    nn_ctx.builder,
                    simulator,
                    nn_ctx.width,
                    nn_ctx.height,
                    nn_ctx.device,
                )
                dummy = np.ones(5) / 5
                root = MCTSNode(
                    game_state=None,
                    prior_policy_p1=dummy,
                    prior_policy_p2=dummy,
                    nn_value_p1=0.0,
                    nn_value_p2=0.0,
                    parent=None,
                    p1_mud_turns_remaining=simulator.player1_mud_turns,
                    p2_mud_turns_remaining=simulator.player2_mud_turns,
                )
                tree = MCTSTree(
                    game=simulator, root=root, gamma=config.mcts.gamma, predict_fn=predict_fn
                )
            else:
                tree = build_tree(game, config.mcts.gamma)

            search = config.mcts.build(tree)
            result = search.search()

            positions += 1
            total_simulations += config.mcts.simulations

            # Sample actions from search policies
            a1 = select_action_from_strategy(result.policy_p1)
            a2 = select_action_from_strategy(result.policy_p2)

            # Record position before making move
            visit_counts_p1, visit_counts_p2 = tree.root.get_marginal_visits_expanded()
            recorder.record_position(
                game=game,
                search_result=result,
                prior_p1=tree.root.prior_policy_p1,
                prior_p2=tree.root.prior_policy_p2,
                visit_counts_p1=visit_counts_p1,
                visit_counts_p2=visit_counts_p2,
                action_p1=a1,
                action_p2=a2,
            )

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
        simulations=total_simulations,
        winner=winner,
        score_p1=score_p1,
        score_p2=score_p2,
        turns=game.turn,
        cheese_available=cheese_available,
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

    Runs until it receives None (sentinel) from the work queue. If a checkpoint
    is configured, loads the model once at startup and uses it for all games.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save game files.
        work_queue: Queue of seeds to process (None = stop).
        results_queue: Queue to send completed game stats.
        device: Device for NN inference (if checkpoint configured).
        use_bundler: If True (default), buffer games and write bundles. If False,
            write individual game files (legacy behavior).
    """
    # Load NN model once per worker if checkpoint configured
    nn_ctx: NNContext | None = None
    if config.checkpoint is not None:
        nn_ctx = load_nn_context(config.checkpoint, device=device)

    # Create bundler for this worker if enabled
    bundler: GameBundler | None = None
    if use_bundler:
        bundler = GameBundler(
            output_dir=games_dir,
            width=config.game.width,
            height=config.game.height,
        )

    while True:
        seed = work_queue.get()
        if seed is None:
            # Sentinel received, flush bundler and exit
            if bundler is not None:
                bundler.flush()
            break

        game_stats, game_data = play_and_record_game(
            config, games_dir, seed, nn_ctx=nn_ctx, auto_save=not use_bundler
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
    batch_dir = exp.create_batch(
        group=config.group,
        mcts_config=config.mcts,
        game=config.game,
        checkpoint_path=config.checkpoint,
        seed_start=0,  # Games use seeds 0, 1, 2, ... N
    )
    games_dir = batch_dir / "games"

    if verbose:
        print(f"Sampling {config.sampling.num_games} games")
        print(f"  Group: {config.group}")
        print(f"  MCTS: {config.mcts.simulations} sims")
        print(
            f"  Game: {config.game.width}x{config.game.height}, {config.game.cheese_count} cheese"
        )
        print(f"  Workers: {config.sampling.workers}")
        if config.checkpoint:
            print(f"  Checkpoint: {config.checkpoint}")
            print(f"  Device: {config.sampling.device}")
        print(f"  Output: {batch_dir}")
        print()

    num_workers = config.sampling.workers
    num_games = config.sampling.num_games

    # Aggregation accumulators
    total_positions = 0
    total_simulations = 0
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

    # Track how many seeds we've submitted and results collected
    next_seed = 0
    completed = 0

    # Initially fill the queue (non-blocking since queue_size > 0)
    while next_seed < num_games and next_seed < queue_size:
        work_queue.put(next_seed)
        next_seed += 1

    # Collect results and submit more work as slots free up
    while completed < num_games:
        game_stats = results_queue.get()
        completed += 1

        # Submit next seed if any remain
        if next_seed < num_games:
            work_queue.put(next_seed)
            next_seed += 1

        # Throughput stats
        total_positions += game_stats.positions
        total_simulations += game_stats.simulations

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
            rate = completed / elapsed
            cheese_util = (
                total_cheese_collected / total_cheese_available
                if total_cheese_available > 0
                else 0.0
            )
            draw_rate = draws / completed
            avg_turns = total_turns / completed
            print(
                f"[{completed}/{num_games}] "
                f"{rate:.1f} games/s | "
                f"W/D/L {p1_wins}/{draws}/{p2_wins} ({draw_rate:.0%} draws) | "
                f"cheese {cheese_util:.0%} | "
                f"avg turns {avg_turns:.1f}"
            )

    # Send sentinel values to stop workers
    for _ in range(num_workers):
        work_queue.put(None)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    elapsed_seconds = time.perf_counter() - start_time

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
    )

    if verbose:
        print()
        print(metrics)

    return batch_dir, metrics
