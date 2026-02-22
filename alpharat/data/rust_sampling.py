"""Rust self-play sampling pipeline wrapper.

Calls the Rust self-play pipeline (via alpharat_sampling) and handles
ExperimentManager integration, ONNX auto-export, and progress display.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpharat.config.game import GameConfig
    from alpharat.mcts.config import RustMCTSConfig

logger = logging.getLogger(__name__)


@dataclass
class RustSamplingMetrics:
    """Metrics from a Rust sampling run."""

    total_games: int
    total_positions: int
    total_simulations: int
    elapsed_seconds: float
    p1_wins: int
    p2_wins: int
    draws: int
    total_cheese_collected: float
    total_cheese_available: int
    min_turns: int
    max_turns: int
    total_nn_evals: int
    total_terminals: int
    total_collisions: int

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
        return self.total_positions / self.total_games if self.total_games > 0 else 0.0

    @property
    def cheese_utilization(self) -> float:
        if self.total_cheese_available == 0:
            return 0.0
        return self.total_cheese_collected / self.total_cheese_available

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0

    @property
    def nn_evals_per_second(self) -> float:
        return self.total_nn_evals / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def nn_eval_fraction(self) -> float:
        return self.total_nn_evals / self.total_simulations if self.total_simulations > 0 else 0.0

    @property
    def terminal_fraction(self) -> float:
        return self.total_terminals / self.total_simulations if self.total_simulations > 0 else 0.0

    @property
    def collision_fraction(self) -> float:
        total = self.total_nn_evals + self.total_terminals + self.total_collisions
        return self.total_collisions / total if total > 0 else 0.0


def _ensure_onnx(checkpoint_path: str) -> str | None:
    """Return path to ONNX model, auto-exporting from .pt if needed.

    Returns None if checkpoint_path is None.
    """
    pt_path = Path(checkpoint_path)
    onnx_path = pt_path.with_suffix(".onnx")

    if onnx_path.exists():
        logger.info("Using existing ONNX model: %s", onnx_path)
        return str(onnx_path)

    logger.info("Exporting ONNX model from %s ...", pt_path)
    from scripts.export_onnx import export_onnx

    result = export_onnx(pt_path, onnx_path)
    return str(result)


def run_rust_sampling(
    *,
    game: GameConfig,
    mcts: RustMCTSConfig,
    num_games: int,
    group: str,
    num_threads: int = 4,
    max_games_per_bundle: int = 32,
    mux_max_batch_size: int = 256,
    checkpoint: str | None = None,
    device: str = "auto",
    experiments_dir: str | Path = "experiments",
    verbose: bool = True,
) -> tuple[Path, RustSamplingMetrics]:
    """Run Rust self-play pipeline with ExperimentManager integration.

    Creates a batch via ExperimentManager, optionally auto-exports ONNX,
    runs multi-threaded Rust self-play, and registers the batch on success.

    Args:
        game: Game configuration (width, height, cheese, etc.).
        mcts: Rust MCTS configuration (simulations, c_puct, etc.).
        num_games: Total games to generate.
        group: Batch group name for ExperimentManager.
        num_threads: Worker threads for Rust self-play.
        max_games_per_bundle: Max games per NPZ bundle file.
        mux_max_batch_size: Max batch size for ONNX mux backend.
        checkpoint: Path to .pt checkpoint for NN-guided sampling.
        device: ONNX execution provider — "auto", "cpu", "coreml", "mps", "cuda".
        experiments_dir: Experiments root directory.
        verbose: Show progress bar.

    Returns:
        Tuple of (batch_dir, metrics).
    """
    from alpharat_sampling import SelfPlayProgress, rust_self_play

    from alpharat.experiments import ExperimentManager

    experiments_dir = Path(experiments_dir)
    exp = ExperimentManager(experiments_dir)

    # Prepare batch directory
    batch_dir, batch_uuid = exp.prepare_batch(
        group=group,
        mcts_config=mcts,
        game=game,
        checkpoint_path=checkpoint,
    )
    output_dir = batch_dir / "games"

    # Auto-export ONNX if checkpoint provided
    onnx_path: str | None = None
    if checkpoint is not None:
        onnx_path = _ensure_onnx(checkpoint)

    kwargs = {
        "width": game.width,
        "height": game.height,
        "cheese_count": game.cheese_count,
        "max_turns": game.max_turns,
        "num_games": num_games,
        "symmetric": game.symmetric,
        "wall_density": game.wall_density,
        "mud_density": game.mud_density,
        "simulations": mcts.simulations,
        "batch_size": mcts.batch_size,
        "c_puct": mcts.c_puct,
        "fpu_reduction": mcts.fpu_reduction,
        "force_k": mcts.force_k,
        "noise_epsilon": mcts.noise_epsilon,
        "noise_concentration": mcts.noise_concentration,
        "max_collisions": mcts.max_collisions,
        "num_threads": num_threads,
        "output_dir": str(output_dir),
        "max_games_per_bundle": max_games_per_bundle,
        "onnx_model_path": onnx_path,
        "device": device,
        "mux_max_batch_size": mux_max_batch_size,
    }

    if verbose:
        stats = _run_with_progress(rust_self_play, kwargs, num_games, SelfPlayProgress)
    else:
        stats = rust_self_play(**kwargs)

    # Register batch on success
    exp.register_batch(
        group=group,
        batch_uuid=batch_uuid,
        mcts_config=mcts,
        game=game,
        checkpoint_path=checkpoint,
    )

    metrics = RustSamplingMetrics(
        total_games=stats.total_games,
        total_positions=stats.total_positions,
        total_simulations=stats.total_simulations,
        elapsed_seconds=stats.elapsed_secs,
        p1_wins=stats.p1_wins,
        p2_wins=stats.p2_wins,
        draws=stats.draws,
        total_cheese_collected=stats.total_cheese_collected,
        total_cheese_available=stats.total_cheese_available,
        min_turns=stats.min_turns,
        max_turns=stats.max_turns,
        total_nn_evals=stats.total_nn_evals,
        total_terminals=stats.total_terminals,
        total_collisions=stats.total_collisions,
    )

    logger.info(
        "Rust self-play complete: %d games, %d positions, "
        "%.0f sims/s (%.0f nn_evals/s, %.0f%% nn, %.1fs)",
        metrics.total_games,
        metrics.total_positions,
        metrics.simulations_per_second,
        metrics.nn_evals_per_second,
        metrics.nn_eval_fraction * 100,
        metrics.elapsed_seconds,
    )

    return batch_dir, metrics


def _run_with_progress(
    rust_self_play_fn: Any,
    kwargs: dict[str, Any],
    num_games: int,
    progress_cls: type[Any],
) -> Any:
    """Run rust_self_play in a background thread with tqdm progress."""
    from tqdm import tqdm

    progress = progress_cls()
    kwargs["progress"] = progress

    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def _worker() -> None:
        try:
            result_holder.append(rust_self_play_fn(**kwargs))
        except BaseException as e:
            error_holder.append(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    with tqdm(total=num_games, desc="Rust self-play", unit="game") as pbar:
        while thread.is_alive():
            completed = progress.games_completed
            pbar.n = completed
            pbar.refresh()
            time.sleep(0.2)
        # Final update
        pbar.n = progress.games_completed
        pbar.refresh()

    thread.join()

    if error_holder:
        raise error_holder[0]
    return result_holder[0]
