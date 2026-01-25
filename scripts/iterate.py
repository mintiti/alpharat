#!/usr/bin/env python3
"""Auto-iteration script for AlphaZero training loop.

Runs the full AlphaZero iteration loop:
    Sample -> Shard -> Train -> (Benchmark) -> repeat

with automatic lineage tracking via ExperimentManager.

Usage:
    # Run forever (Ctrl+C to stop)
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5

    # Run exactly 3 iterations
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --iterations 3

    # Start with an existing checkpoint
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --start-checkpoint path/to/model.pt

    # Resume from iteration 2
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --start-iteration 2

    # Benchmark every 2nd iteration
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --benchmark-every 2

    # Skip all benchmarks
    alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --no-benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import Field

from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.config.loader import load_config
from alpharat.mcts import MCTSConfig  # noqa: TC001
from alpharat.nn.config import ModelConfig, OptimConfig  # noqa: TC001

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Config Models ---


class IterationParams(StrictBaseModel):
    """Per-iteration settings."""

    games: int = 10000
    epochs: int = 100
    checkpoint_every: int = 10


class IterateSamplingParams(StrictBaseModel):
    """Sampling-specific settings for iteration."""

    workers: int = 4


class ShardingParams(StrictBaseModel):
    """Sharding-specific settings."""

    val_ratio: float = 0.1
    positions_per_shard: int = 3_000_000


class BenchmarkParams(StrictBaseModel):
    """Benchmark-specific settings.

    Note: MCTS config is inherited from the top-level `mcts` field,
    ensuring consistency between sampling and benchmarking.
    """

    games_per_matchup: int = 50
    workers: int = 4


class IterateConfig(StrictBaseModel):
    """Full configuration for an iteration run."""

    game: GameConfig
    mcts: MCTSConfig = Field(discriminator="variant")
    model: ModelConfig = Field(discriminator="architecture")
    optim: OptimConfig = Field(discriminator="architecture")
    iteration: IterationParams = Field(default_factory=IterationParams)
    sampling: IterateSamplingParams = Field(default_factory=IterateSamplingParams)
    sharding: ShardingParams = Field(default_factory=ShardingParams)
    benchmark: BenchmarkParams = Field(default_factory=BenchmarkParams)


# --- Phase Functions ---


def run_sampling_phase(
    config: IterateConfig,
    batch_group: str,
    checkpoint_path: Path | None,
    experiments_dir: Path,
    device: str,
) -> Path:
    """Run the sampling phase.

    Args:
        config: Iteration config.
        batch_group: Name for the batch group.
        checkpoint_path: Path to checkpoint for NN-guided sampling, or None.
        experiments_dir: Experiments directory.
        device: Device for NN inference.

    Returns:
        Path to the created batch directory.
    """
    from alpharat.data.sampling import SamplingConfig, SamplingParams, run_sampling

    sampling_config = SamplingConfig(
        mcts=config.mcts,
        game=config.game,
        sampling=SamplingParams(
            num_games=config.iteration.games,
            workers=config.sampling.workers,
            device=device,
        ),
        group=batch_group,
        experiments_dir=str(experiments_dir),
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
    )

    batch_dir, _metrics = run_sampling(sampling_config)
    return batch_dir


def run_sharding_phase(
    config: IterateConfig,
    shard_group: str,
    batch_group: str,
    experiments_dir: Path,
) -> str:
    """Run the sharding phase.

    Args:
        config: Iteration config.
        shard_group: Name for the shard group.
        batch_group: Name of the batch group to process.
        experiments_dir: Experiments directory.

    Returns:
        Shard ID in "group/uuid" format.
    """
    from alpharat.data.sharding import prepare_training_set_with_split
    from alpharat.experiments import ExperimentManager
    from alpharat.experiments.paths import get_shards_dir

    exp = ExperimentManager(experiments_dir)

    # Get batch directories for this group
    all_batches = exp.list_batches()
    batch_ids = [b for b in all_batches if b.startswith(f"{batch_group}/")]
    if not batch_ids:
        raise ValueError(f"No batches found for group '{batch_group}'")

    batch_dirs = [exp.get_batch_path(bid) for bid in batch_ids]
    logger.info(f"Processing {len(batch_ids)} batches from group '{batch_group}'")

    # Get dimensions from first batch
    width, height = _get_dimensions_from_batch(batch_dirs[0])

    # Build observation builder from model config
    builder = config.model.build_observation_builder(width, height)

    # Prepare shards
    shards_group_dir = get_shards_dir(exp.root) / shard_group
    shards_group_dir.mkdir(parents=True, exist_ok=True)

    result = prepare_training_set_with_split(
        batch_dirs=batch_dirs,
        output_dir=shards_group_dir,
        builder=builder,
        val_ratio=config.sharding.val_ratio,
        positions_per_shard=config.sharding.positions_per_shard,
        seed=42,
    )

    # Register in manifest
    exp.register_shards(
        group=shard_group,
        shard_uuid=result.shard_id,
        source_batches=batch_ids,
        total_positions=result.total_positions,
        train_positions=result.train_positions,
        val_positions=result.val_positions,
        shuffle_seed=42,
    )

    shard_id = f"{shard_group}/{result.shard_id}"
    logger.info(
        f"Created shards: {shard_id} (train={result.train_positions}, val={result.val_positions})"
    )
    return shard_id


def run_training_phase(
    config: IterateConfig,
    run_name: str,
    shard_id: str,
    experiments_dir: Path,
    device: str,
    resume_from: Path | None = None,
) -> Path:
    """Run the training phase.

    Args:
        config: Iteration config.
        run_name: Name for this training run.
        shard_id: Shard ID in "group/uuid" format.
        experiments_dir: Experiments directory.
        device: Device for training.
        resume_from: Optional checkpoint to resume from.

    Returns:
        Path to best model checkpoint.
    """
    from alpharat.experiments import ExperimentManager
    from alpharat.nn.config import TrainConfig
    from alpharat.nn.training import run_training
    from alpharat.nn.training.base import DataConfig

    exp = ExperimentManager(experiments_dir)
    shard_path = exp.get_shard_path(shard_id)

    # Build TrainConfig
    train_config = TrainConfig(
        name=run_name,
        model=config.model,
        optim=config.optim,
        data=DataConfig(
            train_dir=str(shard_path / "train"),
            val_dir=str(shard_path / "val"),
        ),
        seed=42,
        resume_from=str(resume_from) if resume_from else None,
    )

    # Create run via ExperimentManager
    run_dir = exp.create_run(
        name=run_name,
        config=train_config.model_dump(),
        source_shards=shard_id,
        parent_checkpoint=str(resume_from) if resume_from else None,
    )
    actual_run_name = run_dir.name

    logger.info(f"Training run: {actual_run_name}")

    checkpoint_path = run_training(
        train_config,
        epochs=config.iteration.epochs,
        checkpoint_every=config.iteration.checkpoint_every,
        device=device,
        output_dir=run_dir.parent,
        run_name=actual_run_name,
        checkpoints_subdir="checkpoints",
    )

    return checkpoint_path


def run_benchmark_phase(
    config: IterateConfig,
    benchmark_name: str,
    checkpoint_path: Path,
    previous_checkpoint: Path | None,
    experiments_dir: Path,
    device: str,
) -> None:
    """Run the benchmark phase.

    Args:
        config: Iteration config.
        benchmark_name: Name for this benchmark.
        checkpoint_path: Path to checkpoint to evaluate.
        previous_checkpoint: Optional path to previous iteration's checkpoint.
        experiments_dir: Experiments directory.
        device: Device for inference.
    """
    from alpharat.eval.benchmark import BenchmarkConfig, build_benchmark_tournament
    from alpharat.eval.elo import compute_elo, from_tournament_result
    from alpharat.eval.tournament import run_tournament
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager(experiments_dir)

    # Use the same MCTS config as sampling for consistency
    benchmark_config = BenchmarkConfig(
        games_per_matchup=config.benchmark.games_per_matchup,
        workers=config.benchmark.workers,
        device=device,
        mcts=config.mcts,
    )

    tournament_config = build_benchmark_tournament(
        benchmark_name,
        checkpoint_path,
        benchmark_config,
        game_config=config.game,
        baseline_checkpoint=previous_checkpoint,
    )

    # Create benchmark via ExperimentManager
    exp.create_benchmark(
        name=benchmark_name,
        config=tournament_config.model_dump(),
        checkpoints=[str(checkpoint_path)],
    )
    logger.info(f"Created benchmark: {benchmark_name}")

    result = run_tournament(tournament_config)

    # Save results
    results_dict = {
        "standings": result.standings(),
        "wdl_matrix": {
            agent: {
                opp: {"wins": wdl[0], "draws": wdl[1], "losses": wdl[2]}
                for opp, wdl in opps.items()
            }
            for agent, opps in result.wdl_matrix().items()
        },
        "cheese_stats": {
            agent: {
                opp: {"scored": cheese[0], "conceded": cheese[1]} for opp, cheese in opps.items()
            }
            for agent, opps in result.cheese_matrix().items()
        },
    }
    exp.save_benchmark_results(benchmark_name, results_dict)

    # Print results
    print()
    print(result.standings_table())
    print()
    print(result.wdl_table())
    print()
    print(result.cheese_table())
    print()

    # Compute and print Elo ratings
    records = from_tournament_result(result)
    elo_result = compute_elo(records, anchor="greedy", anchor_elo=1000, compute_uncertainty=True)
    print(elo_result.format_table())


# --- Helper Functions ---


def _get_dimensions_from_batch(batch_dir: Path) -> tuple[int, int]:
    """Extract width/height from batch metadata."""
    metadata_path = batch_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"No metadata.json in {batch_dir}")

    metadata = json.loads(metadata_path.read_text())
    game_config = metadata["game"]
    return game_config["width"], game_config["height"]


def _artifact_exists(exp_dir: Path, artifact_type: str, name: str) -> bool:
    """Check if an artifact already exists.

    Args:
        exp_dir: Experiments directory.
        artifact_type: One of "batches", "shards", "runs".
        name: Artifact name or group to check.

    Returns:
        True if the artifact exists.
    """
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager(exp_dir)

    if artifact_type == "batches":
        return any(b.startswith(f"{name}/") for b in exp.list_batches())
    elif artifact_type == "shards":
        return any(s.startswith(f"{name}/") for s in exp.list_shards())
    elif artifact_type == "runs":
        return name in exp.list_runs()
    return False


def _get_best_checkpoint(exp_dir: Path, run_name: str) -> Path | None:
    """Get the best checkpoint for a run if it exists."""
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager(exp_dir)
    if run_name not in exp.list_runs():
        return None

    checkpoint_path = exp.get_run_checkpoints_path(run_name) / "best_model.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    return None


# --- Main Loop ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlphaZero iteration loop: Sample -> Shard -> Train -> Benchmark"
    )
    parser.add_argument("config", type=str, help="Path to iteration YAML config")
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Naming prefix for artifacts (e.g., 'mlp_5x5')",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=-1,
        help="Number of iterations (-1 = unlimited, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--start-checkpoint",
        type=Path,
        default=None,
        help="Initial checkpoint for iteration 0 sampling",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Resume from this iteration number",
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=1,
        help="Benchmark frequency (0 = never, 1 = every iteration)",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip all benchmarks",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for training/inference (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()

    # Parse config path
    config_path = Path(args.config)
    config_name = str(config_path.with_suffix(""))
    if config_name.startswith("configs/"):
        config_dir = "configs"
        config_name = config_name[len("configs/") :]
    else:
        config_dir = str(config_path.parent) if config_path.parent.name else "."
        config_name = config_path.stem

    # Load config
    config = load_config(IterateConfig, config_dir, config_name)

    prefix = args.prefix
    experiments_dir = args.experiments_dir
    device = args.device

    # Determine benchmark frequency
    benchmark_every = 0 if args.no_benchmark else args.benchmark_every

    logger.info("=" * 60)
    logger.info("AlphaZero Iteration Loop")
    logger.info("=" * 60)
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Game: {config.game.width}x{config.game.height}")
    logger.info(f"MCTS: {config.mcts.variant}, {config.mcts.simulations} sims")
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Games per iteration: {config.iteration.games}")
    logger.info(f"Epochs per iteration: {config.iteration.epochs}")
    logger.info(f"Benchmark every: {benchmark_every if benchmark_every > 0 else 'never'}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Initialize state
    current_checkpoint = args.start_checkpoint
    start_iteration = args.start_iteration
    max_iterations = args.iterations if args.iterations > 0 else float("inf")

    iteration = start_iteration
    previous_checkpoint: Path | None = None

    try:
        while iteration < max_iterations:
            iter_name = f"{prefix}_iter{iteration}"

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"ITERATION {iteration}")
            logger.info("=" * 60)

            # --- Phase 1: Sampling ---
            batch_group = iter_name
            if _artifact_exists(experiments_dir, "batches", batch_group):
                logger.info(f"Batch group '{batch_group}' exists, skipping sampling")
            else:
                logger.info("")
                logger.info(f"Phase 1: Sampling ({config.iteration.games} games)")
                logger.info("-" * 40)
                if current_checkpoint:
                    logger.info(f"Using checkpoint: {current_checkpoint}")
                else:
                    logger.info("Using uniform priors (no checkpoint)")

                run_sampling_phase(
                    config,
                    batch_group,
                    current_checkpoint,
                    experiments_dir,
                    device,
                )

            # --- Phase 2: Sharding ---
            shard_group = f"{iter_name}_shards"
            if _artifact_exists(experiments_dir, "shards", shard_group):
                logger.info(f"Shard group '{shard_group}' exists, skipping sharding")
                # Need to find the shard ID
                from alpharat.experiments import ExperimentManager

                exp = ExperimentManager(experiments_dir)
                shard_ids = [s for s in exp.list_shards() if s.startswith(f"{shard_group}/")]
                if not shard_ids:
                    raise ValueError(f"No shards found for group '{shard_group}'")
                shard_id = shard_ids[0]
            else:
                logger.info("")
                logger.info("Phase 2: Sharding")
                logger.info("-" * 40)
                shard_id = run_sharding_phase(
                    config,
                    shard_group,
                    batch_group,
                    experiments_dir,
                )

            # --- Phase 3: Training ---
            run_name = iter_name
            existing_checkpoint = _get_best_checkpoint(experiments_dir, run_name)
            if existing_checkpoint:
                logger.info(f"Run '{run_name}' exists, skipping training")
                checkpoint_path = existing_checkpoint
            else:
                logger.info("")
                logger.info(f"Phase 3: Training ({config.iteration.epochs} epochs)")
                logger.info("-" * 40)
                checkpoint_path = run_training_phase(
                    config,
                    run_name,
                    shard_id,
                    experiments_dir,
                    device,
                    resume_from=current_checkpoint,
                )

            logger.info(f"Checkpoint: {checkpoint_path}")

            # --- Phase 4: Benchmark (optional) ---
            if benchmark_every > 0 and (iteration + 1) % benchmark_every == 0:
                benchmark_name = f"{iter_name}_benchmark"
                # Check if benchmark exists
                from alpharat.experiments import ExperimentManager

                exp = ExperimentManager(experiments_dir)
                if benchmark_name in exp.list_benchmarks():
                    logger.info(f"Benchmark '{benchmark_name}' exists, skipping")
                else:
                    logger.info("")
                    logger.info("Phase 4: Benchmark")
                    logger.info("-" * 40)
                    run_benchmark_phase(
                        config,
                        benchmark_name,
                        checkpoint_path,
                        previous_checkpoint,
                        experiments_dir,
                        device,
                    )

            # Update state for next iteration
            previous_checkpoint = current_checkpoint
            current_checkpoint = checkpoint_path
            iteration += 1

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Interrupted at iteration {iteration}")
        logger.info("=" * 60)
        sys.exit(0)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Completed {iteration - start_iteration} iterations")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
