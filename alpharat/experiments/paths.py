"""Path constants and helpers for experiments folder structure.

The experiments folder is NOT in git (added to .gitignore). It contains:
- batches/: Raw game recordings from sampling
- shards/: Processed train/val splits
- runs/: Training runs with checkpoints
- benchmarks/: Tournament/benchmark results
- manifest.yaml: Central index tracking all artifacts
"""

from __future__ import annotations

from pathlib import Path

# Default experiments root (relative to repo root)
DEFAULT_EXPERIMENTS_DIR = "experiments"

# Subfolder names
BATCHES_DIR = "batches"
SHARDS_DIR = "shards"
RUNS_DIR = "runs"
BENCHMARKS_DIR = "benchmarks"

# Standard filenames
MANIFEST_FILE = "manifest.yaml"
METADATA_FILE = "metadata.json"
CONFIG_FILE = "config.yaml"
NOTES_FILE = "notes.txt"
RESULTS_FILE = "results.json"
SHARD_MANIFEST_FILE = "manifest.json"
CHECKPOINTS_DIR = "checkpoints"
GAMES_DIR = "games"
TRAIN_DIR = "train"
VAL_DIR = "val"

# CLAUDE.md files for each subfolder
CLAUDE_MD = "CLAUDE.md"


def get_experiments_root(base_dir: Path | str | None = None) -> Path:
    """Get the experiments root directory.

    Args:
        base_dir: Override for experiments directory. If None, uses DEFAULT_EXPERIMENTS_DIR
                  relative to the current working directory.

    Returns:
        Path to experiments root.
    """
    if base_dir is not None:
        return Path(base_dir)
    return Path(DEFAULT_EXPERIMENTS_DIR)


def get_batches_dir(experiments_root: Path) -> Path:
    """Get the batches directory."""
    return experiments_root / BATCHES_DIR


def get_shards_dir(experiments_root: Path) -> Path:
    """Get the shards directory."""
    return experiments_root / SHARDS_DIR


def get_runs_dir(experiments_root: Path) -> Path:
    """Get the runs directory."""
    return experiments_root / RUNS_DIR


def get_benchmarks_dir(experiments_root: Path) -> Path:
    """Get the benchmarks directory."""
    return experiments_root / BENCHMARKS_DIR


def get_manifest_path(experiments_root: Path) -> Path:
    """Get the manifest.yaml path."""
    return experiments_root / MANIFEST_FILE


def get_batch_path(experiments_root: Path, group: str, batch_uuid: str) -> Path:
    """Get path to a specific batch directory.

    Args:
        experiments_root: Root experiments directory.
        group: Batch group name (e.g., "uniform_5x5").
        batch_uuid: UUID of the batch.

    Returns:
        Path to batch directory: experiments/batches/{group}/{uuid}/
    """
    return get_batches_dir(experiments_root) / group / batch_uuid


def get_shard_path(experiments_root: Path, group: str, shard_uuid: str) -> Path:
    """Get path to a specific shard directory.

    Args:
        experiments_root: Root experiments directory.
        group: Shard group name.
        shard_uuid: UUID of the shard set.

    Returns:
        Path to shard directory: experiments/shards/{group}/{uuid}/
    """
    return get_shards_dir(experiments_root) / group / shard_uuid


def get_run_path(experiments_root: Path, run_name: str) -> Path:
    """Get path to a specific run directory.

    Args:
        experiments_root: Root experiments directory.
        run_name: Human-readable run name.

    Returns:
        Path to run directory: experiments/runs/{run_name}/
    """
    return get_runs_dir(experiments_root) / run_name


def get_benchmark_path(experiments_root: Path, benchmark_name: str) -> Path:
    """Get path to a specific benchmark directory.

    Args:
        experiments_root: Root experiments directory.
        benchmark_name: Human-readable benchmark name.

    Returns:
        Path to benchmark directory: experiments/benchmarks/{benchmark_name}/
    """
    return get_benchmarks_dir(experiments_root) / benchmark_name


def batch_id_from_path(batch_path: Path) -> str:
    """Extract batch ID (group/uuid) from batch path.

    Args:
        batch_path: Path to batch directory.

    Returns:
        Batch ID in format "group/uuid".
    """
    return f"{batch_path.parent.name}/{batch_path.name}"


def shard_id_from_path(shard_path: Path) -> str:
    """Extract shard ID (group/uuid) from shard path.

    Args:
        shard_path: Path to shard directory.

    Returns:
        Shard ID in format "group/uuid".
    """
    return f"{shard_path.parent.name}/{shard_path.name}"


def parse_batch_id(batch_id: str) -> tuple[str, str]:
    """Parse batch ID into group and uuid.

    Args:
        batch_id: Batch ID in format "group/uuid".

    Returns:
        Tuple of (group, uuid).

    Raises:
        ValueError: If batch_id is not in expected format.
    """
    parts = batch_id.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid batch_id format: {batch_id!r}. Expected 'group/uuid'.")
    return parts[0], parts[1]
