"""ExperimentManager: Central API for managing experiment artifacts.

Usage:
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager()
    batch_dir = exp.create_batch(group="uniform_5x5", mcts_config=..., game_params=...)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

import yaml

from alpharat.experiments import paths, templates
from alpharat.experiments.schema import (
    BatchEntry,
    BenchmarkEntry,
    Manifest,
    RunEntry,
    ShardEntry,
)

if TYPE_CHECKING:
    from alpharat.data.batch import GameParams
    from alpharat.mcts import MCTSConfig


class ExperimentManager:
    """Central API for managing experiment artifacts.

    Handles creation of batches, shards, runs, and benchmarks with automatic
    manifest updates and folder structure initialization.

    Args:
        experiments_dir: Root directory for experiments. Defaults to "experiments/".
    """

    def __init__(self, experiments_dir: Path | str | None = None) -> None:
        self.root = paths.get_experiments_root(experiments_dir)

    def init(self) -> None:
        """Initialize the experiments folder structure.

        Creates all subdirectories and CLAUDE.md files if they don't exist.
        Called automatically by other methods when needed.
        """
        # Create directories
        self.root.mkdir(parents=True, exist_ok=True)
        paths.get_batches_dir(self.root).mkdir(exist_ok=True)
        paths.get_shards_dir(self.root).mkdir(exist_ok=True)
        paths.get_runs_dir(self.root).mkdir(exist_ok=True)
        paths.get_benchmarks_dir(self.root).mkdir(exist_ok=True)

        # Write CLAUDE.md files if they don't exist
        self._write_if_missing(self.root / paths.CLAUDE_MD, templates.EXPERIMENTS_CLAUDE_MD)
        self._write_if_missing(
            paths.get_batches_dir(self.root) / paths.CLAUDE_MD,
            templates.BATCHES_CLAUDE_MD,
        )
        self._write_if_missing(
            paths.get_shards_dir(self.root) / paths.CLAUDE_MD,
            templates.SHARDS_CLAUDE_MD,
        )
        self._write_if_missing(
            paths.get_runs_dir(self.root) / paths.CLAUDE_MD,
            templates.RUNS_CLAUDE_MD,
        )
        self._write_if_missing(
            paths.get_benchmarks_dir(self.root) / paths.CLAUDE_MD,
            templates.BENCHMARKS_CLAUDE_MD,
        )

        # Initialize empty manifest if it doesn't exist
        manifest_path = paths.get_manifest_path(self.root)
        if not manifest_path.exists():
            self._save_manifest(Manifest())

    def _ensure_initialized(self) -> None:
        """Ensure the experiments folder is initialized."""
        if not self.root.exists():
            self.init()

    def _write_if_missing(self, path: Path, content: str) -> None:
        """Write content to path if it doesn't exist."""
        if not path.exists():
            path.write_text(content)

    # --- Manifest Operations ---

    def _load_manifest(self) -> Manifest:
        """Load the manifest from disk."""
        manifest_path = paths.get_manifest_path(self.root)
        if not manifest_path.exists():
            return Manifest()
        data = yaml.safe_load(manifest_path.read_text()) or {}
        return Manifest.model_validate(data)

    def _save_manifest(self, manifest: Manifest) -> None:
        """Save the manifest to disk."""
        manifest_path = paths.get_manifest_path(self.root)
        # Use model_dump with mode="json" for datetime serialization
        data = manifest.model_dump(mode="json")
        manifest_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def load_manifest(self) -> Manifest:
        """Load the manifest.

        Returns:
            The current manifest.
        """
        self._ensure_initialized()
        return self._load_manifest()

    # --- Batch Operations ---

    def create_batch(
        self,
        group: str,
        mcts_config: MCTSConfig,
        game_params: GameParams,
        checkpoint_path: str | None = None,
    ) -> Path:
        """Create a new batch directory for sampling.

        Args:
            group: Human-readable grouping name (e.g., "uniform_5x5").
            mcts_config: MCTS algorithm configuration.
            game_params: Game configuration.
            checkpoint_path: Optional path to parent checkpoint for NN-guided sampling.

        Returns:
            Path to the created batch directory.
        """
        self._ensure_initialized()

        batch_uuid = str(uuid.uuid4())
        batch_dir = paths.get_batch_path(self.root, group, batch_uuid)

        # Create directory structure
        batch_dir.mkdir(parents=True, exist_ok=False)
        (batch_dir / paths.GAMES_DIR).mkdir()

        # Save batch metadata (compatible with existing BatchMetadata format)
        from alpharat.data.batch import BatchMetadata, save_batch_metadata

        metadata = BatchMetadata(
            batch_id=batch_uuid,
            created_at=datetime.now(UTC),
            checkpoint_path=checkpoint_path,
            mcts_config=mcts_config,
            game_params=game_params,
        )
        save_batch_metadata(batch_dir, metadata)

        # Update manifest
        batch_id = f"{group}/{batch_uuid}"
        entry = BatchEntry(
            group=group,
            uuid=batch_uuid,
            created_at=metadata.created_at,
            parent_checkpoint=checkpoint_path,
            mcts_config=mcts_config,
            game_params=game_params,
        )
        manifest = self._load_manifest()
        manifest.batches[batch_id] = entry
        self._save_manifest(manifest)

        return batch_dir

    def get_batch_path(self, batch_id: str) -> Path:
        """Get path to a batch by its ID.

        Args:
            batch_id: Batch ID in format "group/uuid".

        Returns:
            Path to batch directory.
        """
        group, batch_uuid = paths.parse_batch_id(batch_id)
        return paths.get_batch_path(self.root, group, batch_uuid)

    def list_batches(self) -> list[str]:
        """List all batch IDs.

        Returns:
            List of batch IDs in format "group/uuid".
        """
        manifest = self.load_manifest()
        return list(manifest.batches.keys())

    # --- Shard Operations ---

    def create_shards(
        self,
        source_batches: list[str],
        total_positions: int,
        train_positions: int,
        val_positions: int,
    ) -> Path:
        """Register a new shard set in the manifest.

        Note: This method registers the shard in the manifest. The actual shard
        files should be created by the sharding code, which then calls this method.

        Args:
            source_batches: List of batch IDs used to create these shards.
            total_positions: Total number of positions.
            train_positions: Number of training positions.
            val_positions: Number of validation positions.

        Returns:
            Path to the shard directory.
        """
        self._ensure_initialized()

        shard_uuid = str(uuid.uuid4())
        shard_dir = paths.get_shard_path(self.root, shard_uuid)

        # Create directory structure
        shard_dir.mkdir(parents=True, exist_ok=False)
        (shard_dir / paths.TRAIN_DIR).mkdir()
        (shard_dir / paths.VAL_DIR).mkdir()

        # Update manifest
        entry = ShardEntry(
            uuid=shard_uuid,
            created_at=datetime.now(UTC),
            source_batches=source_batches,
            total_positions=total_positions,
            train_positions=train_positions,
            val_positions=val_positions,
        )
        manifest = self._load_manifest()
        manifest.shards[shard_uuid] = entry
        self._save_manifest(manifest)

        return shard_dir

    def get_shard_path(self, shard_id: str) -> Path:
        """Get path to a shard set by its ID.

        Args:
            shard_id: Shard UUID.

        Returns:
            Path to shard directory.
        """
        return paths.get_shard_path(self.root, shard_id)

    def list_shards(self) -> list[str]:
        """List all shard IDs.

        Returns:
            List of shard UUIDs.
        """
        manifest = self.load_manifest()
        return list(manifest.shards.keys())

    def register_shards(
        self,
        shard_id: str,
        source_batches: list[str],
        total_positions: int,
        train_positions: int,
        val_positions: int,
    ) -> None:
        """Register an existing shard set in the manifest.

        Use this when shards were created externally (e.g., by sharding code).
        The shard directory must already exist at the expected location.

        Args:
            shard_id: UUID of the shard set.
            source_batches: List of batch IDs used to create these shards.
            total_positions: Total number of positions.
            train_positions: Number of training positions.
            val_positions: Number of validation positions.
        """
        self._ensure_initialized()

        entry = ShardEntry(
            uuid=shard_id,
            created_at=datetime.now(UTC),
            source_batches=source_batches,
            total_positions=total_positions,
            train_positions=train_positions,
            val_positions=val_positions,
        )
        manifest = self._load_manifest()
        manifest.shards[shard_id] = entry
        self._save_manifest(manifest)

    # --- Run Operations ---

    def create_run(
        self,
        name: str,
        config: dict[str, Any],
        source_shards: str,
        parent_checkpoint: str | None = None,
    ) -> Path:
        """Create a new training run directory.

        Args:
            name: Human-readable run name (e.g., "bimatrix_mlp_v1").
            config: Training configuration dict (will be saved as config.yaml).
            source_shards: Shard ID used for training.
            parent_checkpoint: Optional path to checkpoint being resumed from.

        Returns:
            Path to the run directory.

        Raises:
            FileExistsError: If a run with this name already exists.
        """
        self._ensure_initialized()

        run_dir = paths.get_run_path(self.root, name)
        if run_dir.exists():
            raise FileExistsError(f"Run '{name}' already exists at {run_dir}")

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / paths.CHECKPOINTS_DIR).mkdir()

        # Save config.yaml
        config_path = run_dir / paths.CONFIG_FILE
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

        # Create notes.txt with template
        notes_path = run_dir / paths.NOTES_FILE
        notes_path.write_text(templates.NOTES_TEMPLATE)

        # Update manifest
        entry = RunEntry(
            name=name,
            created_at=datetime.now(UTC),
            source_shards=source_shards,
            parent_checkpoint=parent_checkpoint,
            config=config,
        )
        manifest = self._load_manifest()
        manifest.runs[name] = entry
        self._save_manifest(manifest)

        return run_dir

    def update_run_results(
        self,
        name: str,
        best_val_loss: float | None = None,
        final_epoch: int | None = None,
    ) -> None:
        """Update a run's results in the manifest.

        Args:
            name: Run name.
            best_val_loss: Best validation loss achieved.
            final_epoch: Final epoch number.
        """
        manifest = self._load_manifest()
        if name not in manifest.runs:
            raise KeyError(f"Run '{name}' not found in manifest")

        entry = manifest.runs[name]
        if best_val_loss is not None:
            entry.best_val_loss = best_val_loss
        if final_epoch is not None:
            entry.final_epoch = final_epoch

        self._save_manifest(manifest)

    def get_run_path(self, name: str) -> Path:
        """Get path to a run by its name.

        Args:
            name: Run name.

        Returns:
            Path to run directory.
        """
        return paths.get_run_path(self.root, name)

    def get_run_checkpoints_path(self, name: str) -> Path:
        """Get path to a run's checkpoints directory.

        Args:
            name: Run name.

        Returns:
            Path to checkpoints directory.
        """
        return self.get_run_path(name) / paths.CHECKPOINTS_DIR

    def list_runs(self) -> list[str]:
        """List all run names.

        Returns:
            List of run names.
        """
        manifest = self.load_manifest()
        return list(manifest.runs.keys())

    # --- Benchmark Operations ---

    def create_benchmark(
        self,
        name: str,
        config: dict[str, Any],
        checkpoints: list[str],
    ) -> Path:
        """Create a new benchmark directory.

        Args:
            name: Human-readable benchmark name (e.g., "tournament_20260107").
            config: Benchmark configuration dict (will be saved as config.yaml).
            checkpoints: List of checkpoint/run names being evaluated.

        Returns:
            Path to the benchmark directory.

        Raises:
            FileExistsError: If a benchmark with this name already exists.
        """
        self._ensure_initialized()

        bench_dir = paths.get_benchmark_path(self.root, name)
        if bench_dir.exists():
            raise FileExistsError(f"Benchmark '{name}' already exists at {bench_dir}")

        # Create directory
        bench_dir.mkdir(parents=True, exist_ok=False)

        # Save config.yaml
        config_path = bench_dir / paths.CONFIG_FILE
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

        # Create notes.txt with template
        notes_path = bench_dir / paths.NOTES_FILE
        notes_path.write_text(templates.NOTES_TEMPLATE)

        # Update manifest
        entry = BenchmarkEntry(
            name=name,
            created_at=datetime.now(UTC),
            checkpoints=checkpoints,
            config=config,
        )
        manifest = self._load_manifest()
        manifest.benchmarks[name] = entry
        self._save_manifest(manifest)

        return bench_dir

    def save_benchmark_results(self, name: str, results: dict[str, Any]) -> None:
        """Save benchmark results to results.json.

        Args:
            name: Benchmark name.
            results: Results dict to save.
        """
        import json

        bench_dir = paths.get_benchmark_path(self.root, name)
        if not bench_dir.exists():
            raise FileNotFoundError(f"Benchmark '{name}' not found at {bench_dir}")

        results_path = bench_dir / paths.RESULTS_FILE
        results_path.write_text(json.dumps(results, indent=2, default=str))

    def get_benchmark_path(self, name: str) -> Path:
        """Get path to a benchmark by its name.

        Args:
            name: Benchmark name.

        Returns:
            Path to benchmark directory.
        """
        return paths.get_benchmark_path(self.root, name)

    def list_benchmarks(self) -> list[str]:
        """List all benchmark names.

        Returns:
            List of benchmark names.
        """
        manifest = self.load_manifest()
        return list(manifest.benchmarks.keys())
