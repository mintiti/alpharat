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

    def _dump_yaml(self, data: dict[str, Any]) -> str:
        """Serialize data to YAML with consistent formatting."""
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def _setup_config_and_notes(self, artifact_dir: Path, config: dict[str, Any]) -> None:
        """Write config.yaml and notes.txt template to artifact directory."""
        config_path = artifact_dir / paths.CONFIG_FILE
        config_path.write_text(self._dump_yaml(config))

        notes_path = artifact_dir / paths.NOTES_FILE
        notes_path.write_text(templates.NOTES_TEMPLATE)

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
        manifest_path.write_text(self._dump_yaml(data))

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
        seed_start: int = 0,
    ) -> Path:
        """Create a new batch directory for sampling.

        Args:
            group: Human-readable grouping name (e.g., "uniform_5x5").
            mcts_config: MCTS algorithm configuration.
            game_params: Game configuration.
            checkpoint_path: Optional path to parent checkpoint for NN-guided sampling.
            seed_start: Starting seed for game generation (game N uses seed_start + N).

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
            seed_start=seed_start,
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

    def _register_shard_entry(
        self,
        group: str,
        shard_uuid: str,
        source_batches: list[str],
        total_positions: int,
        train_positions: int,
        val_positions: int,
        shuffle_seed: int | None = None,
    ) -> None:
        """Add shard entry to manifest (internal helper)."""
        shard_id = f"{group}/{shard_uuid}"
        entry = ShardEntry(
            group=group,
            uuid=shard_uuid,
            created_at=datetime.now(UTC),
            source_batches=source_batches,
            total_positions=total_positions,
            train_positions=train_positions,
            val_positions=val_positions,
            shuffle_seed=shuffle_seed,
        )
        manifest = self._load_manifest()
        manifest.shards[shard_id] = entry
        self._save_manifest(manifest)

    def create_shards(
        self,
        group: str,
        source_batches: list[str],
        total_positions: int,
        train_positions: int,
        val_positions: int,
        shuffle_seed: int | None = None,
    ) -> Path:
        """Create a new shard directory and register it in the manifest.

        Args:
            group: Shard group name (e.g., "5x5_uniform", "7x7_walls").
            source_batches: List of batch IDs used to create these shards.
            total_positions: Total number of positions.
            train_positions: Number of training positions.
            val_positions: Number of validation positions.
            shuffle_seed: Seed used for train/val split and position shuffling.

        Returns:
            Path to the shard directory.
        """
        self._ensure_initialized()

        shard_uuid = str(uuid.uuid4())
        shard_dir = paths.get_shard_path(self.root, group, shard_uuid)

        # Create directory structure
        shard_dir.mkdir(parents=True, exist_ok=False)
        (shard_dir / paths.TRAIN_DIR).mkdir()
        (shard_dir / paths.VAL_DIR).mkdir()

        self._register_shard_entry(
            group,
            shard_uuid,
            source_batches,
            total_positions,
            train_positions,
            val_positions,
            shuffle_seed,
        )
        return shard_dir

    def get_shard_path(self, shard_id: str) -> Path:
        """Get path to a shard set by its ID.

        Args:
            shard_id: Shard ID in format "group/uuid".

        Returns:
            Path to shard directory.
        """
        group, shard_uuid = paths.parse_shard_id(shard_id)
        return paths.get_shard_path(self.root, group, shard_uuid)

    def list_shards(self) -> list[str]:
        """List all shard IDs.

        Returns:
            List of shard IDs in format "group/uuid".
        """
        manifest = self.load_manifest()
        return list(manifest.shards.keys())

    def shard_id_from_data_path(self, data_path: Path | str) -> str | None:
        """Extract shard ID from a data path.

        Handles paths like:
        - experiments/shards/group/uuid/train
        - experiments/shards/group/uuid/val
        - experiments/shards/group/uuid

        Args:
            data_path: Path to a shard directory or its train/val subdirectory.

        Returns:
            Shard ID in "group/uuid" format, or None if path doesn't match.
        """
        path = Path(data_path)

        # Walk up from train/ or val/ if needed
        if path.name in (paths.TRAIN_DIR, paths.VAL_DIR):
            path = path.parent

        # Now at uuid level: uuid_dir / group_dir / shards_dir
        uuid_dir = path
        group_dir = uuid_dir.parent
        shards_dir = group_dir.parent

        if shards_dir.name != paths.SHARDS_DIR:
            return None

        return f"{group_dir.name}/{uuid_dir.name}"

    def register_shards(
        self,
        group: str,
        shard_uuid: str,
        source_batches: list[str],
        total_positions: int,
        train_positions: int,
        val_positions: int,
        shuffle_seed: int | None = None,
    ) -> None:
        """Register an existing shard set in the manifest.

        Use this when shards were created externally (e.g., by sharding code).
        The shard directory must already exist at the expected location.

        Args:
            group: Shard group name.
            shard_uuid: UUID of the shard set.
            source_batches: List of batch IDs used to create these shards.
            total_positions: Total number of positions.
            train_positions: Number of training positions.
            val_positions: Number of validation positions.
            shuffle_seed: Seed used for train/val split and position shuffling.
        """
        self._ensure_initialized()
        self._register_shard_entry(
            group,
            shard_uuid,
            source_batches,
            total_positions,
            train_positions,
            val_positions,
            shuffle_seed,
        )

    # --- Run Operations ---

    def _configs_equal(self, config1: dict[str, Any], config2: dict[str, Any]) -> bool:
        """Compare configs, ignoring name and resume_from fields."""

        def normalize(cfg: dict[str, Any]) -> dict[str, Any]:
            # Copy and remove fields that aren't part of the "experiment identity"
            c = dict(cfg)
            c.pop("name", None)
            c.pop("resume_from", None)
            return c

        return normalize(config1) == normalize(config2)

    def _find_next_run_name(self, base_name: str) -> str:
        """Find next available run name (base_name, base_name_run2, base_name_run3, ...)."""
        manifest = self._load_manifest()

        # Check if base name is taken
        if base_name not in manifest.runs:
            return base_name

        # Find highest existing run number
        run_num = 2
        while f"{base_name}_run{run_num}" in manifest.runs:
            run_num += 1

        return f"{base_name}_run{run_num}"

    def create_run(
        self,
        name: str,
        config: dict[str, Any],
        source_shards: str,
        parent_checkpoint: str | None = None,
    ) -> Path:
        """Create a new training run directory.

        If a run with this name exists and has the same config, auto-increments
        the name (e.g., mlp_v1 → mlp_v1_run2). If configs differ, raises an error.

        Args:
            name: Human-readable run name (e.g., "bimatrix_mlp_v1").
            config: Training configuration dict (will be saved as config.yaml).
            source_shards: Shard ID used for training.
            parent_checkpoint: Optional path to checkpoint being resumed from.

        Returns:
            Path to the run directory.

        Raises:
            ValueError: If a run with this name exists but has a different config.
        """
        self._ensure_initialized()

        run_dir = paths.get_run_path(self.root, name)
        if run_dir.exists():
            # Check if configs match
            manifest = self._load_manifest()
            existing_config = manifest.runs[name].config

            if self._configs_equal(config, existing_config):
                # Same experiment, auto-increment run number
                name = self._find_next_run_name(name)
                run_dir = paths.get_run_path(self.root, name)
            else:
                raise ValueError(
                    f"Run '{name}' exists with different config. "
                    f"Pick a new name for this experiment."
                )

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / paths.CHECKPOINTS_DIR).mkdir()

        # Save config.yaml and notes.txt
        self._setup_config_and_notes(run_dir, config)

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

        # Save config.yaml and notes.txt
        self._setup_config_and_notes(bench_dir, config)

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

    # --- Query Helpers ---

    def list_batches_with_info(self) -> list[dict[str, Any]]:
        """List batches with metadata for display.

        Returns:
            List of dicts with batch info: id, created, parent_checkpoint, game_params.
        """
        manifest = self.load_manifest()
        return [
            {
                "id": batch_id,
                "created": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "parent": entry.parent_checkpoint or "-",
                "size": f"{entry.game_params.width}x{entry.game_params.height}",
                "simulations": entry.mcts_config.simulations,
            }
            for batch_id, entry in manifest.batches.items()
        ]

    def list_shards_with_info(self) -> list[dict[str, Any]]:
        """List shards with lineage info for display.

        Returns:
            List of dicts with shard info: id, created, source_batches, positions.
        """
        manifest = self.load_manifest()
        return [
            {
                "id": shard_id,
                "created": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "source_batches": entry.source_batches,
                "train": entry.train_positions,
                "val": entry.val_positions,
            }
            for shard_id, entry in manifest.shards.items()
        ]

    def list_runs_with_info(self) -> list[dict[str, Any]]:
        """List runs with metadata for display.

        Returns:
            List of dicts with run info: name, created, source_shards, results.
        """
        manifest = self.load_manifest()
        return [
            {
                "name": name,
                "created": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "source_shards": entry.source_shards,
                "best_val_loss": entry.best_val_loss,
                "final_epoch": entry.final_epoch,
            }
            for name, entry in manifest.runs.items()
        ]
