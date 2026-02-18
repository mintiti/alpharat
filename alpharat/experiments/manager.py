"""ExperimentManager: Central API for managing experiment artifacts.

Usage:
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager()
    batch_dir = exp.create_batch(group="uniform_5x5", mcts_config=..., game_config=...)
"""

from __future__ import annotations

import logging
import shutil
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
    from alpharat.config.game import GameConfig
    from alpharat.mcts.config import MCTSConfigBase

logger = logging.getLogger(__name__)


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

    def _recover_or_raise(
        self, artifact_dir: Path, artifact_key: str, manifest_section: dict[str, Any]
    ) -> None:
        """Handle pre-existing artifact directories.

        If the directory doesn't exist, does nothing.
        If the directory exists AND is registered in manifest, raises FileExistsError.
        If the directory exists but is NOT in manifest (crash leftover), cleans it up.

        Warning: not safe for concurrent access. If two processes call prepare_run()
        with the same name, the second will delete the first's in-progress directory.
        Batches are safe (UUID-based paths prevent collisions).
        """
        if not artifact_dir.exists():
            return
        if artifact_key in manifest_section:
            raise FileExistsError(f"Artifact already registered: {artifact_dir}")
        # Directory exists but no manifest entry = crash leftover
        logger.warning("Cleaning up incomplete artifact: %s", artifact_dir)
        shutil.rmtree(artifact_dir)

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
        data = manifest.model_dump(mode="json")
        content = self._dump_yaml(data)
        # Atomic write: temp file + rename prevents corruption on crash
        tmp_path = manifest_path.with_suffix(".yaml.tmp")
        tmp_path.write_text(content)
        tmp_path.rename(manifest_path)

    def load_manifest(self) -> Manifest:
        """Load the manifest.

        Returns:
            The current manifest.
        """
        self._ensure_initialized()
        return self._load_manifest()

    # --- Batch Operations ---

    def prepare_batch(
        self,
        group: str,
        mcts_config: MCTSConfigBase,
        game: GameConfig,
        checkpoint_path: str | None = None,
    ) -> tuple[Path, str]:
        """Create a batch directory without registering in manifest.

        Use this when the operation (sampling) needs a directory during work.
        Call register_batch() after the operation succeeds.

        If a previous attempt left a directory without a manifest entry,
        it is cleaned up automatically.

        Args:
            group: Human-readable grouping name (e.g., "uniform_5x5").
            mcts_config: MCTS algorithm configuration.
            game: Game configuration (GameConfig).
            checkpoint_path: Optional path to parent checkpoint for NN-guided sampling.

        Returns:
            Tuple of (batch_dir, batch_uuid).
        """
        self._ensure_initialized()

        batch_uuid = str(uuid.uuid4())
        batch_dir = paths.get_batch_path(self.root, group, batch_uuid)

        # New UUID won't collide with manifest, but guard anyway
        batch_id = f"{group}/{batch_uuid}"
        manifest = self._load_manifest()
        self._recover_or_raise(batch_dir, batch_id, manifest.batches)

        # Create directory structure
        batch_dir.mkdir(parents=True, exist_ok=False)
        (batch_dir / paths.GAMES_DIR).mkdir()

        # Save batch metadata (compatible with existing BatchMetadata format)
        from alpharat.data.batch import BatchMetadata, save_batch_metadata

        metadata = BatchMetadata(
            batch_id=batch_uuid,
            created_at=datetime.now(UTC),
            checkpoint_path=checkpoint_path,
            mcts_config=mcts_config,  # type: ignore[arg-type]
            game=game,
        )
        save_batch_metadata(batch_dir, metadata)

        return batch_dir, batch_uuid

    def register_batch(
        self,
        group: str,
        batch_uuid: str,
        mcts_config: MCTSConfigBase,
        game: GameConfig,
        checkpoint_path: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Register a completed batch in the manifest.

        Called after sampling succeeds. The batch directory must already exist.

        Args:
            group: Batch group name.
            batch_uuid: UUID of the batch (from prepare_batch).
            mcts_config: MCTS algorithm configuration.
            game: Game configuration.
            checkpoint_path: Optional parent checkpoint path.
            created_at: Timestamp override (defaults to now).
        """
        batch_id = f"{group}/{batch_uuid}"
        entry = BatchEntry(
            group=group,
            uuid=batch_uuid,
            created_at=created_at or datetime.now(UTC),
            parent_checkpoint=checkpoint_path,
            mcts_config=mcts_config.model_dump(),
            game=game.model_dump(),
        )
        manifest = self._load_manifest()
        manifest.batches[batch_id] = entry
        self._save_manifest(manifest)

    def create_batch(
        self,
        group: str,
        mcts_config: MCTSConfigBase,
        game: GameConfig,
        checkpoint_path: str | None = None,
    ) -> Path:
        """Create a new batch directory and register it in manifest.

        Convenience wrapper around prepare_batch + register_batch.

        Args:
            group: Human-readable grouping name (e.g., "uniform_5x5").
            mcts_config: MCTS algorithm configuration.
            game: Game configuration (GameConfig).
            checkpoint_path: Optional path to parent checkpoint for NN-guided sampling.

        Returns:
            Path to the created batch directory.
        """
        batch_dir, batch_uuid = self.prepare_batch(
            group=group,
            mcts_config=mcts_config,
            game=game,
            checkpoint_path=checkpoint_path,
        )
        self.register_batch(
            group=group,
            batch_uuid=batch_uuid,
            mcts_config=mcts_config,
            game=game,
            checkpoint_path=checkpoint_path,
        )
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

    def _resolve_run_name(self, name: str, config: dict[str, Any]) -> str:
        """Resolve the actual run name, handling auto-increment and conflicts.

        Checks manifest (not filesystem) for name collisions:
        - Name not in manifest → use as-is
        - Name in manifest with same config → auto-increment
        - Name in manifest with different config → raise ValueError

        Args:
            name: Requested run name.
            config: Training configuration dict.

        Returns:
            The resolved run name.
        """
        manifest = self._load_manifest()
        if name in manifest.runs:
            if self._configs_equal(config, manifest.runs[name].config):
                return self._find_next_run_name(name)
            raise ValueError(
                f"Run '{name}' exists with different config. Pick a new name for this experiment."
            )
        return name

    def prepare_run(
        self,
        name: str,
        config: dict[str, Any],
        source_shards: str,
        parent_checkpoint: str | None = None,
    ) -> tuple[Path, str]:
        """Create a run directory without registering in manifest.

        Handles name auto-increment. Call register_run() after training succeeds.

        If a previous attempt left a directory without a manifest entry,
        it is cleaned up automatically.

        Args:
            name: Human-readable run name (e.g., "mlp_v1").
            config: Training configuration dict (will be saved as config.yaml).
            source_shards: Shard ID used for training.
            parent_checkpoint: Optional path to checkpoint being resumed from.

        Returns:
            Tuple of (run_dir, actual_name) where actual_name may be auto-incremented.

        Raises:
            ValueError: If a run with this name exists but has a different config.
        """
        self._ensure_initialized()

        actual_name = self._resolve_run_name(name, config)
        run_dir = paths.get_run_path(self.root, actual_name)

        manifest = self._load_manifest()
        self._recover_or_raise(run_dir, actual_name, manifest.runs)

        # Create directory structure
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / paths.CHECKPOINTS_DIR).mkdir()

        # Save config.yaml and notes.txt
        self._setup_config_and_notes(run_dir, config)

        return run_dir, actual_name

    def register_run(
        self,
        name: str,
        config: dict[str, Any],
        source_shards: str,
        parent_checkpoint: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Register a completed training run in the manifest.

        Called after training succeeds. The run directory must already exist.

        Args:
            name: Run name (the actual_name from prepare_run).
            config: Training configuration dict.
            source_shards: Shard ID used for training.
            parent_checkpoint: Optional parent checkpoint path.
            created_at: Timestamp override (defaults to now).
        """
        entry = RunEntry(
            name=name,
            created_at=created_at or datetime.now(UTC),
            source_shards=source_shards,
            parent_checkpoint=parent_checkpoint,
            config=config,
        )
        manifest = self._load_manifest()
        manifest.runs[name] = entry
        self._save_manifest(manifest)

    def create_run(
        self,
        name: str,
        config: dict[str, Any],
        source_shards: str,
        parent_checkpoint: str | None = None,
    ) -> Path:
        """Create a new training run directory and register it in manifest.

        Convenience wrapper around prepare_run + register_run.

        If a run with this name exists and has the same config, auto-increments
        the name (e.g., mlp_v1 → mlp_v1_run2). If configs differ, raises an error.

        Args:
            name: Human-readable run name (e.g., "mlp_v1").
            config: Training configuration dict (will be saved as config.yaml).
            source_shards: Shard ID used for training.
            parent_checkpoint: Optional path to checkpoint being resumed from.

        Returns:
            Path to the run directory.

        Raises:
            ValueError: If a run with this name exists but has a different config.
        """
        run_dir, actual_name = self.prepare_run(
            name=name,
            config=config,
            source_shards=source_shards,
            parent_checkpoint=parent_checkpoint,
        )
        self.register_run(
            name=actual_name,
            config=config,
            source_shards=source_shards,
            parent_checkpoint=parent_checkpoint,
        )
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
        results: dict[str, Any] | None = None,
    ) -> Path:
        """Create a benchmark directory, save results, and register in manifest.

        Fully transactional: nothing goes to disk until this method is called.
        The tournament should already have completed successfully.

        If a previous attempt left a directory without a manifest entry,
        it is cleaned up automatically.

        Args:
            name: Human-readable benchmark name (e.g., "tournament_20260107").
            config: Benchmark configuration dict (will be saved as config.yaml).
            checkpoints: List of checkpoint/run names being evaluated.
            results: Tournament results dict (saved as results.json if provided).

        Returns:
            Path to the benchmark directory.

        Raises:
            FileExistsError: If a benchmark with this name is already registered.
        """
        import json

        self._ensure_initialized()

        bench_dir = paths.get_benchmark_path(self.root, name)
        manifest = self._load_manifest()
        self._recover_or_raise(bench_dir, name, manifest.benchmarks)

        # Create directory
        bench_dir.mkdir(parents=True, exist_ok=False)

        # Save config.yaml and notes.txt
        self._setup_config_and_notes(bench_dir, config)

        # Save results if provided
        if results is not None:
            results_path = bench_dir / paths.RESULTS_FILE
            results_path.write_text(json.dumps(results, indent=2, default=str))

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
            List of dicts with batch info: id, created, parent_checkpoint, game.
        """
        manifest = self.load_manifest()
        return [
            {
                "id": batch_id,
                "created": entry.created_at.strftime("%Y-%m-%d %H:%M"),
                "parent": entry.parent_checkpoint or "-",
                "size": f"{entry.game.get('width', '?')}x{entry.game.get('height', '?')}",
                "simulations": entry.mcts_config.get("simulations"),
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
