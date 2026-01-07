"""Tests for ExperimentManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from alpharat.data.batch import GameParams
from alpharat.experiments import ExperimentManager
from alpharat.experiments.paths import (
    BATCHES_DIR,
    BENCHMARKS_DIR,
    CHECKPOINTS_DIR,
    CLAUDE_MD,
    CONFIG_FILE,
    GAMES_DIR,
    MANIFEST_FILE,
    METADATA_FILE,
    NOTES_FILE,
    RESULTS_FILE,
    RUNS_DIR,
    SHARDS_DIR,
    TRAIN_DIR,
    VAL_DIR,
)
from alpharat.mcts import DecoupledPUCTConfig


class TestExperimentManagerInit:
    """Tests for ExperimentManager initialization."""

    def test_init_creates_structure(self) -> None:
        """init() creates the full folder structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.init()

            root = Path(tmpdir)
            assert root.exists()
            assert (root / BATCHES_DIR).exists()
            assert (root / SHARDS_DIR).exists()
            assert (root / RUNS_DIR).exists()
            assert (root / BENCHMARKS_DIR).exists()
            assert (root / MANIFEST_FILE).exists()

    def test_init_creates_claude_md_files(self) -> None:
        """init() creates CLAUDE.md files in each subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.init()

            root = Path(tmpdir)
            assert (root / CLAUDE_MD).exists()
            assert (root / BATCHES_DIR / CLAUDE_MD).exists()
            assert (root / SHARDS_DIR / CLAUDE_MD).exists()
            assert (root / RUNS_DIR / CLAUDE_MD).exists()
            assert (root / BENCHMARKS_DIR / CLAUDE_MD).exists()

    def test_init_creates_empty_manifest(self) -> None:
        """init() creates empty manifest.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.init()

            manifest = exp.load_manifest()
            assert manifest.batches == {}
            assert manifest.shards == {}
            assert manifest.runs == {}
            assert manifest.benchmarks == {}

    def test_init_idempotent(self) -> None:
        """init() is idempotent - calling twice doesn't break anything."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.init()

            # Modify a file
            claude_md = Path(tmpdir) / CLAUDE_MD
            original_content = claude_md.read_text()

            # Call init again
            exp.init()

            # File should not be overwritten
            assert claude_md.read_text() == original_content


class TestBatchOperations:
    """Tests for batch-related operations."""

    def test_create_batch_creates_directory(self) -> None:
        """create_batch creates batch directory with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            batch_dir = exp.create_batch(
                group="test_group",
                mcts_config=DecoupledPUCTConfig(simulations=100),
                game_params=GameParams(width=5, height=5, max_turns=30, cheese_count=5),
            )

            assert batch_dir.exists()
            assert (batch_dir / GAMES_DIR).exists()
            assert (batch_dir / METADATA_FILE).exists()

    def test_create_batch_updates_manifest(self) -> None:
        """create_batch adds entry to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            batch_dir = exp.create_batch(
                group="test_group",
                mcts_config=DecoupledPUCTConfig(simulations=100),
                game_params=GameParams(width=5, height=5, max_turns=30, cheese_count=5),
            )

            manifest = exp.load_manifest()
            batch_id = f"test_group/{batch_dir.name}"

            assert batch_id in manifest.batches
            entry = manifest.batches[batch_id]
            assert entry.group == "test_group"
            assert entry.parent_checkpoint is None

    def test_create_batch_with_checkpoint(self) -> None:
        """create_batch stores parent checkpoint path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_batch(
                group="nn_guided",
                mcts_config=DecoupledPUCTConfig(simulations=200, c_puct=2.0),
                game_params=GameParams(width=5, height=5, max_turns=30, cheese_count=5),
                checkpoint_path="/path/to/model.pt",
            )

            manifest = exp.load_manifest()
            entry = list(manifest.batches.values())[0]
            assert entry.parent_checkpoint == "/path/to/model.pt"

    def test_list_batches(self) -> None:
        """list_batches returns all batch IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            game_params = GameParams(width=5, height=5, max_turns=30, cheese_count=5)
            mcts_config = DecoupledPUCTConfig(simulations=100)

            exp.create_batch(group="group_a", mcts_config=mcts_config, game_params=game_params)
            exp.create_batch(group="group_b", mcts_config=mcts_config, game_params=game_params)

            batches = exp.list_batches()
            assert len(batches) == 2
            assert any("group_a" in b for b in batches)
            assert any("group_b" in b for b in batches)

    def test_get_batch_path(self) -> None:
        """get_batch_path returns correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            batch_dir = exp.create_batch(
                group="test_group",
                mcts_config=DecoupledPUCTConfig(simulations=100),
                game_params=GameParams(width=5, height=5, max_turns=30, cheese_count=5),
            )

            batch_id = f"test_group/{batch_dir.name}"
            retrieved_path = exp.get_batch_path(batch_id)
            assert retrieved_path == batch_dir


class TestShardOperations:
    """Tests for shard-related operations."""

    def test_create_shards_creates_directory(self) -> None:
        """create_shards creates shard directory with train/val subdirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            shard_dir = exp.create_shards(
                source_batches=["group/uuid1"],
                total_positions=1000,
                train_positions=900,
                val_positions=100,
            )

            assert shard_dir.exists()
            assert (shard_dir / TRAIN_DIR).exists()
            assert (shard_dir / VAL_DIR).exists()

    def test_create_shards_updates_manifest(self) -> None:
        """create_shards adds entry to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            shard_dir = exp.create_shards(
                source_batches=["group_a/uuid1", "group_b/uuid2"],
                total_positions=5000,
                train_positions=4500,
                val_positions=500,
            )

            manifest = exp.load_manifest()
            shard_id = shard_dir.name

            assert shard_id in manifest.shards
            entry = manifest.shards[shard_id]
            assert entry.source_batches == ["group_a/uuid1", "group_b/uuid2"]
            assert entry.total_positions == 5000
            assert entry.train_positions == 4500
            assert entry.val_positions == 500

    def test_list_shards(self) -> None:
        """list_shards returns all shard IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_shards(["b1"], 100, 90, 10)
            exp.create_shards(["b2"], 200, 180, 20)

            shards = exp.list_shards()
            assert len(shards) == 2


class TestRunOperations:
    """Tests for training run operations."""

    def test_create_run_creates_directory(self) -> None:
        """create_run creates run directory with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            run_dir = exp.create_run(
                name="test_run",
                config={"model": {"hidden_dim": 256}},
                source_shards="shard_uuid",
            )

            assert run_dir.exists()
            assert (run_dir / CHECKPOINTS_DIR).exists()
            assert (run_dir / CONFIG_FILE).exists()
            assert (run_dir / NOTES_FILE).exists()

    def test_create_run_saves_config_yaml(self) -> None:
        """create_run saves config as YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            config = {"model": {"hidden_dim": 256}, "optim": {"lr": 0.001}}
            run_dir = exp.create_run(
                name="test_run",
                config=config,
                source_shards="shard_uuid",
            )

            config_path = run_dir / CONFIG_FILE
            loaded_config = yaml.safe_load(config_path.read_text())
            assert loaded_config == config

    def test_create_run_creates_notes_with_template(self) -> None:
        """create_run creates notes.txt with template content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            run_dir = exp.create_run(
                name="test_run",
                config={},
                source_shards="shard_uuid",
            )

            notes_content = (run_dir / NOTES_FILE).read_text()
            assert "## Goal" in notes_content
            assert "## Observations" in notes_content
            assert "## Results" in notes_content

    def test_create_run_updates_manifest(self) -> None:
        """create_run adds entry to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_run(
                name="my_run",
                config={"model": {"hidden_dim": 128}},
                source_shards="shard_123",
                parent_checkpoint="previous/best.pt",
            )

            manifest = exp.load_manifest()
            assert "my_run" in manifest.runs
            entry = manifest.runs["my_run"]
            assert entry.source_shards == "shard_123"
            assert entry.parent_checkpoint == "previous/best.pt"
            assert entry.config == {"model": {"hidden_dim": 128}}

    def test_create_run_fails_if_exists(self) -> None:
        """create_run raises FileExistsError if run already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_run(name="duplicate", config={}, source_shards="s1")

            with pytest.raises(FileExistsError):
                exp.create_run(name="duplicate", config={}, source_shards="s2")

    def test_update_run_results(self) -> None:
        """update_run_results updates manifest entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_run(name="test_run", config={}, source_shards="s1")

            exp.update_run_results(name="test_run", best_val_loss=0.123, final_epoch=50)

            manifest = exp.load_manifest()
            entry = manifest.runs["test_run"]
            assert entry.best_val_loss == 0.123
            assert entry.final_epoch == 50

    def test_get_run_checkpoints_path(self) -> None:
        """get_run_checkpoints_path returns correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            run_dir = exp.create_run(name="test_run", config={}, source_shards="s1")

            checkpoints_path = exp.get_run_checkpoints_path("test_run")
            assert checkpoints_path == run_dir / CHECKPOINTS_DIR


class TestBenchmarkOperations:
    """Tests for benchmark operations."""

    def test_create_benchmark_creates_directory(self) -> None:
        """create_benchmark creates benchmark directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            bench_dir = exp.create_benchmark(
                name="tournament_001",
                config={"games_per_matchup": 100},
                checkpoints=["run_a", "run_b"],
            )

            assert bench_dir.exists()
            assert (bench_dir / CONFIG_FILE).exists()
            assert (bench_dir / NOTES_FILE).exists()

    def test_create_benchmark_updates_manifest(self) -> None:
        """create_benchmark adds entry to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_benchmark(
                name="my_benchmark",
                config={"games_per_matchup": 50},
                checkpoints=["run_x", "run_y"],
            )

            manifest = exp.load_manifest()
            assert "my_benchmark" in manifest.benchmarks
            entry = manifest.benchmarks["my_benchmark"]
            assert entry.checkpoints == ["run_x", "run_y"]

    def test_save_benchmark_results(self) -> None:
        """save_benchmark_results writes results.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            bench_dir = exp.create_benchmark(
                name="tournament",
                config={},
                checkpoints=["run_a"],
            )

            results = {"elo": {"run_a": 1500}, "win_rate": 0.6}
            exp.save_benchmark_results("tournament", results)

            import json

            results_path = bench_dir / RESULTS_FILE
            loaded = json.loads(results_path.read_text())
            assert loaded == results

    def test_create_benchmark_fails_if_exists(self) -> None:
        """create_benchmark raises FileExistsError if benchmark already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_benchmark(name="duplicate", config={}, checkpoints=[])

            with pytest.raises(FileExistsError):
                exp.create_benchmark(name="duplicate", config={}, checkpoints=[])


class TestManifestPersistence:
    """Tests for manifest persistence and roundtrip."""

    def test_manifest_persists_across_instances(self) -> None:
        """Manifest changes persist across ExperimentManager instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance creates data
            exp1 = ExperimentManager(tmpdir)
            exp1.create_batch(
                group="group1",
                mcts_config=DecoupledPUCTConfig(simulations=100),
                game_params=GameParams(width=5, height=5, max_turns=30, cheese_count=5),
            )
            exp1.create_run(name="run1", config={"lr": 0.001}, source_shards="s1")

            # Second instance can read data
            exp2 = ExperimentManager(tmpdir)
            manifest = exp2.load_manifest()

            assert len(manifest.batches) == 1
            assert "run1" in manifest.runs

    def test_manifest_yaml_is_human_readable(self) -> None:
        """Manifest YAML file is human-readable (not minified)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = ExperimentManager(tmpdir)
            exp.create_run(
                name="readable_test",
                config={"model": {"hidden_dim": 256}},
                source_shards="s1",
            )

            manifest_path = Path(tmpdir) / MANIFEST_FILE
            content = manifest_path.read_text()

            # Should have newlines (not minified)
            assert "\n" in content
            # Should have human-readable keys
            assert "runs:" in content
            assert "readable_test:" in content
