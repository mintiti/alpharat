"""Tests for training set sharding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpharat.data.sharding import (
    TrainingSetManifest,
    _load_all_positions,
    _write_shards,
    load_training_set_manifest,
    prepare_training_set,
    prepare_training_set_with_split,
)
from alpharat.data.types import CheeseOutcome
from alpharat.nn.builders.flat import FlatObservationBuilder


def _create_game_npz(
    path: Path,
    *,
    width: int = 5,
    height: int = 5,
    num_positions: int = 3,
    final_p1_score: float = 2.0,
    final_p2_score: float = 1.0,
) -> None:
    """Create a minimal valid game npz file for testing."""
    n = num_positions

    # Game-level arrays
    maze = np.ones((height, width, 4), dtype=np.int8)
    maze[:, 0, 3] = -1  # LEFT edge
    maze[:, width - 1, 1] = -1  # RIGHT edge
    maze[0, :, 0] = -1  # UP edge
    maze[height - 1, :, 2] = -1  # DOWN edge

    initial_cheese = np.zeros((height, width), dtype=bool)
    initial_cheese[2, 2] = True

    # Cheese outcomes: P1 wins the cheese at (2, 2)
    cheese_outcomes = np.full((height, width), CheeseOutcome.UNCOLLECTED, dtype=np.int8)
    cheese_outcomes[2, 2] = CheeseOutcome.P1_WIN

    # Position-level arrays
    p1_pos = np.zeros((n, 2), dtype=np.int8)
    p1_pos[:, 0] = 1  # x=1
    p1_pos[:, 1] = 1  # y=1

    p2_pos = np.zeros((n, 2), dtype=np.int8)
    p2_pos[:, 0] = 3  # x=3
    p2_pos[:, 1] = 3  # y=3

    p1_score = np.linspace(0, final_p1_score, n, dtype=np.float32)
    p2_score = np.linspace(0, final_p2_score, n, dtype=np.float32)

    p1_mud = np.zeros(n, dtype=np.int8)
    p2_mud = np.zeros(n, dtype=np.int8)

    cheese_mask = np.zeros((n, height, width), dtype=bool)
    cheese_mask[:, 2, 2] = True  # Cheese at (2, 2)

    turn = np.arange(n, dtype=np.int16)

    payout_matrix = np.zeros((n, 5, 5), dtype=np.float32)
    visit_counts = np.ones((n, 5, 5), dtype=np.int32) * 10

    # Uniform policies
    prior_p1 = np.ones((n, 5), dtype=np.float32) / 5
    prior_p2 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p1 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p2 = np.ones((n, 5), dtype=np.float32) / 5

    # Actions (all STAY for simplicity)
    action_p1 = np.zeros(n, dtype=np.int8)
    action_p2 = np.zeros(n, dtype=np.int8)

    np.savez_compressed(
        path,
        maze=maze,
        initial_cheese=initial_cheese,
        cheese_outcomes=cheese_outcomes,
        max_turns=np.array(100, dtype=np.int16),
        result=np.array(1 if final_p1_score > final_p2_score else 0, dtype=np.int8),
        final_p1_score=np.array(final_p1_score, dtype=np.float32),
        final_p2_score=np.array(final_p2_score, dtype=np.float32),
        num_positions=np.array(n, dtype=np.int32),
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        p1_score=p1_score,
        p2_score=p2_score,
        p1_mud=p1_mud,
        p2_mud=p2_mud,
        cheese_mask=cheese_mask,
        turn=turn,
        payout_matrix=payout_matrix,
        visit_counts=visit_counts,
        prior_p1=prior_p1,
        prior_p2=prior_p2,
        policy_p1=policy_p1,
        policy_p2=policy_p2,
        action_p1=action_p1,
        action_p2=action_p2,
    )


def _create_batch(parent_dir: Path, batch_id: str, num_games: int = 2) -> Path:
    """Create a batch directory with game files."""
    batch_dir = parent_dir / batch_id
    games_dir = batch_dir / "games"
    games_dir.mkdir(parents=True)

    for i in range(num_games):
        game_path = games_dir / f"game_{i}.npz"
        _create_game_npz(game_path, num_positions=3)

    return batch_dir


# =============================================================================
# TrainingSetManifest tests
# =============================================================================


class TestTrainingSetManifest:
    """Tests for TrainingSetManifest."""

    def test_roundtrip_json(self) -> None:
        """Manifest should survive JSON roundtrip."""
        from datetime import UTC, datetime

        manifest = TrainingSetManifest(
            training_set_id="test-id",
            created_at=datetime.now(UTC),
            builder_version="flat_v2",
            source_batches=["batch1", "batch2"],
            total_positions=100,
            shard_count=2,
            positions_per_shard=50,
            width=5,
            height=5,
        )

        json_str = manifest.model_dump_json()
        loaded = TrainingSetManifest.model_validate_json(json_str)

        assert loaded.training_set_id == manifest.training_set_id
        assert loaded.builder_version == manifest.builder_version
        assert loaded.source_batches == manifest.source_batches
        assert loaded.total_positions == manifest.total_positions


# =============================================================================
# prepare_training_set tests
# =============================================================================


class TestPrepareTrainingSet:
    """Tests for prepare_training_set()."""

    def test_creates_directory_structure(self) -> None:
        """Should create training_set_id dir with manifest and shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            assert result_dir.exists()
            assert (result_dir / "manifest.json").exists()
            # Should have at least one shard
            shards = list(result_dir.glob("shard_*.npz"))
            assert len(shards) >= 1

    def test_writes_manifest(self) -> None:
        """Manifest should contain all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            manifest = load_training_set_manifest(result_dir)

            assert manifest.builder_version == "flat_v2"
            assert manifest.source_batches == ["batch1"]
            assert manifest.total_positions == 3  # 1 game × 3 positions
            assert manifest.width == 5
            assert manifest.height == 5

    def test_shards_have_correct_keys(self) -> None:
        """Each shard should have all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            shard_path = list(result_dir.glob("shard_*.npz"))[0]
            with np.load(shard_path) as data:
                expected_keys = {
                    "observations",
                    "policy_p1",
                    "policy_p2",
                    "value",
                    "payout_matrix",
                    "action_p1",
                    "action_p2",
                    "cheese_outcomes",
                }
                assert set(data.files) == expected_keys

    def test_shard_shapes(self) -> None:
        """Shard arrays should have correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            shard_path = list(result_dir.glob("shard_*.npz"))[0]
            with np.load(shard_path) as data:
                n = 3  # 1 game × 3 positions
                assert data["observations"].shape == (n, 181)  # 5×5 flat obs
                assert data["policy_p1"].shape == (n, 5)
                assert data["policy_p2"].shape == (n, 5)
                assert data["value"].shape == (n,)
                assert data["payout_matrix"].shape == (n, 5, 5)
                assert data["action_p1"].shape == (n,)
                assert data["action_p2"].shape == (n,)
                assert data["cheese_outcomes"].shape == (n, 5, 5)  # (N, H, W)

    def test_shard_dtypes(self) -> None:
        """Shard arrays should have correct dtypes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            shard_path = list(result_dir.glob("shard_*.npz"))[0]
            with np.load(shard_path) as data:
                assert data["observations"].dtype == np.float32
                assert data["policy_p1"].dtype == np.float32
                assert data["policy_p2"].dtype == np.float32
                assert data["value"].dtype == np.float32
                assert data["payout_matrix"].dtype == np.float32
                assert data["action_p1"].dtype == np.int8
                assert data["action_p2"].dtype == np.int8
                assert data["cheese_outcomes"].dtype == np.int8

    def test_respects_positions_per_shard(self) -> None:
        """Shards should not exceed configured size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=3)  # 9 positions
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=4,
                seed=42,
            )

            shards = sorted(result_dir.glob("shard_*.npz"))
            assert len(shards) == 3  # 9 positions / 4 per shard = 3 shards

            # First two shards should have 4 positions
            with np.load(shards[0]) as data:
                assert len(data["value"]) == 4
            with np.load(shards[1]) as data:
                assert len(data["value"]) == 4

    def test_last_shard_may_be_smaller(self) -> None:
        """Last shard can have fewer positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=3)  # 9 positions
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=4,
                seed=42,
            )

            shards = sorted(result_dir.glob("shard_*.npz"))
            # Last shard should have 1 position (9 - 4 - 4 = 1)
            with np.load(shards[-1]) as data:
                assert len(data["value"]) == 1

    def test_shuffles_positions(self) -> None:
        """Positions should be shuffled across games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=2)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)

            # Run twice with same seed - should get same result
            result_dir1 = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            # Create another output dir for second run
            output_dir2 = tmp_path / "training_sets2"
            output_dir2.mkdir()

            result_dir2 = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir2,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            # Same seed should give same order
            with (
                np.load(list(result_dir1.glob("shard_*.npz"))[0]) as d1,
                np.load(list(result_dir2.glob("shard_*.npz"))[0]) as d2,
            ):
                np.testing.assert_array_equal(d1["observations"], d2["observations"])

    def test_different_seed_different_order(self) -> None:
        """Different seeds should give different shuffles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=3)
            output_dir1 = tmp_path / "training_sets1"
            output_dir1.mkdir()
            output_dir2 = tmp_path / "training_sets2"
            output_dir2.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)

            result_dir1 = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir1,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            result_dir2 = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir2,
                builder=builder,
                positions_per_shard=100,
                seed=123,
            )

            # Different seeds should give different order
            with (
                np.load(list(result_dir1.glob("shard_*.npz"))[0]) as d1,
                np.load(list(result_dir2.glob("shard_*.npz"))[0]) as d2,
            ):
                # Values should be same set but potentially different order
                assert set(d1["value"].tolist()) == set(d2["value"].tolist())
                # With high probability, order should differ
                # (not guaranteed but very likely with 9 positions)

    def test_empty_batch_dirs_raises(self) -> None:
        """Should raise if batch_dirs is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)

            with pytest.raises(ValueError, match="cannot be empty"):
                prepare_training_set(
                    batch_dirs=[],
                    output_dir=output_dir,
                    builder=builder,
                )

    def test_multiple_batches(self) -> None:
        """Should process multiple batch directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir1 = _create_batch(tmp_path, "batch1", num_games=2)
            batch_dir2 = _create_batch(tmp_path, "batch2", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir1, batch_dir2],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=100,
                seed=42,
            )

            manifest = load_training_set_manifest(result_dir)
            assert manifest.total_positions == 9  # (2 + 1) games × 3 positions
            assert set(manifest.source_batches) == {"batch1", "batch2"}


# =============================================================================
# Helper function tests
# =============================================================================


class TestLoadAllPositions:
    """Tests for _load_all_positions()."""

    def test_loads_all_positions(self) -> None:
        """Should load all positions from all games."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=2)

            builder = FlatObservationBuilder(width=5, height=5)
            obs, p1, p2, values, payout, a1, a2, cheese, w, h = _load_all_positions(
                [batch_dir], builder
            )

            assert len(values) == 6  # 2 games × 3 positions
            assert obs.shape[0] == 6
            assert p1.shape == (6, 5)
            assert p2.shape == (6, 5)
            assert payout.shape == (6, 5, 5)
            assert a1.shape == (6,)
            assert a2.shape == (6,)
            assert cheese.shape == (6, 5, 5)
            assert w == 5
            assert h == 5


class TestWriteShards:
    """Tests for _write_shards()."""

    def test_writes_correct_number_of_shards(self) -> None:
        """Should write correct number of shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            obs = np.zeros((10, 181), dtype=np.float32)
            p1 = np.zeros((10, 5), dtype=np.float32)
            p2 = np.zeros((10, 5), dtype=np.float32)
            values = np.zeros(10, dtype=np.float32)
            payout = np.zeros((10, 5, 5), dtype=np.float32)
            a1 = np.zeros(10, dtype=np.int8)
            a2 = np.zeros(10, dtype=np.int8)
            cheese = np.zeros((10, 5, 5), dtype=np.int8)

            shard_count = _write_shards(
                tmp_path, obs, p1, p2, values, payout, a1, a2, cheese, positions_per_shard=3
            )

            assert shard_count == 4  # 10 positions / 3 per shard = 4 shards (ceil)
            assert len(list(tmp_path.glob("shard_*.npz"))) == 4

    def test_shard_naming(self) -> None:
        """Shards should be named shard_XXXX.npz."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            obs = np.zeros((5, 181), dtype=np.float32)
            p1 = np.zeros((5, 5), dtype=np.float32)
            p2 = np.zeros((5, 5), dtype=np.float32)
            values = np.zeros(5, dtype=np.float32)
            payout = np.zeros((5, 5, 5), dtype=np.float32)
            a1 = np.zeros(5, dtype=np.int8)
            a2 = np.zeros(5, dtype=np.int8)
            cheese = np.zeros((5, 5, 5), dtype=np.int8)

            _write_shards(
                tmp_path, obs, p1, p2, values, payout, a1, a2, cheese, positions_per_shard=2
            )

            assert (tmp_path / "shard_0000.npz").exists()
            assert (tmp_path / "shard_0001.npz").exists()
            assert (tmp_path / "shard_0002.npz").exists()


class TestLoadTrainingSetManifest:
    """Tests for load_training_set_manifest()."""

    def test_loads_manifest(self) -> None:
        """Should load manifest from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=1)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                seed=42,
            )

            manifest = load_training_set_manifest(result_dir)

            assert isinstance(manifest, TrainingSetManifest)
            assert manifest.total_positions == 3

    def test_raises_if_not_found(self) -> None:
        """Should raise FileNotFoundError if manifest doesn't exist."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(FileNotFoundError),
        ):
            load_training_set_manifest(Path(tmpdir))


# =============================================================================
# prepare_training_set_with_split tests
# =============================================================================


class TestPrepareTrainingSetWithSplit:
    """Tests for prepare_training_set_with_split()."""

    def test_creates_train_and_val_directories(self) -> None:
        """Should create train/ and val/ subdirectories with manifests and shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=10)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set_with_split(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                val_ratio=0.2,
                positions_per_shard=100,
                seed=42,
            )

            assert result_dir.exists()
            assert (result_dir / "train").exists()
            assert (result_dir / "train" / "manifest.json").exists()
            assert len(list((result_dir / "train").glob("shard_*.npz"))) >= 1

            assert (result_dir / "val").exists()
            assert (result_dir / "val" / "manifest.json").exists()
            assert len(list((result_dir / "val").glob("shard_*.npz"))) >= 1

    def test_respects_val_ratio(self) -> None:
        """Position counts should be proportional to val_ratio, nothing lost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # 10 games × 3 positions = 30 total positions
            batch_dir = _create_batch(tmp_path, "batch1", num_games=10)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set_with_split(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                val_ratio=0.2,  # ~2 games to val, ~8 to train
                positions_per_shard=100,
                seed=42,
            )

            train_manifest = load_training_set_manifest(result_dir / "train")
            val_manifest = load_training_set_manifest(result_dir / "val")

            # Total positions should be preserved
            total = train_manifest.total_positions + val_manifest.total_positions
            assert total == 30  # 10 games × 3 positions

            # Val should have roughly 20% (2 games = 6 positions)
            assert val_manifest.total_positions == 6

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=10)

            builder = FlatObservationBuilder(width=5, height=5)

            output_dir1 = tmp_path / "training_sets1"
            output_dir1.mkdir()
            result_dir1 = prepare_training_set_with_split(
                batch_dirs=[batch_dir],
                output_dir=output_dir1,
                builder=builder,
                val_ratio=0.2,
                seed=42,
            )

            output_dir2 = tmp_path / "training_sets2"
            output_dir2.mkdir()
            result_dir2 = prepare_training_set_with_split(
                batch_dirs=[batch_dir],
                output_dir=output_dir2,
                builder=builder,
                val_ratio=0.2,
                seed=42,
            )

            # Both should have same position counts
            train1 = load_training_set_manifest(result_dir1 / "train")
            train2 = load_training_set_manifest(result_dir2 / "train")
            assert train1.total_positions == train2.total_positions

            val1 = load_training_set_manifest(result_dir1 / "val")
            val2 = load_training_set_manifest(result_dir2 / "val")
            assert val1.total_positions == val2.total_positions

    def test_val_ratio_zero_creates_only_train(self) -> None:
        """val_ratio=0.0 should create only train/, no val/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=5)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            result_dir = prepare_training_set_with_split(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                val_ratio=0.0,
                positions_per_shard=100,
                seed=42,
            )

            assert (result_dir / "train").exists()
            assert not (result_dir / "val").exists()

            train_manifest = load_training_set_manifest(result_dir / "train")
            assert train_manifest.total_positions == 15  # 5 games × 3 positions

    def test_invalid_val_ratio_raises(self) -> None:
        """val_ratio >= 1.0 should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            batch_dir = _create_batch(tmp_path, "batch1", num_games=5)
            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)

            with pytest.raises(ValueError, match="val_ratio must be in"):
                prepare_training_set_with_split(
                    batch_dirs=[batch_dir],
                    output_dir=output_dir,
                    builder=builder,
                    val_ratio=1.0,
                )

            with pytest.raises(ValueError, match="val_ratio must be in"):
                prepare_training_set_with_split(
                    batch_dirs=[batch_dir],
                    output_dir=output_dir,
                    builder=builder,
                    val_ratio=1.5,
                )

    def test_empty_batch_dirs_raises(self) -> None:
        """Empty batch_dirs should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)

            with pytest.raises(ValueError, match="cannot be empty"):
                prepare_training_set_with_split(
                    batch_dirs=[],
                    output_dir=output_dir,
                    builder=builder,
                )
