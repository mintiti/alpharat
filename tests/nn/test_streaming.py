"""Tests for StreamingDataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alpharat.data.sharding import prepare_training_set
from alpharat.data.types import CheeseOutcome
from alpharat.nn.builders.flat import FlatDataset, FlatObservationBuilder
from alpharat.nn.streaming import StreamingDataset


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

    maze = np.ones((height, width, 4), dtype=np.int8)
    maze[:, 0, 3] = -1
    maze[:, width - 1, 1] = -1
    maze[0, :, 0] = -1
    maze[height - 1, :, 2] = -1

    initial_cheese = np.zeros((height, width), dtype=bool)
    initial_cheese[2, 2] = True

    cheese_outcomes = np.full((height, width), CheeseOutcome.UNCOLLECTED, dtype=np.int8)
    cheese_outcomes[2, 2] = CheeseOutcome.P1_WIN

    p1_pos = np.zeros((n, 2), dtype=np.int8)
    p1_pos[:, 0] = 1
    p1_pos[:, 1] = 1

    p2_pos = np.zeros((n, 2), dtype=np.int8)
    p2_pos[:, 0] = 3
    p2_pos[:, 1] = 3

    p1_score = np.linspace(0, final_p1_score, n, dtype=np.float32)
    p2_score = np.linspace(0, final_p2_score, n, dtype=np.float32)

    p1_mud = np.zeros(n, dtype=np.int8)
    p2_mud = np.zeros(n, dtype=np.int8)

    cheese_mask = np.zeros((n, height, width), dtype=bool)
    cheese_mask[:, 2, 2] = True

    turn = np.arange(n, dtype=np.int16)

    value_p1 = np.zeros(n, dtype=np.float32)
    value_p2 = np.zeros(n, dtype=np.float32)
    visit_counts_p1 = np.ones((n, 5), dtype=np.int32) * 10
    visit_counts_p2 = np.ones((n, 5), dtype=np.int32) * 10

    prior_p1 = np.ones((n, 5), dtype=np.float32) / 5
    prior_p2 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p1 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p2 = np.ones((n, 5), dtype=np.float32) / 5
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
        value_p1=value_p1,
        value_p2=value_p2,
        visit_counts_p1=visit_counts_p1,
        visit_counts_p2=visit_counts_p2,
        prior_p1=prior_p1,
        prior_p2=prior_p2,
        policy_p1=policy_p1,
        policy_p2=policy_p2,
        action_p1=action_p1,
        action_p2=action_p2,
    )


def _create_training_set(
    tmp_path: Path, num_games: int = 2, positions_per_shard: int = 100
) -> Path:
    """Create a training set for testing."""
    batch_dir = tmp_path / "batch1"
    games_dir = batch_dir / "games"
    games_dir.mkdir(parents=True)

    for i in range(num_games):
        game_path = games_dir / f"game_{i}.npz"
        _create_game_npz(game_path, num_positions=3)

    output_dir = tmp_path / "training_sets"
    output_dir.mkdir()

    builder = FlatObservationBuilder(width=5, height=5)
    return prepare_training_set(
        batch_dirs=[batch_dir],
        output_dir=output_dir,
        builder=builder,
        positions_per_shard=positions_per_shard,
        seed=42,
    )


class TestStreamingDataset:
    """Tests for StreamingDataset."""

    def test_len_returns_total_positions(self) -> None:
        """len() should return total positions from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = StreamingDataset(training_set_dir)

            assert len(dataset) == 6  # 2 games × 3 positions

    def test_yields_correct_sample_keys(self) -> None:
        """Samples should have same keys as FlatDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = StreamingDataset(training_set_dir)

            sample = next(iter(dataset))

            assert "observation" in sample
            assert "policy_p1" in sample
            assert "policy_p2" in sample
            assert "p1_value" in sample
            assert "p2_value" in sample

    def test_sample_shapes_match_flat_dataset(self) -> None:
        """Sample shapes should match FlatDataset format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = StreamingDataset(training_set_dir)

            sample = next(iter(dataset))

            assert sample["observation"].shape == (181,)
            assert sample["policy_p1"].shape == (5,)
            assert sample["policy_p2"].shape == (5,)
            assert sample["p1_value"].shape == (1,)
            assert sample["p2_value"].shape == (1,)

    def test_samples_are_tensors(self) -> None:
        """Samples should be torch tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = StreamingDataset(training_set_dir)

            sample = next(iter(dataset))

            assert isinstance(sample["observation"], torch.Tensor)
            assert isinstance(sample["policy_p1"], torch.Tensor)
            assert isinstance(sample["policy_p2"], torch.Tensor)
            assert isinstance(sample["p1_value"], torch.Tensor)
            assert isinstance(sample["p2_value"], torch.Tensor)

    def test_iterates_all_positions(self) -> None:
        """Full iteration should yield all positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = StreamingDataset(training_set_dir)

            samples = list(dataset)

            assert len(samples) == 6  # 2 games × 3 positions

    def test_shuffle_shards_deterministic(self) -> None:
        """Same seed should produce same shard order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training set with multiple shards
            training_set_dir = _create_training_set(
                Path(tmpdir), num_games=6, positions_per_shard=3
            )

            dataset1 = StreamingDataset(training_set_dir, shuffle_shards=True, seed=42)
            dataset2 = StreamingDataset(training_set_dir, shuffle_shards=True, seed=42)

            samples1 = [s["observation"].numpy().copy() for s in dataset1]
            samples2 = [s["observation"].numpy().copy() for s in dataset2]

            assert len(samples1) == len(samples2)
            for s1, s2 in zip(samples1, samples2, strict=True):
                np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_different_order(self) -> None:
        """Different seeds should produce different shard order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training set with multiple shards
            training_set_dir = _create_training_set(
                Path(tmpdir), num_games=6, positions_per_shard=3
            )

            dataset1 = StreamingDataset(training_set_dir, shuffle_shards=True, seed=42)
            dataset2 = StreamingDataset(training_set_dir, shuffle_shards=True, seed=123)

            samples1 = [s["observation"].numpy().copy() for s in dataset1]
            samples2 = [s["observation"].numpy().copy() for s in dataset2]

            # Should have same total but different order
            assert len(samples1) == len(samples2)

            # At least some samples should be in different positions
            differences = sum(
                1 for s1, s2 in zip(samples1, samples2, strict=True) if not np.array_equal(s1, s2)
            )
            assert differences > 0

    def test_no_shuffle_same_order(self) -> None:
        """Without shuffle, order should be consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(
                Path(tmpdir), num_games=4, positions_per_shard=3
            )

            dataset1 = StreamingDataset(training_set_dir, shuffle_shards=False)
            dataset2 = StreamingDataset(training_set_dir, shuffle_shards=False)

            samples1 = [s["observation"].numpy().copy() for s in dataset1]
            samples2 = [s["observation"].numpy().copy() for s in dataset2]

            for s1, s2 in zip(samples1, samples2, strict=True):
                np.testing.assert_array_equal(s1, s2)

    def test_manifest_property(self) -> None:
        """manifest should return TrainingSetManifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = StreamingDataset(training_set_dir)

            manifest = dataset.manifest

            assert manifest.width == 5
            assert manifest.height == 5
            assert manifest.total_positions == 6


class TestStreamingDatasetWithDataLoader:
    """Tests for StreamingDataset with DataLoader."""

    def test_works_with_dataloader(self) -> None:
        """Should work with DataLoader for batching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = StreamingDataset(training_set_dir)
            loader = DataLoader(dataset, batch_size=2)

            batches = list(loader)

            assert len(batches) == 3  # 6 samples / batch_size 2
            assert batches[0]["observation"].shape == (2, 181)
            assert batches[0]["policy_p1"].shape == (2, 5)
            assert batches[0]["policy_p2"].shape == (2, 5)
            assert batches[0]["p1_value"].shape == (2, 1)
            assert batches[0]["p2_value"].shape == (2, 1)

    def test_works_with_dataloader_partial_batch(self) -> None:
        """Should handle partial final batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = StreamingDataset(training_set_dir)
            loader = DataLoader(dataset, batch_size=4)

            batches = list(loader)

            assert len(batches) == 2  # 6 samples: batch of 4, batch of 2
            assert batches[0]["observation"].shape == (4, 181)
            assert batches[1]["observation"].shape == (2, 181)

    def test_works_with_dataloader_drop_last(self) -> None:
        """Should work with drop_last=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = StreamingDataset(training_set_dir)
            loader = DataLoader(dataset, batch_size=4, drop_last=True)

            batches = list(loader)

            assert len(batches) == 1  # Only full batch of 4
            assert batches[0]["observation"].shape == (4, 181)

    def test_batch_larger_than_shard(self) -> None:
        """Should handle batch_size > positions_per_shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 12 positions across 4 shards (3 positions each)
            training_set_dir = _create_training_set(
                Path(tmpdir), num_games=4, positions_per_shard=3
            )
            dataset = StreamingDataset(training_set_dir, shuffle_shards=False)

            # batch_size=5 spans multiple shards (each shard has 3 positions)
            loader = DataLoader(dataset, batch_size=5)

            batches = list(loader)

            # 12 positions / batch_size 5 = 2 full batches + 1 partial
            assert len(batches) == 3
            assert batches[0]["observation"].shape == (5, 181)
            assert batches[1]["observation"].shape == (5, 181)
            assert batches[2]["observation"].shape == (2, 181)

            # Verify we got all positions
            total_samples = sum(b["observation"].shape[0] for b in batches)
            assert total_samples == 12


class TestStreamingDatasetVsFlatDataset:
    """Tests comparing StreamingDataset to FlatDataset."""

    def test_yields_same_count_as_flat_dataset(self) -> None:
        """StreamingDataset should yield same number of samples as FlatDataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)

            flat = FlatDataset(training_set_dir)
            streaming = StreamingDataset(training_set_dir, shuffle_shards=False)

            streaming_samples = list(streaming)

            assert len(streaming_samples) == len(flat)

    def test_streaming_samples_have_correct_shapes(self) -> None:
        """All streaming samples should have correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            streaming = StreamingDataset(training_set_dir)

            for sample in streaming:
                assert sample["observation"].shape == (181,)
                assert sample["policy_p1"].shape == (5,)
                assert sample["policy_p2"].shape == (5,)
                assert sample["p1_value"].shape == (1,)
                assert sample["p2_value"].shape == (1,)
                assert sample["action_p1"].shape == (1,)
                assert sample["action_p2"].shape == (1,)
