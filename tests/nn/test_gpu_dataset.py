"""Tests for GPUDataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from alpharat.data.sharding import prepare_training_set
from alpharat.data.types import CheeseOutcome
from alpharat.nn.builders.flat import FlatObservationBuilder
from alpharat.nn.gpu_dataset import GPUDataset


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

    payout_matrix = np.zeros((n, 2, 5, 5), dtype=np.float32)
    visit_counts = np.ones((n, 5, 5), dtype=np.int32) * 10

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
        payout_matrix=payout_matrix,
        visit_counts=visit_counts,
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


class TestGPUDataset:
    """Tests for GPUDataset."""

    def test_len_returns_total_positions(self) -> None:
        """len() should return total positions from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            assert len(dataset) == 6  # 2 games × 3 positions

    def test_data_on_correct_device(self) -> None:
        """Data should be on the specified device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            for batch in dataset.epoch_iter(2, augment=False):
                assert batch["observation"].device == torch.device("cpu")
                assert batch["policy_p1"].device == torch.device("cpu")
                break

    def test_manifest_properties(self) -> None:
        """width/height should come from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            assert dataset.width == 5
            assert dataset.height == 5
            assert dataset.manifest.total_positions == 6

    def test_epoch_iter_yields_correct_batch_size(self) -> None:
        """epoch_iter should yield batches of requested size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=4)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            batches = list(dataset.epoch_iter(3, augment=False))

            # 12 positions / batch_size 3 = 4 complete batches
            assert len(batches) == 4
            for batch in batches:
                assert batch["observation"].shape[0] == 3

    def test_epoch_iter_drops_incomplete_batch(self) -> None:
        """epoch_iter should drop incomplete final batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            # 6 positions / batch_size 4 = 1 complete batch (2 dropped)
            batches = list(dataset.epoch_iter(4, augment=False))

            assert len(batches) == 1
            assert batches[0]["observation"].shape[0] == 4

    def test_batch_has_all_keys(self) -> None:
        """Batches should have all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            batch = next(dataset.epoch_iter(2, augment=False))

            assert "observation" in batch
            assert "policy_p1" in batch
            assert "policy_p2" in batch
            assert "p1_value" in batch
            assert "p2_value" in batch
            assert "payout_matrix" in batch
            assert "action_p1" in batch
            assert "action_p2" in batch

    def test_batch_shapes(self) -> None:
        """Batch tensors should have correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            batch = next(dataset.epoch_iter(2, augment=False))

            assert batch["observation"].shape == (2, 181)
            assert batch["policy_p1"].shape == (2, 5)
            assert batch["policy_p2"].shape == (2, 5)
            assert batch["p1_value"].shape == (2, 1)
            assert batch["p2_value"].shape == (2, 1)
            assert batch["payout_matrix"].shape == (2, 2, 5, 5)
            assert batch["action_p1"].shape == (2, 1)
            assert batch["action_p2"].shape == (2, 1)

    def test_no_shuffle_same_order(self) -> None:
        """Without shuffle, order should be consistent across epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            batches1 = list(dataset.epoch_iter(2, augment=False, shuffle=False))
            batches2 = list(dataset.epoch_iter(2, augment=False, shuffle=False))

            for b1, b2 in zip(batches1, batches2, strict=True):
                torch.testing.assert_close(b1["observation"], b2["observation"])

    def test_shuffle_changes_order(self) -> None:
        """With shuffle, order should change between epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=10)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            # Get observations from two shuffled epochs
            obs1 = torch.cat([b["observation"] for b in dataset.epoch_iter(5, augment=False)])
            obs2 = torch.cat([b["observation"] for b in dataset.epoch_iter(5, augment=False)])

            # Should have same content but likely different order
            assert obs1.shape == obs2.shape

            # Check that at least some positions differ (shuffle happened)
            # Note: There's a tiny chance this fails if shuffle produces same order
            differences = (obs1 != obs2).any(dim=1).sum().item()
            assert differences > 0


class TestGPUDatasetAugmentation:
    """Tests for GPUDataset augmentation."""

    def test_augment_changes_data(self) -> None:
        """Augmentation should modify some data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=10)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            # Get data without augmentation
            batches_no_aug = list(dataset.epoch_iter(5, augment=False, shuffle=False))

            # Get data with augmentation (p_swap=1.0 to guarantee changes)
            batches_aug = list(dataset.epoch_iter(5, augment=True, p_swap=1.0, shuffle=False))

            # Values should be swapped when swapped
            for b_no, b_aug in zip(batches_no_aug, batches_aug, strict=True):
                # All values should be swapped (p_swap=1.0)
                torch.testing.assert_close(b_no["p1_value"], b_aug["p2_value"])
                torch.testing.assert_close(b_no["p2_value"], b_aug["p1_value"])

    def test_augment_swaps_policies(self) -> None:
        """Augmentation should swap p1/p2 policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=4)
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            # Get data with guaranteed swap
            batches_no_aug = list(dataset.epoch_iter(4, augment=False, shuffle=False))
            batches_aug = list(dataset.epoch_iter(4, augment=True, p_swap=1.0, shuffle=False))

            for b_no, b_aug in zip(batches_no_aug, batches_aug, strict=True):
                # Policies should be swapped
                torch.testing.assert_close(b_no["policy_p1"], b_aug["policy_p2"])
                torch.testing.assert_close(b_no["policy_p2"], b_aug["policy_p1"])

    def test_no_augment_preserves_data(self) -> None:
        """augment=False should not modify data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = GPUDataset(training_set_dir, torch.device("cpu"))

            # Get same data twice without augmentation
            batches1 = list(dataset.epoch_iter(2, augment=False, shuffle=False))
            batches2 = list(dataset.epoch_iter(2, augment=False, shuffle=False))

            for b1, b2 in zip(batches1, batches2, strict=True):
                torch.testing.assert_close(b1["p1_value"], b2["p1_value"])
                torch.testing.assert_close(b1["p2_value"], b2["p2_value"])
                torch.testing.assert_close(b1["policy_p1"], b2["policy_p1"])


@pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.backends.mps.is_available(),
    reason="No GPU available",
)
class TestGPUDatasetOnGPU:
    """Tests that require actual GPU."""

    def test_data_on_gpu(self) -> None:
        """Data should be on GPU when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))

            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                pytest.skip("No GPU available")

            dataset = GPUDataset(training_set_dir, device)

            batch = next(dataset.epoch_iter(2, augment=False))
            assert batch["observation"].device.type == device.type
