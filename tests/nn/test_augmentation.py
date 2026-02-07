"""Tests for data augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpharat.nn.augmentation import (
    BatchAugmentation,
    swap_player_perspective,
    swap_player_perspective_batch,
)
from alpharat.nn.builders.flat import MAX_MUD_TURNS, MAX_SCORE, FlatObservationBuilder
from alpharat.nn.types import ObservationInput


def _make_observation_input(
    *,
    width: int = 5,
    height: int = 5,
    p1_pos: tuple[int, int] = (1, 2),
    p2_pos: tuple[int, int] = (3, 4),
    p1_score: float = 5.0,
    p2_score: float = 2.0,
    p1_mud: int = 3,
    p2_mud: int = 0,
    turn: int = 10,
    max_turns: int = 100,
) -> ObservationInput:
    """Create asymmetric ObservationInput for testing."""
    maze = np.ones((height, width, 4), dtype=np.int8)
    # Add some walls at edges
    maze[:, 0, 3] = -1  # Left edge
    maze[:, width - 1, 1] = -1  # Right edge

    cheese_mask = np.zeros((height, width), dtype=bool)
    cheese_mask[2, 2] = True

    return ObservationInput(
        maze=maze,
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        cheese_mask=cheese_mask,
        p1_score=p1_score,
        p2_score=p2_score,
        turn=turn,
        max_turns=max_turns,
        p1_mud=p1_mud,
        p2_mud=p2_mud,
        width=width,
        height=height,
    )


def _make_sample(
    width: int = 5,
    height: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create a complete sample with asymmetric P1/P2 values."""
    builder = FlatObservationBuilder(width=width, height=height)
    obs_input = _make_observation_input(width=width, height=height)
    observation = builder.build(obs_input)

    # Asymmetric policies
    policy_p1 = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
    policy_p2 = np.array([0.1, 0.1, 0.2, 0.3, 0.3], dtype=np.float32)

    # Non-zero asymmetric values (remaining scores)
    value_p1 = np.array([2.5], dtype=np.float32)
    value_p2 = np.array([1.5], dtype=np.float32)

    # Asymmetric actions
    action_p1 = np.array([0], dtype=np.int8)
    action_p2 = np.array([4], dtype=np.int8)

    return (
        observation,
        policy_p1,
        policy_p2,
        value_p1,
        value_p2,
        action_p1,
        action_p2,
    )


class TestSwapPlayerPerspective:
    """Tests for swap_player_perspective."""

    def test_swap_is_involution(self) -> None:
        """swap(swap(x)) = x for all fields."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()
        width, height = 5, 5

        # First swap
        s_obs, s_p1, s_p2, s_p1_val, s_p2_val, s_a1, s_a2 = swap_player_perspective(
            obs, p1, p2, p1_val, p2_val, a1, a2, width, height
        )

        # Second swap
        ss_obs, ss_p1, ss_p2, ss_p1_val, ss_p2_val, ss_a1, ss_a2 = swap_player_perspective(
            s_obs, s_p1, s_p2, s_p1_val, s_p2_val, s_a1, s_a2, width, height
        )

        # Should equal original
        np.testing.assert_array_almost_equal(ss_obs, obs)
        np.testing.assert_array_almost_equal(ss_p1, p1)
        np.testing.assert_array_almost_equal(ss_p2, p2)
        np.testing.assert_array_almost_equal(ss_p1_val, p1_val)
        np.testing.assert_array_almost_equal(ss_p2_val, p2_val)
        np.testing.assert_array_equal(ss_a1, a1)
        np.testing.assert_array_equal(ss_a2, a2)

    def test_swap_is_not_identity(self) -> None:
        """Swap should change asymmetric inputs (not be identity)."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()
        width, height = 5, 5

        s_obs, s_p1, s_p2, s_p1_val, s_p2_val, s_a1, s_a2 = swap_player_perspective(
            obs, p1, p2, p1_val, p2_val, a1, a2, width, height
        )

        # At least some things should be different
        assert not np.allclose(s_obs, obs), "Observation should change"
        assert not np.allclose(s_p1, p1), "Policy P1 should change"
        assert not np.allclose(s_p1_val, p1_val), "P1 value should change"

    def test_swap_positions(self) -> None:
        """P1 and P2 positions are swapped in observation."""
        builder = FlatObservationBuilder(width=5, height=5)
        obs_input = _make_observation_input(p1_pos=(1, 2), p2_pos=(3, 4))
        obs = builder.build(obs_input)

        # Create dummy targets
        p1 = np.ones(5, dtype=np.float32) / 5
        p2 = np.ones(5, dtype=np.float32) / 5
        p1_val = np.array([0.0], dtype=np.float32)
        p2_val = np.array([0.0], dtype=np.float32)
        a1 = np.array([0], dtype=np.int8)
        a2 = np.array([0], dtype=np.int8)

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        spatial = 25
        p1_start = spatial * 4
        p2_start = spatial * 5

        # After swap, p1_pos section should have p2's original position
        # and vice versa
        np.testing.assert_array_equal(
            s_obs[p1_start : p1_start + spatial], obs[p2_start : p2_start + spatial]
        )
        np.testing.assert_array_equal(
            s_obs[p2_start : p2_start + spatial], obs[p1_start : p1_start + spatial]
        )

    def test_swap_score_diff_negated(self) -> None:
        """score_diff is negated after swap."""
        builder = FlatObservationBuilder(width=5, height=5)
        obs_input = _make_observation_input(p1_score=7.0, p2_score=2.0)  # diff = +5
        obs = builder.build(obs_input)

        p1 = np.ones(5, dtype=np.float32) / 5
        p2 = np.ones(5, dtype=np.float32) / 5
        p1_val = np.array([0.0], dtype=np.float32)
        p2_val = np.array([0.0], dtype=np.float32)
        a1 = np.array([0], dtype=np.int8)
        a2 = np.array([0], dtype=np.int8)

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        scalars_start = 25 * 7
        assert obs[scalars_start] == pytest.approx(5.0)  # Original diff
        assert s_obs[scalars_start] == pytest.approx(-5.0)  # Negated

    def test_swap_mud_values(self) -> None:
        """P1 mud and P2 mud are swapped."""
        builder = FlatObservationBuilder(width=5, height=5)
        obs_input = _make_observation_input(p1_mud=5, p2_mud=2)
        obs = builder.build(obs_input)

        p1 = np.ones(5, dtype=np.float32) / 5
        p2 = np.ones(5, dtype=np.float32) / 5
        p1_val = np.array([0.0], dtype=np.float32)
        p2_val = np.array([0.0], dtype=np.float32)
        a1 = np.array([0], dtype=np.int8)
        a2 = np.array([0], dtype=np.int8)

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        scalars_start = 25 * 7
        p1_mud_idx = scalars_start + 2
        p2_mud_idx = scalars_start + 3

        # After swap, p1_mud should have p2's original value and vice versa
        assert obs[p1_mud_idx] == pytest.approx(5.0 / MAX_MUD_TURNS)
        assert obs[p2_mud_idx] == pytest.approx(2.0 / MAX_MUD_TURNS)
        assert s_obs[p1_mud_idx] == pytest.approx(2.0 / MAX_MUD_TURNS)
        assert s_obs[p2_mud_idx] == pytest.approx(5.0 / MAX_MUD_TURNS)

    def test_swap_score_values(self) -> None:
        """P1 score and P2 score are swapped."""
        builder = FlatObservationBuilder(width=5, height=5)
        obs_input = _make_observation_input(p1_score=8.0, p2_score=3.0)
        obs = builder.build(obs_input)

        p1 = np.ones(5, dtype=np.float32) / 5
        p2 = np.ones(5, dtype=np.float32) / 5
        p1_val = np.array([0.0], dtype=np.float32)
        p2_val = np.array([0.0], dtype=np.float32)
        a1 = np.array([0], dtype=np.int8)
        a2 = np.array([0], dtype=np.int8)

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        scalars_start = 25 * 7
        p1_score_idx = scalars_start + 4
        p2_score_idx = scalars_start + 5

        assert obs[p1_score_idx] == pytest.approx(8.0 / MAX_SCORE)
        assert obs[p2_score_idx] == pytest.approx(3.0 / MAX_SCORE)
        assert s_obs[p1_score_idx] == pytest.approx(3.0 / MAX_SCORE)
        assert s_obs[p2_score_idx] == pytest.approx(8.0 / MAX_SCORE)

    def test_swap_policies(self) -> None:
        """P1 and P2 policies are swapped."""
        obs, _, _, p1_val, p2_val, a1, a2 = _make_sample()
        p1 = np.array([0.5, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
        p2 = np.array([0.1, 0.1, 0.2, 0.3, 0.3], dtype=np.float32)

        _, s_p1, s_p2, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        np.testing.assert_array_equal(s_p1, p2)
        np.testing.assert_array_equal(s_p2, p1)

    def test_swap_actions(self) -> None:
        """P1 and P2 actions are swapped."""
        obs, p1, p2, p1_val, p2_val, _, _ = _make_sample()
        a1 = np.array([1], dtype=np.int8)
        a2 = np.array([3], dtype=np.int8)

        *_, s_a1, s_a2 = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        np.testing.assert_array_equal(s_a1, a2)
        np.testing.assert_array_equal(s_a2, a1)

    def test_swap_values_swapped(self) -> None:
        """P1 and P2 values are swapped after swap."""
        obs, p1, p2, _, _, a1, a2 = _make_sample()
        p1_val = np.array([3.5], dtype=np.float32)
        p2_val = np.array([1.5], dtype=np.float32)

        *_, s_p1_val, s_p2_val, _, _ = swap_player_perspective(
            obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5
        )

        assert s_p1_val[0] == pytest.approx(1.5)  # Was p2_val
        assert s_p2_val[0] == pytest.approx(3.5)  # Was p1_val

    def test_swap_preserves_maze(self) -> None:
        """Maze section is unchanged after swap."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        spatial = 25
        maze_end = spatial * 4
        np.testing.assert_array_equal(s_obs[:maze_end], obs[:maze_end])

    def test_swap_preserves_cheese(self) -> None:
        """Cheese section is unchanged after swap."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        spatial = 25
        cheese_start = spatial * 6
        cheese_end = spatial * 7
        np.testing.assert_array_equal(s_obs[cheese_start:cheese_end], obs[cheese_start:cheese_end])

    def test_swap_preserves_progress(self) -> None:
        """Progress scalar is unchanged after swap."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()

        s_obs, *_ = swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        scalars_start = 25 * 7
        progress_idx = scalars_start + 1
        assert s_obs[progress_idx] == obs[progress_idx]

    def test_swap_does_not_mutate_inputs(self) -> None:
        """Swap should not modify input arrays."""
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample()

        # Make copies to compare
        obs_copy = obs.copy()
        p1_copy = p1.copy()
        p1_val_copy = p1_val.copy()

        swap_player_perspective(obs, p1, p2, p1_val, p2_val, a1, a2, 5, 5)

        np.testing.assert_array_equal(obs, obs_copy)
        np.testing.assert_array_equal(p1, p1_copy)
        np.testing.assert_array_equal(p1_val, p1_val_copy)


def _make_batch(batch_size: int = 4, width: int = 5, height: int = 5) -> dict[str, torch.Tensor]:
    """Create a batch of samples for testing batch augmentation."""
    samples = [_make_sample(width, height) for _ in range(batch_size)]

    return {
        "observation": torch.from_numpy(np.stack([s[0] for s in samples])),
        "policy_p1": torch.from_numpy(np.stack([s[1] for s in samples])),
        "policy_p2": torch.from_numpy(np.stack([s[2] for s in samples])),
        "value_p1": torch.from_numpy(np.stack([s[3] for s in samples])),
        "value_p2": torch.from_numpy(np.stack([s[4] for s in samples])),
        "action_p1": torch.from_numpy(np.stack([s[5] for s in samples])),
        "action_p2": torch.from_numpy(np.stack([s[6] for s in samples])),
    }


class TestSwapPlayerPerspectiveBatch:
    """Tests for batch-level swap_player_perspective_batch."""

    def test_batch_swap_matches_single(self) -> None:
        """Batch version with all-True mask matches single-sample version."""
        width, height = 5, 5
        obs, p1, p2, p1_val, p2_val, a1, a2 = _make_sample(width, height)

        # Single sample result
        s_obs, s_p1, s_p2, s_p1_val, s_p2_val, s_a1, s_a2 = swap_player_perspective(
            obs, p1, p2, p1_val, p2_val, a1, a2, width, height
        )

        # Batch of 1 with mask=True
        batch = {
            "observation": torch.from_numpy(obs[np.newaxis, :].copy()),
            "policy_p1": torch.from_numpy(p1[np.newaxis, :].copy()),
            "policy_p2": torch.from_numpy(p2[np.newaxis, :].copy()),
            "value_p1": torch.from_numpy(p1_val[np.newaxis, :].copy()),
            "value_p2": torch.from_numpy(p2_val[np.newaxis, :].copy()),
            "action_p1": torch.from_numpy(a1[np.newaxis, :].copy()),
            "action_p2": torch.from_numpy(a2[np.newaxis, :].copy()),
        }
        mask = torch.tensor([True])

        result = swap_player_perspective_batch(batch, mask, width, height)

        # Compare results
        np.testing.assert_array_almost_equal(result["observation"][0].numpy(), s_obs)
        np.testing.assert_array_almost_equal(result["policy_p1"][0].numpy(), s_p1)
        np.testing.assert_array_almost_equal(result["policy_p2"][0].numpy(), s_p2)
        np.testing.assert_array_almost_equal(result["value_p1"][0].numpy(), s_p1_val)
        np.testing.assert_array_almost_equal(result["value_p2"][0].numpy(), s_p2_val)
        np.testing.assert_array_equal(result["action_p1"][0].numpy(), s_a1)
        np.testing.assert_array_equal(result["action_p2"][0].numpy(), s_a2)

    def test_batch_swap_is_involution(self) -> None:
        """Apply batch swap twice returns original."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)

        # Keep copies of original
        original = {k: v.clone() for k, v in batch.items()}

        # All-True mask for both swaps
        mask = torch.ones(4, dtype=torch.bool)

        # First swap
        swap_player_perspective_batch(batch, mask, width, height)

        # Second swap
        swap_player_perspective_batch(batch, mask, width, height)

        # Should equal original
        for key in batch:
            torch.testing.assert_close(batch[key], original[key])

    def test_batch_partial_mask(self) -> None:
        """Only masked samples are modified."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)

        # Keep copies
        original = {k: v.clone() for k, v in batch.items()}

        # Only modify samples 1 and 3
        mask = torch.tensor([False, True, False, True])

        swap_player_perspective_batch(batch, mask, width, height)

        # Samples 0 and 2 should be unchanged
        for key in batch:
            torch.testing.assert_close(batch[key][0], original[key][0])
            torch.testing.assert_close(batch[key][2], original[key][2])

        # Samples 1 and 3 should be changed
        # Check value swap as a simple indicator
        assert batch["value_p1"][1, 0] == pytest.approx(original["value_p2"][1, 0].item())
        assert batch["value_p1"][3, 0] == pytest.approx(original["value_p2"][3, 0].item())

    def test_batch_empty_mask(self) -> None:
        """All-False mask leaves batch unchanged."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)
        original = {k: v.clone() for k, v in batch.items()}

        mask = torch.zeros(4, dtype=torch.bool)

        swap_player_perspective_batch(batch, mask, width, height)

        for key in batch:
            torch.testing.assert_close(batch[key], original[key])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_on_gpu(self) -> None:
        """Batch swap works on CUDA tensors."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)
        batch = {k: v.cuda() for k, v in batch.items()}

        mask = torch.ones(4, dtype=torch.bool, device="cuda")

        # Should not raise
        result = swap_player_perspective_batch(batch, mask, width, height)

        # Results should still be on GPU
        for key in result:
            assert result[key].is_cuda


class TestBatchAugmentation:
    """Tests for BatchAugmentation class."""

    def test_call_with_p_swap_zero(self) -> None:
        """p_swap=0 leaves batch unchanged."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)
        original = {k: v.clone() for k, v in batch.items()}

        augment = BatchAugmentation(width, height, p_swap=0.0)
        result = augment(batch)

        for key in result:
            torch.testing.assert_close(result[key], original[key])

    def test_call_with_p_swap_one(self) -> None:
        """p_swap=1 transforms all samples."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)
        original_value_p1s = batch["value_p1"].clone()
        original_value_p2s = batch["value_p2"].clone()

        augment = BatchAugmentation(width, height, p_swap=1.0)
        augment(batch)

        # All values should be swapped
        torch.testing.assert_close(batch["value_p1"], original_value_p2s)
        torch.testing.assert_close(batch["value_p2"], original_value_p1s)

    def test_call_preserves_device(self) -> None:
        """BatchAugmentation preserves tensor device."""
        width, height = 5, 5
        batch = _make_batch(batch_size=4, width=width, height=height)

        augment = BatchAugmentation(width, height, p_swap=0.5)
        result = augment(batch)

        for key in result:
            assert result[key].device == batch[key].device

    def test_call_statistical_distribution(self) -> None:
        """p_swap=0.5 augments roughly half the samples over many runs."""
        width, height = 5, 5
        augment = BatchAugmentation(width, height, p_swap=0.5)

        total_augmented = 0
        total_samples = 0
        n_trials = 100
        batch_size = 100

        for _ in range(n_trials):
            batch = _make_batch(batch_size=batch_size, width=width, height=height)
            original_value_p1s = batch["value_p1"].clone()

            augment(batch)

            # Count how many were swapped (augmented)
            # Since value_p1 != value_p2, a swap changes value_p1
            augmented = int((batch["value_p1"] != original_value_p1s).any(dim=-1).sum().item())
            total_augmented += augmented
            total_samples += batch_size

        # Should be close to 50% (allow 5% tolerance)
        ratio = total_augmented / total_samples
        assert 0.45 < ratio < 0.55, f"Expected ~50% augmented, got {ratio * 100:.1f}%"
