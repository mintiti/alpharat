"""Tests for SymmetricMLP model."""

from __future__ import annotations

import torch

from alpharat.nn.models import SymmetricMLP


def _swap_players_in_obs(obs: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Swap P1/P2 in observation tensor.

    Observation layout:
        [maze H*W*4] [p1_pos H*W] [p2_pos H*W] [cheese H*W]
        [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]
    """
    spatial = width * height
    swapped = obs.clone()

    # Swap p1_pos and p2_pos
    p1_pos_start = spatial * 4
    p1_pos_end = spatial * 5
    p2_pos_start = spatial * 5
    p2_pos_end = spatial * 6

    swapped[:, p1_pos_start:p1_pos_end] = obs[:, p2_pos_start:p2_pos_end]
    swapped[:, p2_pos_start:p2_pos_end] = obs[:, p1_pos_start:p1_pos_end]

    # Scalars start after cheese
    scalars_start = spatial * 7

    # Negate score_diff
    swapped[:, scalars_start] = -obs[:, scalars_start]

    # Swap mud (indices 2, 3)
    swapped[:, scalars_start + 2] = obs[:, scalars_start + 3]
    swapped[:, scalars_start + 3] = obs[:, scalars_start + 2]

    # Swap scores (indices 4, 5)
    swapped[:, scalars_start + 4] = obs[:, scalars_start + 5]
    swapped[:, scalars_start + 5] = obs[:, scalars_start + 4]

    return swapped


def _make_valid_obs(width: int, height: int, batch_size: int = 1) -> torch.Tensor:
    """Create valid observation with proper structure.

    Creates observations with:
    - Zero maze (no walls)
    - Random player positions (one-hot)
    - Binary cheese mask
    - Valid scalars
    """
    spatial = width * height
    obs_dim = spatial * 7 + 6
    obs = torch.zeros(batch_size, obs_dim)

    # Random player positions (one-hot)
    p1_pos_start = spatial * 4
    p2_pos_start = spatial * 5
    for i in range(batch_size):
        p1_idx = int(torch.randint(0, spatial, (1,)).item())
        p2_idx = int(torch.randint(0, spatial, (1,)).item())
        obs[i, p1_pos_start + p1_idx] = 1.0
        obs[i, p2_pos_start + p2_idx] = 1.0

    # Binary cheese mask (random ~20% of cells have cheese)
    cheese_start = spatial * 6
    obs[:, cheese_start : cheese_start + spatial] = (torch.rand(batch_size, spatial) > 0.8).float()

    # Valid scalars
    scalars_start = spatial * 7
    obs[:, scalars_start + 1] = 0.5  # progress

    return obs


def _make_symmetric_obs(width: int, height: int, batch_size: int = 1) -> torch.Tensor:
    """Create observation where both players are in identical positions.

    Both players at center, same mud (0), same score (0).
    """
    spatial = width * height
    obs_dim = spatial * 7 + 6

    obs = torch.zeros(batch_size, obs_dim)

    # Random maze (shared)
    obs[:, : spatial * 4] = torch.randn(batch_size, spatial * 4)

    # Both players at center
    center_idx = (height // 2) * width + (width // 2)
    p1_pos_start = spatial * 4
    p2_pos_start = spatial * 5
    obs[:, p1_pos_start + center_idx] = 1.0
    obs[:, p2_pos_start + center_idx] = 1.0

    # Random cheese (shared)
    cheese_start = spatial * 6
    obs[:, cheese_start : cheese_start + spatial] = (torch.rand(batch_size, spatial) > 0.8).float()

    # Scalars: score_diff=0, progress=0.5, p1_mud=0, p2_mud=0, p1_score=0, p2_score=0
    scalars_start = spatial * 7
    obs[:, scalars_start] = 0.0  # score_diff
    obs[:, scalars_start + 1] = 0.5  # progress
    obs[:, scalars_start + 2] = 0.0  # p1_mud
    obs[:, scalars_start + 3] = 0.0  # p2_mud
    obs[:, scalars_start + 4] = 0.0  # p1_score
    obs[:, scalars_start + 5] = 0.0  # p2_score

    return obs


class TestSymmetricMLPForward:
    """Tests for forward() method."""

    def test_output_shapes(self) -> None:
        """Forward should return correct shapes."""
        width, height = 5, 5
        batch_size = 8
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(batch_size, obs_dim)
        logits_p1, logits_p2, payout = model(x)

        assert logits_p1.shape == (batch_size, 5)
        assert logits_p2.shape == (batch_size, 5)
        assert payout.shape == (batch_size, 2, 5, 5)

    def test_single_sample_eval_mode(self) -> None:
        """Should handle single sample in eval mode."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        x = torch.randn(1, obs_dim)
        logits_p1, logits_p2, payout = model(x)

        assert logits_p1.shape == (1, 5)
        assert logits_p2.shape == (1, 5)
        assert payout.shape == (1, 2, 5, 5)

    def test_custom_hidden_dim(self) -> None:
        """Should work with custom hidden dimension."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height, hidden_dim=128)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, payout = model(x)

        assert logits_p1.shape == (4, 5)
        assert logits_p2.shape == (4, 5)
        assert payout.shape == (4, 2, 5, 5)

    def test_different_maze_size(self) -> None:
        """Should work with different maze dimensions."""
        width, height = 7, 6
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, payout = model(x)

        assert logits_p1.shape == (4, 5)
        assert logits_p2.shape == (4, 5)
        assert payout.shape == (4, 2, 5, 5)


class TestSymmetricMLPPredict:
    """Tests for predict() method."""

    def test_output_shapes(self) -> None:
        """Predict should return correct shapes."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(8, obs_dim)
        p1, p2, payout = model.predict(x)

        assert p1.shape == (8, 5)
        assert p2.shape == (8, 5)
        assert payout.shape == (8, 2, 5, 5)

    def test_probs_sum_to_one(self) -> None:
        """Probabilities should sum to 1."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(4, obs_dim)
        p1, p2, _ = model.predict(x)

        torch.testing.assert_close(p1.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p2.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)

    def test_probs_are_valid(self) -> None:
        """Probabilities should be in [0, 1]."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(4, obs_dim)
        p1, p2, _ = model.predict(x)

        assert (p1 >= 0).all()
        assert (p1 <= 1).all()
        assert (p2 >= 0).all()
        assert (p2 <= 1).all()

    def test_payout_non_negative(self) -> None:
        """Payout matrix should be >= 0 (cheese scores can't be negative)."""
        width, height = 5, 5
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        # Use valid observations (payout bounded by remaining_cheese which is non-negative)
        x = _make_valid_obs(width, height, batch_size=32)
        with torch.no_grad():
            _, _, payout = model.predict(x)

        assert (payout >= 0).all(), f"Found negative payouts: min={payout.min().item()}"


class TestSymmetricMLPSymmetry:
    """Tests for structural symmetry guarantees."""

    def test_swap_equivariance_policy(self) -> None:
        """Swapping players in input should swap policy outputs.

        This is the key structural guarantee: model(swap(obs)) produces
        swapped outputs. Not learned - guaranteed by weight sharing.
        """
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        # Random observation
        obs = torch.randn(4, obs_dim)
        obs_swapped = _swap_players_in_obs(obs, width, height)

        with torch.no_grad():
            p1, p2, _ = model.predict(obs)
            p1_swap, p2_swap, _ = model.predict(obs_swapped)

        # After swapping: new P1's policy = old P2's policy
        torch.testing.assert_close(p1, p2_swap, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p2, p1_swap, rtol=1e-5, atol=1e-5)

    def test_swap_equivariance_payout(self) -> None:
        """Swapping players should transform payout matrix correctly.

        When you swap players:
        - new_payout[0] = old_payout[1].T
        - new_payout[1] = old_payout[0].T
        """
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        obs = torch.randn(4, obs_dim)
        obs_swapped = _swap_players_in_obs(obs, width, height)

        with torch.no_grad():
            _, _, payout = model.predict(obs)
            _, _, payout_swap = model.predict(obs_swapped)

        # new_payout[0] = old_payout[1].T
        torch.testing.assert_close(
            payout_swap[:, 0], payout[:, 1].transpose(-1, -2), rtol=1e-5, atol=1e-5
        )
        # new_payout[1] = old_payout[0].T
        torch.testing.assert_close(
            payout_swap[:, 1], payout[:, 0].transpose(-1, -2), rtol=1e-5, atol=1e-5
        )

    def test_symmetric_position_identical_policies(self) -> None:
        """Symmetric positions should produce identical P1/P2 policies.

        If both players are at the same position with same mud/score,
        they should get identical action distributions.
        """
        width, height = 5, 5
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        obs = _make_symmetric_obs(width, height, batch_size=4)

        with torch.no_grad():
            p1, p2, _ = model.predict(obs)

        # Policies should be identical for symmetric positions
        torch.testing.assert_close(p1, p2, rtol=1e-5, atol=1e-5)

    def test_symmetric_position_payout_structure(self) -> None:
        """Symmetric position payouts should have expected structure.

        For symmetric positions:
        - payout[0] should equal payout[1].T (P1's view = P2's view transposed)
        """
        width, height = 5, 5
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        obs = _make_symmetric_obs(width, height, batch_size=4)

        with torch.no_grad():
            _, _, payout = model.predict(obs)

        # payout[0] = payout[1].T for symmetric positions
        torch.testing.assert_close(
            payout[:, 0], payout[:, 1].transpose(-1, -2), rtol=1e-5, atol=1e-5
        )

    def test_swap_is_involution(self) -> None:
        """Swapping twice should return to original outputs."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        obs = torch.randn(4, obs_dim)
        obs_swapped = _swap_players_in_obs(obs, width, height)
        obs_double_swapped = _swap_players_in_obs(obs_swapped, width, height)

        with torch.no_grad():
            p1, p2, payout = model.predict(obs)
            p1_ds, p2_ds, payout_ds = model.predict(obs_double_swapped)

        torch.testing.assert_close(p1, p1_ds, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p2, p2_ds, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(payout, payout_ds, rtol=1e-5, atol=1e-5)


class TestSymmetricMLPGradients:
    """Tests for gradient flow."""

    def test_gradients_flow_to_all_parameters(self) -> None:
        """All parameters should receive gradients."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, payout = model(x)

        loss = logits_p1.sum() + logits_p2.sum() + payout.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_policy_head_shared(self) -> None:
        """Loss on P1 should affect same policy head as loss on P2.

        Unlike PyRatMLP which has separate heads, SymmetricMLP shares the head.
        """
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, _ = model(x)

        # Loss only on P1
        loss_p1 = logits_p1.sum()
        loss_p1.backward()
        assert model.policy_head.weight.grad is not None
        grad_from_p1 = model.policy_head.weight.grad.clone()

        # Reset
        model.zero_grad()
        logits_p1, logits_p2, _ = model(x)

        # Loss only on P2
        loss_p2 = logits_p2.sum()
        loss_p2.backward()
        assert model.policy_head.weight.grad is not None
        grad_from_p2 = model.policy_head.weight.grad.clone()

        # Both should produce gradients (same head is used)
        assert grad_from_p1.abs().sum() > 0
        assert grad_from_p2.abs().sum() > 0


class TestSymmetricMLPInitialization:
    """Tests for weight initialization."""

    def test_initial_policy_near_uniform(self) -> None:
        """Fresh model should output near-uniform policies."""
        width, height = 5, 5
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        x = _make_valid_obs(width, height, batch_size=32)
        with torch.no_grad():
            p1, p2, _ = model.predict(x)

        uniform = 1.0 / 5
        assert (p1.mean(dim=0) - uniform).abs().max() < 0.1
        assert (p2.mean(dim=0) - uniform).abs().max() < 0.1

    def test_initial_payout_near_zero(self) -> None:
        """Fresh model should predict small non-negative payouts."""
        width, height = 5, 5
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        x = _make_valid_obs(width, height, batch_size=32)
        with torch.no_grad():
            _, _, payout = model.predict(x)

        # softplus ensures non-negative, small init keeps values small
        assert (payout >= 0).all()
        # With std=0.01 init, softplus(x≈0) ≈ 0.69, should be well under 5
        assert payout.max() < 5.0


class TestSymmetricMLPConsistency:
    """Tests for consistency between forward and predict."""

    def test_forward_and_predict_consistent(self) -> None:
        """softmax(forward_logits) should equal predict for policies."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        x = torch.randn(4, obs_dim)

        with torch.no_grad():
            logits_p1, logits_p2, payout_forward = model(x)
            p1, p2, payout_predict = model.predict(x)

        torch.testing.assert_close(torch.softmax(logits_p1, dim=-1), p1, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(torch.softmax(logits_p2, dim=-1), p2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(payout_forward, payout_predict)

    def test_deterministic_in_eval_mode(self) -> None:
        """Same input should produce same output in eval mode."""
        width, height = 5, 5
        obs_dim = width * height * 7 + 6
        model = SymmetricMLP(width=width, height=height)
        model.eval()

        x = torch.randn(4, obs_dim)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        for o1, o2 in zip(out1, out2, strict=True):
            torch.testing.assert_close(o1, o2)
