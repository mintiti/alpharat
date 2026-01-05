"""Tests for PyRatMLP model."""

from __future__ import annotations

import torch

from alpharat.nn.models import PyRatMLP


class TestPyRatMLPForward:
    """Tests for forward() method."""

    def test_output_shapes(self) -> None:
        """Forward should return correct shapes."""
        obs_dim = 181  # 5x5 maze
        batch_size = 8
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(batch_size, obs_dim)
        log_p1, log_p2, payout = model(x)

        assert log_p1.shape == (batch_size, 5)
        assert log_p2.shape == (batch_size, 5)
        assert payout.shape == (batch_size, 2, 5, 5)

    def test_single_sample(self) -> None:
        """Should handle single sample (batch_size=1) in eval mode.

        BatchNorm requires batch_size > 1 in training mode. Single sample
        inference uses eval mode which relies on running statistics.
        """
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)
        model.eval()

        x = torch.randn(1, obs_dim)
        log_p1, log_p2, payout = model(x)

        assert log_p1.shape == (1, 5)
        assert log_p2.shape == (1, 5)
        assert payout.shape == (1, 2, 5, 5)

    def test_softmax_of_logits_sums_to_one(self) -> None:
        """softmax(logits) should sum to 1."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, _ = model(x)

        probs_p1 = torch.softmax(logits_p1, dim=-1)
        probs_p2 = torch.softmax(logits_p2, dim=-1)

        torch.testing.assert_close(probs_p1.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(probs_p2.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)

    def test_log_softmax_of_logits_negative(self) -> None:
        """log_softmax(logits) should be <= 0."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        logits_p1, logits_p2, _ = model(x)

        log_p1 = torch.log_softmax(logits_p1, dim=-1)
        log_p2 = torch.log_softmax(logits_p2, dim=-1)

        assert (log_p1 <= 0).all()
        assert (log_p2 <= 0).all()

    def test_custom_hidden_dim(self) -> None:
        """Should work with custom hidden dimension."""
        obs_dim = 181
        hidden_dim = 128
        model = PyRatMLP(obs_dim=obs_dim, hidden_dim=hidden_dim)

        x = torch.randn(4, obs_dim)
        log_p1, log_p2, payout = model(x)

        assert log_p1.shape == (4, 5)
        assert log_p2.shape == (4, 5)
        assert payout.shape == (4, 2, 5, 5)

    def test_different_obs_dim(self) -> None:
        """Should work with different observation dimensions."""
        obs_dim = 304  # larger maze
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        log_p1, log_p2, payout = model(x)

        assert log_p1.shape == (4, 5)
        assert log_p2.shape == (4, 5)
        assert payout.shape == (4, 2, 5, 5)


class TestPyRatMLPPredict:
    """Tests for predict() method."""

    def test_output_shapes(self) -> None:
        """Predict should return correct shapes."""
        obs_dim = 181
        batch_size = 8
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(batch_size, obs_dim)
        p1, p2, payout = model.predict(x)

        assert p1.shape == (batch_size, 5)
        assert p2.shape == (batch_size, 5)
        assert payout.shape == (batch_size, 2, 5, 5)

    def test_probs_sum_to_one(self) -> None:
        """Probabilities should sum to 1."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        p1, p2, _ = model.predict(x)

        torch.testing.assert_close(p1.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p2.sum(dim=-1), torch.ones(4), rtol=1e-5, atol=1e-5)

    def test_probs_are_positive(self) -> None:
        """Probabilities should be >= 0."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        p1, p2, _ = model.predict(x)

        assert (p1 >= 0).all()
        assert (p2 >= 0).all()

    def test_probs_at_most_one(self) -> None:
        """Probabilities should be <= 1."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        p1, p2, _ = model.predict(x)

        assert (p1 <= 1).all()
        assert (p2 <= 1).all()


class TestPyRatMLPGradients:
    """Tests for gradient flow."""

    def test_gradients_flow_to_all_parameters(self) -> None:
        """All parameters should receive gradients."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        log_p1, log_p2, payout = model(x)

        # Combined loss
        loss = log_p1.sum() + log_p2.sum() + payout.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_policy_heads_are_independent(self) -> None:
        """Loss on one head should not affect the other head's parameters."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        log_p1, log_p2, _ = model(x)

        # Loss only on P1
        loss = log_p1.sum()
        loss.backward()

        # P1 head should have gradient
        assert model.policy_p1_head.weight.grad is not None
        assert not torch.isnan(model.policy_p1_head.weight.grad).any()

        # P2 head should have NO gradient (not in computation graph)
        assert model.policy_p2_head.weight.grad is None

        # Trunk should have gradient (shared)
        assert model.trunk[0].weight.grad is not None

    def test_payout_gradients(self) -> None:
        """Payout head should receive gradients."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)

        x = torch.randn(4, obs_dim)
        _, _, payout = model(x)

        loss = payout.sum()
        loss.backward()

        assert model.payout_head.weight.grad is not None
        assert not torch.isnan(model.payout_head.weight.grad).any()


class TestPyRatMLPConsistency:
    """Tests for consistency between forward and predict."""

    def test_forward_and_predict_consistent(self) -> None:
        """softmax(forward_logits) should equal predict for policies."""
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)
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
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)
        model.eval()

        x = torch.randn(4, obs_dim)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        for o1, o2 in zip(out1, out2, strict=True):
            torch.testing.assert_close(o1, o2)


class TestPyRatMLPInitialization:
    """Tests for weight initialization."""

    def test_initial_policy_near_uniform(self) -> None:
        """Fresh model should output near-uniform policies.

        With small head init, softmax of near-zero logits → ~uniform.
        """
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)
        model.eval()

        x = torch.randn(32, obs_dim)
        with torch.no_grad():
            p1, p2, _ = model.predict(x)

        # Each action should get roughly 1/5 = 0.2 probability
        uniform = 1.0 / 5
        assert (p1.mean(dim=0) - uniform).abs().max() < 0.1
        assert (p2.mean(dim=0) - uniform).abs().max() < 0.1

    def test_initial_payout_near_zero(self) -> None:
        """Fresh model should predict payouts near zero.

        This reflects uncertainty / predicting draws initially.
        """
        obs_dim = 181
        model = PyRatMLP(obs_dim=obs_dim)
        model.eval()

        x = torch.randn(32, obs_dim)
        with torch.no_grad():
            _, _, payout = model.predict(x)

        # Mean payout should be near zero
        assert payout.mean().abs() < 1.0
        # Individual predictions shouldn't be extreme
        assert payout.abs().max() < 5.0
