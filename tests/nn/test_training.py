"""Tests for training loss functions."""

from __future__ import annotations

import torch

from alpharat.nn.symmetric_training import sparse_payout_loss
from alpharat.nn.training import constant_sum_loss, nash_consistency_loss


class TestNashConsistencyLoss:
    """Tests for nash_consistency_loss function."""

    def test_consistent_payout_zero_loss(self) -> None:
        """When payout matrix is consistent with policy, loss should be ~0."""
        batch_size = 4

        # Simple case: both players play action 0 with prob 1
        target_pi1 = torch.zeros(batch_size, 5)
        target_pi1[:, 0] = 1.0
        target_pi2 = torch.zeros(batch_size, 5)
        target_pi2[:, 0] = 1.0

        # Payout matrix where action 0 is best for both
        # P1[0,j] >= P1[i,j] for all i, j=0 (since P2 plays j=0)
        # P2[i,0] >= P2[i,j] for all i=0, j (since P1 plays i=0)
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0, 0, :] = 5.0  # P1 gets 5 for action 0 against any P2 action
        pred_payout[:, 0, 1:, :] = 3.0  # P1 gets 3 for other actions
        pred_payout[:, 1, :, 0] = 5.0  # P2 gets 5 for action 0 against any P1 action
        pred_payout[:, 1, :, 1:] = 3.0  # P2 gets 3 for other actions

        total, indiff, dev = nash_consistency_loss(pred_payout, target_pi1, target_pi2)

        # Indifference: only action 0 in support, so no variance to penalize
        assert indiff.item() == 0.0

        # Deviation: actions 1-4 have lower payoff than action 0, so no deviation penalty
        # P1: E[action 0] = 5, E[action i] = 3 < 5 for i > 0 (since P2 plays action 0)
        # P2: E[action 0] = 5, E[action j] = 3 < 5 for j > 0 (since P1 plays action 0)
        assert dev.item() == 0.0
        assert total.item() == 0.0

    def test_profitable_deviation_penalized(self) -> None:
        """When outside-support action is better than equilibrium, should penalize."""
        batch_size = 2

        # P1 plays action 0 with prob 1
        target_pi1 = torch.zeros(batch_size, 5)
        target_pi1[:, 0] = 1.0
        # P2 plays action 0 with prob 1
        target_pi2 = torch.zeros(batch_size, 5)
        target_pi2[:, 0] = 1.0

        # Payout matrix where action 1 is actually better for P1 (profitable deviation!)
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0, 0, :] = 3.0  # P1 gets 3 for action 0
        pred_payout[:, 0, 1, :] = 10.0  # P1 gets 10 for action 1 (better!)
        pred_payout[:, 1, :, :] = 5.0  # P2 gets 5 for everything

        total, indiff, dev = nash_consistency_loss(pred_payout, target_pi1, target_pi2)

        # P1's expected payoff per action (against P2 playing action 0):
        # E[action 0] = 3, E[action 1] = 10
        # Equilibrium value V = 3 (since P1 plays action 0)
        # Deviation loss for action 1: ReLU(10 - 3)^2 = 49
        assert indiff.item() == 0.0  # Only one action in support
        assert dev.item() > 0.0  # Action 1 is a profitable deviation

    def test_indifference_violated_penalized(self) -> None:
        """When support actions have different expected payoffs, should penalize."""
        batch_size = 2

        # P1 plays mixed strategy: 50% action 0, 50% action 1
        target_pi1 = torch.zeros(batch_size, 5)
        target_pi1[:, 0] = 0.5
        target_pi1[:, 1] = 0.5
        # P2 plays action 0 with prob 1
        target_pi2 = torch.zeros(batch_size, 5)
        target_pi2[:, 0] = 1.0

        # Payout matrix where actions 0 and 1 have different payoffs for P1
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0, 0, :] = 2.0  # P1 gets 2 for action 0
        pred_payout[:, 0, 1, :] = 8.0  # P1 gets 8 for action 1 (different!)
        pred_payout[:, 1, :, :] = 5.0  # P2 gets 5 for everything

        total, indiff, dev = nash_consistency_loss(pred_payout, target_pi1, target_pi2)

        # P1's expected payoffs: E[0] = 2, E[1] = 8
        # Equilibrium value V = 0.5 * 2 + 0.5 * 8 = 5
        # Indifference loss: 0.5*(2-5)^2 + 0.5*(8-5)^2 = 0.5*9 + 0.5*9 = 9
        # But we divide by total elements (batch * actions), so per-element is smaller
        assert indiff.item() > 0.0  # Actions in support have different payoffs
        assert dev.item() == 0.0  # No actions outside support for P1

    def test_blocked_actions_outside_support(self) -> None:
        """Blocked actions (pi=0) should be in outside-support and checked for deviation."""
        batch_size = 1

        # P1 only plays action 2 (others blocked or not chosen)
        target_pi1 = torch.zeros(batch_size, 5)
        target_pi1[:, 2] = 1.0
        # P2 only plays action 3
        target_pi2 = torch.zeros(batch_size, 5)
        target_pi2[:, 3] = 1.0

        # Payout where blocked action 0 is much better for P1
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0, 2, 3] = 1.0  # P1 gets 1 for (2,3)
        pred_payout[:, 0, 0, 3] = 100.0  # P1 would get 100 for (0,3) - blocked but better!
        pred_payout[:, 1, 2, 3] = 1.0  # P2 gets 1

        total, indiff, dev = nash_consistency_loss(pred_payout, target_pi1, target_pi2)

        # V1 = 1 (P1 plays action 2, P2 plays action 3)
        # E[action 0] = 100 >> 1 = V → deviation penalty
        assert dev.item() > 0.0

    def test_batch_independence(self) -> None:
        """Each sample in batch should be computed independently."""
        # Create two different scenarios
        pi1_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        pi2_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        payout_a = torch.zeros(1, 2, 5, 5)
        payout_a[:, 0, 0, :] = 5.0
        payout_a[:, 1, :, 0] = 5.0

        pi1_b = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
        pi2_b = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        payout_b = torch.zeros(1, 2, 5, 5)
        payout_b[:, 0, 0, :] = 2.0
        payout_b[:, 0, 1, :] = 8.0
        payout_b[:, 1, :, :] = 5.0

        # Compute separately
        loss_a, _, _ = nash_consistency_loss(payout_a, pi1_a, pi2_a)
        loss_b, _, _ = nash_consistency_loss(payout_b, pi1_b, pi2_b)

        # Compute batched
        pi1_batch = torch.cat([pi1_a, pi1_b], dim=0)
        pi2_batch = torch.cat([pi2_a, pi2_b], dim=0)
        payout_batch = torch.cat([payout_a, payout_b], dim=0)
        loss_batch, _, _ = nash_consistency_loss(payout_batch, pi1_batch, pi2_batch)

        # Batched loss should be average of individual losses
        expected = (loss_a.item() + loss_b.item()) / 2
        # Allow small numerical tolerance due to mean() behavior
        assert abs(loss_batch.item() - expected) < 0.01

    def test_symmetric_for_both_players(self) -> None:
        """Loss should apply symmetrically to both players."""
        batch_size = 1

        # P1 plays action 0, P2 plays mixed 0/1
        target_pi1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        target_pi2 = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

        # P2 has indifference violation (actions 0 and 1 have different payoffs)
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0, :, :] = 5.0  # P1 gets 5 everywhere
        pred_payout[:, 1, 0, 0] = 2.0  # P2 gets 2 for action 0
        pred_payout[:, 1, 0, 1] = 8.0  # P2 gets 8 for action 1

        total, indiff, dev = nash_consistency_loss(pred_payout, target_pi1, target_pi2)

        # P1 has no indifference violation (only one action in support)
        # P2 has indifference violation: E[0]=2, E[1]=8, V=5
        assert indiff.item() > 0.0  # From P2's indifference violation

    def test_gradient_flow(self) -> None:
        """Loss should have gradients w.r.t. payout matrix."""
        target_pi1 = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])
        target_pi2 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

        # Non-zero payout that creates indifference violation
        pred_payout = torch.zeros(1, 2, 5, 5)
        pred_payout[:, 0, 0, :] = 2.0  # P1 gets 2 for action 0
        pred_payout[:, 0, 1, :] = 8.0  # P1 gets 8 for action 1 (different!)
        pred_payout = pred_payout.clone().requires_grad_(True)

        total, _, _ = nash_consistency_loss(pred_payout, target_pi1, target_pi2)
        total.backward()

        assert pred_payout.grad is not None
        # Should have non-zero gradients for P1's payout entries
        assert pred_payout.grad.abs().sum() > 0


class TestConstantSumLoss:
    """Tests for constant_sum_loss function."""

    def test_perfect_constant_sum_zero_loss(self) -> None:
        """When P1 + P2 = total collected for all pairs, loss should be ~0."""
        batch_size = 4
        total_collected = 3.0  # Total cheese collected

        # Payout matrix where all pairs sum to total_collected
        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        pred_payout[:, 0] = 1.5  # P1 gets half
        pred_payout[:, 1] = 1.5  # P2 gets half

        p1_value = torch.full((batch_size,), total_collected / 2)
        p2_value = torch.full((batch_size,), total_collected / 2)

        loss = constant_sum_loss(pred_payout, p1_value, p2_value)
        assert loss.item() < 1e-6

    def test_non_constant_sum_positive_loss(self) -> None:
        """When P1 + P2 varies across pairs, loss should be > 0."""
        batch_size = 2

        pred_payout = torch.zeros(batch_size, 2, 5, 5)
        # Some pairs sum to 5, not 3
        pred_payout[:, 0] = 3.0  # P1 gets 3
        pred_payout[:, 1] = 2.0  # P2 gets 2 (sum = 5, not 3)

        p1_value = torch.full((batch_size,), 1.5)
        p2_value = torch.full((batch_size,), 1.5)

        loss = constant_sum_loss(pred_payout, p1_value, p2_value)
        assert loss.item() > 0.0

    def test_gradient_flow(self) -> None:
        """Loss should have gradients w.r.t. payout matrix."""
        pred_payout = torch.zeros(1, 2, 5, 5, requires_grad=True)

        p1_value = torch.tensor([2.0])
        p2_value = torch.tensor([1.0])

        loss = constant_sum_loss(pred_payout, p1_value, p2_value)
        loss.backward()

        assert pred_payout.grad is not None
        assert pred_payout.grad.abs().sum() > 0


class TestSparsePayoutLoss:
    """Tests for sparse_payout_loss."""

    def test_basic_loss(self) -> None:
        """Basic MSE at played action pair."""
        batch_size = 4
        pred_payout = torch.randn(batch_size, 2, 5, 5) + 2.0  # Ensure positive

        action_p1 = torch.randint(0, 5, (batch_size, 1))
        action_p2 = torch.randint(0, 5, (batch_size, 1))
        p1_value = torch.randn(batch_size)
        p2_value = torch.randn(batch_size)

        loss = sparse_payout_loss(pred_payout, action_p1, action_p2, p1_value, p2_value)

        assert loss.item() >= 0
        assert loss.dim() == 0  # Scalar
