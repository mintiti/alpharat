"""Tests for architecture-specific loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from alpharat.nn.architectures.local_value.loss import compute_local_value_losses
from alpharat.nn.architectures.mlp.config import MLPOptimConfig
from alpharat.nn.architectures.mlp.loss import compute_mlp_losses
from alpharat.nn.architectures.symmetric.config import SymmetricOptimConfig
from alpharat.nn.architectures.symmetric.loss import compute_symmetric_losses
from alpharat.nn.training.keys import LossKey, ModelOutput


@pytest.fixture
def batch_size() -> int:
    """Standard batch size for tests."""
    return 8


@pytest.fixture
def model_output(batch_size: int) -> dict[str, torch.Tensor]:
    """Create model output with logits and scalar values."""
    return {
        ModelOutput.LOGITS_P1: torch.randn(batch_size, 5),
        ModelOutput.LOGITS_P2: torch.randn(batch_size, 5),
        ModelOutput.VALUE_P1: torch.randn(batch_size),
        ModelOutput.VALUE_P2: torch.randn(batch_size),
    }


@pytest.fixture
def batch(batch_size: int) -> dict[str, torch.Tensor]:
    """Create training batch with policy and value targets."""
    return {
        "policy_p1": F.softmax(torch.randn(batch_size, 5), dim=-1),
        "policy_p2": F.softmax(torch.randn(batch_size, 5), dim=-1),
        "value_p1": torch.rand(batch_size, 1) * 5,  # Values in [0, 5]
        "value_p2": torch.rand(batch_size, 1) * 5,
        "action_p1": torch.randint(0, 5, (batch_size, 1)),
        "action_p2": torch.randint(0, 5, (batch_size, 1)),
    }


class TestMLPLoss:
    """Tests for MLP architecture loss function."""

    def test_returns_total_loss(self, model_output: dict, batch: dict) -> None:
        """Loss function should return LossKey.TOTAL."""
        config = MLPOptimConfig()
        losses = compute_mlp_losses(model_output, batch, config)

        assert LossKey.TOTAL in losses
        assert losses[LossKey.TOTAL].shape == ()  # Scalar

    def test_returns_component_losses(self, model_output: dict, batch: dict) -> None:
        """Loss function should return individual loss components."""
        config = MLPOptimConfig()
        losses = compute_mlp_losses(model_output, batch, config)

        assert LossKey.POLICY_P1 in losses
        assert LossKey.POLICY_P2 in losses
        assert LossKey.VALUE in losses
        assert LossKey.VALUE_P1 in losses
        assert LossKey.VALUE_P2 in losses

    def test_value_loss_uses_mse(self, batch_size: int) -> None:
        """Value loss should use MSE on scalar values."""
        pred_v1 = torch.tensor([1.0, 2.0, 3.0])
        pred_v2 = torch.tensor([0.5, 1.5, 2.5])
        target_v1 = torch.tensor([[1.5], [2.5], [3.5]])
        target_v2 = torch.tensor([[1.0], [2.0], [3.0]])

        model_output: dict[str, torch.Tensor] = {
            ModelOutput.LOGITS_P1: torch.randn(3, 5),
            ModelOutput.LOGITS_P2: torch.randn(3, 5),
            ModelOutput.VALUE_P1: pred_v1,
            ModelOutput.VALUE_P2: pred_v2,
        }
        batch: dict[str, torch.Tensor] = {
            "policy_p1": F.softmax(torch.randn(3, 5), dim=-1),
            "policy_p2": F.softmax(torch.randn(3, 5), dim=-1),
            "value_p1": target_v1,
            "value_p2": target_v2,
        }

        config = MLPOptimConfig()
        losses = compute_mlp_losses(model_output, batch, config)

        # Compute expected MSE manually
        expected_mse_p1 = F.mse_loss(pred_v1, target_v1.squeeze(-1))
        expected_mse_p2 = F.mse_loss(pred_v2, target_v2.squeeze(-1))
        expected_value_loss = 0.5 * (expected_mse_p1 + expected_mse_p2)

        torch.testing.assert_close(losses[LossKey.VALUE_P1], expected_mse_p1)
        torch.testing.assert_close(losses[LossKey.VALUE_P2], expected_mse_p2)
        torch.testing.assert_close(losses[LossKey.VALUE], expected_value_loss)

    def test_zero_value_weight_removes_value_loss(self, model_output: dict, batch: dict) -> None:
        """With value_weight=0, value loss should not affect total."""
        config_with_value = MLPOptimConfig(value_weight=1.0)
        config_no_value = MLPOptimConfig(value_weight=0.0)

        losses_with = compute_mlp_losses(model_output, batch, config_with_value)
        losses_without = compute_mlp_losses(model_output, batch, config_no_value)

        # Total should differ when value weight changes
        assert losses_with[LossKey.TOTAL] != losses_without[LossKey.TOTAL]
        # But individual value loss should be the same
        torch.testing.assert_close(losses_with[LossKey.VALUE], losses_without[LossKey.VALUE])


class TestSymmetricLoss:
    """Tests for Symmetric architecture loss function."""

    def test_returns_total_loss(self, model_output: dict, batch: dict) -> None:
        """Loss function should return LossKey.TOTAL."""
        config = SymmetricOptimConfig()
        losses = compute_symmetric_losses(model_output, batch, config)

        assert LossKey.TOTAL in losses
        assert losses[LossKey.TOTAL].shape == ()

    def test_returns_component_losses(self, model_output: dict, batch: dict) -> None:
        """Loss function should return individual loss components."""
        config = SymmetricOptimConfig()
        losses = compute_symmetric_losses(model_output, batch, config)

        assert LossKey.POLICY_P1 in losses
        assert LossKey.POLICY_P2 in losses
        assert LossKey.VALUE in losses
        assert LossKey.VALUE_P1 in losses
        assert LossKey.VALUE_P2 in losses

    def test_value_loss_uses_mse(self) -> None:
        """Value loss should use MSE on scalar values."""
        pred_v1 = torch.tensor([2.0, 4.0])
        pred_v2 = torch.tensor([1.0, 3.0])
        target_v1 = torch.tensor([[2.5], [4.5]])
        target_v2 = torch.tensor([[1.5], [3.5]])

        model_output: dict[str, torch.Tensor] = {
            ModelOutput.LOGITS_P1: torch.randn(2, 5),
            ModelOutput.LOGITS_P2: torch.randn(2, 5),
            ModelOutput.VALUE_P1: pred_v1,
            ModelOutput.VALUE_P2: pred_v2,
        }
        batch: dict[str, torch.Tensor] = {
            "policy_p1": F.softmax(torch.randn(2, 5), dim=-1),
            "policy_p2": F.softmax(torch.randn(2, 5), dim=-1),
            "value_p1": target_v1,
            "value_p2": target_v2,
        }

        config = SymmetricOptimConfig()
        losses = compute_symmetric_losses(model_output, batch, config)

        expected_mse_p1 = F.mse_loss(pred_v1, target_v1.squeeze(-1))
        expected_mse_p2 = F.mse_loss(pred_v2, target_v2.squeeze(-1))
        expected_value_loss = 0.5 * (expected_mse_p1 + expected_mse_p2)

        torch.testing.assert_close(losses[LossKey.VALUE], expected_value_loss)


class TestLocalValueLoss:
    """Tests for LocalValue architecture loss function."""

    @pytest.fixture
    def local_value_output(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Create model output with ownership logits."""
        return {
            ModelOutput.LOGITS_P1: torch.randn(batch_size, 5),
            ModelOutput.LOGITS_P2: torch.randn(batch_size, 5),
            ModelOutput.VALUE_P1: torch.randn(batch_size),
            ModelOutput.VALUE_P2: torch.randn(batch_size),
            ModelOutput.OWNERSHIP_LOGITS: torch.randn(batch_size, 5, 5, 4),
        }

    @pytest.fixture
    def local_value_batch(self, batch: dict, batch_size: int) -> dict[str, torch.Tensor]:
        """Add cheese_outcomes to batch for LocalValue."""
        batch = dict(batch)  # Copy to avoid modifying original
        batch["cheese_outcomes"] = torch.randint(-1, 4, (batch_size, 5, 5))
        return batch

    def test_returns_total_loss(self, local_value_output: dict, local_value_batch: dict) -> None:
        """Loss function should return LossKey.TOTAL."""
        from alpharat.nn.architectures.local_value.config import LocalValueOptimConfig

        config = LocalValueOptimConfig()
        losses = compute_local_value_losses(local_value_output, local_value_batch, config)

        assert LossKey.TOTAL in losses
        assert losses[LossKey.TOTAL].shape == ()

    def test_returns_ownership_loss(
        self, local_value_output: dict, local_value_batch: dict
    ) -> None:
        """Loss function should return ownership loss component."""
        from alpharat.nn.architectures.local_value.config import LocalValueOptimConfig

        config = LocalValueOptimConfig()
        losses = compute_local_value_losses(local_value_output, local_value_batch, config)

        assert LossKey.OWNERSHIP in losses

    def test_value_loss_uses_mse(self, local_value_batch: dict) -> None:
        """Value loss should use MSE on scalar values."""
        from alpharat.nn.architectures.local_value.config import LocalValueOptimConfig

        pred_v1 = torch.tensor([1.0, 2.0])
        pred_v2 = torch.tensor([0.5, 1.5])
        target_v1 = torch.tensor([[1.5], [2.5]])
        target_v2 = torch.tensor([[1.0], [2.0]])

        model_output: dict[str, torch.Tensor] = {
            ModelOutput.LOGITS_P1: torch.randn(2, 5),
            ModelOutput.LOGITS_P2: torch.randn(2, 5),
            ModelOutput.VALUE_P1: pred_v1,
            ModelOutput.VALUE_P2: pred_v2,
            ModelOutput.OWNERSHIP_LOGITS: torch.randn(2, 5, 5, 4),
        }
        batch: dict[str, torch.Tensor] = {
            "policy_p1": F.softmax(torch.randn(2, 5), dim=-1),
            "policy_p2": F.softmax(torch.randn(2, 5), dim=-1),
            "value_p1": target_v1,
            "value_p2": target_v2,
            "cheese_outcomes": torch.randint(-1, 4, (2, 5, 5)),
        }

        config = LocalValueOptimConfig()
        losses = compute_local_value_losses(model_output, batch, config)

        expected_mse_p1 = F.mse_loss(pred_v1, target_v1.squeeze(-1))
        expected_mse_p2 = F.mse_loss(pred_v2, target_v2.squeeze(-1))
        expected_value_loss = 0.5 * (expected_mse_p1 + expected_mse_p2)

        torch.testing.assert_close(losses[LossKey.VALUE], expected_value_loss)
