"""Tests for training metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpharat.nn.metrics import (
    MetricsAccumulator,
    compute_payout_metrics,
    compute_policy_metrics,
    explained_variance,
    payout_correlation,
    policy_entropy,
    target_entropy,
    top_k_accuracy,
)


class TestTopKAccuracy:
    """Tests for top_k_accuracy()."""

    def test_top1_perfect_match(self) -> None:
        """Top-1 accuracy is 1.0 when argmax matches."""
        logits = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]])

        result = top_k_accuracy(logits, target, k=1)

        assert result.item() == pytest.approx(1.0)

    def test_top1_no_match(self) -> None:
        """Top-1 accuracy is 0.0 when argmax differs."""
        logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]])  # argmax=1
        target = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]])  # argmax=0

        result = top_k_accuracy(logits, target, k=1)

        assert result.item() == pytest.approx(0.0)

    def test_top2_target_in_top2(self) -> None:
        """Top-2 accuracy is 1.0 when target's argmax is in top-2."""
        logits = torch.tensor([[2.0, 1.5, 0.0, 0.0, 0.0]])  # top-2: [0, 1]
        target = torch.tensor([[0.1, 0.8, 0.05, 0.03, 0.02]])  # argmax=1

        result = top_k_accuracy(logits, target, k=2)

        assert result.item() == pytest.approx(1.0)

    def test_top2_target_not_in_top2(self) -> None:
        """Top-2 accuracy is 0.0 when target's argmax is not in top-2."""
        logits = torch.tensor([[2.0, 1.5, 0.0, 0.0, 0.0]])  # top-2: [0, 1]
        target = torch.tensor([[0.1, 0.1, 0.7, 0.05, 0.05]])  # argmax=2

        result = top_k_accuracy(logits, target, k=2)

        assert result.item() == pytest.approx(0.0)

    def test_batch_averaging(self) -> None:
        """Should average across batch."""
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, 0.0, 0.0],  # argmax=0, matches
                [0.0, 2.0, 0.0, 0.0, 0.0],  # argmax=1, doesn't match
            ]
        )
        target = torch.tensor(
            [
                [0.8, 0.1, 0.05, 0.03, 0.02],  # argmax=0
                [0.8, 0.1, 0.05, 0.03, 0.02],  # argmax=0
            ]
        )

        result = top_k_accuracy(logits, target, k=1)

        assert result.item() == pytest.approx(0.5)


class TestPolicyEntropy:
    """Tests for policy_entropy()."""

    def test_uniform_distribution_max_entropy(self) -> None:
        """Uniform distribution should have max entropy (log(5) ≈ 1.609)."""
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])

        result = policy_entropy(logits)

        expected = np.log(5)  # Max entropy for 5 actions
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_deterministic_distribution_zero_entropy(self) -> None:
        """Very peaked distribution should have near-zero entropy."""
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0]])

        result = policy_entropy(logits)

        assert result.item() == pytest.approx(0.0, abs=1e-4)

    def test_batch_averaging(self) -> None:
        """Should average entropy across batch."""
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],  # uniform, high entropy
                [100.0, 0.0, 0.0, 0.0, 0.0],  # peaked, low entropy
            ]
        )

        result = policy_entropy(logits)

        # Average of max_entropy and ~0
        expected = np.log(5) / 2
        assert result.item() == pytest.approx(expected, rel=1e-3)


class TestTargetEntropy:
    """Tests for target_entropy()."""

    def test_uniform_distribution_max_entropy(self) -> None:
        """Uniform probabilities should have max entropy."""
        target = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])

        result = target_entropy(target)

        expected = np.log(5)
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_deterministic_distribution_zero_entropy(self) -> None:
        """Single action with probability 1 should have zero entropy."""
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

        result = target_entropy(target)

        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_handles_zeros_in_target(self) -> None:
        """Should handle zeros in target without NaN."""
        target = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0]])

        result = target_entropy(target)

        assert not torch.isnan(result)
        expected = np.log(2)  # Binary distribution entropy
        assert result.item() == pytest.approx(expected, rel=1e-4)


class TestExplainedVariance:
    """Tests for explained_variance()."""

    def test_perfect_predictions(self) -> None:
        """EV = 1.0 for perfect predictions."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = explained_variance(pred, target)

        assert result.item() == pytest.approx(1.0)

    def test_predicting_mean(self) -> None:
        """EV = 0.0 when predicting the mean."""
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mean_pred = torch.full_like(target, target.mean().item())

        result = explained_variance(mean_pred, target)

        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_worse_than_mean_clamped(self) -> None:
        """EV clamped to -1 when predictions are worse than predicting mean."""
        pred = torch.tensor([4.0, 3.0, 2.0, 1.0])  # Reversed
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = explained_variance(pred, target)

        # Would be -3.0 without clamping, clamped to -1.0
        assert result.item() == pytest.approx(-1.0)

    def test_constant_target_returns_zero(self) -> None:
        """EV = 0 when target has no variance."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 2.0])

        result = explained_variance(pred, target)

        assert result.item() == pytest.approx(0.0)

    def test_works_with_matrices(self) -> None:
        """Should flatten and work with 2D inputs."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        result = explained_variance(pred, target)

        assert result.item() == pytest.approx(1.0)


class TestPayoutCorrelation:
    """Tests for payout_correlation()."""

    def test_perfect_correlation(self) -> None:
        """Correlation = 1.0 for identical matrices."""
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        result = payout_correlation(pred, target)

        assert result.item() == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        """Correlation = -1.0 for negated matrices."""
        pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.tensor([[[-1.0, -2.0], [-3.0, -4.0]]])

        result = payout_correlation(pred, target)

        assert result.item() == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        """Correlation near 0 for unrelated matrices."""
        # Construct orthogonal patterns
        pred = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        target = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])

        result = payout_correlation(pred, target)

        assert result.item() == pytest.approx(0.0, abs=0.1)

    def test_constant_input_returns_zero(self) -> None:
        """Correlation = 0 when input has no variance."""
        pred = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
        target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        result = payout_correlation(pred, target)

        assert result.item() == pytest.approx(0.0)


class TestComputePolicyMetrics:
    """Tests for compute_policy_metrics()."""

    def test_returns_all_metrics(self) -> None:
        """Should return dict with all four policy metrics."""
        logits = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]])

        result = compute_policy_metrics(logits, target)

        assert "top1_accuracy" in result
        assert "top2_accuracy" in result
        assert "entropy_pred" in result
        assert "entropy_target" in result

    def test_returns_floats(self) -> None:
        """All values should be Python floats."""
        logits = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]])

        result = compute_policy_metrics(logits, target)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"


class TestComputePayoutMetrics:
    """Tests for compute_payout_metrics()."""

    def test_returns_all_metrics(self) -> None:
        """Should return dict with explained_variance and correlation."""
        pred = torch.randn(4, 5, 5)
        target = torch.randn(4, 5, 5)

        result = compute_payout_metrics(pred, target)

        assert "explained_variance" in result
        assert "correlation" in result

    def test_returns_floats(self) -> None:
        """All values should be Python floats."""
        pred = torch.randn(4, 5, 5)
        target = torch.randn(4, 5, 5)

        result = compute_payout_metrics(pred, target)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"


class TestMetricsAccumulator:
    """Tests for MetricsAccumulator."""

    def test_single_update(self) -> None:
        """Single update should return same values."""
        acc = MetricsAccumulator()
        acc.update({"loss": 1.0, "accuracy": 0.9}, batch_size=10)

        result = acc.compute()

        assert result["loss"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(0.9)

    def test_multiple_updates_equal_weight(self) -> None:
        """Multiple equal-weight updates should average."""
        acc = MetricsAccumulator()
        acc.update({"loss": 1.0}, batch_size=10)
        acc.update({"loss": 2.0}, batch_size=10)

        result = acc.compute()

        assert result["loss"] == pytest.approx(1.5)

    def test_weighted_average(self) -> None:
        """Should compute weighted average by batch size."""
        acc = MetricsAccumulator()
        acc.update({"loss": 1.0}, batch_size=30)
        acc.update({"loss": 2.0}, batch_size=10)

        result = acc.compute()

        # (1.0 * 30 + 2.0 * 10) / 40 = 50/40 = 1.25
        assert result["loss"] == pytest.approx(1.25)

    def test_reset_clears_state(self) -> None:
        """Reset should clear accumulated values."""
        acc = MetricsAccumulator()
        acc.update({"loss": 1.0}, batch_size=10)
        acc.reset()
        acc.update({"loss": 2.0}, batch_size=10)

        result = acc.compute()

        assert result["loss"] == pytest.approx(2.0)

    def test_different_metrics_per_update(self) -> None:
        """Should handle metrics appearing in different updates."""
        acc = MetricsAccumulator()
        acc.update({"loss": 1.0, "accuracy": 0.8}, batch_size=10)
        acc.update({"loss": 2.0}, batch_size=10)  # No accuracy

        result = acc.compute()

        assert result["loss"] == pytest.approx(1.5)
        assert result["accuracy"] == pytest.approx(0.8)  # Only one update

    def test_empty_accumulator(self) -> None:
        """Empty accumulator should return empty dict."""
        acc = MetricsAccumulator()

        result = acc.compute()

        assert result == {}
