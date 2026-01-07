"""Tests for forced playout functionality."""

import numpy as np

from alpharat.mcts.selection import compute_forced_threshold


class TestForcedThreshold:
    """Tests for compute_forced_threshold."""

    def test_basic_computation(self) -> None:
        """Test threshold formula: sqrt(k * prior * total_visits)."""
        prior = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        total_visits = 100
        k = 2.0

        threshold = compute_forced_threshold(prior, total_visits, k)

        # sqrt(2 * 0.4 * 100) = sqrt(80) ≈ 8.94
        expected_0 = np.sqrt(2.0 * 0.4 * 100)
        assert abs(threshold[0] - expected_0) < 1e-6

        # sqrt(2 * 0.3 * 100) = sqrt(60) ≈ 7.75
        expected_1 = np.sqrt(2.0 * 0.3 * 100)
        assert abs(threshold[1] - expected_1) < 1e-6

    def test_zero_k_disables_forcing(self) -> None:
        """With k=0, all thresholds should be 0 (no forcing)."""
        prior = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        total_visits = 100
        k = 0.0

        threshold = compute_forced_threshold(prior, total_visits, k)

        np.testing.assert_array_equal(threshold, 0.0)

    def test_zero_prior_gives_zero_threshold(self) -> None:
        """Actions with 0 prior should have 0 threshold (never forced)."""
        prior = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        total_visits = 100
        k = 2.0

        threshold = compute_forced_threshold(prior, total_visits, k)

        assert threshold[0] > 0
        assert threshold[1] > 0
        assert threshold[2] == 0.0
        assert threshold[3] == 0.0
        assert threshold[4] == 0.0

    def test_sublinear_scaling(self) -> None:
        """Threshold scales sublinearly with visits (sqrt)."""
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        k = 2.0

        thresh_100 = compute_forced_threshold(prior, 100, k)
        thresh_400 = compute_forced_threshold(prior, 400, k)

        # With 4x visits, threshold should only double (sqrt scaling)
        np.testing.assert_array_almost_equal(thresh_400, 2 * thresh_100)

    def test_uniform_prior(self) -> None:
        """Uniform prior gives equal thresholds for all actions."""
        prior = np.ones(5) / 5  # [0.2, 0.2, 0.2, 0.2, 0.2]
        total_visits = 100
        k = 2.0

        threshold = compute_forced_threshold(prior, total_visits, k)

        # All thresholds should be equal
        assert np.allclose(threshold, threshold[0])

        # sqrt(2 * 0.2 * 100) = sqrt(40) ≈ 6.32
        expected = np.sqrt(2.0 * 0.2 * 100)
        np.testing.assert_array_almost_equal(threshold, expected)


class TestForcedMask:
    """Tests for forced action mask creation."""

    def test_undervisited_actions_forced(self) -> None:
        """Actions with visits below threshold should be forced."""
        prior = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        total_visits = 100
        k = 2.0

        threshold = compute_forced_threshold(prior, total_visits, k)
        marginal_visits = np.array([5.0, 10.0, 20.0, 0.0, 0.0])

        # threshold[0] = sqrt(2 * 0.5 * 100) = 10, visits=5 < 10 → forced
        # threshold[1] = sqrt(2 * 0.3 * 100) ≈ 7.75, visits=10 > 7.75 → not forced
        # threshold[2] = sqrt(2 * 0.2 * 100) ≈ 6.32, visits=20 > 6.32 → not forced
        forced_mask = marginal_visits < threshold

        assert forced_mask[0] == True  # noqa: E712
        assert forced_mask[1] == False  # noqa: E712
        assert forced_mask[2] == False  # noqa: E712
        assert forced_mask[3] == False  # noqa: E712 (0 threshold, 0 visits)
        assert forced_mask[4] == False  # noqa: E712

    def test_all_visited_enough_none_forced(self) -> None:
        """When all actions have enough visits, no forcing needed."""
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        total_visits = 100
        k = 2.0

        threshold = compute_forced_threshold(prior, total_visits, k)
        # Each threshold is sqrt(40) ≈ 6.32
        marginal_visits = np.array([20.0, 20.0, 20.0, 20.0, 20.0])

        forced_mask = marginal_visits < threshold

        assert not forced_mask.any()
