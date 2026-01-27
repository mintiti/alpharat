"""Tests for payout matrix filtering."""

import numpy as np

from alpharat.mcts.payout_filter import filter_low_visit_payout


class TestFilterLowVisitPayout:
    """Tests for filter_low_visit_payout()."""

    def test_basic_filtering(self) -> None:
        """Cells with visits < min_visits are zeroed, others preserved."""
        payout = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
            ]
        )
        visits = np.array([[10.0, 0.0, 5.0], [1.0, 3.0, 0.0]])

        result = filter_low_visit_payout(payout, visits, min_visits=2)

        # Cells with visits >= 2 are preserved
        assert result[0, 0, 0] == 1.0  # visits=10
        assert result[0, 0, 2] == 3.0  # visits=5
        assert result[0, 1, 1] == 5.0  # visits=3

        # Cells with visits < 2 are zeroed
        assert result[0, 0, 1] == 0.0  # visits=0
        assert result[0, 1, 0] == 0.0  # visits=1
        assert result[0, 1, 2] == 0.0  # visits=0

        # Both player slices are affected
        assert result[1, 0, 0] == 0.5  # visits=10, preserved
        assert result[1, 0, 1] == 0.0  # visits=0, zeroed

    def test_shape_and_dtype_preserved(self) -> None:
        """Output has same shape and dtype as input."""
        payout = np.ones((2, 4, 3), dtype=np.float32)
        visits = np.ones((4, 3))

        result = filter_low_visit_payout(payout, visits, min_visits=2)

        assert result.shape == payout.shape
        assert result.dtype == payout.dtype

    def test_original_not_mutated(self) -> None:
        """Original payout array is not modified."""
        payout = np.ones((2, 3, 3))
        original_copy = payout.copy()
        visits = np.zeros((3, 3))

        filter_low_visit_payout(payout, visits, min_visits=2)

        np.testing.assert_array_equal(payout, original_copy)

    def test_reduced_matrices(self) -> None:
        """Works with reduced [2, n1, n2] matrices (not just [2, 5, 5])."""
        payout = np.ones((2, 3, 4)) * 5.0
        visits = np.array([[10, 0, 0, 5], [0, 10, 0, 0], [0, 0, 10, 0]])

        result = filter_low_visit_payout(payout, visits, min_visits=2)

        assert result.shape == (2, 3, 4)
        # Only cells with visits >= 2 should have non-zero payout
        assert result[0, 0, 0] == 5.0
        assert result[0, 0, 3] == 5.0
        assert result[0, 1, 1] == 5.0
        assert result[0, 2, 2] == 5.0
        # All other cells zeroed
        assert result[0, 0, 1] == 0.0
        assert result[0, 0, 2] == 0.0

    def test_all_below_threshold(self) -> None:
        """All cells below threshold → all zeros."""
        payout = np.ones((2, 3, 3)) * 10.0
        visits = np.ones((3, 3))  # all visits = 1

        result = filter_low_visit_payout(payout, visits, min_visits=2)

        np.testing.assert_array_equal(result, 0.0)

    def test_min_visits_zero(self) -> None:
        """min_visits=0 → nothing filtered (all visits >= 0)."""
        payout = np.ones((2, 3, 3)) * 7.0
        visits = np.zeros((3, 3))

        result = filter_low_visit_payout(payout, visits, min_visits=0)

        np.testing.assert_array_equal(result, payout)

    def test_default_min_visits(self) -> None:
        """Default min_visits=2 filters visits < 2."""
        payout = np.ones((2, 2, 2))
        visits = np.array([[1.0, 2.0], [0.0, 3.0]])

        result = filter_low_visit_payout(payout, visits)

        assert result[0, 0, 0] == 0.0  # visits=1 < 2
        assert result[0, 0, 1] == 1.0  # visits=2 >= 2
        assert result[0, 1, 0] == 0.0  # visits=0 < 2
        assert result[0, 1, 1] == 1.0  # visits=3 >= 2
