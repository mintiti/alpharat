"""Tests for Numba JIT-compiled MCTS operations."""

import numpy as np
import pytest

from alpharat.mcts.numba_ops import compute_puct_scores, select_max_with_tiebreak


class TestComputePuctScores:
    """Tests for compute_puct_scores."""

    def test_basic_puct_formula(self) -> None:
        """Verify PUCT = Q + c * prior * sqrt(N) / (1 + n)."""
        q_values = np.array([1.0, 2.0, 3.0])
        prior = np.array([0.5, 0.3, 0.2])
        visit_counts = np.array([10.0, 5.0, 0.0])
        total_visits = 100
        c_puct = 1.5
        force_k = 0.0  # Disabled
        is_root = False

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # Manual calculation: sqrt(N) = sqrt(100) = 10
        expected_0 = 1.0 + 1.5 * 0.5 * 10 / (1 + 10)  # 1.0 + 0.68
        expected_1 = 2.0 + 1.5 * 0.3 * 10 / (1 + 5)  # 2.0 + 0.75
        expected_2 = 3.0 + 1.5 * 0.2 * 10 / (1 + 0)  # 3.0 + 3.0

        assert scores[0] == pytest.approx(expected_0)
        assert scores[1] == pytest.approx(expected_1)
        assert scores[2] == pytest.approx(expected_2)

    def test_zero_visits_high_exploration(self) -> None:
        """Unvisited actions get high exploration bonus."""
        q_values = np.zeros(3)
        prior = np.array([0.33, 0.33, 0.34])
        visit_counts = np.array([100.0, 100.0, 0.0])
        total_visits = 200
        c_puct = 1.5
        force_k = 0.0
        is_root = False

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # Unvisited action should have highest score due to exploration
        assert scores[2] > scores[0]
        assert scores[2] > scores[1]


class TestForcedPlayouts:
    """Tests for forced playout behavior in compute_puct_scores."""

    def test_undervisited_actions_forced_at_root(self) -> None:
        """Undervisited actions get boosted score at root with force_k > 0."""
        q_values = np.zeros(5)
        prior = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        visit_counts = np.array([5.0, 10.0, 20.0, 0.0, 0.0])
        total_visits = 100
        c_puct = 1.5
        force_k = 2.0
        is_root = True

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # threshold[0] = sqrt(2 * 0.5 * 100) = 10, visits=5 < 10 → forced
        # threshold[1] = sqrt(2 * 0.3 * 100) ≈ 7.75, visits=10 > 7.75 → not forced
        # threshold[2] = sqrt(2 * 0.2 * 100) ≈ 6.32, visits=20 > 6.32 → not forced
        assert scores[0] == pytest.approx(1e20)  # Forced
        assert scores[1] < 1e10  # Not forced
        assert scores[2] < 1e10  # Not forced

    def test_zero_force_k_disables_forcing(self) -> None:
        """With force_k=0, no actions are forced regardless of visits."""
        q_values = np.zeros(3)
        prior = np.array([0.5, 0.3, 0.2])
        visit_counts = np.array([0.0, 0.0, 0.0])  # All undervisited
        total_visits = 100
        c_puct = 1.5
        force_k = 0.0  # Disabled
        is_root = True

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # No scores should be 1e20
        assert all(s < 1e10 for s in scores)

    def test_forcing_only_at_root(self) -> None:
        """Forced playouts only apply at root node."""
        q_values = np.zeros(3)
        prior = np.array([0.5, 0.3, 0.2])
        visit_counts = np.array([0.0, 0.0, 0.0])  # All undervisited
        total_visits = 100
        c_puct = 1.5
        force_k = 2.0
        is_root = False  # Not root

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # No forcing at non-root
        assert all(s < 1e10 for s in scores)

    def test_zero_prior_never_forced(self) -> None:
        """Actions with zero prior are never forced."""
        q_values = np.zeros(3)
        prior = np.array([0.5, 0.5, 0.0])  # Third has zero prior
        visit_counts = np.array([0.0, 0.0, 0.0])
        total_visits = 100
        c_puct = 1.5
        force_k = 2.0
        is_root = True

        scores = compute_puct_scores(
            q_values, prior, visit_counts, total_visits, c_puct, force_k, is_root
        )

        # First two forced, third not (zero prior)
        assert scores[0] == pytest.approx(1e20)
        assert scores[1] == pytest.approx(1e20)
        assert scores[2] < 1e10

    def test_sublinear_threshold_scaling(self) -> None:
        """Threshold scales sublinearly with visits (sqrt)."""
        q_values = np.zeros(1)
        prior = np.array([0.2])
        c_puct = 1.5
        force_k = 2.0
        is_root = True

        # At 100 visits: threshold = sqrt(2 * 0.2 * 100) = sqrt(40) ≈ 6.32
        # At 400 visits: threshold = sqrt(2 * 0.2 * 400) = sqrt(160) ≈ 12.65 (2x)

        # 5 visits: forced at 100, forced at 400
        visit_counts = np.array([5.0])
        scores_100 = compute_puct_scores(
            q_values, prior, visit_counts, 100, c_puct, force_k, is_root
        )
        scores_400 = compute_puct_scores(
            q_values, prior, visit_counts, 400, c_puct, force_k, is_root
        )
        assert scores_100[0] == pytest.approx(1e20)
        assert scores_400[0] == pytest.approx(1e20)

        # 10 visits: not forced at 100, forced at 400
        visit_counts = np.array([10.0])
        scores_100 = compute_puct_scores(
            q_values, prior, visit_counts, 100, c_puct, force_k, is_root
        )
        scores_400 = compute_puct_scores(
            q_values, prior, visit_counts, 400, c_puct, force_k, is_root
        )
        assert scores_100[0] < 1e10  # Not forced (10 > 6.32)
        assert scores_400[0] == pytest.approx(1e20)  # Forced (10 < 12.65)


class TestSelectMaxWithTiebreak:
    """Tests for select_max_with_tiebreak."""

    def test_clear_maximum(self) -> None:
        """Returns index of clear maximum."""
        scores = np.array([1.0, 5.0, 3.0, 2.0])
        assert select_max_with_tiebreak(scores) == 1

    def test_first_element_max(self) -> None:
        """Works when first element is max."""
        scores = np.array([10.0, 5.0, 3.0, 2.0])
        assert select_max_with_tiebreak(scores) == 0

    def test_last_element_max(self) -> None:
        """Works when last element is max."""
        scores = np.array([1.0, 2.0, 3.0, 10.0])
        assert select_max_with_tiebreak(scores) == 3

    def test_ties_select_one_of_maxes(self) -> None:
        """With ties, returns one of the tied indices."""
        scores = np.array([1.0, 5.0, 5.0, 2.0])
        # Run multiple times - should only return 1 or 2
        results = {select_max_with_tiebreak(scores) for _ in range(100)}
        assert results <= {1, 2}
        # Should hit both eventually (probabilistic but very likely)
        assert len(results) == 2

    def test_all_equal_random_selection(self) -> None:
        """All equal scores gives random selection."""
        scores = np.array([3.0, 3.0, 3.0, 3.0])
        results = {select_max_with_tiebreak(scores) for _ in range(200)}
        # Should hit all indices eventually
        assert results == {0, 1, 2, 3}

    def test_single_element(self) -> None:
        """Single element array returns 0."""
        scores = np.array([5.0])
        assert select_max_with_tiebreak(scores) == 0
