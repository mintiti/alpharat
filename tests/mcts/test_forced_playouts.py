"""Tests for forced playout pruning (KataGo-style).

The forced playout *threshold* is now handled by numba_ops.compute_puct_scores.
These tests cover the post-search *pruning* of forced visits.
"""

import numpy as np

from alpharat.mcts.selection import (
    compute_pruning_adjustment,
    prune_visit_counts,
)


class TestPruningAdjustment:
    """Tests for compute_pruning_adjustment (reduced-space inputs)."""

    def test_best_action_not_pruned(self) -> None:
        """The best outcome (most visits) should never be pruned."""
        q_values = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([50.0, 20.0, 15.0, 10.0, 5.0])
        total_visits = 100
        c_puct = 1.5

        delta = compute_pruning_adjustment(q_values, prior, marginal_visits, total_visits, c_puct)

        # Outcome 0 has most visits, should not be pruned
        assert delta[0] == 0.0

    def test_high_q_actions_not_pruned(self) -> None:
        """Outcomes with Q >= PUCT* should not be pruned."""
        q_values = np.array([0.5, 2.0, 0.2, 0.1, 0.0])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([50.0, 10.0, 15.0, 10.0, 5.0])
        total_visits = 90
        c_puct = 1.5

        delta = compute_pruning_adjustment(q_values, prior, marginal_visits, total_visits, c_puct)

        # Outcome 1 has Q = 2.0, which exceeds any PUCT score
        # PUCT* (for outcome 0) = 0.5 + 1.5 * 0.2 * sqrt(90) / 51 ≈ 0.56
        # Since Q[1]=2.0 > PUCT*, outcome 1 should not be pruned
        assert delta[1] == 0.0

    def test_low_visit_actions_pruned(self) -> None:
        """Outcomes with low Q and excessive visits get pruned."""
        q_values = np.array([0.5, 0.1, 0.1, 0.1, 0.1])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([40.0, 30.0, 10.0, 10.0, 10.0])
        total_visits = 100
        c_puct = 1.5

        delta = compute_pruning_adjustment(q_values, prior, marginal_visits, total_visits, c_puct)

        # Outcome 1 has low Q (0.1) vs PUCT* for outcome 0
        assert delta[1] > 0, "Low-Q outcome with excessive visits should be pruned"

    def test_single_outcome_no_pruning(self) -> None:
        """With only one unique outcome, no pruning should occur."""
        # Reduced to a single outcome (e.g. all actions blocked except STAY)
        q_values = np.array([0.5])
        prior = np.array([1.0])
        marginal_visits = np.array([100.0])
        total_visits = 100
        c_puct = 1.5

        delta = compute_pruning_adjustment(q_values, prior, marginal_visits, total_visits, c_puct)

        np.testing.assert_array_equal(delta, 0.0)

    def test_reduced_space_with_fewer_outcomes(self) -> None:
        """Works correctly when reduced space has fewer than 5 outcomes."""
        # 4 unique outcomes (one action was blocked → collapsed into STAY)
        q_values = np.array([0.5, 0.3, 0.2, 0.0])
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        marginal_visits = np.array([50.0, 20.0, 15.0, 5.0])
        total_visits = 90
        c_puct = 1.5

        delta = compute_pruning_adjustment(q_values, prior, marginal_visits, total_visits, c_puct)

        assert len(delta) == 4
        assert delta[0] == 0.0  # Best outcome not pruned


class TestPruneVisitCounts:
    """Tests for prune_visit_counts (reduced-space inputs)."""

    def test_no_adjustment_preserves_visits(self) -> None:
        """With zero adjustments, visits should be unchanged."""
        action_visits = np.array(
            [
                [10, 5, 3, 1, 1],
                [8, 4, 2, 1, 0],
                [5, 3, 2, 0, 0],
                [2, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        delta_p1 = np.zeros(5)
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        prior_p2 = np.array([0.4, 0.3, 0.2, 0.05, 0.05])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        np.testing.assert_array_equal(pruned, action_visits)

    def test_adjustment_reduces_visits(self) -> None:
        """Pruning adjustments should reduce visit counts."""
        action_visits = np.array(
            [
                [20, 10, 5, 3, 2],
                [15, 8, 4, 2, 1],
                [10, 5, 3, 1, 1],
                [5, 3, 2, 1, 0],
                [2, 1, 1, 0, 0],
            ],
            dtype=float,
        )
        delta_p1 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.5, 0.3, 0.1, 0.05, 0.05])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        # Row 0 should be reduced by delta_p1[0] * prior_p2
        assert pruned[0, 0] < action_visits[0, 0]
        # Other rows unchanged (delta_p1[1:] = 0)
        np.testing.assert_array_equal(pruned[1:, :], action_visits[1:, :])

    def test_clamps_to_zero(self) -> None:
        """Visits should never go negative."""
        action_visits = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        delta_p1 = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        assert (pruned >= 0).all()
        np.testing.assert_array_equal(pruned[0, :], 0.0)

    def test_both_players_adjusted(self) -> None:
        """Adjustments from both players should combine additively."""
        action_visits = np.full((5, 5), 10.0)
        delta_p1 = np.array([2.0, 0.0, 0.0, 0.0, 0.0])
        delta_p2 = np.array([0.0, 3.0, 0.0, 0.0, 0.0])
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        # Row 0 reduced by delta_p1[0] * prior_p2[j] = 2 * 0.2 = 0.4 per cell
        assert pruned[0, 0] < 10.0
        # Column 1 reduced by delta_p2[1] * prior_p1[i] = 3 * 0.2 = 0.6 per cell
        assert pruned[0, 1] < pruned[0, 0]  # (0,1) gets both adjustments

    def test_result_is_fractional(self) -> None:
        """Pruned visits can be fractional."""
        action_visits = np.full((5, 5), 10.0)
        delta_p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.3, 0.3, 0.2, 0.1, 0.1])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        # Row 0: pruned[0, j] = 10 - 1 * prior_p2[j]
        # pruned[0, 0] = 10 - 0.3 = 9.7
        assert abs(pruned[0, 0] - 9.7) < 1e-6

    def test_reduced_dimensions(self) -> None:
        """Works correctly with non-square reduced matrices."""
        # 3 outcomes for P1, 4 outcomes for P2
        action_visits = np.full((3, 4), 10.0)
        delta_p1 = np.array([2.0, 0.0, 0.0])
        delta_p2 = np.array([0.0, 1.0, 0.0, 0.0])
        prior_p1 = np.array([0.4, 0.3, 0.3])
        prior_p2 = np.array([0.25, 0.25, 0.25, 0.25])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        assert pruned.shape == (3, 4)
        assert (pruned >= 0).all()
        # Row 0 should be reduced
        assert pruned[0, 0] < 10.0
