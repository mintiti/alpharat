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
    """Tests for compute_pruning_adjustment."""

    def test_best_action_not_pruned(self) -> None:
        """The best action (most visits) should never be pruned."""
        q_values = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([50.0, 20.0, 15.0, 10.0, 5.0])
        total_visits = 100
        c_puct = 1.5
        effective = [0, 1, 2, 3, 4]  # All actions effective

        delta = compute_pruning_adjustment(
            q_values, prior, marginal_visits, total_visits, c_puct, effective
        )

        # Action 0 has most visits, should not be pruned
        assert delta[0] == 0.0

    def test_high_q_actions_not_pruned(self) -> None:
        """Actions with Q >= PUCT* should not be pruned."""
        # Set up so action 1 has very high Q (genuinely good)
        q_values = np.array([0.5, 2.0, 0.2, 0.1, 0.0])  # Action 1 has Q > any PUCT
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([50.0, 10.0, 15.0, 10.0, 5.0])
        total_visits = 90
        c_puct = 1.5
        effective = [0, 1, 2, 3, 4]

        delta = compute_pruning_adjustment(
            q_values, prior, marginal_visits, total_visits, c_puct, effective
        )

        # Action 1 has Q = 2.0, which exceeds any PUCT score
        # PUCT* (for action 0) = 0.5 + 1.5 * 0.2 * sqrt(90) / 51 ≈ 0.56
        # Since Q[1]=2.0 > PUCT*, action 1 should not be pruned
        assert delta[1] == 0.0

    def test_low_visit_actions_pruned(self) -> None:
        """Actions with low Q and excessive visits get pruned."""
        q_values = np.array([0.5, 0.1, 0.1, 0.1, 0.1])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        # Action 1 has way more visits than PUCT would justify
        marginal_visits = np.array([40.0, 30.0, 10.0, 10.0, 10.0])
        total_visits = 100
        c_puct = 1.5
        effective = [0, 1, 2, 3, 4]

        delta = compute_pruning_adjustment(
            q_values, prior, marginal_visits, total_visits, c_puct, effective
        )

        # Action 1 has low Q (0.1) vs PUCT* for action 0
        # PUCT* = 0.5 + 1.5 * 0.2 * sqrt(100) / 41 ≈ 0.57
        # For action 1: N'_min = 1.5 * 0.2 * 10 / (0.57 - 0.1) - 1 ≈ 5.4
        # Delta[1] = max(0, 30 - 5.4) ≈ 24.6
        assert delta[1] > 0, "Low-Q action with excessive visits should be pruned"

    def test_blocked_actions_ignored(self) -> None:
        """Blocked actions (not in effective) should have zero adjustment."""
        q_values = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        marginal_visits = np.array([50.0, 20.0, 15.0, 10.0, 5.0])
        total_visits = 100
        c_puct = 1.5
        # Action 0 is blocked (maps to STAY=4)
        effective = [4, 1, 2, 3, 4]

        delta = compute_pruning_adjustment(
            q_values, prior, marginal_visits, total_visits, c_puct, effective
        )

        # Action 0 is not effective, should be 0
        assert delta[0] == 0.0

    def test_single_effective_action_no_pruning(self) -> None:
        """With only one effective action, no pruning should occur."""
        q_values = np.array([0.5, 0.3, 0.2, 0.1, 0.0])
        prior = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        marginal_visits = np.array([0.0, 0.0, 0.0, 0.0, 100.0])
        total_visits = 100
        c_puct = 1.5
        # All actions blocked except STAY
        effective = [4, 4, 4, 4, 4]

        delta = compute_pruning_adjustment(
            q_values, prior, marginal_visits, total_visits, c_puct, effective
        )

        # No pruning when only one effective action
        np.testing.assert_array_equal(delta, 0.0)


class TestPruneVisitCounts:
    """Tests for prune_visit_counts."""

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
        delta_p1 = np.array([5.0, 0.0, 0.0, 0.0, 0.0])  # Prune 5 from action 0
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.5, 0.3, 0.1, 0.05, 0.05])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        # Row 0 should be reduced by delta_p1[0] * prior_p2
        # pruned[0, j] = max(0, visits[0, j] - 5 * prior_p2[j])
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
        delta_p1 = np.array([100.0, 0.0, 0.0, 0.0, 0.0])  # Large adjustment
        delta_p2 = np.zeros(5)
        prior_p1 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        prior_p2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        pruned = prune_visit_counts(action_visits, delta_p1, delta_p2, prior_p1, prior_p2)

        # Row 0 would go negative, but should be clamped to 0
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
