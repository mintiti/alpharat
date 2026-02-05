"""Tests for forced playout pruning."""

from __future__ import annotations

import numpy as np

from alpharat.mcts.forced_playouts import compute_pruned_visits


class TestComputePrunedVisits:
    """Tests for compute_pruned_visits function."""

    def test_best_outcome_unchanged(self) -> None:
        """Best outcome (highest visits) should keep all its visits."""
        q_values = np.array([0.5, 0.3, 0.2])
        prior = np.array([0.5, 0.3, 0.2])
        visits = np.array([100.0, 30.0, 10.0])  # outcome 0 is best
        total_visits = 140
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Best outcome unchanged
        assert pruned[0] == visits[0]

    def test_high_q_outcomes_unchanged(self) -> None:
        """Outcomes with Q >= PUCT* should keep all visits."""
        # Set up so outcome 1 has very high Q (above PUCT*)
        q_values = np.array([0.5, 10.0, 0.2])  # outcome 1 has very high Q
        prior = np.array([0.5, 0.3, 0.2])
        visits = np.array([100.0, 30.0, 10.0])
        total_visits = 140
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Outcome 1 has Q >= PUCT*, should be unchanged
        assert pruned[1] == visits[1]

    def test_forced_visits_pruned(self) -> None:
        """Low-Q outcomes with high visits should have visits capped."""
        # outcome 0: high visits, decent Q (best)
        # outcome 1: low visits, low Q (should be kept)
        # outcome 2: high visits, very low Q (should be pruned - these are "forced")
        q_values = np.array([0.5, 0.3, -0.5])
        prior = np.array([0.4, 0.3, 0.3])
        visits = np.array([80.0, 20.0, 40.0])  # outcome 2 has suspiciously high visits
        total_visits = 140
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Outcome 2 should be pruned (low Q but high visits = forced)
        assert pruned[2] < visits[2]
        # Best outcome unchanged
        assert pruned[0] == visits[0]

    def test_single_outcome_no_pruning(self) -> None:
        """Single outcome should return unchanged."""
        q_values = np.array([0.5])
        prior = np.array([1.0])
        visits = np.array([100.0])
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        np.testing.assert_array_equal(pruned, visits)

    def test_zero_total_visits(self) -> None:
        """Zero total visits should return unchanged."""
        q_values = np.array([0.5, 0.3, 0.2])
        prior = np.array([0.5, 0.3, 0.2])
        visits = np.array([0.0, 0.0, 0.0])
        total_visits = 0
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        np.testing.assert_array_equal(pruned, visits)

    def test_zero_prior_action_pruned_to_zero(self) -> None:
        """Action with zero prior should be pruned to zero visits."""
        q_values = np.array([0.5, 0.3, 0.2])
        prior = np.array([0.5, 0.5, 0.0])  # outcome 2 has zero prior
        visits = np.array([50.0, 30.0, 20.0])  # but somehow has visits (forced)
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Zero prior means this action shouldn't have been visited
        assert pruned[2] == 0.0

    def test_no_negative_visits(self) -> None:
        """Pruned visits should never go negative."""
        q_values = np.array([0.5, -10.0])  # outcome 1 has very bad Q
        prior = np.array([0.5, 0.5])
        visits = np.array([50.0, 50.0])
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        assert np.all(pruned >= 0)

    def test_visits_never_increase(self) -> None:
        """Pruning should never increase visit counts."""
        q_values = np.array([0.5, 0.3, 0.2, 0.1])
        prior = np.array([0.25, 0.25, 0.25, 0.25])
        visits = np.array([40.0, 30.0, 20.0, 10.0])
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        assert np.all(pruned <= visits)

    def test_returns_copy_not_view(self) -> None:
        """Should return a copy, not modify the input."""
        q_values = np.array([0.5, 0.3])
        prior = np.array([0.5, 0.5])
        visits = np.array([50.0, 50.0])
        total_visits = 100
        c_puct = 1.5

        original_visits = visits.copy()
        _ = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        np.testing.assert_array_equal(visits, original_visits)

    def test_near_zero_q_gap(self) -> None:
        """Near-zero Q gap should handle numerical stability."""
        q_values = np.array([0.5, 0.4999999])  # Very close Q values
        prior = np.array([0.5, 0.5])
        visits = np.array([60.0, 40.0])
        total_visits = 100
        c_puct = 1.5

        # Should not crash or produce NaN/Inf
        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        assert np.all(np.isfinite(pruned))
        assert np.all(pruned >= 0)

    def test_all_equal_q_values(self) -> None:
        """Equal Q-values: only best by visits unchanged, others may be pruned."""
        q_values = np.array([0.5, 0.5, 0.5])
        prior = np.array([0.4, 0.3, 0.3])
        visits = np.array([50.0, 30.0, 20.0])
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Best by visits unchanged
        assert pruned[0] == visits[0]
        # With equal Q, others depend on their priors vs PUCT*
        assert np.all(pruned <= visits)


class TestPruningMathematicalProperties:
    """Tests verifying the mathematical properties of the pruning formula."""

    def test_puct_formula_derivation(self) -> None:
        """Verify the pruning formula matches PUCT derivation.

        For action i with Q(i) < PUCT*, the max visits is where:
        Q(i) + c * P(i) * sqrt(N) / (1 + n) = PUCT*

        Solving: n = c * P(i) * sqrt(N) / (PUCT* - Q(i)) - 1
        """
        q_values = np.array([0.6, 0.2])
        prior = np.array([0.7, 0.3])
        visits = np.array([70.0, 30.0])  # Outcome 0 is best
        total_visits = 100
        c_puct = 2.0

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        # Manually compute PUCT* for best action
        sqrt_n = np.sqrt(total_visits)
        puct_best = q_values[0] + c_puct * prior[0] * sqrt_n / (1 + visits[0])

        # Compute expected n_min for action 1
        gap = puct_best - q_values[1]
        expected_n_min = c_puct * prior[1] * sqrt_n / gap - 1
        expected_pruned = min(visits[1], max(0.0, expected_n_min))

        np.testing.assert_almost_equal(pruned[1], expected_pruned, decimal=10)

    def test_pruned_action_puct_equals_best(self) -> None:
        """After pruning, non-best action's PUCT should equal PUCT* (if visits were capped)."""
        q_values = np.array([0.5, -0.3])  # Action 1 has low Q
        prior = np.array([0.5, 0.5])
        visits = np.array([80.0, 20.0])
        total_visits = 100
        c_puct = 1.5

        pruned = compute_pruned_visits(q_values, prior, visits, total_visits, c_puct)

        sqrt_n = np.sqrt(total_visits)
        puct_best = q_values[0] + c_puct * prior[0] * sqrt_n / (1 + visits[0])

        # If action 1 was pruned (visits reduced), its new PUCT should match PUCT*
        if pruned[1] < visits[1]:
            puct_1 = q_values[1] + c_puct * prior[1] * sqrt_n / (1 + pruned[1])
            np.testing.assert_almost_equal(puct_1, puct_best, decimal=10)
