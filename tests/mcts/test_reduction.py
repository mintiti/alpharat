"""Tests for boundary translation between 5-action and outcome-indexed space."""

import numpy as np
import pytest

from alpharat.mcts.reduction import (
    build_action_to_outcome_map,
    expand_payout,
    expand_prior,
    expand_visits,
    get_unique_outcomes,
    outcome_to_effective,
    reduce_payout,
    reduce_prior,
)


class TestGetUniqueOutcomes:
    """Tests for get_unique_outcomes."""

    def test_no_equivalence(self) -> None:
        """All actions are distinct outcomes."""
        effective = [0, 1, 2, 3, 4]
        assert get_unique_outcomes(effective) == [0, 1, 2, 3, 4]

    def test_one_blocked_action(self) -> None:
        """UP blocked → maps to STAY."""
        effective = [4, 1, 2, 3, 4]  # Action 0 → 4
        assert get_unique_outcomes(effective) == [1, 2, 3, 4]

    def test_corner_position(self) -> None:
        """Corner: UP and LEFT both blocked."""
        effective = [4, 1, 2, 4, 4]  # Actions 0, 3 → 4
        assert get_unique_outcomes(effective) == [1, 2, 4]

    def test_mud_all_equivalent(self) -> None:
        """Player in mud: all actions equivalent to STAY."""
        effective = [4, 4, 4, 4, 4]
        assert get_unique_outcomes(effective) == [4]


class TestBuildActionToOutcomeMap:
    """Tests for build_action_to_outcome_map."""

    def test_no_equivalence(self) -> None:
        """Identity mapping when no equivalence."""
        effective = [0, 1, 2, 3, 4]
        result = build_action_to_outcome_map(effective)
        assert result == [0, 1, 2, 3, 4]

    def test_one_blocked_action(self) -> None:
        """UP blocked maps action 0 to same outcome index as STAY."""
        effective = [4, 1, 2, 3, 4]
        # Unique outcomes: [1, 2, 3, 4] → indices 0, 1, 2, 3
        # Action 0 → effective 4 → outcome index 3
        # Action 1 → effective 1 → outcome index 0
        # Action 4 → effective 4 → outcome index 3
        result = build_action_to_outcome_map(effective)
        assert result[0] == result[4]  # Both map to STAY outcome
        assert result == [3, 0, 1, 2, 3]

    def test_mud_all_same(self) -> None:
        """All actions map to same outcome index."""
        effective = [4, 4, 4, 4, 4]
        result = build_action_to_outcome_map(effective)
        assert result == [0, 0, 0, 0, 0]


class TestOutcomeToEffective:
    """Tests for outcome_to_effective."""

    def test_no_equivalence(self) -> None:
        """Outcome index equals effective action when no equivalence."""
        effective = [0, 1, 2, 3, 4]
        for i in range(5):
            assert outcome_to_effective(i, effective) == i

    def test_with_equivalence(self) -> None:
        """Outcome index maps to correct effective action."""
        effective = [4, 1, 2, 3, 4]
        # Unique outcomes: [1, 2, 3, 4]
        assert outcome_to_effective(0, effective) == 1
        assert outcome_to_effective(1, effective) == 2
        assert outcome_to_effective(2, effective) == 3
        assert outcome_to_effective(3, effective) == 4


class TestReducePrior:
    """Tests for reduce_prior."""

    def test_no_equivalence(self) -> None:
        """No reduction when no equivalence."""
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        effective = [0, 1, 2, 3, 4]
        result = reduce_prior(prior, effective)
        np.testing.assert_array_almost_equal(result, prior)

    def test_sums_equivalent_probabilities(self) -> None:
        """Equivalent actions' probabilities are summed."""
        prior = np.array([0.1, 0.2, 0.3, 0.2, 0.2])  # Sum = 1.0
        effective = [4, 1, 2, 3, 4]  # Action 0 → 4
        # Unique outcomes: [1, 2, 3, 4]
        # Outcome 3 (effective=4) gets prior[0] + prior[4] = 0.1 + 0.2 = 0.3
        result = reduce_prior(prior, effective)
        assert len(result) == 4
        np.testing.assert_array_almost_equal(result, [0.2, 0.3, 0.2, 0.3])
        assert result.sum() == pytest.approx(1.0)

    def test_mud_all_equivalent(self) -> None:
        """All probabilities sum to single outcome."""
        prior = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        effective = [4, 4, 4, 4, 4]
        result = reduce_prior(prior, effective)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_corner_position(self) -> None:
        """Corner: two blocked actions."""
        prior = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
        effective = [4, 1, 2, 4, 4]  # 0, 3, 4 all → 4
        # Unique outcomes: [1, 2, 4]
        # Outcome 2 (effective=4) gets prior[0] + prior[3] + prior[4] = 0.5
        result = reduce_prior(prior, effective)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.3, 0.2, 0.5])


class TestReducePayout:
    """Tests for reduce_payout."""

    def test_no_equivalence(self) -> None:
        """No reduction when no equivalence."""
        payout = np.random.rand(2, 5, 5)
        p1_eff = [0, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]
        result = reduce_payout(payout, p1_eff, p2_eff)
        np.testing.assert_array_almost_equal(result, payout)

    def test_p1_equivalence_averages(self) -> None:
        """P1 has equivalence - should average equivalent rows."""
        payout = np.zeros((2, 5, 5))
        # Row 0 and row 4 are equivalent (both map to STAY outcome)
        # Give them different values to test averaging
        payout[0, 0, :] = [1.0, 2.0, 3.0, 4.0, 5.0]  # action 0
        payout[0, 4, :] = [3.0, 4.0, 5.0, 6.0, 7.0]  # action 4 (STAY)
        # Other rows
        payout[0, 1, :] = [10.0, 10.0, 10.0, 10.0, 10.0]
        payout[0, 2, :] = [20.0, 20.0, 20.0, 20.0, 20.0]
        payout[0, 3, :] = [30.0, 30.0, 30.0, 30.0, 30.0]

        p1_eff = [4, 1, 2, 3, 4]  # 0 → 4
        p2_eff = [0, 1, 2, 3, 4]

        result = reduce_payout(payout, p1_eff, p2_eff)
        assert result.shape == (2, 4, 5)

        # Outcome 3 (STAY) should be average of rows 0 and 4
        expected_stay_row = (payout[0, 0, :] + payout[0, 4, :]) / 2
        np.testing.assert_array_almost_equal(result[0, 3, :], expected_stay_row)

    def test_both_equivalence_averages_all(self) -> None:
        """Both players have equivalence - averages all 25 cells."""
        payout = np.zeros((2, 5, 5))
        # Fill with different values
        payout[0, :, :] = np.arange(25).reshape(5, 5).astype(float)
        payout[1, :, :] = np.arange(25, 50).reshape(5, 5).astype(float)

        p1_eff = [4, 4, 4, 4, 4]  # All → 4
        p2_eff = [4, 4, 4, 4, 4]

        result = reduce_payout(payout, p1_eff, p2_eff)
        assert result.shape == (2, 1, 1)

        # Should be average of all 25 values
        assert result[0, 0, 0] == pytest.approx(payout[0].mean())
        assert result[1, 0, 0] == pytest.approx(payout[1].mean())

    def test_noisy_nn_prediction_averaged(self) -> None:
        """Noisy NN predictions are averaged for equivalent pairs."""
        # Simulate noisy NN that doesn't perfectly respect equivalence
        payout = np.zeros((2, 5, 5))

        # Action 0 is blocked (maps to STAY=4)
        # NN gives slightly different values for equivalent pairs
        payout[0, 0, 2] = 5.0  # (UP, DOWN) - blocked
        payout[0, 4, 2] = 5.2  # (STAY, DOWN) - effective
        payout[1, 0, 2] = 3.0
        payout[1, 4, 2] = 3.4

        p1_eff = [4, 1, 2, 3, 4]  # 0 → 4
        p2_eff = [0, 1, 2, 3, 4]

        result = reduce_payout(payout, p1_eff, p2_eff)

        # Outcome (STAY, DOWN) should average the noisy values
        # STAY outcome is index 3 in reduced (outcomes = [1,2,3,4])
        assert result[0, 3, 2] == pytest.approx(5.1)  # (5.0 + 5.2) / 2
        assert result[1, 3, 2] == pytest.approx(3.2)  # (3.0 + 3.4) / 2

    def test_identical_values_unchanged(self) -> None:
        """When equivalent pairs already have identical values, result is same."""
        payout = np.zeros((2, 5, 5))
        # Set up so equivalent pairs have identical values (like after backup)
        payout[0, 0, :] = [5.0, 5.0, 5.0, 5.0, 5.0]
        payout[0, 4, :] = [5.0, 5.0, 5.0, 5.0, 5.0]  # Same as row 0

        p1_eff = [4, 1, 2, 3, 4]  # 0 → 4
        p2_eff = [0, 1, 2, 3, 4]

        result = reduce_payout(payout, p1_eff, p2_eff)

        # Average of identical values = same value
        np.testing.assert_array_almost_equal(result[0, 3, :], [5.0, 5.0, 5.0, 5.0, 5.0])


class TestExpandPrior:
    """Tests for expand_prior."""

    def test_no_equivalence(self) -> None:
        """No expansion when no equivalence."""
        prior_n = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        effective = [0, 1, 2, 3, 4]
        result = expand_prior(prior_n, effective)
        np.testing.assert_array_almost_equal(result, prior_n)

    def test_blocked_action_gets_zero(self) -> None:
        """Blocked action gets 0, effective action gets full probability."""
        prior_n = np.array([0.3, 0.2, 0.2, 0.3])  # [1, 2, 3, 4] outcomes
        effective = [4, 1, 2, 3, 4]

        result = expand_prior(prior_n, effective)

        # Action 0 blocked → gets 0
        assert result[0] == 0.0
        # Action 4 (STAY) gets the STAY outcome probability
        assert result[4] == 0.3
        # Other actions get their probabilities
        assert result[1] == 0.3
        assert result[2] == 0.2
        assert result[3] == 0.2

    def test_mud_single_outcome(self) -> None:
        """All probability on single effective action."""
        prior_n = np.array([1.0])
        effective = [4, 4, 4, 4, 4]

        result = expand_prior(prior_n, effective)

        # Only STAY gets probability
        assert result[4] == 1.0
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0


class TestExpandPayout:
    """Tests for expand_payout."""

    def test_no_equivalence(self) -> None:
        """No expansion when no equivalence."""
        payout = np.random.rand(2, 5, 5)
        p1_eff = [0, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]
        result = expand_payout(payout, p1_eff, p2_eff)
        np.testing.assert_array_almost_equal(result, payout)

    def test_equivalent_pairs_get_same_value(self) -> None:
        """Equivalent action pairs all get the same payout value."""
        payout_n1n2 = np.zeros((2, 4, 5))
        payout_n1n2[0, 3, 2] = 7.0  # Outcome (3, 2) = effective (4, 2)
        payout_n1n2[1, 3, 2] = 3.5

        p1_eff = [4, 1, 2, 3, 4]  # 0, 4 → 4 (outcome 3)
        p2_eff = [0, 1, 2, 3, 4]

        result = expand_payout(payout_n1n2, p1_eff, p2_eff)

        # Both (0, 2) and (4, 2) should have the same value
        assert result[0, 0, 2] == 7.0
        assert result[0, 4, 2] == 7.0
        assert result[1, 0, 2] == 3.5
        assert result[1, 4, 2] == 3.5

    def test_mud_single_value_expanded(self) -> None:
        """Single outcome value expanded to all 25 pairs."""
        payout_n1n2 = np.array([[[10.0]], [[5.0]]])

        p1_eff = [4, 4, 4, 4, 4]
        p2_eff = [4, 4, 4, 4, 4]

        result = expand_payout(payout_n1n2, p1_eff, p2_eff)

        assert result.shape == (2, 5, 5)
        np.testing.assert_array_equal(result[0], np.full((5, 5), 10.0))
        np.testing.assert_array_equal(result[1], np.full((5, 5), 5.0))


class TestExpandVisits:
    """Tests for expand_visits."""

    def test_no_equivalence(self) -> None:
        """No expansion when no equivalence."""
        visits = np.random.randint(0, 100, (5, 5))
        p1_eff = [0, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]
        result = expand_visits(visits, p1_eff, p2_eff)
        np.testing.assert_array_equal(result, visits)

    def test_equivalent_pairs_same_visits(self) -> None:
        """Equivalent action pairs get same visit count."""
        visits_n1n2 = np.zeros((4, 5), dtype=np.int32)
        visits_n1n2[3, 2] = 10  # Outcome (3, 2)

        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        result = expand_visits(visits_n1n2, p1_eff, p2_eff)

        assert result[0, 2] == 10  # Action 0 → outcome 3
        assert result[4, 2] == 10  # Action 4 → outcome 3


class TestRoundTrip:
    """Tests verifying expand(reduce(x)) ≈ x with equivalence-aware comparison."""

    def test_prior_roundtrip_no_equivalence(self) -> None:
        """Prior unchanged when no equivalence."""
        prior = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        effective = [0, 1, 2, 3, 4]

        reduced = reduce_prior(prior, effective)
        expanded = expand_prior(reduced, effective)

        np.testing.assert_array_almost_equal(expanded, prior)

    def test_prior_roundtrip_with_equivalence(self) -> None:
        """Reduced prior expands to canonical form."""
        prior = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        effective = [4, 1, 2, 3, 4]

        reduced = reduce_prior(prior, effective)
        expanded = expand_prior(reduced, effective)

        # Blocked action 0 should now be 0
        assert expanded[0] == 0.0
        # STAY should have combined probability
        assert expanded[4] == pytest.approx(0.3)  # 0.1 + 0.2
        # Other actions unchanged
        assert expanded[1] == pytest.approx(0.2)
        assert expanded[2] == pytest.approx(0.3)
        assert expanded[3] == pytest.approx(0.2)
        # Total still sums to 1
        assert expanded.sum() == pytest.approx(1.0)

    def test_payout_roundtrip_equivalence_invariant(self) -> None:
        """Payout roundtrip preserves equivalence structure.

        When payout matrix already respects equivalence (equivalent pairs have
        identical values), roundtrip should be exact.
        """
        # Create payout that respects equivalence:
        # Actions 0 and 4 (both → STAY) have identical rows
        payout = np.zeros((2, 5, 5))
        for a1 in range(5):
            for a2 in range(5):
                payout[0, a1, a2] = a1 * 10 + a2
                payout[1, a1, a2] = a1 * 10 + a2 + 100

        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        # Make rows 0 and 4 identical (equivalence invariant)
        payout[:, 0, :] = payout[:, 4, :]

        reduced = reduce_payout(payout, p1_eff, p2_eff)
        expanded = expand_payout(reduced, p1_eff, p2_eff)

        np.testing.assert_array_almost_equal(expanded, payout)

    def test_payout_roundtrip_noisy_input(self) -> None:
        """Payout roundtrip averages noisy equivalent pairs.

        When input doesn't respect equivalence (NN noise), roundtrip
        produces a matrix where equivalent pairs ARE identical.
        """
        # Create noisy payout that doesn't respect equivalence
        payout = np.zeros((2, 5, 5))
        payout[0, 0, :] = [1.0, 2.0, 3.0, 4.0, 5.0]  # Row 0
        payout[0, 4, :] = [3.0, 4.0, 5.0, 6.0, 7.0]  # Row 4 (different!)
        payout[0, 1, :] = [10.0] * 5
        payout[0, 2, :] = [20.0] * 5
        payout[0, 3, :] = [30.0] * 5

        p1_eff = [4, 1, 2, 3, 4]  # 0 → 4
        p2_eff = [0, 1, 2, 3, 4]

        reduced = reduce_payout(payout, p1_eff, p2_eff)
        expanded = expand_payout(reduced, p1_eff, p2_eff)

        # After roundtrip, rows 0 and 4 should be identical (averaged)
        np.testing.assert_array_almost_equal(expanded[0, 0, :], expanded[0, 4, :])

        # The averaged value
        expected_avg = (payout[0, 0, :] + payout[0, 4, :]) / 2
        np.testing.assert_array_almost_equal(expanded[0, 0, :], expected_avg)

    def test_visits_roundtrip(self) -> None:
        """Visit counts roundtrip preserves equivalence structure."""
        visits = np.zeros((5, 5), dtype=np.int32)
        visits[1, 2] = 5
        visits[4, 3] = 10

        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        # Make equivalent positions have same values
        visits[0, :] = visits[4, :]

        # Manually reduce by selecting unique outcome positions
        p1_outcomes = get_unique_outcomes(p1_eff)  # [1, 2, 3, 4]
        p2_outcomes = get_unique_outcomes(p2_eff)  # [0, 1, 2, 3, 4]
        reduced_visits = visits[np.ix_(p1_outcomes, p2_outcomes)]

        expanded = expand_visits(reduced_visits, p1_eff, p2_eff)

        np.testing.assert_array_equal(expanded, visits)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_3_action_space(self) -> None:
        """Works with non-5 action spaces (for test fixtures)."""
        prior = np.array([0.3, 0.3, 0.4])
        effective = [2, 1, 2]  # Actions 0 and 2 equivalent

        reduced = reduce_prior(prior, effective)
        assert len(reduced) == 2
        assert reduced.sum() == pytest.approx(1.0)

        expanded = expand_prior(reduced, effective)
        assert expanded[0] == 0.0  # Not the canonical action
        assert expanded[1] == pytest.approx(0.3)
        assert expanded[2] == pytest.approx(0.7)  # 0.3 + 0.4

    def test_single_outcome(self) -> None:
        """Single outcome (all equivalent) edge case."""
        prior = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        effective = [4, 4, 4, 4, 4]

        reduced = reduce_prior(prior, effective)
        assert reduced.shape == (1,)
        assert reduced[0] == pytest.approx(1.0)

        expanded = expand_prior(reduced, effective)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(expanded, expected)

    def test_preserves_dtype(self) -> None:
        """Output dtypes match input dtypes."""
        prior_f32 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        effective = [0, 1, 2, 3, 4]

        reduced = reduce_prior(prior_f32, effective)
        assert reduced.dtype == np.float32

        expanded = expand_prior(reduced, effective)
        assert expanded.dtype == np.float32
