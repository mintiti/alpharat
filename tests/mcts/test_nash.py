"""Tests for Nash equilibrium computation."""

import numpy as np
import pytest

from alpharat.mcts.nash import (
    aggregate_equilibria,
    compute_nash_equilibrium,
    compute_nash_value,
    select_action_from_strategy,
)


def make_bimatrix(p1_payout: np.ndarray) -> np.ndarray:
    """Helper to create bimatrix from P1 payout (zero-sum: P2 = -P1)."""
    return np.stack([p1_payout, -p1_payout])


class TestNashEquilibrium:
    """Tests for Nash equilibrium computation."""

    def test_matching_pennies(self) -> None:
        """Test Nash equilibrium for matching pennies game.

        In matching pennies:
        - P1 wants to match (both heads or both tails)
        - P2 wants to mismatch
        - Nash equilibrium is (0.5, 0.5) for both players
        - Expected value is 0
        """
        # Payout matrix from P1's perspective (wants to match)
        #           P2: Heads  Tails
        # P1: Heads    1      -1
        #     Tails   -1       1
        p1_payout = np.array([[1.0, -1.0], [-1.0, 1.0]])
        payout = make_bimatrix(p1_payout)

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # Both should play uniformly random (0.5, 0.5)
        np.testing.assert_array_almost_equal(p1_strategy, [0.5, 0.5])
        np.testing.assert_array_almost_equal(p2_strategy, [0.5, 0.5])

        # Expected value should be 0
        value = compute_nash_value(p1_payout, p1_strategy, p2_strategy)
        assert abs(value) < 1e-6

    def test_rock_paper_scissors(self) -> None:
        """Test Nash equilibrium for rock-paper-scissors.

        Nash equilibrium should be uniform (1/3, 1/3, 1/3) for both players.
        Expected value is 0.
        """
        # Payout matrix from P1's perspective
        #           P2: Rock  Paper  Scissors
        # P1: Rock      0     -1       1
        #     Paper     1      0      -1
        #     Scissors -1      1       0
        p1_payout = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
        payout = make_bimatrix(p1_payout)

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # Both should play uniformly random (1/3, 1/3, 1/3)
        expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        np.testing.assert_array_almost_equal(p1_strategy, expected)
        np.testing.assert_array_almost_equal(p2_strategy, expected)

        # Expected value should be 0
        value = compute_nash_value(p1_payout, p1_strategy, p2_strategy)
        assert abs(value) < 1e-6

    def test_pure_strategy_nash(self) -> None:
        """Test game with pure strategy Nash equilibrium.

        Simple coordination game where (action 0, action 0) dominates.
        """
        # P1 strongly prefers action 0 regardless of P2's choice
        p1_payout = np.array([[10.0, 5.0], [1.0, 1.0]])
        payout = make_bimatrix(p1_payout)

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # P1 should strongly prefer action 0
        assert p1_strategy[0] > 0.9

        # Verify it's a valid probability distribution
        assert abs(p1_strategy.sum() - 1.0) < 1e-6
        assert abs(p2_strategy.sum() - 1.0) < 1e-6

    def test_no_equilibrium_fallback(self) -> None:
        """Test fallback to uniform strategy when no equilibrium is found.

        This is a degenerate case that shouldn't happen with valid zero-sum games,
        but we test the fallback behavior anyway.
        """
        # Create an invalid/degenerate payout matrix that might cause issues
        # Note: In practice, support_enumeration should always find equilibria for valid games
        # This test mainly ensures the fallback code path exists
        p1_payout = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        payout = make_bimatrix(p1_payout)

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # Should fall back to uniform distribution
        np.testing.assert_array_almost_equal(p1_strategy, [0.5, 0.5])
        np.testing.assert_array_almost_equal(p2_strategy, [0.5, 0.5])


class TestActionSelection:
    """Tests for action selection from strategies."""

    def test_deterministic_selection(self) -> None:
        """Test deterministic action selection (temperature=0)."""
        strategy = np.array([0.1, 0.7, 0.2])

        # Should always select action 1 (highest probability)
        for _ in range(10):
            action = select_action_from_strategy(strategy, temperature=0.0)
            assert action == 1

    def test_stochastic_selection(self) -> None:
        """Test stochastic action selection (temperature=1)."""
        strategy = np.array([0.5, 0.5])

        # Should sample both actions over many trials
        actions = [select_action_from_strategy(strategy, temperature=1.0) for _ in range(1000)]

        # Check that both actions were sampled
        assert 0 in actions
        assert 1 in actions

        # Check approximate distribution (should be close to 50/50)
        count_0 = sum(1 for a in actions if a == 0)
        count_1 = sum(1 for a in actions if a == 1)
        assert 400 < count_0 < 600  # Roughly 50% with some variance
        assert 400 < count_1 < 600

    def test_temperature_based_selection(self) -> None:
        """Test action selection with temperature != 0 and != 1."""
        strategy = np.array([0.1, 0.2, 0.7])

        # High temperature (2.0) should make selection more uniform
        actions_high_temp = [
            select_action_from_strategy(strategy, temperature=2.0) for _ in range(1000)
        ]

        # Low temperature (0.5) should make selection more peaked
        actions_low_temp = [
            select_action_from_strategy(strategy, temperature=0.5) for _ in range(1000)
        ]

        # Both should sample all actions
        for actions in [actions_high_temp, actions_low_temp]:
            assert 0 in actions
            assert 1 in actions
            assert 2 in actions

        # Low temp should select action 2 (highest prob) more often than high temp
        count_2_high = sum(1 for a in actions_high_temp if a == 2)
        count_2_low = sum(1 for a in actions_low_temp if a == 2)
        assert count_2_low > count_2_high

    def test_strategy_sums_to_one(self) -> None:
        """Test that strategies are valid probability distributions."""
        p1_payout = np.array([[1.0, -1.0], [-1.0, 1.0]])
        payout = make_bimatrix(p1_payout)
        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        assert abs(p1_strategy.sum() - 1.0) < 1e-6
        assert abs(p2_strategy.sum() - 1.0) < 1e-6
        assert np.all(p1_strategy >= 0)
        assert np.all(p2_strategy >= 0)


class TestNashWithEquivalence:
    """Tests for Nash equilibrium with action equivalence."""

    def test_blocked_actions_get_zero_probability(self) -> None:
        """Blocked actions should have probability 0 in Nash strategy."""
        # 5x5 matching pennies variant
        # Row 0 = Row 4 (both blocked for P1)
        # Col 1 = Col 4 (both blocked for P2)
        p1_payout = np.array(
            [
                [1.0, 2.0, -1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0, -1.0, 1.0],
                [-1.0, 0.0, 1.0, 2.0, 0.0],
                [2.0, -1.0, 0.0, 1.0, -1.0],
                [1.0, 2.0, -1.0, 0.0, 2.0],  # Same as row 0
            ]
        )
        payout = make_bimatrix(p1_payout)

        p1_effective = [4, 1, 2, 3, 4]  # Action 0 -> 4
        p2_effective = [0, 4, 2, 3, 4]  # Action 1 -> 4

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # Blocked actions must have 0 probability
        assert p1_strategy[0] == 0.0, "P1's blocked action 0 should have 0 probability"
        assert p2_strategy[1] == 0.0, "P2's blocked action 1 should have 0 probability"

        # Strategies should still be valid distributions
        assert abs(p1_strategy.sum() - 1.0) < 1e-6
        assert abs(p2_strategy.sum() - 1.0) < 1e-6
        assert np.all(p1_strategy >= 0)
        assert np.all(p2_strategy >= 0)

    def test_equivalence_preserves_nash_value(self) -> None:
        """Nash value should be same with or without equivalence reduction."""
        # Create a matrix where equivalent rows/cols are identical
        p1_payout = np.array(
            [
                [1.0, -1.0, 0.0, 0.5, -1.0],  # Row 0 = Row 4
                [0.0, 1.0, -1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -0.5, 0.0],
                [0.5, 0.0, -0.5, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.5, -1.0],  # Row 4 = Row 0
            ]
        )
        # Make col 1 = col 4
        p1_payout[:, 1] = p1_payout[:, 4]
        payout = make_bimatrix(p1_payout)

        p1_effective = [4, 1, 2, 3, 4]
        p2_effective = [0, 4, 2, 3, 4]

        # Compute with equivalence
        p1_eq, p2_eq = compute_nash_equilibrium(payout, p1_effective, p2_effective)
        value_eq = compute_nash_value(p1_payout, p1_eq, p2_eq)

        # Compute without equivalence
        p1_raw, p2_raw = compute_nash_equilibrium(payout)
        value_raw = compute_nash_value(p1_payout, p1_raw, p2_raw)

        # Values should be approximately equal (Nash value is unique in zero-sum games)
        assert abs(value_eq - value_raw) < 1e-6

    def test_all_actions_equivalent_gives_deterministic(self) -> None:
        """When all actions are equivalent, all probability goes to effective."""
        p1_payout = np.ones((5, 5))  # Trivial game, all payoffs equal
        payout = make_bimatrix(p1_payout)

        p1_effective = [4, 4, 4, 4, 4]  # All -> 4
        p2_effective = [4, 4, 4, 4, 4]  # All -> 4

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # All probability should be on action 4
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(p1_strategy, expected)
        np.testing.assert_array_almost_equal(p2_strategy, expected)

    def test_without_effective_maps_uses_full_matrix(self) -> None:
        """Without effective maps, computes on full matrix (backwards compatible)."""
        p1_payout = np.array([[1.0, -1.0], [-1.0, 1.0]])
        payout = make_bimatrix(p1_payout)

        # Without effective maps
        p1, p2 = compute_nash_equilibrium(payout)

        # Should still work (matching pennies)
        np.testing.assert_array_almost_equal(p1, [0.5, 0.5])
        np.testing.assert_array_almost_equal(p2, [0.5, 0.5])

    def test_equilibrium_property_with_equivalence(self) -> None:
        """Verify output is actually a Nash equilibrium: no player can improve by deviating.

        For a Nash equilibrium over the EFFECTIVE action space:
        - P1's strategy is a best response to P2's strategy (among effective actions)
        - P2's strategy is a best response to P1's strategy (among effective actions)

        Deviating to a blocked action = deviating to STAY, so we only check effective actions.
        """
        # Use rock-paper-scissors with one blocked action each
        # Effective 2x2 game embedded in 3x3
        # RPS payoffs: win=1, lose=-1, tie=0
        p1_payout = np.array(
            [
                [0.0, -1.0, 1.0],  # Rock: ties rock, loses to paper, beats scissors
                [1.0, 0.0, -1.0],  # Paper: beats rock, ties paper, loses to scissors
                [0.0, -1.0, 1.0],  # Blocked -> same as Rock (action 0)
            ]
        )
        payout = make_bimatrix(p1_payout)

        # Action 2 is blocked for P1 (maps to action 0)
        # All actions valid for P2
        p1_effective = [0, 1, 0]  # Action 2 -> 0
        p2_effective = [0, 1, 2]  # No blocking

        # Get the unique effective actions for each player
        p1_effective_actions = sorted(set(p1_effective))  # [0, 1]
        p2_effective_actions = sorted(set(p2_effective))  # [0, 1, 2]

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # Blocked action must have 0 probability
        assert p1_strategy[2] == 0.0

        # Compute expected value at equilibrium
        equilibrium_value = compute_nash_value(p1_payout, p1_strategy, p2_strategy)

        # Check P1 cannot improve by deviating to any EFFECTIVE action
        for i in p1_effective_actions:
            pure_p1 = np.zeros(3)
            pure_p1[i] = 1.0
            deviation_value = compute_nash_value(p1_payout, pure_p1, p2_strategy)
            assert deviation_value <= equilibrium_value + 1e-6, (
                f"P1 can improve by playing effective action {i}: "
                f"{deviation_value} > {equilibrium_value}"
            )

        # Check P2 cannot improve by deviating to any EFFECTIVE action
        for j in p2_effective_actions:
            pure_p2 = np.zeros(3)
            pure_p2[j] = 1.0
            deviation_value = compute_nash_value(p1_payout, p1_strategy, pure_p2)
            assert deviation_value >= equilibrium_value - 1e-6, (
                f"P2 can improve by playing effective action {j}: "
                f"{deviation_value} < {equilibrium_value}"
            )

    def test_equilibrium_only_on_effective_actions(self) -> None:
        """Verify equilibrium uses only effective actions, not blocked ones.

        Even if a blocked action would be "better", Nash should not use it
        because blocked actions are equivalent to STAY.
        """
        # Construct a game where action 0 looks attractive but is blocked
        # Row 0 has high payoffs, but it's equivalent to row 4
        p1_payout = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],  # Looks great for P1
                [0.0, 1.0, -1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0, 1.0, -1.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],  # Same as row 0 (equivalence)
            ]
        )
        payout = make_bimatrix(p1_payout)

        p1_effective = [4, 1, 2, 3, 4]  # Action 0 -> 4 (blocked)
        p2_effective = [0, 1, 2, 3, 4]  # No blocking for P2

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # P1's blocked action 0 must have 0 probability
        assert p1_strategy[0] == 0.0

        # The equilibrium value should reflect the ACTUAL game (where action 0 = action 4)
        # not the "fantasy" where P1 could independently play action 0
        equilibrium_value = compute_nash_value(p1_payout, p1_strategy, p2_strategy)

        # Since rows 0 and 4 are identical (equivalence invariant), and action 0 is blocked,
        # all P1's probability on row 0 goes to row 4 in the effective game.
        # The value should be 100.0 (P1 effectively plays row 4 which has payoff 100 everywhere)
        assert equilibrium_value == pytest.approx(100.0)

    def test_degenerate_zero_matrix_gives_uniform(self) -> None:
        """All-zero payout matrix should return uniform, not arbitrary first equilibrium.

        This tests the fix for P1 win rate bias: when the payout matrix is all zeros
        (common with few simulations), we return uniform instead of letting nashpy
        pick an arbitrary equilibrium that systematically favors P1.
        """
        payout = np.zeros((2, 5, 5))  # Both P1 and P2 payoff matrices are zeros
        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        expected = np.ones(5) / 5
        np.testing.assert_array_almost_equal(p1_strategy, expected)
        np.testing.assert_array_almost_equal(p2_strategy, expected)

    def test_degenerate_zero_matrix_with_equivalence(self) -> None:
        """Zero matrix with equivalence reduction should give uniform over effective actions.

        When action 0 is blocked (maps to 4), the uniform distribution should only
        cover the 4 effective actions [1,2,3,4], with action 0 getting probability 0.
        """
        payout = np.zeros((2, 5, 5))
        p1_effective = [4, 1, 2, 3, 4]  # Action 0 blocked (maps to STAY=4)
        p2_effective = [0, 1, 2, 3, 4]  # All actions effective for P2

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # P1: action 0 should be 0 (blocked), others uniform (1/4 each)
        assert p1_strategy[0] == 0.0
        np.testing.assert_array_almost_equal(p1_strategy[1:], 0.25)

        # P2: uniform over all 5 actions
        np.testing.assert_array_almost_equal(p2_strategy, 0.2)

    def test_multiple_equilibria_averaged(self) -> None:
        """When multiple Nash equilibria exist, return their centroid.

        This matrix has two pure equilibria at (0,0) and (1,1).
        Without averaging, nashpy picks one arbitrarily.
        With averaging, we get the centroid of these equilibria.
        """
        from alpharat.mcts.nash import _compute_nash_raw

        # Battle of the sexes (zero-sum variant)
        # P1 prefers (0,0), P2 prefers (1,1) but must coordinate
        # Has two pure equilibria: (0,0) and (1,1)
        p1_payout = np.array([[3.0, 0.0], [0.0, 2.0]])
        payout = make_bimatrix(p1_payout)

        p1_strategy, p2_strategy = _compute_nash_raw(payout)

        # With averaging, we expect the centroid of pure equilibria
        # Pure equilibrium 1: p1=[1,0], p2=[1,0] (both play action 0)
        # Pure equilibrium 2: p1=[0,1], p2=[0,1] (both play action 1)
        # Plus a mixed equilibrium: p1=[0.4, 0.6], p2=[0.6, 0.4]
        # Centroid of all three: approximately mixed

        # The key property: strategies sum to 1 and are valid probabilities
        assert abs(p1_strategy.sum() - 1.0) < 1e-6
        assert abs(p2_strategy.sum() - 1.0) < 1e-6
        assert np.all(p1_strategy >= 0)
        assert np.all(p2_strategy >= 0)

        # Nash value should be computable (strategies are valid)
        compute_nash_value(p1_payout, p1_strategy, p2_strategy)


class TestAggregateEquilibria:
    """Tests for aggregate_equilibria function — mathematical properties of centroid."""

    # --- Centroid Properties ---

    def test_centroid_is_arithmetic_mean(self) -> None:
        """Centroid should be the arithmetic mean of input vectors."""
        v1 = np.array([0.2, 0.3, 0.5])
        v2 = np.array([0.4, 0.4, 0.2])
        v3 = np.array([0.1, 0.6, 0.3])

        equilibria = [(v1, v1), (v2, v2), (v3, v3)]
        p1_result, p2_result = aggregate_equilibria(equilibria)

        expected = (v1 + v2 + v3) / 3
        np.testing.assert_array_almost_equal(p1_result, expected)
        np.testing.assert_array_almost_equal(p2_result, expected)

    def test_centroid_of_single_vector(self) -> None:
        """Centroid of a single vector is that vector."""
        v = np.array([0.1, 0.2, 0.7])
        u = np.array([0.5, 0.5])

        p1_result, p2_result = aggregate_equilibria([(v, u)])

        np.testing.assert_array_almost_equal(p1_result, v)
        np.testing.assert_array_almost_equal(p2_result, u)

    def test_centroid_preserves_probability_sum(self) -> None:
        """If all inputs sum to 1, output sums to 1."""
        equilibria = [
            (np.array([0.3, 0.7]), np.array([0.2, 0.8])),
            (np.array([0.6, 0.4]), np.array([0.5, 0.5])),
            (np.array([0.1, 0.9]), np.array([0.9, 0.1])),
        ]

        p1_result, p2_result = aggregate_equilibria(equilibria)

        assert abs(p1_result.sum() - 1.0) < 1e-10
        assert abs(p2_result.sum() - 1.0) < 1e-10

    def test_centroid_preserves_nonnegativity(self) -> None:
        """If all inputs are non-negative, output is non-negative."""
        equilibria = [
            (np.array([0.0, 0.5, 0.5]), np.array([1.0, 0.0])),
            (np.array([0.5, 0.0, 0.5]), np.array([0.0, 1.0])),
            (np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5])),
        ]

        p1_result, p2_result = aggregate_equilibria(equilibria)

        assert np.all(p1_result >= 0)
        assert np.all(p2_result >= 0)

    def test_centroid_in_convex_hull(self) -> None:
        """Centroid is a convex combination of inputs (equal weights)."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        equilibria = [(v1, v1), (v2, v2), (v3, v3)]
        p1_result, _ = aggregate_equilibria(equilibria)

        # Centroid of three corners of simplex is the center
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_array_almost_equal(p1_result, expected)

    def test_centroid_idempotent_on_identical(self) -> None:
        """Centroid of identical vectors is that vector."""
        v = np.array([0.25, 0.25, 0.5])
        u = np.array([0.4, 0.6])

        equilibria = [(v, u), (v, u), (v, u)]
        p1_result, p2_result = aggregate_equilibria(equilibria)

        np.testing.assert_array_almost_equal(p1_result, v)
        np.testing.assert_array_almost_equal(p2_result, u)

    # --- Policy Aggregation Properties ---

    def test_aggregate_independent_per_player(self) -> None:
        """P1's result depends only on P1 strategies, not P2's."""
        p1_a = np.array([0.3, 0.7])
        p1_b = np.array([0.5, 0.5])
        p2_x = np.array([0.1, 0.9])
        p2_y = np.array([0.8, 0.2])

        # Different P2 strategies should not affect P1 result
        result1, _ = aggregate_equilibria([(p1_a, p2_x), (p1_b, p2_x)])
        result2, _ = aggregate_equilibria([(p1_a, p2_y), (p1_b, p2_y)])

        np.testing.assert_array_almost_equal(result1, result2)

    def test_aggregate_returns_valid_distributions(self) -> None:
        """Both outputs are valid probability distributions."""
        equilibria = [
            (np.array([0.2, 0.3, 0.5]), np.array([0.6, 0.4])),
            (np.array([0.4, 0.4, 0.2]), np.array([0.3, 0.7])),
        ]

        p1_result, p2_result = aggregate_equilibria(equilibria)

        # Sum to 1
        assert abs(p1_result.sum() - 1.0) < 1e-10
        assert abs(p2_result.sum() - 1.0) < 1e-10
        # Non-negative
        assert np.all(p1_result >= 0)
        assert np.all(p2_result >= 0)

    def test_aggregate_handles_different_action_counts(self) -> None:
        """Works when P1 and P2 have different numbers of actions."""
        p1_strat = np.array([0.5, 0.3, 0.2])  # 3 actions
        p2_strat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 5 actions

        equilibria = [(p1_strat, p2_strat)]
        p1_result, p2_result = aggregate_equilibria(equilibria)

        assert len(p1_result) == 3
        assert len(p2_result) == 5

    # --- Edge Cases ---

    def test_aggregate_empty_raises(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot aggregate empty"):
            aggregate_equilibria([])

    def test_aggregate_pure_strategies(self) -> None:
        """Pure strategies average to mixed correctly."""
        # Two pure strategies: (1,0) and (0,1)
        eq1 = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        eq2 = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        p1_result, p2_result = aggregate_equilibria([eq1, eq2])

        # Should average to (0.5, 0.5)
        np.testing.assert_array_almost_equal(p1_result, [0.5, 0.5])
        np.testing.assert_array_almost_equal(p2_result, [0.5, 0.5])
