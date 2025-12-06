"""Tests for Nash equilibrium computation."""

import numpy as np
import pytest

from alpharat.mcts.nash import (
    compute_nash_equilibrium,
    compute_nash_value,
    select_action_from_strategy,
)


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
        payout = np.array([[1.0, -1.0], [-1.0, 1.0]])

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # Both should play uniformly random (0.5, 0.5)
        np.testing.assert_array_almost_equal(p1_strategy, [0.5, 0.5])
        np.testing.assert_array_almost_equal(p2_strategy, [0.5, 0.5])

        # Expected value should be 0
        value = compute_nash_value(payout, p1_strategy, p2_strategy)
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
        payout = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout)

        # Both should play uniformly random (1/3, 1/3, 1/3)
        expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        np.testing.assert_array_almost_equal(p1_strategy, expected)
        np.testing.assert_array_almost_equal(p2_strategy, expected)

        # Expected value should be 0
        value = compute_nash_value(payout, p1_strategy, p2_strategy)
        assert abs(value) < 1e-6

    def test_pure_strategy_nash(self) -> None:
        """Test game with pure strategy Nash equilibrium.

        Simple coordination game where (action 0, action 0) dominates.
        """
        # P1 strongly prefers action 0 regardless of P2's choice
        payout = np.array([[10.0, 5.0], [1.0, 1.0]])

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
        payout = np.array([[np.nan, np.nan], [np.nan, np.nan]])

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
        payout = np.array([[1.0, -1.0], [-1.0, 1.0]])
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
        payout = np.array(
            [
                [1.0, 2.0, -1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0, -1.0, 1.0],
                [-1.0, 0.0, 1.0, 2.0, 0.0],
                [2.0, -1.0, 0.0, 1.0, -1.0],
                [1.0, 2.0, -1.0, 0.0, 2.0],  # Same as row 0
            ]
        )

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
        payout = np.array(
            [
                [1.0, -1.0, 0.0, 0.5, -1.0],  # Row 0 = Row 4
                [0.0, 1.0, -1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -0.5, 0.0],
                [0.5, 0.0, -0.5, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.5, -1.0],  # Row 4 = Row 0
            ]
        )
        # Make col 1 = col 4
        payout[:, 1] = payout[:, 4]

        p1_effective = [4, 1, 2, 3, 4]
        p2_effective = [0, 4, 2, 3, 4]

        # Compute with equivalence
        p1_eq, p2_eq = compute_nash_equilibrium(payout, p1_effective, p2_effective)
        value_eq = compute_nash_value(payout, p1_eq, p2_eq)

        # Compute without equivalence
        p1_raw, p2_raw = compute_nash_equilibrium(payout)
        value_raw = compute_nash_value(payout, p1_raw, p2_raw)

        # Values should be approximately equal (Nash value is unique in zero-sum games)
        assert abs(value_eq - value_raw) < 1e-6

    def test_all_actions_equivalent_gives_deterministic(self) -> None:
        """When all actions are equivalent, all probability goes to effective."""
        payout = np.ones((5, 5))  # Trivial game, all payoffs equal

        p1_effective = [4, 4, 4, 4, 4]  # All -> 4
        p2_effective = [4, 4, 4, 4, 4]  # All -> 4

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # All probability should be on action 4
        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(p1_strategy, expected)
        np.testing.assert_array_almost_equal(p2_strategy, expected)

    def test_without_effective_maps_uses_full_matrix(self) -> None:
        """Without effective maps, computes on full matrix (backwards compatible)."""
        payout = np.array([[1.0, -1.0], [-1.0, 1.0]])

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
        payout = np.array(
            [
                [0.0, -1.0, 1.0],  # Rock: ties rock, loses to paper, beats scissors
                [1.0, 0.0, -1.0],  # Paper: beats rock, ties paper, loses to scissors
                [0.0, -1.0, 1.0],  # Blocked -> same as Rock (action 0)
            ]
        )

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
        equilibrium_value = compute_nash_value(payout, p1_strategy, p2_strategy)

        # Check P1 cannot improve by deviating to any EFFECTIVE action
        for i in p1_effective_actions:
            pure_p1 = np.zeros(3)
            pure_p1[i] = 1.0
            deviation_value = compute_nash_value(payout, pure_p1, p2_strategy)
            assert deviation_value <= equilibrium_value + 1e-6, (
                f"P1 can improve by playing effective action {i}: "
                f"{deviation_value} > {equilibrium_value}"
            )

        # Check P2 cannot improve by deviating to any EFFECTIVE action
        for j in p2_effective_actions:
            pure_p2 = np.zeros(3)
            pure_p2[j] = 1.0
            deviation_value = compute_nash_value(payout, p1_strategy, pure_p2)
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
        payout = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],  # Looks great for P1
                [0.0, 1.0, -1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0, 1.0, -1.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],  # Same as row 0 (equivalence)
            ]
        )

        p1_effective = [4, 1, 2, 3, 4]  # Action 0 -> 4 (blocked)
        p2_effective = [0, 1, 2, 3, 4]  # No blocking for P2

        p1_strategy, p2_strategy = compute_nash_equilibrium(payout, p1_effective, p2_effective)

        # P1's blocked action 0 must have 0 probability
        assert p1_strategy[0] == 0.0

        # The equilibrium value should reflect the ACTUAL game (where action 0 = action 4)
        # not the "fantasy" where P1 could independently play action 0
        equilibrium_value = compute_nash_value(payout, p1_strategy, p2_strategy)

        # Since rows 0 and 4 are identical (equivalence invariant), and action 0 is blocked,
        # all P1's probability on row 0 goes to row 4 in the effective game.
        # The value should be 100.0 (P1 effectively plays row 4 which has payoff 100 everywhere)
        assert equilibrium_value == pytest.approx(100.0)
