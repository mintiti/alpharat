"""Tests for Nash equilibrium computation."""

import numpy as np

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
