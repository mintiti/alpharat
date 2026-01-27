"""Tests for policy strategy classes."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from alpharat.mcts.policy_strategy import (
    NashPolicyConfig,
    NashPolicyStrategy,
    VisitPolicyConfig,
    VisitPolicyStrategy,
    apply_katago_temperature,
)


class TestNashPolicyStrategy:
    """Tests for NashPolicyStrategy."""

    def test_returns_nash_unchanged(self) -> None:
        """Should return Nash policy unchanged."""
        nash = np.array([0.6, 0.2, 0.1, 0.1, 0.0], dtype=np.float32)
        visits = np.array([100, 30, 20, 10, 0], dtype=np.float32)

        strategy = NashPolicyStrategy()
        result = strategy.derive_policy(visits, nash)

        np.testing.assert_array_almost_equal(result, nash)

    def test_returns_copy_not_reference(self) -> None:
        """Should return a copy, not the original array."""
        nash = np.array([0.5, 0.3, 0.2, 0.0, 0.0], dtype=np.float32)
        visits = np.zeros(5, dtype=np.float32)

        strategy = NashPolicyStrategy()
        result = strategy.derive_policy(visits, nash)

        # Modify result, should not affect original
        result[0] = 1.0
        assert nash[0] == pytest.approx(0.5)


class TestVisitPolicyStrategy:
    """Tests for VisitPolicyStrategy."""

    def test_proportional_to_visits_at_temp_1(self) -> None:
        """At temperature 1.0, should be proportional to visits."""
        visits = np.array([10.0, 5.0, 3.0, 2.0, 0.0], dtype=np.float32)
        nash = np.ones(5) / 5  # Unused

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        result = strategy.derive_policy(visits, nash)

        expected = visits / visits.sum()
        np.testing.assert_array_almost_equal(result, expected)

    def test_sharper_at_low_temp(self) -> None:
        """Temperature < 1 should produce sharper distribution."""
        visits = np.array([10.0, 5.0, 3.0, 2.0, 0.0], dtype=np.float32)
        nash = np.ones(5) / 5

        strategy_hot = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        strategy_cold = VisitPolicyStrategy(
            temperature=0.5,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )

        hot_result = strategy_hot.derive_policy(visits, nash)
        cold_result = strategy_cold.derive_policy(visits, nash)

        # Cold should concentrate more mass on top action
        assert cold_result[0] > hot_result[0]

    def test_argmax_at_near_zero_temp(self) -> None:
        """Near-zero temperature should return argmax."""
        visits = np.array([10.0, 5.0, 3.0, 2.0, 0.0], dtype=np.float32)
        nash = np.ones(5) / 5

        strategy = VisitPolicyStrategy(
            temperature=1e-6,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        result = strategy.derive_policy(visits, nash)

        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_uniform_on_zero_visits(self) -> None:
        """Zero visits should return uniform distribution."""
        visits = np.zeros(5, dtype=np.float32)
        nash = np.ones(5) / 5

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        result = strategy.derive_policy(visits, nash)

        expected = np.ones(5, dtype=np.float32) / 5
        np.testing.assert_array_almost_equal(result, expected)

    def test_prune_threshold(self) -> None:
        """Prune threshold should zero out low-visit actions."""
        visits = np.array([10.0, 5.0, 0.5, 0.3, 0.0], dtype=np.float32)
        nash = np.ones(5) / 5

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=1.0,  # Prune visits < 1
            subtract_visits=0.0,
        )
        result = strategy.derive_policy(visits, nash)

        # Actions 2, 3, 4 should be pruned (0.5, 0.3, 0.0 < 1)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        # Sum should still be 1
        assert result.sum() == pytest.approx(1.0)

    def test_subtract_visits(self) -> None:
        """Subtract visits should reduce all counts."""
        visits = np.array([100.0, 50.0, 10.0, 5.0, 0.0], dtype=np.float32)
        nash = np.ones(5) / 5

        strategy_no_sub = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        strategy_sub = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=5.0,
        )

        result_no_sub = strategy_no_sub.derive_policy(visits, nash)
        result_sub = strategy_sub.derive_policy(visits, nash)

        # With subtraction, low-visit actions get even less probability
        # because subtracting a constant relatively hurts low counts more
        assert result_sub[0] > result_no_sub[0]  # Top action gains
        assert result_sub[3] < result_no_sub[3]  # Low action loses

    def test_output_is_float32(self) -> None:
        """Output should be float32."""
        visits = np.array([10.0, 5.0, 3.0, 2.0, 0.0], dtype=np.float64)
        nash = np.ones(5) / 5

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        result = strategy.derive_policy(visits, nash)

        assert result.dtype == np.float32

    def test_sums_to_one(self) -> None:
        """Policy should sum to 1.0."""
        visits = np.array([37.0, 22.0, 15.0, 8.0, 3.0], dtype=np.float32)
        nash = np.ones(5) / 5

        strategy = VisitPolicyStrategy(
            temperature=0.8,
            temperature_only_below_prob=0.5,
            prune_threshold=1.0,
            subtract_visits=2.0,
        )
        result = strategy.derive_policy(visits, nash)

        assert result.sum() == pytest.approx(1.0, rel=1e-5)


class TestApplyKatagoTemperature:
    """Tests for apply_katago_temperature function."""

    def test_only_below_prob_preserves_top_moves(self) -> None:
        """only_below_prob should preserve top move probabilities."""
        visits = np.array([80.0, 10.0, 5.0, 3.0, 2.0], dtype=np.float32)

        # Full temperature: affects all
        result_full = apply_katago_temperature(
            visits,
            temperature=2.0,
            only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )

        # Partial temperature: only affects low moves
        result_partial = apply_katago_temperature(
            visits,
            temperature=2.0,
            only_below_prob=0.1,  # Only affect moves with <10% prob
            prune_threshold=0.0,
            subtract_visits=0.0,
        )

        # Top move should retain more probability with partial temp
        assert result_partial[0] > result_full[0]

    def test_handles_single_nonzero_visit(self) -> None:
        """Should handle case where only one action has visits."""
        visits = np.array([0.0, 0.0, 100.0, 0.0, 0.0], dtype=np.float32)

        result = apply_katago_temperature(
            visits,
            temperature=1.0,
            only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )

        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestNashPolicyConfig:
    """Tests for NashPolicyConfig."""

    def test_default_strategy_is_nash(self) -> None:
        """Default strategy field should be 'nash'."""
        config = NashPolicyConfig()
        assert config.strategy == "nash"

    def test_build_returns_nash_strategy(self) -> None:
        """build() should return NashPolicyStrategy instance."""
        config = NashPolicyConfig()
        strategy = config.build()
        assert isinstance(strategy, NashPolicyStrategy)

    def test_rejects_extra_fields(self) -> None:
        """Should reject extra fields (StrictBaseModel)."""
        with pytest.raises(ValidationError):
            NashPolicyConfig(extra_field="value")  # type: ignore[call-arg]


class TestVisitPolicyConfig:
    """Tests for VisitPolicyConfig."""

    def test_default_strategy_is_visits(self) -> None:
        """Default strategy field should be 'visits'."""
        config = VisitPolicyConfig()
        assert config.strategy == "visits"

    def test_default_values(self) -> None:
        """Should have sensible defaults."""
        config = VisitPolicyConfig()
        assert config.temperature == 1.0
        assert config.temperature_only_below_prob == 1.0
        assert config.prune_threshold == 0.0
        assert config.subtract_visits == 0.0

    def test_build_returns_visit_strategy(self) -> None:
        """build() should return VisitPolicyStrategy instance."""
        config = VisitPolicyConfig()
        strategy = config.build()
        assert isinstance(strategy, VisitPolicyStrategy)

    def test_build_passes_params(self) -> None:
        """build() should pass config values to strategy."""
        config = VisitPolicyConfig(
            temperature=0.5,
            temperature_only_below_prob=0.2,
            prune_threshold=1.5,
            subtract_visits=3.0,
        )
        strategy = config.build()
        assert isinstance(strategy, VisitPolicyStrategy)

        assert strategy.temperature == 0.5
        assert strategy.temperature_only_below_prob == 0.2
        assert strategy.prune_threshold == 1.5
        assert strategy.subtract_visits == 3.0

    def test_validates_temperature_positive(self) -> None:
        """Temperature must be > 0."""
        with pytest.raises(ValidationError):
            VisitPolicyConfig(temperature=0.0)

        with pytest.raises(ValidationError):
            VisitPolicyConfig(temperature=-0.5)

    def test_validates_temperature_only_below_prob(self) -> None:
        """temperature_only_below_prob must be in (0, 1]."""
        with pytest.raises(ValidationError):
            VisitPolicyConfig(temperature_only_below_prob=0.0)

        with pytest.raises(ValidationError):
            VisitPolicyConfig(temperature_only_below_prob=1.5)

    def test_validates_prune_threshold_non_negative(self) -> None:
        """prune_threshold must be >= 0."""
        with pytest.raises(ValidationError):
            VisitPolicyConfig(prune_threshold=-1.0)

        # Zero is valid
        config = VisitPolicyConfig(prune_threshold=0.0)
        assert config.prune_threshold == 0.0

    def test_validates_subtract_visits_non_negative(self) -> None:
        """subtract_visits must be >= 0."""
        with pytest.raises(ValidationError):
            VisitPolicyConfig(subtract_visits=-1.0)

        # Zero is valid
        config = VisitPolicyConfig(subtract_visits=0.0)
        assert config.subtract_visits == 0.0

    def test_rejects_extra_fields(self) -> None:
        """Should reject extra fields (StrictBaseModel)."""
        with pytest.raises(ValidationError):
            VisitPolicyConfig(extra_field="value")  # type: ignore[call-arg]
