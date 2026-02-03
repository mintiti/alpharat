"""Tests for policy strategy classes."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from alpharat.mcts.node import MCTSNode
from alpharat.mcts.policy_strategy import (
    NashPolicyConfig,
    NashPolicyStrategy,
    VisitPolicyConfig,
    VisitPolicyStrategy,
    apply_katago_temperature,
)


def make_node(
    visits: np.ndarray | None = None,
    payout_p1: np.ndarray | None = None,
    payout_p2: np.ndarray | None = None,
    prior_p1: np.ndarray | None = None,
    prior_p2: np.ndarray | None = None,
) -> MCTSNode:
    """Create a minimal MCTSNode for testing with optional pre-set statistics.

    The node is created with all 5 actions effective (no blocked actions),
    so visits and payout matrices should be [5, 5] to match the reduced space.

    Args:
        visits: Optional visit matrix [5, 5] to set on the node.
        payout_p1: Optional P1 payout matrix [5, 5] to set.
        payout_p2: Optional P2 payout matrix [5, 5] to set.
        prior_p1: Optional P1 prior [5] to use.
        prior_p2: Optional P2 prior [5] to use.

    Returns:
        MCTSNode with the specified statistics.
    """
    if prior_p1 is None:
        prior_p1 = np.ones(5) / 5
    if prior_p2 is None:
        prior_p2 = np.ones(5) / 5

    node = MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1.astype(np.float32),
        prior_policy_p2=prior_p2.astype(np.float32),
        nn_payout_prediction=np.zeros((2, 5, 5)),
    )

    # Override internal state if provided
    # Note: visits and payout should be [5, 5] for a node with all actions effective
    if visits is not None:
        node._visits = visits.astype(np.float64)
        node._total_visits = int(visits.sum())
    if payout_p1 is not None:
        node._payout_p1 = payout_p1.astype(np.float64)
    if payout_p2 is not None:
        node._payout_p2 = payout_p2.astype(np.float64)

    return node


class TestNashPolicyStrategy:
    """Tests for NashPolicyStrategy."""

    def test_returns_valid_policies(self) -> None:
        """Should return valid probability distributions."""
        # [5, 5] visit matrix to match node's reduced space (all actions effective)
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [2, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = NashPolicyStrategy()
        p1, p2 = strategy.derive_policies(node)

        # Policies should sum to 1
        assert p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert p2.sum() == pytest.approx(1.0, abs=1e-6)

        # Policies should be non-negative
        assert np.all(p1 >= 0)
        assert np.all(p2 >= 0)

    def test_respects_reduced_space(self) -> None:
        """Should return policies in reduced space [n1], [n2]."""
        # [5, 5] visit matrix to match node's reduced space
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [2, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = NashPolicyStrategy()
        p1, p2 = strategy.derive_policies(node)

        # Reduced space is [5] for this node (all actions effective)
        assert p1.shape == (node.n1,)
        assert p2.shape == (node.n2,)


class TestVisitPolicyStrategy:
    """Tests for VisitPolicyStrategy."""

    def test_proportional_to_visits_at_temp_1(self) -> None:
        """At temperature 1.0, should be proportional to marginal visits."""
        # [5, 5] visit matrix to match node's reduced space
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        p1, p2 = strategy.derive_policies(node)

        # Expected marginals
        marginal_p1 = visits.sum(axis=1)  # [18, 6, 2, 1, 0]
        marginal_p2 = visits.sum(axis=0)  # [14, 9, 3, 1, 0]

        expected_p1 = marginal_p1 / marginal_p1.sum()
        expected_p2 = marginal_p2 / marginal_p2.sum()

        np.testing.assert_array_almost_equal(p1, expected_p1)
        np.testing.assert_array_almost_equal(p2, expected_p2)

    def test_sharper_at_low_temp(self) -> None:
        """Temperature < 1 should produce sharper distribution."""
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

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

        hot_p1, _ = strategy_hot.derive_policies(node)
        cold_p1, _ = strategy_cold.derive_policies(node)

        # Cold should concentrate more mass on top action
        assert cold_p1[0] > hot_p1[0]

    def test_argmax_at_near_zero_temp(self) -> None:
        """Near-zero temperature should return argmax."""
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = VisitPolicyStrategy(
            temperature=1e-6,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        p1, p2 = strategy.derive_policies(node)

        # Action 0 has most marginal visits for P1
        expected_p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(p1, expected_p1)

    def test_uniform_on_zero_visits(self) -> None:
        """Zero visits should return uniform distribution."""
        visits = np.zeros((5, 5), dtype=np.float64)
        node = make_node(visits=visits)

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        p1, p2 = strategy.derive_policies(node)

        expected = np.ones(5, dtype=np.float32) / 5
        np.testing.assert_array_almost_equal(p1, expected)
        np.testing.assert_array_almost_equal(p2, expected)

    def test_output_is_float32(self) -> None:
        """Output should be float32."""
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        p1, p2 = strategy.derive_policies(node)

        assert p1.dtype == np.float32
        assert p2.dtype == np.float32

    def test_sums_to_one(self) -> None:
        """Policy should sum to 1.0."""
        visits = np.array(
            [
                [37, 22, 5, 2, 1],
                [15, 8, 3, 1, 0],
                [3, 1, 1, 0, 0],
                [2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        node = make_node(visits=visits)

        strategy = VisitPolicyStrategy(
            temperature=0.8,
            temperature_only_below_prob=0.5,
            prune_threshold=1.0,
            subtract_visits=2.0,
        )
        p1, p2 = strategy.derive_policies(node)

        assert p1.sum() == pytest.approx(1.0, rel=1e-5)
        assert p2.sum() == pytest.approx(1.0, rel=1e-5)

    def test_skips_nash_computation(self) -> None:
        """VisitPolicyStrategy should not use payout matrix at all.

        This verifies the key optimization: visit-based policy skips Nash.
        """
        visits = np.array(
            [
                [10, 5, 2, 1, 0],
                [3, 2, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        # Set nonsensical payout that would produce different Nash
        payout_p1 = np.ones((5, 5)) * 1000
        payout_p2 = np.ones((5, 5)) * -1000

        node = make_node(visits=visits, payout_p1=payout_p1, payout_p2=payout_p2)

        strategy = VisitPolicyStrategy(
            temperature=1.0,
            temperature_only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        p1, p2 = strategy.derive_policies(node)

        # Should still be proportional to visits, not affected by payout
        marginal_p1 = visits.sum(axis=1)
        expected_p1 = marginal_p1 / marginal_p1.sum()

        np.testing.assert_array_almost_equal(p1, expected_p1)


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

    def test_prune_threshold(self) -> None:
        """Prune threshold should zero out low-visit actions."""
        visits = np.array([10.0, 5.0, 0.5, 0.3, 0.0], dtype=np.float32)

        result = apply_katago_temperature(
            visits,
            temperature=1.0,
            only_below_prob=1.0,
            prune_threshold=1.0,  # Prune visits < 1
            subtract_visits=0.0,
        )

        # Actions 2, 3, 4 should be pruned (0.5, 0.3, 0.0 < 1)
        assert result[2] == pytest.approx(0.0)
        assert result[3] == pytest.approx(0.0)
        assert result[4] == pytest.approx(0.0)
        # Sum should still be 1
        assert result.sum() == pytest.approx(1.0)

    def test_subtract_visits(self) -> None:
        """Subtract visits should reduce all counts."""
        visits = np.array([100.0, 50.0, 10.0, 5.0, 0.0], dtype=np.float32)

        result_no_sub = apply_katago_temperature(
            visits,
            temperature=1.0,
            only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=0.0,
        )
        result_sub = apply_katago_temperature(
            visits,
            temperature=1.0,
            only_below_prob=1.0,
            prune_threshold=0.0,
            subtract_visits=5.0,
        )

        # With subtraction, low-visit actions get even less probability
        # because subtracting a constant relatively hurts low counts more
        assert result_sub[0] > result_no_sub[0]  # Top action gains
        assert result_sub[3] < result_no_sub[3]  # Low action loses


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

    def test_build_passes_search_params(self) -> None:
        """build() should pass force_k and c_puct to strategy."""
        config = NashPolicyConfig()
        strategy = config.build(force_k=3.0, c_puct=2.0)
        assert isinstance(strategy, NashPolicyStrategy)
        assert strategy._force_k == 3.0
        assert strategy._c_puct == 2.0

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
        """build() should pass config values and search params to strategy."""
        config = VisitPolicyConfig(
            temperature=0.5,
            temperature_only_below_prob=0.2,
            prune_threshold=1.5,
            subtract_visits=3.0,
        )
        strategy = config.build(force_k=2.5, c_puct=1.8)
        assert isinstance(strategy, VisitPolicyStrategy)

        assert strategy.temperature == 0.5
        assert strategy.temperature_only_below_prob == 0.2
        assert strategy.prune_threshold == 1.5
        assert strategy.subtract_visits == 3.0
        assert strategy._force_k == 2.5
        assert strategy._c_puct == 1.8

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
