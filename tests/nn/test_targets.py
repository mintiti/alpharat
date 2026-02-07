"""Tests for training target extraction."""

from __future__ import annotations

import numpy as np
import pytest

from alpharat.data.types import CheeseOutcome, GameData, PositionData
from alpharat.nn.targets import build_targets
from alpharat.nn.types import TargetBundle


def _make_game_data(
    *,
    final_p1_score: float = 3.0,
    final_p2_score: float = 2.0,
) -> GameData:
    """Create GameData with defaults for testing."""
    maze = np.ones((5, 5, 4), dtype=np.int8)
    initial_cheese = np.zeros((5, 5), dtype=bool)
    cheese_outcomes = np.full((5, 5), CheeseOutcome.UNCOLLECTED, dtype=np.int8)

    return GameData(
        maze=maze,
        initial_cheese=initial_cheese,
        max_turns=100,
        width=5,
        height=5,
        positions=[],
        result=1,
        final_p1_score=final_p1_score,
        final_p2_score=final_p2_score,
        cheese_outcomes=cheese_outcomes,
    )


def _make_position_data(
    *,
    policy_p1: np.ndarray | None = None,
    policy_p2: np.ndarray | None = None,
    p1_score: float = 0.0,
    p2_score: float = 0.0,
    turn: int = 0,
    action_p1: int = 0,
    action_p2: int = 0,
) -> PositionData:
    """Create PositionData with defaults for testing."""
    if policy_p1 is None:
        policy_p1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float32)
    if policy_p2 is None:
        policy_p2 = np.array([0.3, 0.3, 0.2, 0.1, 0.1], dtype=np.float32)

    return PositionData(
        p1_pos=(1, 1),
        p2_pos=(3, 3),
        p1_score=p1_score,
        p2_score=p2_score,
        p1_mud=0,
        p2_mud=0,
        cheese_positions=[(2, 2)],
        turn=turn,
        value_p1=0.0,
        value_p2=0.0,
        visit_counts_p1=np.zeros(5, dtype=np.int32),
        visit_counts_p2=np.zeros(5, dtype=np.int32),
        prior_p1=np.ones(5, dtype=np.float32) / 5,
        prior_p2=np.ones(5, dtype=np.float32) / 5,
        policy_p1=policy_p1,
        policy_p2=policy_p2,
        action_p1=action_p1,
        action_p2=action_p2,
    )


class TestBuildTargets:
    """Tests for build_targets()."""

    def test_policy_targets_from_position(self) -> None:
        """Policy targets should come from visit-proportional policy in position."""
        policy_p1 = np.array([0.6, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        policy_p2 = np.array([0.2, 0.4, 0.2, 0.1, 0.1], dtype=np.float32)

        game = _make_game_data()
        position = _make_position_data(policy_p1=policy_p1, policy_p2=policy_p2)

        result = build_targets(game, position)

        np.testing.assert_array_almost_equal(result.policy_p1, policy_p1)
        np.testing.assert_array_almost_equal(result.policy_p2, policy_p2)

    def test_policy_targets_are_float32(self) -> None:
        """Policy targets should be float32."""
        # Use float64 input to verify conversion
        policy_p1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
        policy_p2 = np.array([0.3, 0.3, 0.2, 0.1, 0.1], dtype=np.float64)

        game = _make_game_data()
        position = _make_position_data(policy_p1=policy_p1, policy_p2=policy_p2)

        result = build_targets(game, position)

        assert result.policy_p1.dtype == np.float32
        assert result.policy_p2.dtype == np.float32

    def test_values_are_remaining_scores(self) -> None:
        """Values should be remaining scores for each player."""
        # Final: p1=5, p2=2, Current: p1=1, p2=0
        # p1_value = 5-1 = 4, p2_value = 2-0 = 2
        game = _make_game_data(final_p1_score=5.0, final_p2_score=2.0)
        position = _make_position_data(p1_score=1.0, p2_score=0.0)

        result = build_targets(game, position)

        assert result.p1_value == pytest.approx(4.0)
        assert result.p2_value == pytest.approx(2.0)

    def test_values_at_game_start(self) -> None:
        """At game start (score 0-0), values equal final scores."""
        game = _make_game_data(final_p1_score=5.0, final_p2_score=2.0)
        position = _make_position_data(p1_score=0.0, p2_score=0.0)

        result = build_targets(game, position)

        assert result.p1_value == pytest.approx(5.0)
        assert result.p2_value == pytest.approx(2.0)

    def test_values_at_game_end(self) -> None:
        """At game end (current = final), values are zero."""
        game = _make_game_data(final_p1_score=5.0, final_p2_score=2.0)
        position = _make_position_data(p1_score=5.0, p2_score=2.0)

        result = build_targets(game, position)

        assert result.p1_value == pytest.approx(0.0)
        assert result.p2_value == pytest.approx(0.0)

    def test_values_p1_ahead_will_extend_lead(self) -> None:
        """P1 ahead and will extend lead."""
        # Current: p1=2, p2=1, Final: p1=5, p2=2
        # p1_value = 5-2 = 3, p2_value = 2-1 = 1
        game = _make_game_data(final_p1_score=5.0, final_p2_score=2.0)
        position = _make_position_data(p1_score=2.0, p2_score=1.0)

        result = build_targets(game, position)

        assert result.p1_value > result.p2_value  # P1 will gain more
        assert result.p1_value == pytest.approx(3.0)
        assert result.p2_value == pytest.approx(1.0)

    def test_values_p1_ahead_will_lose_lead(self) -> None:
        """P1 ahead but will lose lead."""
        # Current: p1=3, p2=1, Final: p1=3, p2=4
        # p1_value = 3-3 = 0, p2_value = 4-1 = 3
        game = _make_game_data(final_p1_score=3.0, final_p2_score=4.0)
        position = _make_position_data(p1_score=3.0, p2_score=1.0)

        result = build_targets(game, position)

        assert result.p2_value > result.p1_value  # P2 will gain more
        assert result.p1_value == pytest.approx(0.0)
        assert result.p2_value == pytest.approx(3.0)

    def test_values_p2_ahead_will_extend_lead(self) -> None:
        """P2 ahead and will extend lead."""
        # Current: p1=1, p2=2, Final: p1=2, p2=5
        # p1_value = 2-1 = 1, p2_value = 5-2 = 3
        game = _make_game_data(final_p1_score=2.0, final_p2_score=5.0)
        position = _make_position_data(p1_score=1.0, p2_score=2.0)

        result = build_targets(game, position)

        assert result.p2_value > result.p1_value  # P2 will gain more
        assert result.p1_value == pytest.approx(1.0)
        assert result.p2_value == pytest.approx(3.0)

    def test_values_draw_from_tied_position(self) -> None:
        """Draw game from tied position: equal remaining values."""
        # Current: p1=2, p2=2, Final: p1=4, p2=4
        # p1_value = 4-2 = 2, p2_value = 4-2 = 2
        game = _make_game_data(final_p1_score=4.0, final_p2_score=4.0)
        position = _make_position_data(p1_score=2.0, p2_score=2.0)

        result = build_targets(game, position)

        assert result.p1_value == pytest.approx(2.0)
        assert result.p2_value == pytest.approx(2.0)

    def test_values_half_point_scores(self) -> None:
        """Should handle 0.5 point scores (simultaneous cheese collection)."""
        # Current: p1=1.5, p2=1.0, Final: p1=3.5, p2=2.5
        # p1_value = 3.5-1.5 = 2.0, p2_value = 2.5-1.0 = 1.5
        game = _make_game_data(final_p1_score=3.5, final_p2_score=2.5)
        position = _make_position_data(p1_score=1.5, p2_score=1.0)

        result = build_targets(game, position)

        assert result.p1_value == pytest.approx(2.0)
        assert result.p2_value == pytest.approx(1.5)

    def test_values_vary_by_position_in_same_game(self) -> None:
        """Different positions in same game should have different values."""
        game = _make_game_data(final_p1_score=4.0, final_p2_score=2.0)

        # Early position: p1=0, p2=0 -> p1_value=4, p2_value=2
        early = _make_position_data(p1_score=0.0, p2_score=0.0, turn=0)
        # Mid position: p1=2, p2=1 -> p1_value=2, p2_value=1
        mid = _make_position_data(p1_score=2.0, p2_score=1.0, turn=50)
        # Late position: p1=3, p2=2 -> p1_value=1, p2_value=0
        late = _make_position_data(p1_score=3.0, p2_score=2.0, turn=90)
        # End position: p1=4, p2=2 -> p1_value=0, p2_value=0
        end = _make_position_data(p1_score=4.0, p2_score=2.0, turn=100)

        early_result = build_targets(game, early)
        mid_result = build_targets(game, mid)
        late_result = build_targets(game, late)
        end_result = build_targets(game, end)

        assert early_result.p1_value == pytest.approx(4.0)
        assert early_result.p2_value == pytest.approx(2.0)
        assert mid_result.p1_value == pytest.approx(2.0)
        assert mid_result.p2_value == pytest.approx(1.0)
        assert late_result.p1_value == pytest.approx(1.0)
        assert late_result.p2_value == pytest.approx(0.0)
        assert end_result.p1_value == pytest.approx(0.0)
        assert end_result.p2_value == pytest.approx(0.0)

    def test_returns_target_bundle(self) -> None:
        """Should return TargetBundle dataclass."""
        game = _make_game_data()
        position = _make_position_data()

        result = build_targets(game, position)

        assert isinstance(result, TargetBundle)

    def test_policy_shape(self) -> None:
        """Policy targets should have shape (5,)."""
        game = _make_game_data()
        position = _make_position_data()

        result = build_targets(game, position)

        assert result.policy_p1.shape == (5,)
        assert result.policy_p2.shape == (5,)
