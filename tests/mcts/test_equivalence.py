"""Tests for action equivalence utilities."""

import numpy as np
import pytest

from alpharat.mcts.equivalence import (
    compute_effective_marginals,
    compute_effective_total_visits,
    expand_strategy,
    get_effective_actions,
    get_equivalence_classes,
    reduce_and_expand_nash,
    reduce_matrix,
)


class TestGetEffectiveActions:
    """Tests for get_effective_actions."""

    def test_no_equivalence(self) -> None:
        """Identity mapping returns all actions."""
        effective_map = [0, 1, 2, 3, 4]
        assert get_effective_actions(effective_map) == [0, 1, 2, 3, 4]

    def test_one_blocked(self) -> None:
        """One blocked action reduces to 4 effective."""
        effective_map = [4, 1, 2, 3, 4]  # UP blocked
        assert get_effective_actions(effective_map) == [1, 2, 3, 4]

    def test_multiple_blocked(self) -> None:
        """Multiple blocked actions in corner."""
        effective_map = [4, 1, 2, 4, 4]  # UP and LEFT blocked
        assert get_effective_actions(effective_map) == [1, 2, 4]

    def test_all_equivalent(self) -> None:
        """All actions equivalent (mud case)."""
        effective_map = [4, 4, 4, 4, 4]
        assert get_effective_actions(effective_map) == [4]


class TestGetEquivalenceClasses:
    """Tests for get_equivalence_classes."""

    def test_no_equivalence(self) -> None:
        """Each action is its own class."""
        effective_map = [0, 1, 2, 3, 4]
        classes = get_equivalence_classes(effective_map)
        assert classes == {0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}

    def test_one_blocked(self) -> None:
        """Blocked action joins STAY class."""
        effective_map = [4, 1, 2, 3, 4]
        classes = get_equivalence_classes(effective_map)
        assert classes == {4: [0, 4], 1: [1], 2: [2], 3: [3]}

    def test_multiple_blocked(self) -> None:
        """Multiple blocked actions in same class."""
        effective_map = [4, 1, 2, 4, 4]
        classes = get_equivalence_classes(effective_map)
        assert classes == {4: [0, 3, 4], 1: [1], 2: [2]}


class TestReduceMatrix:
    """Tests for reduce_matrix."""

    def test_no_reduction(self) -> None:
        """No equivalence means same matrix."""
        matrix = np.arange(25).reshape(5, 5).astype(float)
        p1_effective = [0, 1, 2, 3, 4]
        p2_effective = [0, 1, 2, 3, 4]

        reduced, p1_actions, p2_actions = reduce_matrix(matrix, p1_effective, p2_effective)

        np.testing.assert_array_equal(reduced, matrix)
        assert p1_actions == [0, 1, 2, 3, 4]
        assert p2_actions == [0, 1, 2, 3, 4]

    def test_reduce_one_player(self) -> None:
        """Reduce when P1 has blocked action."""
        matrix = np.arange(25).reshape(5, 5).astype(float)
        p1_effective = [4, 1, 2, 3, 4]  # Row 0 equivalent to row 4
        p2_effective = [0, 1, 2, 3, 4]  # No reduction

        reduced, p1_actions, p2_actions = reduce_matrix(matrix, p1_effective, p2_effective)

        assert reduced.shape == (4, 5)  # 4 effective P1 actions
        assert p1_actions == [1, 2, 3, 4]
        assert p2_actions == [0, 1, 2, 3, 4]

        # Verify correct rows extracted
        expected = matrix[[1, 2, 3, 4], :]
        np.testing.assert_array_equal(reduced, expected)

    def test_reduce_both_players(self) -> None:
        """Reduce when both players have blocked actions."""
        matrix = np.arange(25).reshape(5, 5).astype(float)
        p1_effective = [4, 1, 2, 4, 4]  # Rows 0,3,4 equivalent
        p2_effective = [0, 4, 2, 3, 4]  # Cols 1,4 equivalent

        reduced, p1_actions, p2_actions = reduce_matrix(matrix, p1_effective, p2_effective)

        assert reduced.shape == (3, 4)  # 3 effective P1, 4 effective P2
        assert p1_actions == [1, 2, 4]
        assert p2_actions == [0, 2, 3, 4]

        # Verify correct submatrix
        expected = matrix[np.ix_([1, 2, 4], [0, 2, 3, 4])]
        np.testing.assert_array_equal(reduced, expected)

    def test_extreme_reduction(self) -> None:
        """Reduce to 1x1 when all actions equivalent."""
        matrix = np.arange(25).reshape(5, 5).astype(float)
        p1_effective = [4, 4, 4, 4, 4]  # All P1 -> 4
        p2_effective = [4, 4, 4, 4, 4]  # All P2 -> 4

        reduced, p1_actions, p2_actions = reduce_matrix(matrix, p1_effective, p2_effective)

        assert reduced.shape == (1, 1)
        assert p1_actions == [4]
        assert p2_actions == [4]
        assert reduced[0, 0] == matrix[4, 4]


class TestExpandStrategy:
    """Tests for expand_strategy."""

    def test_no_expansion(self) -> None:
        """Full effective list means no change."""
        reduced = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        effective_actions = [0, 1, 2, 3, 4]
        effective_map = [0, 1, 2, 3, 4]

        full = expand_strategy(reduced, effective_actions, effective_map)

        np.testing.assert_array_almost_equal(full, reduced)

    def test_expand_with_blocked(self) -> None:
        """Blocked action gets 0, effective gets probability."""
        reduced = np.array([0.3, 0.3, 0.2, 0.2])  # Over [1,2,3,4]
        effective_actions = [1, 2, 3, 4]
        effective_map = [4, 1, 2, 3, 4]  # Action 0 -> 4

        full = expand_strategy(reduced, effective_actions, effective_map)

        expected = np.array([0.0, 0.3, 0.3, 0.2, 0.2])
        np.testing.assert_array_almost_equal(full, expected)

    def test_expand_multiple_blocked(self) -> None:
        """Multiple blocked actions all get 0."""
        reduced = np.array([0.4, 0.3, 0.3])  # Over [1,2,4]
        effective_actions = [1, 2, 4]
        effective_map = [4, 1, 2, 4, 4]  # Actions 0,3,4 -> 4

        full = expand_strategy(reduced, effective_actions, effective_map)

        expected = np.array([0.0, 0.4, 0.3, 0.0, 0.3])
        np.testing.assert_array_almost_equal(full, expected)

    def test_expand_extreme(self) -> None:
        """All equivalent, one action gets all probability."""
        reduced = np.array([1.0])  # Over [4]
        effective_actions = [4]
        effective_map = [4, 4, 4, 4, 4]

        full = expand_strategy(reduced, effective_actions, effective_map)

        expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(full, expected)


class TestReduceAndExpandNash:
    """Tests for reduce_and_expand_nash."""

    def test_with_simple_nash(self) -> None:
        """Integration test with a simple Nash computation."""
        # Create a simple 5x5 matrix where rows 0,4 and cols 1,4 are equivalent
        # (filled with same values to satisfy equivalence invariant)
        matrix = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 2.0],  # Row 0 equivalent to row 4
                [5.0, 6.0, 7.0, 8.0, 6.0],
                [9.0, 10.0, 11.0, 12.0, 10.0],
                [13.0, 14.0, 15.0, 16.0, 14.0],
                [1.0, 2.0, 3.0, 4.0, 2.0],  # Row 4 equivalent to row 0
            ]
        )

        p1_effective = [4, 1, 2, 3, 4]  # Actions 0,4 -> 4
        p2_effective = [0, 4, 2, 3, 4]  # Actions 1,4 -> 4

        # Mock Nash function that returns uniform over effective actions
        def mock_nash(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n_rows, n_cols = m.shape
            return np.ones(n_rows) / n_rows, np.ones(n_cols) / n_cols

        p1_strat, p2_strat = reduce_and_expand_nash(mock_nash, matrix, p1_effective, p2_effective)

        # P1: 4 effective actions [1,2,3,4], uniform = 0.25 each
        # Actions 0 should be 0, 1,2,3,4 should be 0.25
        assert p1_strat[0] == pytest.approx(0.0)
        assert p1_strat[1] == pytest.approx(0.25)
        assert p1_strat[4] == pytest.approx(0.25)

        # P2: 4 effective actions [0,2,3,4], uniform = 0.25 each
        # Action 1 should be 0
        assert p2_strat[1] == pytest.approx(0.0)
        assert p2_strat[0] == pytest.approx(0.25)
        assert p2_strat[4] == pytest.approx(0.25)

        # Strategies should still sum to 1
        assert np.sum(p1_strat) == pytest.approx(1.0)
        assert np.sum(p2_strat) == pytest.approx(1.0)

    def test_blocked_actions_get_zero(self) -> None:
        """Verify blocked actions always get probability 0."""
        matrix = np.random.rand(5, 5)

        # Make matrix satisfy equivalence invariant
        # Row 0 = Row 4, Col 1 = Col 4
        matrix[0, :] = matrix[4, :]
        matrix[:, 1] = matrix[:, 4]

        p1_effective = [4, 1, 2, 3, 4]
        p2_effective = [0, 4, 2, 3, 4]

        def mock_nash(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # Return arbitrary valid distributions
            n_rows, n_cols = m.shape
            p1 = np.random.rand(n_rows)
            p1 /= p1.sum()
            p2 = np.random.rand(n_cols)
            p2 /= p2.sum()
            return p1, p2

        p1_strat, p2_strat = reduce_and_expand_nash(mock_nash, matrix, p1_effective, p2_effective)

        # Blocked actions must be 0
        assert p1_strat[0] == pytest.approx(0.0)  # P1's blocked action
        assert p2_strat[1] == pytest.approx(0.0)  # P2's blocked action


class TestComputeEffectiveMarginals:
    """Tests for compute_effective_marginals."""

    def test_no_equivalence(self) -> None:
        """No equivalence — marginals equal raw sums."""
        p1_eff = [0, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        visits = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 5],
            ],
            dtype=np.int32,
        )

        n1, n2 = compute_effective_marginals(visits, p1_eff, p2_eff)

        # Row sums
        np.testing.assert_array_equal(n1, [1, 2, 3, 4, 5])
        # Column sums
        np.testing.assert_array_equal(n2, [1, 2, 3, 4, 5])

    def test_with_equivalence_does_not_double_count(self) -> None:
        """Marginals should not double-count equivalent actions."""
        # P1 action 0 blocked (maps to STAY/4)
        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]  # P2 has no walls

        # Simulate: 1 visit to effective pair (STAY, UP).
        # Backup writes to [0,0] and [4,0] (equivalent rows).
        visits = np.zeros((5, 5), dtype=np.int32)
        visits[0, 0] = 1
        visits[4, 0] = 1

        n1, n2 = compute_effective_marginals(visits, p1_eff, p2_eff)

        # P1's marginals: actions 0 and 4 both map to effective=4
        # so they should both show the marginal of effective action 4
        assert n1[0] == 1
        assert n1[4] == 1
        assert n1[1] == 0
        assert n1[2] == 0
        assert n1[3] == 0

        # P2's marginal for action 0: should be 1, not 2
        assert n2[0] == 1  # This is the fix!
        assert n2[1] == 0
        assert n2[2] == 0
        assert n2[3] == 0
        assert n2[4] == 0

    def test_both_players_have_equivalence(self) -> None:
        """Both players have blocked actions."""
        # P1: action 0 blocked (maps to 4)
        # P2: action 1 blocked (maps to 4)
        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 4, 2, 3, 4]

        # Simulate: 1 visit to effective pair (STAY, STAY).
        # Backup writes to all 4 equivalent cells: [0,1], [0,4], [4,1], [4,4]
        visits = np.zeros((5, 5), dtype=np.int32)
        visits[0, 1] = 1
        visits[0, 4] = 1
        visits[4, 1] = 1
        visits[4, 4] = 1

        n1, n2 = compute_effective_marginals(visits, p1_eff, p2_eff)

        # P1: actions 0,4 → effective 4, should both be 1
        assert n1[0] == 1
        assert n1[4] == 1

        # P2: actions 1,4 → effective 4, should both be 1
        assert n2[1] == 1
        assert n2[4] == 1


class TestComputeEffectiveTotalVisits:
    """Tests for compute_effective_total_visits."""

    def test_no_equivalence(self) -> None:
        """No equivalence — total equals raw sum."""
        p1_eff = [0, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        visits = np.ones((5, 5), dtype=np.int32)
        total = compute_effective_total_visits(visits, p1_eff, p2_eff)
        assert total == 25

    def test_with_equivalence_does_not_double_count(self) -> None:
        """Total visits should count each simulation once."""
        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 1, 2, 3, 4]

        # 1 simulation to effective (STAY, UP), written to 2 cells
        visits = np.zeros((5, 5), dtype=np.int32)
        visits[0, 0] = 1
        visits[4, 0] = 1

        total = compute_effective_total_visits(visits, p1_eff, p2_eff)
        assert total == 1  # Not 2!

    def test_both_players_have_equivalence(self) -> None:
        """Both players blocked — 4 cells but 1 visit."""
        p1_eff = [4, 1, 2, 3, 4]
        p2_eff = [0, 4, 2, 3, 4]

        visits = np.zeros((5, 5), dtype=np.int32)
        visits[0, 1] = 1
        visits[0, 4] = 1
        visits[4, 1] = 1
        visits[4, 4] = 1

        total = compute_effective_total_visits(visits, p1_eff, p2_eff)
        assert total == 1  # Not 4!
