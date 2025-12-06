"""Tests for MCTS Node implementation."""

import numpy as np
import pytest

from alpharat.mcts.node import MCTSNode


@pytest.fixture
def simple_node() -> MCTSNode:
    """Create a simple 5x5 node with no mud and zero NN predictions."""
    prior_p1 = np.ones(5) / 5
    prior_p2 = np.ones(5) / 5
    nn_payout = np.zeros((5, 5))

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
        p1_mud_turns_remaining=0,
        p2_mud_turns_remaining=0,
    )


@pytest.fixture
def node_with_nn_predictions() -> MCTSNode:
    """Create a 3x3 node with non-zero NN predictions."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3
    nn_payout = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
    )


@pytest.fixture
def node_p1_in_mud() -> MCTSNode:
    """Create a 3x3 node with P1 stuck in mud."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3
    nn_payout = np.zeros((3, 3))

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
        p1_mud_turns_remaining=2,
        p2_mud_turns_remaining=0,
    )


@pytest.fixture
def node_p2_in_mud() -> MCTSNode:
    """Create a 3x3 node with P2 stuck in mud."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3
    nn_payout = np.zeros((3, 3))

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
        p1_mud_turns_remaining=0,
        p2_mud_turns_remaining=1,
    )


@pytest.fixture
def node_both_in_mud() -> MCTSNode:
    """Create a 2x2 node with both players stuck in mud."""
    prior_p1 = np.ones(2) / 2
    prior_p2 = np.ones(2) / 2
    nn_payout = np.zeros((2, 2))

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_payout_prediction=nn_payout,
        parent=None,
        p1_mud_turns_remaining=1,
        p2_mud_turns_remaining=1,
    )


class TestNodeBackup:
    """Tests for the backup method with incremental mean updates."""

    def test_basic_backup_single_visit(self, simple_node: MCTSNode) -> None:
        """Test backing up a single value to an unvisited action pair.

        Incremental mean formula: Q_new = Q_old + (G - Q_old) / (n + 1)
        With Q_old = 0, n = 0: Q_new = 0 + (5.0 - 0) / 1 = 5.0
        """
        assert simple_node.action_visits[1, 2] == 0
        assert simple_node.payout_matrix[1, 2] == 0.0

        simple_node.backup(action_p1=1, action_p2=2, value=5.0)

        assert simple_node.action_visits[1, 2] == 1
        assert simple_node.payout_matrix[1, 2] == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "backups,expected_q",
        [
            # Single backup
            ([(10.0,)], 10.0),
            # Two backups: Q = 10.0 + (6.0 - 10.0) / 2 = 8.0
            ([(10.0,), (6.0,)], 8.0),
            # Three backups: 10.0 -> 8.0 -> 7.0
            ([(10.0,), (6.0,), (5.0,)], 7.0),
            # Four backups: average of [10, 6, 5, 3] = 6.0
            ([(10.0,), (6.0,), (5.0,), (3.0,)], 6.0),
        ],
    )
    def test_incremental_mean_multiple_backups(
        self, simple_node: MCTSNode, backups: list[tuple[float]], expected_q: float
    ) -> None:
        """Test incremental mean update over multiple backups."""
        for value_tuple in backups:
            (value,) = value_tuple
            simple_node.backup(action_p1=0, action_p2=0, value=value)

        assert simple_node.action_visits[0, 0] == len(backups)
        assert simple_node.payout_matrix[0, 0] == pytest.approx(expected_q)

    def test_p1_in_mud_updates_entire_column(self, node_p1_in_mud: MCTSNode) -> None:
        """Test that when P1 is in mud, entire column is updated.

        P1 is stuck, so regardless of P1's action choice, if P2 plays action 1,
        the outcome is the same. All entries in column [:, 1] should be updated.
        """
        node_p1_in_mud.backup(action_p1=0, action_p2=1, value=4.0)

        # All entries in column 1 should be updated
        for i in range(3):
            assert node_p1_in_mud.action_visits[i, 1] == 1
            assert node_p1_in_mud.payout_matrix[i, 1] == pytest.approx(4.0)

        # Other columns should be unchanged
        assert node_p1_in_mud.action_visits[0, 0] == 0
        assert node_p1_in_mud.action_visits[0, 2] == 0

    def test_p2_in_mud_updates_entire_row(self, node_p2_in_mud: MCTSNode) -> None:
        """Test that when P2 is in mud, entire row is updated.

        P2 is stuck, so if P1 plays action 2, regardless of P2's action,
        the outcome is the same. All entries in row [2, :] should be updated.
        """
        node_p2_in_mud.backup(action_p1=2, action_p2=0, value=7.0)

        # All entries in row 2 should be updated
        for j in range(3):
            assert node_p2_in_mud.action_visits[2, j] == 1
            assert node_p2_in_mud.payout_matrix[2, j] == pytest.approx(7.0)

        # Other rows should be unchanged
        assert node_p2_in_mud.action_visits[0, 0] == 0
        assert node_p2_in_mud.action_visits[1, 0] == 0

    def test_both_in_mud_updates_full_matrix(self, node_both_in_mud: MCTSNode) -> None:
        """Test that when both players are in mud, entire matrix is updated.

        Both stuck, so all action combinations lead to the same outcome.
        """
        node_both_in_mud.backup(action_p1=0, action_p2=0, value=3.0)

        # All entries should be updated
        assert np.all(node_both_in_mud.action_visits == 1)
        assert np.all(node_both_in_mud.payout_matrix == pytest.approx(3.0))

    def test_mud_incremental_updates_column(self, node_p1_in_mud: MCTSNode) -> None:
        """Test incremental mean updates work correctly with mud states (column)."""
        # First backup: entire column [:, 0] updated to 10.0
        node_p1_in_mud.backup(action_p1=0, action_p2=0, value=10.0)
        assert node_p1_in_mud.payout_matrix[0, 0] == pytest.approx(10.0)
        assert node_p1_in_mud.payout_matrix[1, 0] == pytest.approx(10.0)
        assert node_p1_in_mud.payout_matrix[2, 0] == pytest.approx(10.0)

        # Second backup to same column: Q = 10.0 + (6.0 - 10.0) / 2 = 8.0
        node_p1_in_mud.backup(action_p1=1, action_p2=0, value=6.0)
        for i in range(3):
            assert node_p1_in_mud.action_visits[i, 0] == 2
            assert node_p1_in_mud.payout_matrix[i, 0] == pytest.approx(8.0)

    def test_preserves_nn_prediction_until_visited(
        self, node_with_nn_predictions: MCTSNode
    ) -> None:
        """Test that NN predictions remain until action pair is visited."""
        # Initially, payout matrix should equal NN prediction
        expected_nn = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_array_equal(node_with_nn_predictions.payout_matrix, expected_nn)

        # After backing up to (1, 1), only that entry changes
        # Q = 5.0 + (100.0 - 5.0) / 1 = 100.0
        node_with_nn_predictions.backup(action_p1=1, action_p2=1, value=100.0)

        assert node_with_nn_predictions.payout_matrix[1, 1] == pytest.approx(100.0)

        # Other entries still have NN predictions
        assert node_with_nn_predictions.payout_matrix[0, 0] == pytest.approx(1.0)
        assert node_with_nn_predictions.payout_matrix[2, 2] == pytest.approx(9.0)
        assert node_with_nn_predictions.payout_matrix[0, 2] == pytest.approx(3.0)

    def test_different_actions_independent(self, simple_node: MCTSNode) -> None:
        """Test that backing up different action pairs are independent."""
        simple_node.backup(action_p1=0, action_p2=0, value=5.0)
        simple_node.backup(action_p1=1, action_p2=1, value=10.0)
        simple_node.backup(action_p1=2, action_p2=2, value=15.0)

        # Each action pair should have its own value
        assert simple_node.payout_matrix[0, 0] == pytest.approx(5.0)
        assert simple_node.payout_matrix[1, 1] == pytest.approx(10.0)
        assert simple_node.payout_matrix[2, 2] == pytest.approx(15.0)

        # Visit counts should be independent
        assert simple_node.action_visits[0, 0] == 1
        assert simple_node.action_visits[1, 1] == 1
        assert simple_node.action_visits[2, 2] == 1
        assert simple_node.action_visits[0, 1] == 0

    @pytest.mark.parametrize(
        "values,expected",
        [
            # Single negative value
            ([(-5.0,)], -5.0),
            # Mix negative and positive: Q = -5.0 + (3.0 - (-5.0)) / 2 = -1.0
            ([(-5.0,), (3.0,)], -1.0),
            # All negative
            ([(-10.0,), (-6.0,)], -8.0),
            # Zero crossing: [-4, 0, 4] -> average = 0.0
            ([(-4.0,), (0.0,), (4.0,)], 0.0),
        ],
    )
    def test_negative_values(
        self, simple_node: MCTSNode, values: list[tuple[float]], expected: float
    ) -> None:
        """Test that backup works correctly with negative values."""
        for value_tuple in values:
            (value,) = value_tuple
            simple_node.backup(action_p1=0, action_p2=0, value=value)

        assert simple_node.payout_matrix[0, 0] == pytest.approx(expected)


class TestNodeProperties:
    """Tests for node properties and state."""

    def test_total_visits_zero_initially(self, simple_node: MCTSNode) -> None:
        """Test that total_visits is 0 for a new node."""
        assert simple_node.total_visits == 0

    def test_total_visits_sums_all_action_visits(self, simple_node: MCTSNode) -> None:
        """Test that total_visits correctly sums all action pair visits."""
        simple_node.backup(action_p1=0, action_p2=0, value=1.0)
        simple_node.backup(action_p1=1, action_p2=1, value=2.0)
        simple_node.backup(action_p1=0, action_p2=0, value=3.0)

        # Should have 2 visits to (0,0) and 1 visit to (1,1) = 3 total
        assert simple_node.total_visits == 3

    def test_total_visits_with_mud(self, node_p1_in_mud: MCTSNode) -> None:
        """Test that total_visits counts correctly with mud updates."""
        # P1 in mud, so backing up column [:, 0] increments 3 entries
        node_p1_in_mud.backup(action_p1=0, action_p2=0, value=5.0)

        # 3 entries in column, each visited once = 3 total
        assert node_p1_in_mud.total_visits == 3

    def test_is_expanded_false_initially(self, simple_node: MCTSNode) -> None:
        """Test that is_expanded is False for a node with no children."""
        assert simple_node.is_expanded is False

    def test_is_expanded_true_with_children(self, simple_node: MCTSNode) -> None:
        """Test that is_expanded is True when children are added."""
        # Create a dummy child node
        child_prior_p1 = np.ones(5) / 5
        child_prior_p2 = np.ones(5) / 5
        child_nn_payout = np.zeros((5, 5))

        child = MCTSNode(
            game_state=None,
            prior_policy_p1=child_prior_p1,
            prior_policy_p2=child_prior_p2,
            nn_payout_prediction=child_nn_payout,
            parent=simple_node,
        )

        # Add child to parent
        simple_node.children[(0, 0)] = child

        assert simple_node.is_expanded is True
        assert len(simple_node.children) == 1


class TestActionEquivalence:
    """Tests for action equivalence (wall-based effective action mapping)."""

    @pytest.fixture
    def node_with_p1_wall(self) -> MCTSNode:
        """Node where P1's action 0 (UP) is blocked, equivalent to STAY."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        # P1: UP(0) blocked -> maps to STAY(4)
        # Actions 1,2,3 are valid, 0 and 4 are equivalent
        p1_effective = [4, 1, 2, 3, 4]  # 0 -> 4, others -> themselves
        p2_effective = [0, 1, 2, 3, 4]  # P2 has no walls

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    @pytest.fixture
    def node_with_corner_walls(self) -> MCTSNode:
        """Node where both players have multiple blocked directions (corner)."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        # P1: UP(0) and LEFT(3) blocked -> both map to STAY(4)
        # Equivalence class: {0, 3, 4}
        p1_effective = [4, 1, 2, 4, 4]

        # P2: DOWN(2) blocked -> maps to STAY(4)
        # Equivalence class: {2, 4}
        p2_effective = [0, 1, 4, 3, 4]

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    def test_backup_updates_equivalent_actions(self, node_with_p1_wall: MCTSNode) -> None:
        """Backing up action 0 should also update action 4 (both are STAY for P1)."""
        node_with_p1_wall.backup(action_p1=0, action_p2=1, value=10.0)

        # Both action 0 and 4 for P1 should be updated (same column for P2's action 1)
        assert node_with_p1_wall.payout_matrix[0, 1] == pytest.approx(10.0)
        assert node_with_p1_wall.payout_matrix[4, 1] == pytest.approx(10.0)
        assert node_with_p1_wall.action_visits[0, 1] == 1
        assert node_with_p1_wall.action_visits[4, 1] == 1

        # Other P1 actions should not be updated
        assert node_with_p1_wall.action_visits[1, 1] == 0
        assert node_with_p1_wall.action_visits[2, 1] == 0

    def test_backup_updates_rectangular_region(self, node_with_corner_walls: MCTSNode) -> None:
        """Backing up should update all combinations of equivalent actions."""
        node_with_corner_walls.backup(action_p1=0, action_p2=2, value=5.0)

        # P1's {0, 3, 4} x P2's {2, 4} = 6 entries should be updated
        for p1_action in [0, 3, 4]:
            for p2_action in [2, 4]:
                assert node_with_corner_walls.payout_matrix[p1_action, p2_action] == pytest.approx(
                    5.0
                )
                assert node_with_corner_walls.action_visits[p1_action, p2_action] == 1

        # Non-equivalent actions should not be updated
        assert node_with_corner_walls.action_visits[1, 1] == 0
        assert node_with_corner_walls.action_visits[2, 0] == 0

    def test_equivalence_incremental_mean(self, node_with_p1_wall: MCTSNode) -> None:
        """Multiple backups to equivalent actions should use incremental mean."""
        # First backup via action 0
        node_with_p1_wall.backup(action_p1=0, action_p2=0, value=10.0)

        # Second backup via action 4 (equivalent to 0)
        node_with_p1_wall.backup(action_p1=4, action_p2=0, value=6.0)

        # Both entries should have 2 visits and Q = (10 + 6) / 2 = 8
        assert node_with_p1_wall.action_visits[0, 0] == 2
        assert node_with_p1_wall.action_visits[4, 0] == 2
        assert node_with_p1_wall.payout_matrix[0, 0] == pytest.approx(8.0)
        assert node_with_p1_wall.payout_matrix[4, 0] == pytest.approx(8.0)

    def test_mud_overrides_wall_equivalence(self) -> None:
        """Mud should make all actions equivalent, overriding wall-based mapping."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        # Even though we pass wall-based effective mapping, mud should override
        p1_effective = [4, 1, 2, 3, 4]  # Would only merge 0 and 4
        p2_effective = [0, 1, 2, 3, 4]

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=None,
            p1_mud_turns_remaining=2,  # P1 in mud
            p2_mud_turns_remaining=0,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

        # Mud should have overridden p1_effective to all STAY
        assert node.p1_effective == [4, 4, 4, 4, 4]

        # Backup should update entire column
        node.backup(action_p1=1, action_p2=0, value=7.0)
        for i in range(5):
            assert node.action_visits[i, 0] == 1


class TestEquivalenceInvariants:
    """Tests verifying that equivalent actions always have identical statistics.

    The core invariant: after any sequence of backups, all equivalent action pairs
    must have exactly the same payout_matrix and action_visits values. This ensures
    Nash equilibrium computation treats them as truly equivalent.
    """

    @pytest.fixture
    def complex_equivalence_node(self) -> MCTSNode:
        """Node with multiple equivalence classes for both players.

        P1: {0, 3, 4} equivalent (corner - UP and LEFT blocked)
        P2: {1, 4} equivalent (edge - RIGHT blocked)
        """
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        p1_effective = [4, 1, 2, 4, 4]  # 0,3,4 -> 4
        p2_effective = [0, 4, 2, 3, 4]  # 1,4 -> 4

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    def _get_equivalence_class(self, effective_map: list[int], action: int) -> list[int]:
        """Get all actions equivalent to the given action."""
        target = effective_map[action]
        return [a for a, c in enumerate(effective_map) if c == target]

    def _verify_equivalence_invariant(self, node: MCTSNode) -> None:
        """Verify that all equivalent action pairs have identical statistics."""
        for a1 in range(5):
            for a2 in range(5):
                p1_equiv = self._get_equivalence_class(node.p1_effective, a1)
                p2_equiv = self._get_equivalence_class(node.p2_effective, a2)

                # All pairs in the equivalence class should have same stats
                reference_payout = node.payout_matrix[a1, a2]
                reference_visits = node.action_visits[a1, a2]

                for eq_a1 in p1_equiv:
                    for eq_a2 in p2_equiv:
                        assert node.payout_matrix[eq_a1, eq_a2] == pytest.approx(
                            reference_payout
                        ), f"Payout mismatch: ({a1},{a2}) vs ({eq_a1},{eq_a2})"
                        assert node.action_visits[eq_a1, eq_a2] == reference_visits, (
                            f"Visit mismatch: ({a1},{a2}) vs ({eq_a1},{eq_a2})"
                        )

    def test_invariant_after_single_backup(self, complex_equivalence_node: MCTSNode) -> None:
        """Single backup should maintain equivalence invariant."""
        complex_equivalence_node.backup(action_p1=0, action_p2=1, value=5.0)
        self._verify_equivalence_invariant(complex_equivalence_node)

    def test_invariant_after_multiple_backups_same_class(
        self, complex_equivalence_node: MCTSNode
    ) -> None:
        """Multiple backups through same equivalence class maintain invariant."""
        # All these go to the same equivalence class: P1's {0,3,4} x P2's {1,4}
        complex_equivalence_node.backup(action_p1=0, action_p2=1, value=10.0)
        self._verify_equivalence_invariant(complex_equivalence_node)

        complex_equivalence_node.backup(action_p1=4, action_p2=4, value=6.0)
        self._verify_equivalence_invariant(complex_equivalence_node)

        complex_equivalence_node.backup(action_p1=3, action_p2=1, value=8.0)
        self._verify_equivalence_invariant(complex_equivalence_node)

        # After 3 backups, all 6 entries (3x2) should have same stats
        # Q = (10 + 6 + 8) / 3 = 8.0
        for p1_action in [0, 3, 4]:
            for p2_action in [1, 4]:
                assert complex_equivalence_node.payout_matrix[
                    p1_action, p2_action
                ] == pytest.approx(8.0)
                assert complex_equivalence_node.action_visits[p1_action, p2_action] == 3

    def test_invariant_after_backups_to_different_classes(
        self, complex_equivalence_node: MCTSNode
    ) -> None:
        """Backups to different equivalence classes maintain invariant."""
        # Class 1: P1's {0,3,4} x P2's {1,4}
        complex_equivalence_node.backup(action_p1=0, action_p2=1, value=10.0)

        # Class 2: P1's {1} x P2's {0}
        complex_equivalence_node.backup(action_p1=1, action_p2=0, value=20.0)

        # Class 3: P1's {2} x P2's {2}
        complex_equivalence_node.backup(action_p1=2, action_p2=2, value=30.0)

        self._verify_equivalence_invariant(complex_equivalence_node)

        # Verify specific values
        assert complex_equivalence_node.payout_matrix[0, 1] == pytest.approx(10.0)
        assert complex_equivalence_node.payout_matrix[1, 0] == pytest.approx(20.0)
        assert complex_equivalence_node.payout_matrix[2, 2] == pytest.approx(30.0)

    def test_invariant_with_randomized_backups(self, complex_equivalence_node: MCTSNode) -> None:
        """Randomized backup sequence maintains invariant."""
        import random

        random.seed(42)

        # Do 50 random backups
        for _ in range(50):
            a1 = random.randint(0, 4)
            a2 = random.randint(0, 4)
            value = random.uniform(-10.0, 10.0)
            complex_equivalence_node.backup(action_p1=a1, action_p2=a2, value=value)

        # Invariant should still hold
        self._verify_equivalence_invariant(complex_equivalence_node)

    def test_invariant_stress_test(self) -> None:
        """Stress test with extreme equivalence (all actions equivalent for one player)."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5
        nn_payout = np.zeros((5, 5))

        # P1 in mud: all actions equivalent
        # P2: only action 2 blocked
        p1_effective = [4, 4, 4, 4, 4]  # All -> 4
        p2_effective = [0, 1, 4, 3, 4]  # 2,4 -> 4

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_payout_prediction=nn_payout,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

        import random

        random.seed(123)

        # 100 random backups
        for _ in range(100):
            a1 = random.randint(0, 4)
            a2 = random.randint(0, 4)
            value = random.uniform(-5.0, 5.0)
            node.backup(action_p1=a1, action_p2=a2, value=value)

        # For P1, entire rows should be identical (all P1 actions equivalent)
        for a2 in range(5):
            reference = node.payout_matrix[0, a2]
            ref_visits = node.action_visits[0, a2]
            for a1 in range(5):
                assert node.payout_matrix[a1, a2] == pytest.approx(reference)
                assert node.action_visits[a1, a2] == ref_visits
