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
