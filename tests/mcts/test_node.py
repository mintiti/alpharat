"""Tests for MCTS Node implementation with LC0-style scalar values."""

import numpy as np
import pytest

from alpharat.mcts.node import MCTSNode


def add_child(
    parent: MCTSNode,
    idx1: int,
    idx2: int,
    v1: float,
    v2: float,
    edge_r1: float = 0.0,
    edge_r2: float = 0.0,
) -> MCTSNode:
    """Helper to add a child node for testing Q computation.

    Sets child._visits = 1 to mark it as visited (otherwise get_q_values uses FPU).
    Also sets edge rewards for Q = edge_r + gamma * V computation.
    """
    child = MCTSNode(
        game_state=None,
        prior_policy_p1=np.ones(parent.n1) / parent.n1,
        prior_policy_p2=np.ones(parent.n2) / parent.n2,
        nn_value_p1=v1,
        nn_value_p2=v2,
        parent=parent,
    )
    child._visits = 1  # Mark as visited
    child._edge_r1 = edge_r1
    child._edge_r2 = edge_r2
    parent.children[(idx1, idx2)] = child
    return child


@pytest.fixture
def simple_node() -> MCTSNode:
    """Create a simple 5x5 node with no mud and zero NN predictions."""
    prior_p1 = np.ones(5) / 5
    prior_p2 = np.ones(5) / 5

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=0,
        p2_mud_turns_remaining=0,
    )


@pytest.fixture
def node_with_nn_predictions() -> MCTSNode:
    """Create a 3x3 node with non-zero NN predictions."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_value_p1=5.0,
        nn_value_p2=2.5,
        parent=None,
    )


@pytest.fixture
def node_p1_in_mud() -> MCTSNode:
    """Create a 3x3 node with P1 stuck in mud."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=2,
        p2_mud_turns_remaining=0,
    )


@pytest.fixture
def node_p2_in_mud() -> MCTSNode:
    """Create a 3x3 node with P2 stuck in mud."""
    prior_p1 = np.ones(3) / 3
    prior_p2 = np.ones(3) / 3

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=0,
        p2_mud_turns_remaining=1,
    )


@pytest.fixture
def node_both_in_mud() -> MCTSNode:
    """Create a 2x2 node with both players stuck in mud."""
    prior_p1 = np.ones(2) / 2
    prior_p2 = np.ones(2) / 2

    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior_p1,
        prior_policy_p2=prior_p2,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
        parent=None,
        p1_mud_turns_remaining=1,
        p2_mud_turns_remaining=1,
    )


class TestNodeBackup:
    """Tests for the backup method with decoupled incremental mean updates.

    Backup updates both child.V and self.V:
    - child.V updated with child_value (child's expected return)
    - self.V updated with self_value (discounted return Q = r + gamma * child.V)
    """

    def test_basic_backup_single_visit(self, simple_node: MCTSNode) -> None:
        """Test backing up a single value to an unvisited action pair.

        Incremental mean formula: V_new = V_old + (value - V_old) / (n + 1)
        With V_old = 0, n = 0: V_new = 0 + (value - 0) / 1 = value
        """
        q1, q2 = simple_node.get_q_values()
        n1, n2 = simple_node.get_visit_counts()

        # Initially Q = NN value = 0, N = 0
        assert n1[1] == 0
        assert n2[2] == 0
        assert q1[1] == 0.0
        assert q2[2] == 0.0

        # Backup: child_value is child's V, self_value is Q = r + gamma * V
        # For unit test without child, only self_value matters
        simple_node.backup(action_p1=1, action_p2=2, child_value=(0.0, 0.0), self_value=(5.0, 3.0))

        q1, q2 = simple_node.get_q_values()
        n1, n2 = simple_node.get_visit_counts()

        # Marginal updates: Q1[1] updated for P1, Q2[2] updated for P2
        assert n1[1] == 1
        assert n2[2] == 1
        assert q1[1] == pytest.approx(5.0)
        assert q2[2] == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "backups,expected_p1,expected_p2",
        [
            # Single backup
            ([(10.0, 5.0)], 10.0, 5.0),
            # Two backups: Q = old + (new - old) / n
            # P1: 10.0 + (6.0 - 10.0) / 2 = 8.0
            # P2: 5.0 + (3.0 - 5.0) / 2 = 4.0
            ([(10.0, 5.0), (6.0, 3.0)], 8.0, 4.0),
            # Three backups: average
            ([(10.0, 6.0), (6.0, 4.0), (5.0, 2.0)], 7.0, 4.0),
            # Four backups
            ([(10.0, 4.0), (6.0, 8.0), (5.0, 6.0), (3.0, 6.0)], 6.0, 6.0),
        ],
    )
    def test_incremental_mean_multiple_backups(
        self,
        simple_node: MCTSNode,
        backups: list[tuple[float, float]],
        expected_p1: float,
        expected_p2: float,
    ) -> None:
        """Test incremental mean update over multiple backups for both players."""
        for p1_value, p2_value in backups:
            simple_node.backup(
                action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(p1_value, p2_value)
            )

        q1, q2 = simple_node.get_q_values()
        n1, n2 = simple_node.get_visit_counts()

        assert n1[0] == len(backups)
        assert n2[0] == len(backups)
        assert q1[0] == pytest.approx(expected_p1)
        assert q2[0] == pytest.approx(expected_p2)

    def test_p1_in_mud_single_outcome(self, node_p1_in_mud: MCTSNode) -> None:
        """Test that when P1 is in mud, P1 has a single outcome.

        P1 is stuck, so all P1 actions map to STAY. P1's marginal Q has size 1.
        """
        # P1 in mud means n1 = 1 (all actions equivalent to STAY)
        assert node_p1_in_mud.n1 == 1
        assert node_p1_in_mud.n2 == 3

        node_p1_in_mud.backup(
            action_p1=0, action_p2=1, child_value=(0.0, 0.0), self_value=(4.0, 2.0)
        )

        q1, q2 = node_p1_in_mud.get_q_values()
        n1, n2 = node_p1_in_mud.get_visit_counts()

        # P1 has single Q value, P2's action 1 updated
        assert q1[0] == pytest.approx(4.0)
        assert q2[1] == pytest.approx(2.0)
        assert n1[0] == 1
        assert n2[1] == 1

    def test_p2_in_mud_single_outcome(self, node_p2_in_mud: MCTSNode) -> None:
        """Test that when P2 is in mud, P2 has a single outcome."""
        assert node_p2_in_mud.n1 == 3
        assert node_p2_in_mud.n2 == 1

        node_p2_in_mud.backup(
            action_p1=2, action_p2=0, child_value=(0.0, 0.0), self_value=(7.0, 3.0)
        )

        q1, q2 = node_p2_in_mud.get_q_values()
        n1, n2 = node_p2_in_mud.get_visit_counts()

        assert q1[2] == pytest.approx(7.0)
        assert q2[0] == pytest.approx(3.0)
        assert n1[2] == 1
        assert n2[0] == 1

    def test_both_in_mud_single_outcome_each(self, node_both_in_mud: MCTSNode) -> None:
        """Test that when both players are in mud, both have single outcomes."""
        assert node_both_in_mud.n1 == 1
        assert node_both_in_mud.n2 == 1

        node_both_in_mud.backup(
            action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(3.0, 1.0)
        )

        q1, q2 = node_both_in_mud.get_q_values()
        n1, n2 = node_both_in_mud.get_visit_counts()

        assert q1[0] == pytest.approx(3.0)
        assert q2[0] == pytest.approx(1.0)
        assert n1[0] == 1
        assert n2[0] == 1

    def test_fpu_uses_node_value(self, node_with_nn_predictions: MCTSNode) -> None:
        """Test that FPU (first play urgency) uses node's current value.

        LC0-style: Q for unvisited outcomes = node's v1/v2 (not original NN value).
        """
        q1, q2 = node_with_nn_predictions.get_q_values()

        # Initially Q = FPU = node's v1/v2 = (5.0, 2.5)
        np.testing.assert_array_almost_equal(q1, [5.0, 5.0, 5.0])
        np.testing.assert_array_almost_equal(q2, [2.5, 2.5, 2.5])

        # Add a child and back up through it
        child = add_child(node_with_nn_predictions, idx1=1, idx2=1, v1=100.0, v2=50.0)
        node_with_nn_predictions.backup(
            action_p1=1, action_p2=1, child_value=(100.0, 50.0), self_value=(100.0, 50.0)
        )

        q1, q2 = node_with_nn_predictions.get_q_values()

        # Q for visited outcome comes from child
        assert q1[1] == pytest.approx(100.0)
        assert q2[1] == pytest.approx(50.0)
        assert child.visits == 2  # add_child sets _visits=1, backup increments to 2

        # FPU for unvisited outcomes = node's updated v1/v2 = (100.0, 50.0)
        assert q1[0] == pytest.approx(100.0)
        assert q1[2] == pytest.approx(100.0)
        assert q2[0] == pytest.approx(50.0)
        assert q2[2] == pytest.approx(50.0)

    def test_different_actions_independent(self, simple_node: MCTSNode) -> None:
        """Test that Q for different outcomes comes from different children."""
        # Add children for each outcome pair
        add_child(simple_node, idx1=0, idx2=0, v1=5.0, v2=2.0)
        add_child(simple_node, idx1=1, idx2=1, v1=10.0, v2=4.0)
        add_child(simple_node, idx1=2, idx2=2, v1=15.0, v2=6.0)

        # Back up through each child (child_value and self_value same for gamma=1, no reward)
        simple_node.backup(action_p1=0, action_p2=0, child_value=(5.0, 2.0), self_value=(5.0, 2.0))
        simple_node.backup(
            action_p1=1, action_p2=1, child_value=(10.0, 4.0), self_value=(10.0, 4.0)
        )
        simple_node.backup(
            action_p1=2, action_p2=2, child_value=(15.0, 6.0), self_value=(15.0, 6.0)
        )

        q1, q2 = simple_node.get_q_values()

        # Each outcome gets Q from its child
        assert q1[0] == pytest.approx(5.0)
        assert q1[1] == pytest.approx(10.0)
        assert q1[2] == pytest.approx(15.0)
        assert q2[0] == pytest.approx(2.0)
        assert q2[1] == pytest.approx(4.0)
        assert q2[2] == pytest.approx(6.0)

    @pytest.mark.parametrize(
        "values,expected_p1,expected_p2",
        [
            # Single negative value
            ([(-5.0, -2.0)], -5.0, -2.0),
            # Mix negative and positive: Q = -5.0 + (3.0 - (-5.0)) / 2 = -1.0
            ([(-5.0, -3.0), (3.0, 1.0)], -1.0, -1.0),
            # All negative
            ([(-10.0, -4.0), (-6.0, -2.0)], -8.0, -3.0),
            # Zero crossing: [-4, 0, 4] -> average = 0.0
            ([(-4.0, -2.0), (0.0, 0.0), (4.0, 2.0)], 0.0, 0.0),
        ],
    )
    def test_negative_values(
        self,
        simple_node: MCTSNode,
        values: list[tuple[float, float]],
        expected_p1: float,
        expected_p2: float,
    ) -> None:
        """Test that backup works correctly with negative values."""
        for p1_val, p2_val in values:
            simple_node.backup(
                action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(p1_val, p2_val)
            )

        q1, q2 = simple_node.get_q_values()
        assert q1[0] == pytest.approx(expected_p1)
        assert q2[0] == pytest.approx(expected_p2)


class TestNodeProperties:
    """Tests for node properties and state."""

    def test_total_visits_zero_initially(self, simple_node: MCTSNode) -> None:
        """Test that total_visits is 0 for a new node."""
        assert simple_node.total_visits == 0

    def test_total_visits_sums_all_action_visits(self, simple_node: MCTSNode) -> None:
        """Test that total_visits correctly counts simulations."""
        simple_node.backup(action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(1.0, 0.5))
        simple_node.backup(action_p1=1, action_p2=1, child_value=(0.0, 0.0), self_value=(2.0, 1.0))
        simple_node.backup(action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(3.0, 1.5))

        # Should have 3 total simulations
        assert simple_node.total_visits == 3

    def test_total_visits_with_mud(self, node_p1_in_mud: MCTSNode) -> None:
        """Test that total_visits counts simulations, not matrix cells."""
        node_p1_in_mud.backup(
            action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(5.0, 2.0)
        )

        # 1 simulation
        assert node_p1_in_mud.total_visits == 1

    def test_is_expanded_false_initially(self, simple_node: MCTSNode) -> None:
        """Test that is_expanded is False for a node with no children."""
        assert simple_node.is_expanded is False

    def test_is_expanded_true_with_children(self, simple_node: MCTSNode) -> None:
        """Test that is_expanded is True when children are added."""
        # Create a dummy child node
        child_prior_p1 = np.ones(5) / 5
        child_prior_p2 = np.ones(5) / 5

        child = MCTSNode(
            game_state=None,
            prior_policy_p1=child_prior_p1,
            prior_policy_p2=child_prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=simple_node,
        )

        # Add child to parent
        simple_node.children[(0, 0)] = child

        assert simple_node.is_expanded is True
        assert len(simple_node.children) == 1

    def test_v1_v2_properties(self, node_with_nn_predictions: MCTSNode) -> None:
        """Test that v1 and v2 return node values (initially NN predictions)."""
        assert node_with_nn_predictions.v1 == pytest.approx(5.0)
        assert node_with_nn_predictions.v2 == pytest.approx(2.5)


class TestActionEquivalence:
    """Tests for action equivalence (wall-based effective action mapping)."""

    @pytest.fixture
    def node_with_p1_wall(self) -> MCTSNode:
        """Node where P1's action 0 (UP) is blocked, equivalent to STAY."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        # P1: UP(0) blocked -> maps to STAY(4)
        p1_effective = [4, 1, 2, 3, 4]
        p2_effective = [0, 1, 2, 3, 4]

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    @pytest.fixture
    def node_with_corner_walls(self) -> MCTSNode:
        """Node where both players have multiple blocked directions (corner)."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        # P1: UP(0) and LEFT(3) blocked -> both map to STAY(4)
        p1_effective = [4, 1, 2, 4, 4]

        # P2: DOWN(2) blocked -> maps to STAY(4)
        p2_effective = [0, 1, 4, 3, 4]

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    def test_reduced_dimensions_with_equivalence(self, node_with_p1_wall: MCTSNode) -> None:
        """Test that equivalence reduces the number of outcomes."""
        # P1: actions 0 and 4 both map to STAY -> 4 unique outcomes
        # P2: all 5 actions unique
        assert node_with_p1_wall.n1 == 4
        assert node_with_p1_wall.n2 == 5

    def test_backup_maps_equivalent_actions_to_same_outcome(
        self, node_with_p1_wall: MCTSNode
    ) -> None:
        """Backing up action 0 or 4 for P1 should update the same Q1 entry."""
        node_with_p1_wall.backup(
            action_p1=0, action_p2=1, child_value=(0.0, 0.0), self_value=(10.0, 5.0)
        )

        q1_before, _ = node_with_p1_wall.get_q_values()
        q1_outcome_for_0 = q1_before[node_with_p1_wall.action_to_outcome(1, 0)]
        q1_outcome_for_4 = q1_before[node_with_p1_wall.action_to_outcome(1, 4)]

        # Both actions 0 and 4 map to the same outcome index
        assert node_with_p1_wall.action_to_outcome(1, 0) == node_with_p1_wall.action_to_outcome(
            1, 4
        )
        assert q1_outcome_for_0 == pytest.approx(10.0)
        assert q1_outcome_for_4 == pytest.approx(10.0)

    def test_equivalence_incremental_mean(self, node_with_p1_wall: MCTSNode) -> None:
        """Multiple backups via equivalent actions should share the same outcome."""
        # First backup via action 0
        node_with_p1_wall.backup(
            action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(10.0, 6.0)
        )

        # Second backup via action 4 (equivalent to 0 for P1)
        node_with_p1_wall.backup(
            action_p1=4, action_p2=0, child_value=(0.0, 0.0), self_value=(6.0, 2.0)
        )

        q1, q2 = node_with_p1_wall.get_q_values()
        n1, n2 = node_with_p1_wall.get_visit_counts()

        # P1's outcome for actions 0 and 4 should have 2 visits, Q = 8.0
        outcome_idx = node_with_p1_wall.action_to_outcome(1, 0)
        assert n1[outcome_idx] == 2
        assert q1[outcome_idx] == pytest.approx(8.0)

        # P2's action 0 outcome should have 2 visits, Q = 4.0
        assert n2[0] == 2
        assert q2[0] == pytest.approx(4.0)

    def test_mud_overrides_wall_equivalence(self) -> None:
        """Mud should make all actions equivalent, overriding wall-based mapping."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        p1_effective = [4, 1, 2, 3, 4]  # Would only merge 0 and 4
        p2_effective = [0, 1, 2, 3, 4]

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=None,
            p1_mud_turns_remaining=2,  # P1 in mud
            p2_mud_turns_remaining=0,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

        # Mud should have overridden p1_effective to all STAY
        assert node.p1_effective == [4, 4, 4, 4, 4]
        assert node.n1 == 1  # Single outcome for P1


class TestEquivalenceInvariants:
    """Tests verifying that equivalent actions always map to the same outcome."""

    @pytest.fixture
    def complex_equivalence_node(self) -> MCTSNode:
        """Node with multiple equivalence classes for both players.

        P1: {0, 3, 4} equivalent (corner - UP and LEFT blocked)
        P2: {1, 4} equivalent (edge - RIGHT blocked)
        """
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        p1_effective = [4, 1, 2, 4, 4]  # 0,3,4 -> 4
        p2_effective = [0, 4, 2, 3, 4]  # 1,4 -> 4

        return MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=None,
            p1_effective=p1_effective,
            p2_effective=p2_effective,
        )

    def test_outcome_indices_consistent(self, complex_equivalence_node: MCTSNode) -> None:
        """Equivalent actions should map to the same outcome index."""
        node = complex_equivalence_node

        # P1's equivalence class {0, 3, 4} should all map to same outcome
        assert node.action_to_outcome(1, 0) == node.action_to_outcome(1, 3)
        assert node.action_to_outcome(1, 0) == node.action_to_outcome(1, 4)

        # P2's equivalence class {1, 4} should map to same outcome
        assert node.action_to_outcome(2, 1) == node.action_to_outcome(2, 4)

        # Non-equivalent actions should have different outcomes
        assert node.action_to_outcome(1, 1) != node.action_to_outcome(1, 2)
        assert node.action_to_outcome(2, 0) != node.action_to_outcome(2, 2)

    def test_invariant_with_randomized_backups(self, complex_equivalence_node: MCTSNode) -> None:
        """Randomized backup sequence maintains outcome consistency."""
        import random

        random.seed(42)
        node = complex_equivalence_node

        # Do 50 random backups
        for _ in range(50):
            a1 = random.randint(0, 4)
            a2 = random.randint(0, 4)
            p1_val = random.uniform(-10.0, 10.0)
            p2_val = random.uniform(-5.0, 5.0)
            node.backup(
                action_p1=a1, action_p2=a2, child_value=(0.0, 0.0), self_value=(p1_val, p2_val)
            )

        # Verify that equivalent actions have same outcome index
        assert node.action_to_outcome(1, 0) == node.action_to_outcome(1, 4)
        assert node.action_to_outcome(2, 1) == node.action_to_outcome(2, 4)

    def test_invariant_stress_test(self) -> None:
        """Stress test with extreme equivalence (all actions equivalent for one player)."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        p1_effective = [4, 4, 4, 4, 4]  # All -> 4
        p2_effective = [0, 1, 4, 3, 4]  # 2,4 -> 4

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
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
            p1_val = random.uniform(-5.0, 5.0)
            p2_val = random.uniform(-3.0, 3.0)
            node.backup(
                action_p1=a1, action_p2=a2, child_value=(0.0, 0.0), self_value=(p1_val, p2_val)
            )

        # P1 should have single outcome
        assert node.n1 == 1
        # All P1 actions should map to outcome 0
        for a1 in range(5):
            assert node.action_to_outcome(1, a1) == 0


class TestUpdateEffectiveActions:
    """Tests for update_effective_actions method."""

    def test_resets_statistics_to_zero(self) -> None:
        """Updating effective actions should reset all statistics."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=1.0,
            nn_value_p2=0.5,
        )

        # Do some backups
        node.backup(action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(10.0, 5.0))
        node.backup(action_p1=1, action_p2=1, child_value=(0.0, 0.0), self_value=(20.0, 10.0))
        assert node.total_visits == 2

        # Update effective actions (introduce wall for P1)
        node.update_effective_actions(p1_effective=[4, 1, 2, 3, 4])

        # Statistics should be reset
        assert node.total_visits == 0

        n1, n2 = node.get_visit_counts()
        assert np.all(n1 == 0)
        assert np.all(n2 == 0)

    def test_reduced_matrices_resize_correctly(self) -> None:
        """Updating effective actions should resize reduced matrices."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        # Start with identity mapping (5x5 reduced)
        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )

        assert node.n1 == 5
        assert node.n2 == 5

        # Update P1 with blocked action (4x5 reduced)
        node.update_effective_actions(p1_effective=[4, 1, 2, 3, 4])
        assert node.n1 == 4  # Actions 0,4 merged
        assert node.n2 == 5

        # Update both players with walls
        node.update_effective_actions(
            p1_effective=[4, 1, 4, 4, 4],  # Only 1,4 unique
            p2_effective=[4, 4, 2, 4, 4],  # Only 2,4 unique
        )
        assert node.n1 == 2
        assert node.n2 == 2

    def test_action_to_outcome_mappings_update(self) -> None:
        """Verify action_to_outcome returns correct indices after update."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )

        # Initially identity: action i -> outcome i
        for i in range(5):
            assert node.action_to_outcome(1, i) == i
            assert node.action_to_outcome(2, i) == i

        # Update P1: actions 0,4 -> effective 4
        node.update_effective_actions(p1_effective=[4, 1, 2, 3, 4])

        # Now P1's outcomes are [1, 2, 3, 4] (sorted)
        # action 0 -> effective 4 -> outcome index 3
        # action 1 -> effective 1 -> outcome index 0
        # action 4 -> effective 4 -> outcome index 3
        assert node.action_to_outcome(1, 0) == node.action_to_outcome(1, 4)  # Same outcome
        assert node.action_to_outcome(1, 1) != node.action_to_outcome(1, 2)  # Different

    def test_p1_outcomes_and_p2_outcomes_update(self) -> None:
        """Verify p1_outcomes and p2_outcomes lists update correctly."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )

        # Initially all unique
        assert node.p1_outcomes == [0, 1, 2, 3, 4]
        assert node.p2_outcomes == [0, 1, 2, 3, 4]

        # Update with walls
        node.update_effective_actions(
            p1_effective=[4, 1, 2, 3, 4],  # 0->4
            p2_effective=[0, 4, 2, 3, 4],  # 1->4
        )

        assert node.p1_outcomes == [1, 2, 3, 4]  # 0 merged with 4
        assert node.p2_outcomes == [0, 2, 3, 4]  # 1 merged with 4


class TestMarginalVisitsExpanded:
    """Tests for get_marginal_visits_expanded method."""

    def test_initial_zeros(self, simple_node: MCTSNode) -> None:
        """Expanded visits should be zero initially."""
        n1, n2 = simple_node.get_marginal_visits_expanded()
        np.testing.assert_array_equal(n1, np.zeros(5))
        np.testing.assert_array_equal(n2, np.zeros(5))

    def test_visits_expanded_correctly(self, simple_node: MCTSNode) -> None:
        """Visits should expand to [5] space correctly."""
        simple_node.backup(action_p1=1, action_p2=2, child_value=(0.0, 0.0), self_value=(5.0, 3.0))
        simple_node.backup(action_p1=1, action_p2=2, child_value=(0.0, 0.0), self_value=(7.0, 4.0))

        n1, n2 = simple_node.get_marginal_visits_expanded()

        assert n1[1] == 2
        assert n2[2] == 2
        assert n1[0] == 0
        assert n2[0] == 0

    def test_with_equivalence(self) -> None:
        """Expanded visits with equivalence should only show canonical actions."""
        prior_p1 = np.ones(5) / 5
        prior_p2 = np.ones(5) / 5

        # P1: actions 0 and 4 equivalent
        node = MCTSNode(
            game_state=None,
            prior_policy_p1=prior_p1,
            prior_policy_p2=prior_p2,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            p1_effective=[4, 1, 2, 3, 4],
            p2_effective=[0, 1, 2, 3, 4],
        )

        node.backup(action_p1=0, action_p2=0, child_value=(0.0, 0.0), self_value=(5.0, 3.0))

        n1, n2 = node.get_marginal_visits_expanded()

        # Only canonical action 4 gets the visit count (action 0 is blocked)
        assert n1[0] == 0  # Not canonical
        assert n1[4] == 1  # Canonical
        assert n2[0] == 1
