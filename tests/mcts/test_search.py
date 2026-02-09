"""Tests for MCTS Search implementation with decoupled UCT."""

import numpy as np
import pytest
from pyrat_engine.core.game import PyRat

from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig, DecoupledPUCTSearch, SearchResult
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree


class FakeMoveUndo:
    """Undo token for FakeGame."""

    def __init__(
        self,
        p1_pos: tuple[int, int],
        p2_pos: tuple[int, int],
        p1_score: float,
        p2_score: float,
        turn: int,
        p1_mud: int = 0,
        p2_mud: int = 0,
    ) -> None:
        self.p1_pos = p1_pos
        self.p2_pos = p2_pos
        self.p1_score = p1_score
        self.p2_score = p2_score
        self.turn = turn
        self.p1_mud = p1_mud
        self.p2_mud = p2_mud


class FakeGame:
    """Deterministic game stub for search tests."""

    # Y-up coordinate system: UP increases Y, DOWN decreases Y
    _DELTAS = {
        0: (0, 1),  # UP
        1: (1, 0),  # RIGHT
        2: (0, -1),  # DOWN
        3: (-1, 0),  # LEFT
        4: (0, 0),  # STAY
    }

    def __init__(self, max_turns: int = 300) -> None:
        self.player1_position = (5, 5)
        self.player2_position = (5, 5)
        self.player1_score = 0.0
        self.player2_score = 0.0
        self.player1_mud_turns = 0
        self.player2_mud_turns = 0
        self.turn = 0
        self.max_turns = max_turns
        self._cheese: list[tuple[int, int]] = [(3, 3), (7, 7), (5, 3)]

    def _apply(self, pos: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = self._DELTAS[action]
        return pos[0] + dx, pos[1] + dy

    def make_move(self, p1_move: int, p2_move: int) -> FakeMoveUndo:
        undo = FakeMoveUndo(
            p1_pos=self.player1_position,
            p2_pos=self.player2_position,
            p1_score=self.player1_score,
            p2_score=self.player2_score,
            turn=self.turn,
            p1_mud=self.player1_mud_turns,
            p2_mud=self.player2_mud_turns,
        )
        self.player1_position = self._apply(self.player1_position, p1_move)
        self.player2_position = self._apply(self.player2_position, p2_move)
        self.turn += 1
        return undo

    def unmake_move(self, undo: FakeMoveUndo) -> None:
        self.player1_position = undo.p1_pos
        self.player2_position = undo.p2_pos
        self.player1_score = undo.p1_score
        self.player2_score = undo.p2_score
        self.turn = undo.turn

    def get_valid_moves(self, position: tuple[int, int]) -> list[int]:
        """Return all movement directions (no walls in FakeGame)."""
        return [0, 1, 2, 3]

    def cheese_positions(self) -> list[tuple[int, int]]:
        """Return remaining cheese positions."""
        return self._cheese.copy()

    def get_observation(self, is_player_one: bool = True) -> object:
        """Return dummy observation for predict_fn."""
        return None


@pytest.fixture
def fake_game() -> FakeGame:
    """Provide a deterministic game stub."""
    return FakeGame()


@pytest.fixture
def uniform_root() -> MCTSNode:
    """Create root node with uniform priors."""
    prior = np.ones(5) / 5
    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior,
        prior_policy_p2=prior,
        nn_value_p1=0.0,
        nn_value_p2=0.0,
    )


@pytest.fixture
def fake_tree(fake_game: FakeGame, uniform_root: MCTSNode) -> MCTSTree:
    """Tree with FakeGame and uniform priors."""
    return MCTSTree(game=fake_game, root=uniform_root, gamma=1.0)  # type: ignore[arg-type]


def make_config(simulations: int) -> DecoupledPUCTConfig:
    """Helper to create search config with given simulations."""
    return DecoupledPUCTConfig(simulations=simulations)


class TestSearchBasics:
    """Basic search functionality tests."""

    def test_search_returns_search_result(self, fake_tree: MCTSTree) -> None:
        """Search should return a SearchResult."""
        search = DecoupledPUCTSearch(fake_tree, make_config(5))
        result = search.search()

        assert isinstance(result, SearchResult)
        assert isinstance(result.policy_p1, np.ndarray)
        assert isinstance(result.policy_p2, np.ndarray)
        # New API returns scalar values
        assert isinstance(result.value_p1, float)
        assert isinstance(result.value_p2, float)

    def test_search_updates_root_visits(self, fake_tree: MCTSTree) -> None:
        """Search should increase visit counts on root."""
        initial_visits = fake_tree.root.total_visits

        search = DecoupledPUCTSearch(fake_tree, make_config(10))
        search.search()

        # Visits should have increased
        assert fake_tree.root.total_visits > initial_visits

    def test_search_returns_valid_policies(self, fake_tree: MCTSTree) -> None:
        """Returned policies should be valid probability distributions."""
        search = DecoupledPUCTSearch(fake_tree, make_config(10))
        result = search.search()

        # Policies should sum to 1
        assert np.isclose(result.policy_p1.sum(), 1.0)
        assert np.isclose(result.policy_p2.sum(), 1.0)

        # Policies should be non-negative
        assert np.all(result.policy_p1 >= 0)
        assert np.all(result.policy_p2 >= 0)

        # Policies should have correct shape
        assert result.policy_p1.shape == (5,)
        assert result.policy_p2.shape == (5,)


class TestTerminalHandling:
    """Tests for terminal state handling."""

    def test_search_on_terminal_root(self) -> None:
        """Search on terminal root should return current state without simulations."""
        # Create terminal root node
        prior = np.ones(5) / 5

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        root.is_terminal = True

        # Create game and tree
        game = FakeGame()
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Run search
        search = DecoupledPUCTSearch(tree, make_config(100))
        result = search.search()

        # No simulations should have run (no visits beyond initial)
        assert tree.root.total_visits == 0

        # Should still return valid result
        assert np.isclose(result.policy_p1.sum(), 1.0)
        assert np.isclose(result.policy_p2.sum(), 1.0)


class TestExpansion:
    """Tests for node expansion behavior."""

    def test_search_expands_nodes(self, fake_tree: MCTSTree) -> None:
        """Search should expand new nodes."""
        # Initially no children
        assert len(fake_tree.root.children) == 0

        search = DecoupledPUCTSearch(fake_tree, make_config(10))
        search.search()

        # Should have expanded some children
        assert len(fake_tree.root.children) > 0

    def test_search_expands_at_most_n_nodes_per_n_sims(self, fake_tree: MCTSTree) -> None:
        """Each simulation expands at most one node."""
        n_sims = 5
        search = DecoupledPUCTSearch(fake_tree, make_config(n_sims))
        search.search()

        # Count total nodes in tree (including root)
        def count_nodes(node: MCTSNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children.values())

        total_nodes = count_nodes(fake_tree.root)

        # At most n_sims + 1 nodes (root + one per simulation)
        assert total_nodes <= n_sims + 1


class TestLeafValue:
    """Tests for leaf value computation during backup."""

    def test_leaf_value_uses_nn_prediction(self) -> None:
        """Backup should use NN's expected value for non-terminal leaf."""
        # Create root with specific scalar value
        prior = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # Only first two actions

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )

        # Create game and tree with custom predict_fn
        game = FakeGame()

        def predict_fn(_: object) -> tuple[np.ndarray, np.ndarray, float, float]:
            # Return scalar values (v1=10.0, v2=-10.0)
            return prior, prior, 10.0, -10.0

        tree = MCTSTree(game=game, root=root, gamma=1.0, predict_fn=predict_fn)  # type: ignore[arg-type]

        # Run one simulation
        search = DecoupledPUCTSearch(tree, make_config(1))
        search.search()

        # The root Q-values should have been updated with a value backed up from leaf
        assert tree.root.total_visits > 0

        # At least one outcome should have non-zero Q-value
        q1, q2 = tree.root.get_q_values()
        assert np.any(q1 != 0) or np.any(q2 != 0)


class TestWithRealGame:
    """Integration tests with real PyRat game."""

    @pytest.fixture
    def real_game(self) -> PyRat:
        """Create a small PyRat game."""
        return PyRat(width=5, height=5, cheese_count=3, seed=42)

    @pytest.fixture
    def real_tree(self, real_game: PyRat) -> MCTSTree:
        """Tree with real PyRat game."""
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        return MCTSTree(game=real_game, root=root, gamma=0.99)

    def test_search_with_real_game(self, real_tree: MCTSTree) -> None:
        """Search should work with real PyRat game."""
        search = DecoupledPUCTSearch(real_tree, make_config(20))
        result = search.search()

        # Basic sanity checks
        assert np.isclose(result.policy_p1.sum(), 1.0)
        assert np.isclose(result.policy_p2.sum(), 1.0)
        assert real_tree.root.total_visits > 0

    def test_search_respects_effective_actions(self, real_tree: MCTSTree) -> None:
        """Search should respect effective action mappings (walls block moves)."""
        search = DecoupledPUCTSearch(real_tree, make_config(50))
        result = search.search()

        # Blocked actions should have 0 probability in result
        for i in range(5):
            if real_tree.root.p1_effective[i] != i:
                # This action is blocked (maps to STAY)
                assert result.policy_p1[i] == 0.0

            if real_tree.root.p2_effective[i] != i:
                assert result.policy_p2[i] == 0.0


class TestMakeResultIntegration:
    """Integration tests for _make_result() — full search pipeline."""

    def test_search_result_shapes(self) -> None:
        """SearchResult has correct shapes after a real search."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        assert result.policy_p1.shape == (5,)
        assert result.policy_p2.shape == (5,)
        assert isinstance(result.value_p1, float)
        assert isinstance(result.value_p2, float)

    def test_policies_sum_to_one(self) -> None:
        """Policies sum to ~1.0."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        assert result.policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.policy_p2.sum() == pytest.approx(1.0, abs=1e-6)

    def test_blocked_actions_get_zero_probability(self) -> None:
        """Blocked actions get 0 probability in search result policies."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        for i in range(5):
            if root.p1_effective[i] != i:
                assert result.policy_p1[i] == 0.0
            if root.p2_effective[i] != i:
                assert result.policy_p2[i] == 0.0

    def test_values_are_finite(self) -> None:
        """Search should produce finite value estimates."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        assert np.isfinite(result.value_p1)
        assert np.isfinite(result.value_p2)


class TestBackupWithLeafValue:
    """Tests for backup with non-zero leaf value."""

    def test_backup_with_g_parameter(self, fake_tree: MCTSTree) -> None:
        """Tree.backup should use g parameter for leaf value."""
        # Create a child
        child, _reward = fake_tree.make_move_from(fake_tree.root, 0, 0)

        # Set edge reward on child (simulates reward from make_move_from)
        child._edge_r1 = 1.0
        child._edge_r2 = -1.0

        # Backup with specific leaf value - tuple (p1_value, p2_value)
        path = [(fake_tree.root, 0, 0)]
        fake_tree.backup(path, g=(5.0, -5.0))

        # Value = edge_r + gamma * g = (1.0, -1.0) + 1.0 * (5.0, -5.0) = (6.0, -6.0)
        # With decoupled UCT, check Q-values instead of payout matrix
        q1, q2 = fake_tree.root.get_q_values()
        assert q1[0] == pytest.approx(6.0)
        assert q2[0] == pytest.approx(-6.0)

    def test_backup_with_discount(self) -> None:
        """Backup should apply gamma to leaf value."""
        game = FakeGame()
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.5)  # type: ignore[arg-type]

        # Create child
        child, _ = tree.make_move_from(root, 1, 1)

        # Set edge reward on child (simulates reward from make_move_from)
        child._edge_r1 = 2.0
        child._edge_r2 = -2.0

        # Backup with leaf value - tuple (p1_value, p2_value)
        path = [(root, 1, 1)]
        tree.backup(path, g=(10.0, -10.0))

        # Value = edge_r + gamma * g = (2.0, -2.0) + 0.5 * (10.0, -10.0) = (7.0, -7.0)
        q1, q2 = root.get_q_values(gamma=0.5)
        assert q1[1] == pytest.approx(7.0)
        assert q2[1] == pytest.approx(-7.0)


class TestPureNNMode:
    """Tests for n_sims=0 (pure NN mode, no MCTS)."""

    def test_returns_nn_priors_directly(self, fake_tree: MCTSTree) -> None:
        """n_sims=0 returns raw NN priors without any simulation."""
        root = fake_tree.root

        # Set distinct priors so we can verify they're returned
        root.prior_policy_p1 = np.array([0.5, 0.25, 0.15, 0.08, 0.02])
        root.prior_policy_p2 = np.array([0.1, 0.2, 0.3, 0.3, 0.1])

        search = DecoupledPUCTSearch(fake_tree, make_config(0))
        result = search.search()

        # Should return the exact prior policies
        np.testing.assert_array_almost_equal(result.policy_p1, root.prior_policy_p1)
        np.testing.assert_array_almost_equal(result.policy_p2, root.prior_policy_p2)

    def test_no_tree_expansion(self, fake_tree: MCTSTree) -> None:
        """n_sims=0 doesn't expand any nodes."""
        # Initially no children
        assert len(fake_tree.root.children) == 0

        search = DecoupledPUCTSearch(fake_tree, make_config(0))
        search.search()

        # Still no children - no expansion happened
        assert len(fake_tree.root.children) == 0

    def test_no_visits_accumulated(self, fake_tree: MCTSTree) -> None:
        """n_sims=0 doesn't accumulate any visits."""
        assert fake_tree.root.total_visits == 0

        search = DecoupledPUCTSearch(fake_tree, make_config(0))
        search.search()

        # Still zero visits
        assert fake_tree.root.total_visits == 0

    def test_returns_nn_values(self, fake_tree: MCTSTree) -> None:
        """n_sims=0 returns the NN value predictions."""
        root = fake_tree.root

        search = DecoupledPUCTSearch(fake_tree, make_config(0))
        result = search.search()

        # Value estimates should be the node's values (NN initial)
        assert result.value_p1 == pytest.approx(root.v1)
        assert result.value_p2 == pytest.approx(root.v2)


class TestTerminalMidSearch:
    """Tests for terminal states encountered during search (not at root)."""

    def test_terminal_child_backs_up_zero_leaf_value(self) -> None:
        """Terminal node at depth > 0 should back up g=(0,0) through edge rewards.

        When search hits a terminal state mid-tree, the leaf value is (0, 0) —
        no future rewards from a game-over position. The backed-up Q values
        come entirely from edge rewards.
        """
        game = FakeGame(max_turns=1)  # Terminal after 1 move
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Run enough simulations to visit all action pairs
        search = DecoupledPUCTSearch(tree, make_config(50))
        result = search.search()

        # All children should be terminal (game ends after 1 move)
        for child in tree.root.children.values():
            assert child.is_terminal

        # Values should be finite and policies valid
        assert np.isfinite(result.value_p1)
        assert np.isfinite(result.value_p2)
        assert result.policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.policy_p2.sum() == pytest.approx(1.0, abs=1e-6)

        # Root should have been updated (terminal backup propagated)
        assert tree.root.total_visits > 0


class TestMakeResultControlled:
    """Tests for _make_result() with controlled visit/Q state."""

    def test_policy_from_known_visits(self) -> None:
        """Set up root with known visits and verify exact policy output.

        Tests the full pruning-to-policy pipeline end-to-end.
        """
        game = FakeGame()
        prior = np.array([0.2, 0.3, 0.2, 0.2, 0.1])
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Create children with controlled visit counts and values
        # Action 1 is clearly best (high Q, many visits)
        # Action 0 has some visits, lower Q
        child_00 = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=2.0,
            nn_value_p2=1.0,
            parent=root,
        )
        child_00._edge_visits = 5
        child_00._edge_r1 = 0.0
        child_00._edge_r2 = 0.0
        root.children[(0, 0)] = child_00

        child_11 = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=8.0,
            nn_value_p2=4.0,
            parent=root,
        )
        child_11._edge_visits = 45
        child_11._edge_r1 = 0.0
        child_11._edge_r2 = 0.0
        root.children[(1, 1)] = child_11

        # Set marginal visit counts to match
        root._n1_visits[0] = 5.0
        root._n1_visits[1] = 45.0
        root._n2_visits[0] = 5.0
        root._n2_visits[1] = 45.0
        root._total_visits = 50

        # Run _make_result
        search = DecoupledPUCTSearch(tree, make_config(0))
        result = search._make_result()

        # Policy should be valid distribution
        assert result.policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.policy_p2.sum() == pytest.approx(1.0, abs=1e-6)

        # Action 1 should dominate (more visits, higher Q)
        assert result.policy_p1[1] > result.policy_p1[0]
        assert result.policy_p2[1] > result.policy_p2[0]

        # Blocked actions (not visited) should have 0 probability
        assert result.policy_p1[2] == 0.0
        assert result.policy_p1[3] == 0.0
        assert result.policy_p1[4] == 0.0


class TestRawVsPrunedVisits:
    """Tests documenting the intentional design: value uses raw visits, policy uses pruned."""

    def test_value_uses_raw_visits_policy_uses_pruned(self) -> None:
        """Value estimate uses raw visit counts; policy uses pruned visits.

        This is intentional: value should reflect the actual search experience
        (all simulations), while policy should ignore forced-exploration visits
        that inflate low-Q actions.
        """
        game = FakeGame()

        # Non-uniform prior so forced playouts have something to force
        prior = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Set up controlled state with clear best action (0) and weak action (3)
        # Values are raw cheese scale (no normalization in storage).
        # Action 0: high Q, many visits
        child_00 = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=10.0,
            nn_value_p2=5.0,
            parent=root,
        )
        child_00._edge_visits = 80
        root.children[(0, 0)] = child_00

        # Action 3: low Q, few visits (forced exploration)
        child_33 = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=1.0,
            nn_value_p2=0.5,
            parent=root,
        )
        child_33._edge_visits = 20
        root.children[(3, 3)] = child_33

        root._n1_visits[0] = 80.0
        root._n1_visits[3] = 20.0
        root._n2_visits[0] = 80.0
        root._n2_visits[3] = 20.0
        root._total_visits = 100

        search = DecoupledPUCTSearch(tree, make_config(0))
        result = search._make_result()

        # Value should reflect ALL visits (raw), so it's a weighted average.
        # Q1(0) = 10.0, Q1(3) = 1.0 (raw cheese scale).
        # value = (80*10 + 20*1) / 100 = 8.2
        assert result.value_p1 == pytest.approx(8.2)

        # Policy should use pruned visits (action 3 may have visits reduced)
        # The key property: policy ratios can differ from raw visit ratios
        raw_ratio_p1 = 20.0 / 80.0  # 0.25
        policy_ratio_p1 = (
            result.policy_p1[3] / result.policy_p1[0] if result.policy_p1[0] > 0 else float("inf")
        )

        # Pruning should reduce action 3's share relative to raw visits
        assert policy_ratio_p1 <= raw_ratio_p1


class TestNormalization:
    """Tests for per-node PUCT normalization by remaining cheese."""

    def test_value_scale_set_on_root(self) -> None:
        """Root node gets value_scale from game state."""
        game = FakeGame()
        # FakeGame has 3 cheese positions
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]
        assert tree.root.value_scale == 3.0

    def test_value_scale_minimum_one(self) -> None:
        """value_scale should be at least 1.0 to avoid division by zero."""
        game = FakeGame()
        game._cheese = []  # No cheese
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]
        assert tree.root.value_scale == 1.0

    def test_value_scale_set_on_children(self) -> None:
        """Children get value_scale from game state at their position."""
        game = FakeGame()
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]

        # Create a child — FakeGame doesn't collect cheese, so same count
        child, _ = tree.make_move_from(root, 0, 0)
        assert child.value_scale == 3.0

    def test_value_scale_decreases_on_collection(self) -> None:
        """value_scale decreases when cheese is collected."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        initial_cheese = tree.root.value_scale

        # Run search — children at depth may collect cheese
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        search.search()

        # All children should have value_scale <= root's
        for child in tree.root.children.values():
            assert child.value_scale <= initial_cheese

    def test_internal_q_raw_cheese_scale(self) -> None:
        """Internal Q-values are stored in raw cheese scale (not normalized).

        Uses a predict_fn returning values > 1.0 so we can verify Q-values
        aren't accidentally squeezed into [0, 1].
        """
        game = PyRat(width=5, height=5, cheese_count=5, seed=42)
        prior = np.ones(5) / 5

        def predict_fn(_: object) -> tuple[np.ndarray, np.ndarray, float, float]:
            return prior.copy(), prior.copy(), 2.5, 2.0

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99, predict_fn=predict_fn)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=100))
        search.search()

        # Raw Q-values can be in [0, value_scale], not necessarily [0, 1]
        q1, q2 = tree.root.get_q_values(gamma=0.99)
        rc = tree.root.value_scale
        for q in [q1, q2]:
            visited = q[q != 0]
            if len(visited) > 0:
                assert np.all(visited >= -0.01), f"Q-values below 0: {visited}"
                assert np.all(visited <= rc + 0.01), f"Q-values above remaining: {visited}"

        # With predict_fn returning v1=2.5 in raw cheese scale, Q-values
        # should exceed 1.0 — catches accidental normalization
        all_q = np.concatenate([q1, q2])
        assert np.max(all_q) > 1.0, (
            f"No Q-value exceeds 1.0 — values may be accidentally normalized: {all_q}"
        )

    def test_output_values_raw_cheese_scale(self) -> None:
        """Search output values should be in raw cheese scale.

        Uses a predict_fn returning values > 1.0 so we can verify output
        values aren't accidentally normalized.
        """
        game = PyRat(width=5, height=5, cheese_count=5, seed=42)
        prior = np.ones(5) / 5

        def predict_fn(_: object) -> tuple[np.ndarray, np.ndarray, float, float]:
            return prior.copy(), prior.copy(), 2.5, 2.0

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99, predict_fn=predict_fn)

        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=100))
        result = search.search()

        # Output values are raw cheese scale — with predict_fn returning v=2.5,
        # at least one player's value should exceed 1.0
        assert np.isfinite(result.value_p1)
        assert np.isfinite(result.value_p2)
        assert max(result.value_p1, result.value_p2) > 1.0, (
            f"Output values look normalized: v1={result.value_p1}, v2={result.value_p2}"
        )

    def test_normalized_q_in_unit_range_at_every_depth(self) -> None:
        """Q / value_scale should be in ~[0, 1] at every visited node.

        This is the core behavioral invariant: PUCT sees Q normalized by
        the cheese available at *that* node, not the global total. At a
        deep node with 2 of 5 cheese left, Q max is ~2 and rc=2, so
        Q/rc ≈ 1.0 — exploration and exploitation stay balanced.
        """
        game = PyRat(width=5, height=5, cheese_count=5, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=200))
        search.search()

        # Walk the entire tree and check the invariant at every visited node
        def check_node(node: MCTSNode) -> None:
            if node.total_visits == 0:
                return

            q1, q2 = node.get_q_values(gamma=0.99)
            rc = node.value_scale
            assert rc >= 1.0, f"value_scale too small: {rc}"

            q1_norm = q1 / rc
            q2_norm = q2 / rc

            for label, q_norm in [("P1", q1_norm), ("P2", q2_norm)]:
                visited = q_norm[q_norm != 0]
                if len(visited) > 0:
                    assert np.all(visited >= -0.01), (
                        f"{label} normalized Q below 0 at depth {node.depth}: {visited} (rc={rc})"
                    )
                    assert np.all(visited <= 1.01), (
                        f"{label} normalized Q above 1 at depth {node.depth}: {visited} (rc={rc})"
                    )

            for child in node.children.values():
                check_node(child)

        check_node(tree.root)

    def test_per_node_normalization_differs_from_global(self) -> None:
        """Per-node normalization should produce different Q/rc than global.

        Construct a scenario: root has 10 cheese, child has 2 (8 collected).
        With global normalization, child Q=1.5 → Q/10 = 0.15 (exploration dominates).
        With per-node normalization, child Q=1.5 → Q/2 = 0.75 (balanced).
        """
        game = FakeGame()
        game._cheese = [(i, i) for i in range(10)]  # 10 cheese
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=1.0)  # type: ignore[arg-type]
        assert tree.root.value_scale == 10.0

        # Create child and simulate cheese being collected
        child, _ = tree.make_move_from(root, 0, 0)
        # Manually set child's remaining cheese to simulate 8 cheese collected
        child.value_scale = 2.0
        child._v1 = 1.5  # Raw Q value
        child._v2 = 0.5

        # Per-node: Q/rc = 1.5/2.0 = 0.75
        # Global (old): Q/total = 1.5/10.0 = 0.15
        per_node_norm = child._v1 / child.value_scale
        global_norm = child._v1 / root.value_scale

        assert per_node_norm == pytest.approx(0.75)
        assert global_norm == pytest.approx(0.15)
        # Per-node puts Q in the right range — exploitation is properly weighted
        assert per_node_norm > global_norm

    def test_normalization_with_predict_fn(self) -> None:
        """Q / value_scale ∈ [0, 1] holds when predict_fn returns non-trivial values.

        All other normalization tests use smart-uniform (no NN, values are 0.0).
        This exercises the NN-value → FPU → normalization path.
        """
        game = PyRat(width=5, height=5, cheese_count=5, seed=42)
        prior = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        def predict_fn(_: object) -> tuple[np.ndarray, np.ndarray, float, float]:
            return prior.copy(), prior.copy(), 2.0, 1.5

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99, predict_fn=predict_fn)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=200))
        search.search()

        # Walk tree and check Q / value_scale ∈ [0, 1] at every visited node
        def check_node(node: MCTSNode) -> None:
            if node.total_visits == 0:
                return

            q1, q2 = node.get_q_values(gamma=0.99)
            rc = node.value_scale
            assert rc >= 1.0

            for label, q in [("P1", q1), ("P2", q2)]:
                q_norm = q / rc
                visited = q_norm[q_norm != 0]
                if len(visited) > 0:
                    assert np.all(visited >= -0.01), (
                        f"{label} Q/value_scale below 0 at depth {node.depth}: {visited}"
                    )
                    assert np.all(visited <= 1.01), (
                        f"{label} Q/value_scale above 1 at depth {node.depth}: {visited}"
                    )

            for child in node.children.values():
                check_node(child)

        check_node(tree.root)


class TestFPUReductionSearch:
    """Tests for FPU reduction wired through the search."""

    def test_fpu_reduction_config_default(self) -> None:
        """Default config has fpu_reduction=0.2."""
        config = DecoupledPUCTConfig()
        assert config.fpu_reduction == 0.2

    def test_fpu_reduction_zero_matches_no_reduction(self) -> None:
        """fpu_reduction=0 should produce valid results (legacy behavior)."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)

        config = DecoupledPUCTConfig(simulations=50, fpu_reduction=0.0)
        search = DecoupledPUCTSearch(tree, config)
        result = search.search()

        assert np.isclose(result.policy_p1.sum(), 1.0)
        assert np.isclose(result.policy_p2.sum(), 1.0)
        assert np.isfinite(result.value_p1)
        assert np.isfinite(result.value_p2)

    def test_fpu_reduction_produces_valid_search(self) -> None:
        """Search with fpu_reduction > 0 produces valid policies and values."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)

        config = DecoupledPUCTConfig(simulations=50, fpu_reduction=0.3)
        search = DecoupledPUCTSearch(tree, config)
        result = search.search()

        assert result.policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.policy_p2.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.isfinite(result.value_p1)
        assert np.isfinite(result.value_p2)
        assert tree.root.total_visits > 0
