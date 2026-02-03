"""Tests for MCTS Search implementation."""

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
    nn_payout = np.zeros((2, 5, 5))  # Bimatrix format: [p1_payoffs, p2_payoffs]
    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior,
        prior_policy_p2=prior,
        nn_payout_prediction=nn_payout,
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
        assert isinstance(result.payout_matrix, np.ndarray)
        assert isinstance(result.policy_p1, np.ndarray)
        assert isinstance(result.policy_p2, np.ndarray)

    def test_search_updates_root_visits(self, fake_tree: MCTSTree) -> None:
        """Search should increase visit counts on root."""
        initial_visits = fake_tree.root.total_visits

        search = DecoupledPUCTSearch(fake_tree, make_config(10))
        search.search()

        # Visits should have increased
        assert fake_tree.root.total_visits > initial_visits

    def test_search_returns_valid_nash(self, fake_tree: MCTSTree) -> None:
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
        nn_payout_full = np.zeros((2, 5, 5))  # Bimatrix format

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=nn_payout_full,
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
        # Create root with specific payout prediction
        prior = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # Only first two actions
        # Bimatrix: P1 gets +10, P2 gets -10 (zero-sum for simplicity)
        nn_payout = np.zeros((2, 5, 5))
        nn_payout[0] = 10.0  # P1's payoffs
        nn_payout[1] = -10.0  # P2's payoffs

        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),  # Root starts at 0
        )

        # Create game and tree with custom predict_fn
        game = FakeGame()

        def predict_fn(_: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return prior, prior, nn_payout

        tree = MCTSTree(game=game, root=root, gamma=1.0, predict_fn=predict_fn)  # type: ignore[arg-type]

        # Run one simulation
        search = DecoupledPUCTSearch(tree, make_config(1))
        search.search()

        # The root payout should have been updated with a value backed up from leaf
        assert tree.root.total_visits > 0

        # At least one action pair should have non-zero payout
        assert np.any(tree.root.payout_matrix != 0)


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
        nn_payout = np.zeros((2, 5, 5))  # Bimatrix format
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=nn_payout,
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

        # Blocked actions should have 0 probability in Nash result
        for i in range(5):
            if real_tree.root.p1_effective[i] != i:
                # This action is blocked (maps to STAY)
                assert result.policy_p1[i] == 0.0

            if real_tree.root.p2_effective[i] != i:
                assert result.policy_p2[i] == 0.0


class TestMakeResultIntegration:
    """Integration tests for _make_result() — full search pipeline."""

    def test_search_result_shapes(self) -> None:
        """SearchResult arrays have correct shapes after a real search."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        assert result.policy_p1.shape == (5,)
        assert result.policy_p2.shape == (5,)
        assert result.payout_matrix.shape == (2, 5, 5)
        assert result.action_visits.shape == (5, 5)

    def test_policies_sum_to_one(self) -> None:
        """Both acting and learning policies sum to ~1.0."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        assert result.policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.policy_p2.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.learning_policy_p1.sum() == pytest.approx(1.0, abs=1e-6)
        assert result.learning_policy_p2.sum() == pytest.approx(1.0, abs=1e-6)

    def test_blocked_actions_get_zero_probability(self) -> None:
        """Blocked actions get 0 probability in search result policies."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=50))
        result = search.search()

        for i in range(5):
            if root.p1_effective[i] != i:
                assert result.policy_p1[i] == 0.0
                assert result.learning_policy_p1[i] == 0.0
            if root.p2_effective[i] != i:
                assert result.policy_p2[i] == 0.0
                assert result.learning_policy_p2[i] == 0.0

    def test_payout_zeros_for_low_visits(self) -> None:
        """Payout matrix has zeros where visit count was too low."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)
        # Few simulations so some pairs will have < 2 visits
        search = DecoupledPUCTSearch(tree, DecoupledPUCTConfig(simulations=10))
        result = search.search()

        # Cells with 0 visits in the expanded visit matrix should have 0 payout
        for i in range(5):
            for j in range(5):
                if result.action_visits[i, j] == 0:
                    assert result.payout_matrix[0, i, j] == 0.0
                    assert result.payout_matrix[1, i, j] == 0.0

    def test_forced_playouts_produce_valid_visits(self) -> None:
        """With force_k > 0, search produces valid non-negative visit counts."""
        game = PyRat(width=5, height=5, cheese_count=3, seed=42)
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),
        )
        tree = MCTSTree(game=game, root=root, gamma=0.99)

        config = DecoupledPUCTConfig(simulations=50, force_k=2.0)
        search = DecoupledPUCTSearch(tree, config)
        result = search.search()

        # Pruned visits should be non-negative and have some mass
        assert np.all(result.action_visits >= 0)
        assert result.action_visits.sum() > 0
        assert result.action_visits.shape == (5, 5)


class TestBackupWithLeafValue:
    """Tests for backup with non-zero leaf value."""

    def test_backup_with_g_parameter(self, fake_tree: MCTSTree) -> None:
        """Tree.backup should use g parameter for leaf value."""
        # Create a child
        child, reward = fake_tree.make_move_from(fake_tree.root, 0, 0)

        # Backup with specific leaf value - tuple (p1_value, p2_value)
        path = [(fake_tree.root, 0, 0, (1.0, -1.0))]
        fake_tree.backup(path, g=(5.0, -5.0))

        # Value = reward + gamma * g = (1.0, -1.0) + 1.0 * (5.0, -5.0) = (6.0, -6.0)
        assert fake_tree.root.payout_matrix[0, 0, 0] == pytest.approx(6.0)  # P1
        assert fake_tree.root.payout_matrix[1, 0, 0] == pytest.approx(-6.0)  # P2

    def test_backup_with_discount(self) -> None:
        """Backup should apply gamma to leaf value."""
        game = FakeGame()
        prior = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=prior,
            prior_policy_p2=prior,
            nn_payout_prediction=np.zeros((2, 5, 5)),  # Bimatrix format
        )
        tree = MCTSTree(game=game, root=root, gamma=0.5)  # type: ignore[arg-type]

        # Create child
        child, _ = tree.make_move_from(root, 1, 1)

        # Backup with leaf value - tuple (p1_value, p2_value)
        path = [(root, 1, 1, (2.0, -2.0))]
        tree.backup(path, g=(10.0, -10.0))

        # Value = reward + gamma * g = (2.0, -2.0) + 0.5 * (10.0, -10.0) = (7.0, -7.0)
        assert root.payout_matrix[0, 1, 1] == pytest.approx(7.0)  # P1
        assert root.payout_matrix[1, 1, 1] == pytest.approx(-7.0)  # P2


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

    def test_returns_nn_payout_matrix(self, fake_tree: MCTSTree) -> None:
        """n_sims=0 returns the NN payout prediction unchanged."""
        root = fake_tree.root

        # The root's payout_matrix should be the NN prediction
        original_payout = root.payout_matrix.copy()

        search = DecoupledPUCTSearch(fake_tree, make_config(0))
        result = search.search()

        # Payout matrix should be unchanged (no MCTS backup)
        np.testing.assert_array_almost_equal(result.payout_matrix, original_payout)
