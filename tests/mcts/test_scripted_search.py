"""Scripted MCTS search tests with predictable outcomes.

These tests use custom PyRat game setups where optimal play is known,
verifying that MCTS converges to correct Nash equilibria.
"""

import numpy as np
import pytest
from pyrat_engine.core.game import PyRat
from pyrat_engine.core.types import Direction

from alpharat.mcts.node import MCTSNode
from alpharat.mcts.search import MCTSSearch
from alpharat.mcts.tree import MCTSTree

# High simulation count - PyRat is fast (Rust backend)
# With 10k sims on tiny mazes and gamma=1, we converge to exact values.
N_SIMS = 10_000
GAMMA = 1.0
VALUE_TOL = 0.01


def create_uniform_root(game: PyRat) -> MCTSNode:
    """Create root node with uniform priors."""
    prior = np.ones(5) / 5
    nn_payout = np.zeros((5, 5))
    return MCTSNode(
        game_state=None,
        prior_policy_p1=prior,
        prior_policy_p2=prior,
        nn_payout_prediction=nn_payout,
    )


class TestClearWinner:
    """Scenario 1: P1 adjacent to cheese, P2 far away."""

    @pytest.fixture
    def game(self) -> PyRat:
        """P1 one step above cheese, P2 in far corner."""
        return PyRat.create_custom(
            width=5,
            height=5,
            cheese=[(2, 2)],
            player1_pos=(2, 1),
            player2_pos=(4, 4),
            symmetric=False,
        )

    def test_p1_goes_up(self, game: PyRat) -> None:
        """P1 should strongly prefer UP to grab cheese.

        Note: In PyRat, UP increases y. P1 at (2,1), cheese at (2,2).
        """
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # P1 should go UP (pure strategy)
        assert result.policy_p1[Direction.UP] == 1.0, (
            f"Expected policy_p1[UP] == 1.0, got {result.policy_p1[Direction.UP]:.6f}\n"
            f"Full policy: {result.policy_p1}"
        )

        # Value should be exactly 1 (P1 gets the only cheese)
        value = result.policy_p1 @ result.payout_matrix @ result.policy_p2
        assert value == pytest.approx(1.0, abs=VALUE_TOL), f"Expected value ≈ 1.0, got {value:.6f}"


class TestNonCompetingCheeses:
    """Scenario 2: Each player has their own nearby cheese."""

    @pytest.fixture
    def game(self) -> PyRat:
        """P1 near (1,0), P2 near (3,4), no conflict."""
        return PyRat.create_custom(
            width=5,
            height=5,
            cheese=[(1, 0), (3, 4)],
            player1_pos=(0, 0),
            player2_pos=(4, 4),
            symmetric=False,
        )

    def test_each_goes_to_own_cheese(self, game: PyRat) -> None:
        """Each player should go toward their nearest cheese."""
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # P1 should go RIGHT toward (1,0)
        assert result.policy_p1[Direction.RIGHT] == 1.0, (
            f"Expected policy_p1[RIGHT] == 1.0, got {result.policy_p1[Direction.RIGHT]:.6f}\n"
            f"Full policy: {result.policy_p1}"
        )

        # P2 should go LEFT toward (3,4)
        assert result.policy_p2[Direction.LEFT] == 1.0, (
            f"Expected policy_p2[LEFT] == 1.0, got {result.policy_p2[Direction.LEFT]:.6f}\n"
            f"Full policy: {result.policy_p2}"
        )

        # Value should be exactly 0 (each gets 1 cheese)
        value = result.policy_p1 @ result.payout_matrix @ result.policy_p2
        assert value == pytest.approx(0.0, abs=VALUE_TOL), f"Expected value ≈ 0.0, got {value:.6f}"


class TestMirrorSymmetric:
    """Scenario 3: Mirror symmetric setup.

    Setup: P1 and P2 on same row, each closer to opposite cheese.
    - P1 at (1, 2), cheese at (0, 2) is 1 step LEFT
    - P2 at (3, 2), cheese at (4, 2) is 1 step RIGHT

    This is mirror symmetric: each player should go to their closest cheese.
    Equivalent to "non-competing" but tests the mirror symmetry property.
    """

    @pytest.fixture
    def game(self) -> PyRat:
        """P1 closer to left cheese, P2 closer to right cheese."""
        return PyRat.create_custom(
            width=5,
            height=5,
            cheese=[(0, 2), (4, 2)],
            player1_pos=(1, 2),
            player2_pos=(3, 2),
            symmetric=False,
        )

    def test_each_goes_to_closest(self, game: PyRat) -> None:
        """Each player should go to their closest cheese."""
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # P1 should go LEFT toward (0, 2)
        assert result.policy_p1[Direction.LEFT] == 1.0, (
            f"Expected policy_p1[LEFT] == 1.0, got {result.policy_p1[Direction.LEFT]:.6f}\n"
            f"Full policy: {result.policy_p1}"
        )

        # P2 should go RIGHT toward (4, 2)
        assert result.policy_p2[Direction.RIGHT] == 1.0, (
            f"Expected policy_p2[RIGHT] == 1.0, got {result.policy_p2[Direction.RIGHT]:.6f}\n"
            f"Full policy: {result.policy_p2}"
        )

        # Value should be exactly 0 (symmetric game, each gets one cheese)
        value = result.policy_p1 @ result.payout_matrix @ result.policy_p2
        assert value == pytest.approx(0.0, abs=VALUE_TOL), f"Expected value ≈ 0.0, got {value:.6f}"


class TestBlockedDirections:
    """Scenario 4: Walls block certain directions."""

    @pytest.fixture
    def game(self) -> PyRat:
        """P1 at (1,1) with walls blocking UP and LEFT, cheese to the RIGHT."""
        return PyRat.create_custom(
            width=5,
            height=5,
            walls=[
                ((1, 1), (1, 0)),  # Wall blocking UP from (1,1)
                ((1, 1), (0, 1)),  # Wall blocking LEFT from (1,1)
            ],
            cheese=[(2, 1)],
            player1_pos=(1, 1),
            player2_pos=(4, 4),
            symmetric=False,
        )

    def test_blocked_actions_zero_probability(self, game: PyRat) -> None:
        """Blocked actions must have exactly 0 probability."""
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # Blocked actions must be exactly 0
        assert result.policy_p1[Direction.UP] == 0.0, (
            f"Expected policy_p1[UP] == 0.0 (blocked), got {result.policy_p1[Direction.UP]}"
        )
        assert result.policy_p1[Direction.LEFT] == 0.0, (
            f"Expected policy_p1[LEFT] == 0.0 (blocked), got {result.policy_p1[Direction.LEFT]}"
        )

        # P1 should go RIGHT toward cheese
        assert result.policy_p1[Direction.RIGHT] == 1.0, (
            f"Expected policy_p1[RIGHT] == 1.0, got {result.policy_p1[Direction.RIGHT]:.6f}\n"
            f"Full policy: {result.policy_p1}"
        )


class TestMatchingPenniesNash:
    """Sanity check: verify Nash solver finds 50/50 for matching pennies.

    This doesn't use PyRat — it directly tests that our Nash computation
    correctly finds mixed equilibria when they exist.

    Matching pennies structure:
    - Same choice → P1 wins (+1)
    - Different choice → P2 wins (-1)

    Matrix: [[+1, -1], [-1, +1]]
    Unique Nash: both players 50/50, value = 0
    """

    def test_matching_pennies_gives_fifty_fifty(self) -> None:
        """Nash solver should find 50/50 for matching pennies."""
        from alpharat.mcts.nash import compute_nash_equilibrium, compute_nash_value

        matching_pennies = np.array(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )

        p1_strat, p2_strat = compute_nash_equilibrium(matching_pennies)

        # Both should be 50/50
        assert p1_strat[0] == pytest.approx(0.5, abs=0.01), (
            f"Expected P1[0] = 0.5, got {p1_strat[0]:.3f}"
        )
        assert p1_strat[1] == pytest.approx(0.5, abs=0.01), (
            f"Expected P1[1] = 0.5, got {p1_strat[1]:.3f}"
        )
        assert p2_strat[0] == pytest.approx(0.5, abs=0.01), (
            f"Expected P2[0] = 0.5, got {p2_strat[0]:.3f}"
        )
        assert p2_strat[1] == pytest.approx(0.5, abs=0.01), (
            f"Expected P2[1] = 0.5, got {p2_strat[1]:.3f}"
        )

        # Value should be 0
        value = compute_nash_value(matching_pennies, p1_strat, p2_strat)
        assert value == pytest.approx(0.0, abs=0.01), f"Expected value = 0, got {value:.3f}"


class TestSymmetricGameValue:
    """Test that symmetric games have value 0 regardless of equilibrium chosen.

    Setup: P1 and P2 at opposite corners, two corner cheeses equidistant.
    The game is symmetric under player swap + board rotation.

    Multiple pure Nash equilibria exist (each player goes to "their" corner),
    but all equilibria have value 0.
    """

    @pytest.fixture
    def game(self) -> PyRat:
        """Symmetric corner setup: contested corners."""
        return PyRat.create_custom(
            width=5,
            height=5,
            cheese=[(0, 4), (4, 0)],  # Two corner cheeses
            player1_pos=(0, 0),
            player2_pos=(4, 4),
            symmetric=False,
        )

    def test_symmetric_game_has_zero_value(self, game: PyRat) -> None:
        """Value should be 0 for symmetric game regardless of equilibrium."""
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # Value should be 0 (symmetric game)
        value = result.policy_p1 @ result.payout_matrix @ result.policy_p2
        assert value == pytest.approx(0.0, abs=VALUE_TOL), (
            f"Expected value ≈ 0.0 for symmetric game, got {value:.6f}"
        )

        # Players should go toward cheese (UP or RIGHT for P1, DOWN or LEFT for P2)
        p1_toward_cheese = result.policy_p1[Direction.UP] + result.policy_p1[Direction.RIGHT]
        assert p1_toward_cheese == 1.0, (
            f"Expected P1 to go toward cheese, got UP+RIGHT={p1_toward_cheese:.6f}"
        )


class TestRaceToSameCheese:
    """Scenario 5: Both players equidistant from single cheese.

    Game theory analysis:
    - If both go: split cheese (0.5 each), score diff = 0
    - If P1 goes alone: P1 gets 1, score diff = +1
    - If P2 goes alone: P2 gets 1, score diff = -1
    - If neither: continue (suboptimal)

    This is NOT a mixed strategy game. P1 playing RIGHT dominates:
    - RIGHT guarantees ≥0 (shared or alone)
    - Any other action risks -1 if P2 goes LEFT

    Nash equilibrium: (P1: RIGHT, P2: LEFT) — both go, split cheese.
    """

    @pytest.fixture
    def game(self) -> PyRat:
        """P1 one step LEFT of cheese, P2 one step RIGHT."""
        return PyRat.create_custom(
            width=5,
            height=5,
            cheese=[(2, 2)],
            player1_pos=(1, 2),
            player2_pos=(3, 2),
            symmetric=False,
        )

    def test_both_go_to_cheese(self, game: PyRat) -> None:
        """Nash equilibrium is both players going toward cheese (pure strategy)."""
        np.random.seed(42)

        root = create_uniform_root(game)
        tree = MCTSTree(game=game, root=root, gamma=GAMMA)
        search = MCTSSearch(tree, n_sims=N_SIMS)
        result = search.search()

        # P1 should go RIGHT (dominant strategy)
        assert result.policy_p1[Direction.RIGHT] == 1.0, (
            f"Expected policy_p1[RIGHT] == 1.0, got {result.policy_p1[Direction.RIGHT]:.6f}\n"
            f"Full policy: {result.policy_p1}"
        )

        # P2 should go LEFT (best response to P1's RIGHT)
        assert result.policy_p2[Direction.LEFT] == 1.0, (
            f"Expected policy_p2[LEFT] == 1.0, got {result.policy_p2[Direction.LEFT]:.6f}\n"
            f"Full policy: {result.policy_p2}"
        )

        # Value should be exactly 0 (both get half the cheese)
        value = result.policy_p1 @ result.payout_matrix @ result.policy_p2
        assert value == pytest.approx(0.0, abs=VALUE_TOL), f"Expected value ≈ 0.0, got {value:.6f}"
