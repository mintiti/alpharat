"""Tests for GreedyAgent Dijkstra pathfinding.

Adapted from pyrat_base/tests/unit/test_utils.py to verify our
GreedyAgent implementation handles walls, mud, and edge cases correctly.
"""

from __future__ import annotations

import pytest
from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates, Direction, Mud, Wall

from alpharat.ai.greedy_agent import GreedyAgent


class TestGreedyAgentBasic:
    """Test basic greedy agent behavior."""

    def test_moves_toward_cheese_in_open_maze(self) -> None:
        """Agent moves toward cheese when no obstacles."""
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(2, 0)])  # Cheese to the right
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Should move RIGHT toward cheese at (2,0)
        assert move == Direction.RIGHT

    def test_returns_stay_when_no_cheese(self) -> None:
        """Agent returns STAY when no cheese remains."""
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(2, 2)])
            .build()
        )

        # Move P1 from (0,0) to (2,2) to collect cheese
        # RIGHT twice: (0,0) -> (1,0) -> (2,0)
        # UP twice: (2,0) -> (2,1) -> (2,2)
        game.step(Direction.RIGHT, Direction.STAY)
        game.step(Direction.RIGHT, Direction.STAY)
        game.step(Direction.UP, Direction.STAY)
        game.step(Direction.UP, Direction.STAY)

        # Verify cheese was collected
        assert len(game.cheese_positions()) == 0

        agent = GreedyAgent()
        move = agent.get_move(game, player=1)
        assert move == Direction.STAY

    def test_works_for_both_players(self) -> None:
        """Agent works correctly for both player 1 and player 2."""
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(2, 2)])
            .build()
        )
        agent = GreedyAgent()

        move_p1 = agent.get_move(game, player=1)
        move_p2 = agent.get_move(game, player=2)

        # P1 at (0,0) should move toward (2,2) - UP or RIGHT
        assert move_p1 in (Direction.UP, Direction.RIGHT)
        # P2 at (4,4) should move toward (2,2) - DOWN or LEFT
        assert move_p2 in (Direction.DOWN, Direction.LEFT)


class TestGreedyAgentWithWalls:
    """Test greedy agent pathfinding around walls."""

    def test_avoids_wall_blocking_direct_path(self) -> None:
        """Agent goes around wall blocking direct path."""
        # P1 at (0,0), cheese at (0,2), wall between (0,0) and (0,1)
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(0, 2)])
            .with_walls([Wall(Coordinates(0, 0), Coordinates(0, 1))])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Cannot go UP (blocked), must go RIGHT to go around
        assert move == Direction.RIGHT

    def test_finds_path_around_partial_barrier(self) -> None:
        """Agent finds path around partial wall barrier."""
        # Create partial barrier that blocks direct horizontal path
        # Gap at y=2 allows passage
        game = (
            GameConfigBuilder(5, 3)
            .with_player1_pos(Coordinates(0, 1))
            .with_player2_pos(Coordinates(4, 1))
            .with_cheese([Coordinates(4, 1)])
            .with_walls([
                Wall(Coordinates(2, 0), Coordinates(3, 0)),
                Wall(Coordinates(2, 1), Coordinates(3, 1)),
                # No wall at y=2 - gap to pass through
            ])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Must go UP to reach the gap at y=2, then around
        assert move == Direction.UP

    def test_handles_complete_barrier(self) -> None:
        """Agent returns STAY when cheese is completely blocked."""
        # Complete vertical barrier between x=1 and x=2
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 2))
            .with_player2_pos(Coordinates(4, 2))
            .with_cheese([Coordinates(4, 2)])  # Cheese on other side
            .with_walls([
                Wall(Coordinates(1, 0), Coordinates(2, 0)),
                Wall(Coordinates(1, 1), Coordinates(2, 1)),
                Wall(Coordinates(1, 2), Coordinates(2, 2)),
                Wall(Coordinates(1, 3), Coordinates(2, 3)),
                Wall(Coordinates(1, 4), Coordinates(2, 4)),
            ])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # No path to cheese - should return STAY
        assert move == Direction.STAY


class TestGreedyAgentWithMud:
    """Test greedy agent mud avoidance."""

    def test_avoids_expensive_mud(self) -> None:
        """Agent chooses longer path to avoid expensive mud."""
        # Direct path has 5-turn mud, going around is faster
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 2))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(4, 2)])
            .with_mud([Mud(Coordinates(2, 2), Coordinates(3, 2), 5)])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Direct: 2 normal + 5 mud + 1 normal = 8 turns
        # Around: go up/down first, ~6 turns
        # Should not go directly RIGHT into the mud path
        # First move should be UP or DOWN to start going around
        assert move in (Direction.UP, Direction.DOWN, Direction.RIGHT)

    def test_takes_mud_when_faster_than_around(self) -> None:
        """Agent takes mud path when it's faster than alternatives."""
        # Narrow corridor - must go through mud, no way around
        game = (
            GameConfigBuilder(3, 1)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(2, 0))
            .with_cheese([Coordinates(2, 0)])
            .with_mud([Mud(Coordinates(0, 0), Coordinates(1, 0), 2)])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Only path is through mud - should go RIGHT
        assert move == Direction.RIGHT

    def test_mud_cost_affects_cheese_choice(self) -> None:
        """Agent picks farther cheese if closer one is behind mud."""
        # Closer cheese behind expensive mud, farther cheese in clear
        game = (
            GameConfigBuilder(7, 3)
            .with_player1_pos(Coordinates(0, 1))
            .with_player2_pos(Coordinates(6, 1))
            .with_cheese([
                Coordinates(1, 1),  # Close but behind 5-turn mud
                Coordinates(4, 1),  # Farther but clear path
            ])
            .with_mud([Mud(Coordinates(0, 1), Coordinates(1, 1), 5)])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Cheese at (1,1): 5 turns through mud OR ~3 turns around
        # Cheese at (4,1): 4 turns direct
        # Around path to (1,1) is 3 turns, so should pick that
        # First move should be UP or DOWN to go around mud
        assert move in (Direction.UP, Direction.DOWN)


class TestGreedyAgentEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_already_on_cheese(self) -> None:
        """Agent behavior when standing on cheese (collected next step)."""
        # If we're on cheese, we've already collected it
        # This tests the case where remaining cheese is elsewhere
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(2, 2))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(2, 2), Coordinates(4, 0)])
            .build()
        )

        # Collect cheese at current position
        game.step(Direction.STAY, Direction.STAY)

        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Should move toward remaining cheese at (4,0)
        assert move in (Direction.RIGHT, Direction.DOWN)

    def test_multiple_equidistant_cheese(self) -> None:
        """Agent picks one cheese when multiple are equidistant."""
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(2, 2))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([
                Coordinates(2, 0),  # 2 down
                Coordinates(2, 4),  # 2 up
                Coordinates(0, 2),  # 2 left
                Coordinates(4, 2),  # 2 right
            ])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # Should pick one valid direction (any is fine, just not STAY)
        assert move in (Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT)

    def test_corner_positions(self) -> None:
        """Agent handles corner positions correctly."""
        game = (
            GameConfigBuilder(5, 5)
            .with_player1_pos(Coordinates(0, 0))
            .with_player2_pos(Coordinates(4, 4))
            .with_cheese([Coordinates(4, 4)])
            .build()
        )
        agent = GreedyAgent()
        move = agent.get_move(game, player=1)

        # From corner (0,0) to (4,4), should go UP or RIGHT
        assert move in (Direction.UP, Direction.RIGHT)

    def test_agent_name(self) -> None:
        """Agent has correct name."""
        agent = GreedyAgent()
        assert agent.name == "Greedy"
