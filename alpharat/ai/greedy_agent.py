"""Greedy agent that moves toward the closest cheese using Dijkstra pathfinding."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from pyrat_engine.core.types import Coordinates, Direction

from alpharat.ai.base import Agent

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


class GreedyAgent(Agent):
    """Agent that moves toward the cheese reachable in minimum turns.

    Uses Dijkstra's algorithm to find the fastest path, accounting for
    walls (impassable) and mud (costs multiple turns).
    """

    def get_move(self, game: PyRat, player: int) -> int:
        """Move toward closest cheese by travel time (Dijkstra)."""
        pos = game.player1_position if player == 1 else game.player2_position

        cheeses = set(game.cheese_positions())
        if not cheeses:
            return Direction.STAY

        first_move = self._find_move_to_nearest_cheese(game, pos, cheeses)
        return first_move if first_move is not None else Direction.STAY

    def _find_move_to_nearest_cheese(
        self, game: PyRat, start: Coordinates, cheeses: set[Coordinates]
    ) -> int | None:
        """Dijkstra search to find first move toward nearest cheese by time.

        Args:
            game: PyRat game instance
            start: Starting position
            cheeses: Set of cheese positions

        Returns:
            First action to take, or None if no cheese reachable
        """
        # Movement matrix for mud costs (same regardless of player perspective)
        obs = game.get_observation(is_player_one=True)
        movement_matrix = obs.movement_matrix

        # Priority queue: (cost, counter, position, first_move)
        # counter breaks ties to avoid comparing Coordinates
        counter = 0
        pq: list[tuple[int, int, Coordinates, int | None]] = [(0, counter, start, None)]
        best_cost: dict[Coordinates, int] = {start: 0}

        while pq:
            cost, _, pos, first_move = heapq.heappop(pq)

            # Skip if we found a better path already
            if cost > best_cost.get(pos, float("inf")):
                continue

            # Found cheese — return the first move we took to get here
            if pos in cheeses:
                return first_move

            # Explore only valid moves (skips walls and out-of-bounds)
            for direction in game.get_valid_moves(pos):  # type: ignore[attr-defined]
                neighbor = pos.get_neighbor(direction)

                # Cost: 1 for normal move, or mud value for mud
                move_cost = movement_matrix[pos.x, pos.y, direction]
                edge_cost = 1 if move_cost == 0 else int(move_cost)
                new_cost = cost + edge_cost

                if new_cost < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = new_cost
                    # Track first move: if this is from start, it's `direction`
                    new_first_move = first_move if first_move is not None else int(direction)
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, neighbor, new_first_move))

        return None  # No cheese reachable

    @property
    def name(self) -> str:
        return "Greedy"
