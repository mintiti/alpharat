"""Single game execution for PyRat agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrat_engine.core.game import PyRat

if TYPE_CHECKING:
    from alpharat.ai.base import Agent


@dataclass
class GameResult:
    """Result of a single game.

    Attributes:
        p1_score: Player 1 (Rat) final score.
        p2_score: Player 2 (Python) final score.
        turns: Number of turns played.
        winner: 1 if P1 won, 2 if P2 won, 0 if draw.
    """

    p1_score: float
    p2_score: float
    turns: int
    winner: int  # 1=P1, 2=P2, 0=draw


def is_terminal(game: PyRat) -> bool:
    """Check if the game is over."""
    # Turn limit reached
    if game.turn >= game.max_turns:
        return True

    # All cheese collected
    remaining = len(game.cheese_positions())
    if remaining == 0:
        return True

    # Majority win
    total = game.player1_score + game.player2_score + remaining
    return game.player1_score > total / 2 or game.player2_score > total / 2


def play_game(
    agent_p1: Agent,
    agent_p2: Agent,
    *,
    game: PyRat | None = None,
    width: int = 15,
    height: int = 11,
    cheese_count: int = 21,
    max_turns: int = 300,
    seed: int | None = None,
    wall_density: float | None = None,
    mud_density: float | None = None,
) -> GameResult:
    """Play a single game between two agents.

    Args:
        agent_p1: Agent playing as Player 1 (Rat).
        agent_p2: Agent playing as Player 2 (Python).
        game: Pre-configured PyRat game. If provided, ignores other game params.
        width: Maze width (ignored if game provided).
        height: Maze height (ignored if game provided).
        cheese_count: Number of cheese pieces (ignored if game provided).
        max_turns: Maximum turns before game ends (ignored if game provided).
        seed: Random seed for maze generation (ignored if game provided).
        wall_density: Wall density 0.0-1.0, None for pyrat default (ignored if game provided).
        mud_density: Mud density 0.0-1.0, None for pyrat default (ignored if game provided).

    Returns:
        GameResult with scores and winner.
    """
    # Reset agents for new game
    agent_p1.reset()
    agent_p2.reset()

    # Create game if not provided
    if game is None:
        import random

        actual_seed = seed if seed is not None else random.randint(0, 2**31)

        # Build kwargs, only including density params if explicitly set
        kwargs: dict[str, int | float] = {
            "width": width,
            "height": height,
            "cheese_count": cheese_count,
            "max_turns": max_turns,
            "seed": actual_seed,
        }
        if wall_density is not None:
            kwargs["wall_density"] = wall_density
        if mud_density is not None:
            kwargs["mud_density"] = mud_density

        game = PyRat(**kwargs)  # type: ignore[arg-type]

    # Game loop
    while not is_terminal(game):
        # Both agents select actions
        action_p1 = agent_p1.get_move(game, player=1)
        action_p2 = agent_p2.get_move(game, player=2)

        # Execute simultaneous moves
        game.make_move(action_p1, action_p2)

    # Determine winner
    p1, p2 = game.player1_score, game.player2_score
    if p1 > p2:
        winner = 1
    elif p2 > p1:
        winner = 2
    else:
        winner = 0

    return GameResult(p1_score=p1, p2_score=p2, turns=game.turn, winner=winner)
