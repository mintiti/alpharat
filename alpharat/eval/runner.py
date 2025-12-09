"""Multi-game evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alpharat.eval.game import GameResult, play_game

if TYPE_CHECKING:
    from alpharat.ai.base import Agent


@dataclass
class EvalResult:
    """Aggregated results from multiple games.

    Attributes:
        n_games: Total games played.
        agent1_wins: Games won by agent 1.
        agent2_wins: Games won by agent 2.
        draws: Games ending in a draw.
        agent1_avg_score: Average score for agent 1.
        agent2_avg_score: Average score for agent 2.
        games: List of individual game results.
    """

    n_games: int
    agent1_wins: int
    agent2_wins: int
    draws: int
    agent1_avg_score: float
    agent2_avg_score: float
    games: list[GameResult]

    @property
    def agent1_win_rate(self) -> float:
        """Win rate for agent 1 (excluding draws)."""
        wins = self.agent1_wins + self.agent2_wins
        return self.agent1_wins / wins if wins > 0 else 0.5

    def summary(self, agent1_name: str = "Agent1", agent2_name: str = "Agent2") -> str:
        """Human-readable summary."""
        lines = [
            f"Evaluation: {agent1_name} vs {agent2_name}",
            f"Games: {self.n_games}",
            f"{agent1_name} wins: {self.agent1_wins} ({self.agent1_wins / self.n_games:.1%})",
            f"{agent2_name} wins: {self.agent2_wins} ({self.agent2_wins / self.n_games:.1%})",
            f"Draws: {self.draws} ({self.draws / self.n_games:.1%})",
            f"Avg score: {agent1_name}={self.agent1_avg_score:.1f}, "
            f"{agent2_name}={self.agent2_avg_score:.1f}",
        ]
        return "\n".join(lines)


def evaluate(
    agent1: Agent,
    agent2: Agent,
    n_games: int = 10,
    *,
    alternate_sides: bool = True,
    verbose: bool = True,
    width: int = 15,
    height: int = 11,
    cheese_count: int = 21,
    max_turns: int = 300,
) -> EvalResult:
    """Evaluate two agents across multiple games.

    Args:
        agent1: First agent.
        agent2: Second agent.
        n_games: Number of games to play.
        alternate_sides: If True, agents swap P1/P2 roles each game.
        verbose: If True, print progress.
        width: Maze width.
        height: Maze height.
        cheese_count: Number of cheese pieces.
        max_turns: Maximum turns before game ends.

    Returns:
        EvalResult with aggregated statistics.
    """
    games: list[GameResult] = []
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    agent1_total_score = 0.0
    agent2_total_score = 0.0

    for i in range(n_games):
        # Alternate sides if requested
        if alternate_sides and i % 2 == 1:
            p1_agent, p2_agent = agent2, agent1
            agent1_is_p1 = False
        else:
            p1_agent, p2_agent = agent1, agent2
            agent1_is_p1 = True

        # Use game index as seed for reproducibility
        result = play_game(
            p1_agent,
            p2_agent,
            seed=i,
            width=width,
            height=height,
            cheese_count=cheese_count,
            max_turns=max_turns,
        )
        games.append(result)

        # Track scores from agent1's perspective
        if agent1_is_p1:
            agent1_score, agent2_score = result.p1_score, result.p2_score
            if result.winner == 1:
                agent1_wins += 1
            elif result.winner == 2:
                agent2_wins += 1
            else:
                draws += 1
        else:
            agent1_score, agent2_score = result.p2_score, result.p1_score
            if result.winner == 2:
                agent1_wins += 1
            elif result.winner == 1:
                agent2_wins += 1
            else:
                draws += 1

        agent1_total_score += agent1_score
        agent2_total_score += agent2_score

        if verbose:
            side = "P1" if agent1_is_p1 else "P2"
            winner_str = (
                f"{agent1.name}" if (result.winner == 1) == agent1_is_p1 else f"{agent2.name}"
            )
            if result.winner == 0:
                winner_str = "Draw"
            score_str = f"{result.p1_score:.1f}-{result.p2_score:.1f}"
            print(
                f"Game {i + 1}/{n_games}: {agent1.name}({side}) - "
                f"{score_str} in {result.turns} turns - {winner_str}"
            )

    return EvalResult(
        n_games=n_games,
        agent1_wins=agent1_wins,
        agent2_wins=agent2_wins,
        draws=draws,
        agent1_avg_score=agent1_total_score / n_games,
        agent2_avg_score=agent2_total_score / n_games,
        games=games,
    )
