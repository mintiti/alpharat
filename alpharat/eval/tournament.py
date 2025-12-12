"""Round-robin tournament for comparing MCTS agents."""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from alpharat.ai import GreedyAgent, MCTSAgent, RandomAgent
from alpharat.data.batch import GameParams  # noqa: TC001
from alpharat.eval.game import play_game
from alpharat.mcts import MCTSConfig  # noqa: TC001

if TYPE_CHECKING:
    from alpharat.ai.base import Agent


class TournamentConfig(BaseModel):
    """Round-robin tournament configuration."""

    agents: dict[str, MCTSConfig]
    games_per_matchup: int
    game: GameParams
    workers: int = 4


@dataclass
class MatchupResult:
    """Result of one matchup (A vs B over N games)."""

    agent_a: str
    agent_b: str
    wins_a: int
    draws: int
    wins_b: int
    avg_cheese_a: float
    avg_cheese_b: float


@dataclass
class TournamentResult:
    """Full tournament results."""

    matchups: list[MatchupResult]
    agent_names: list[str]

    def _get_matchup(self, a: str, b: str) -> MatchupResult | None:
        """Get matchup result for agents a vs b."""
        for m in self.matchups:
            if m.agent_a == a and m.agent_b == b:
                return m
            if m.agent_a == b and m.agent_b == a:
                # Flip perspective
                return MatchupResult(
                    agent_a=a,
                    agent_b=b,
                    wins_a=m.wins_b,
                    draws=m.draws,
                    wins_b=m.wins_a,
                    avg_cheese_a=m.avg_cheese_b,
                    avg_cheese_b=m.avg_cheese_a,
                )
        return None

    def wdl_table(self) -> str:
        """Format WDL matrix as string."""
        # Determine column width
        col_width = max(len(name) for name in self.agent_names)
        col_width = max(col_width, 10)  # minimum width for WDL values

        lines = ["Tournament Results (W/D/L from row's perspective)", "=" * 50]

        # Header row
        header = " " * (col_width + 2)
        for name in self.agent_names:
            header += f"{name:>{col_width}}  "
        lines.append(header)

        # Data rows
        for row_name in self.agent_names:
            row = f"{row_name:<{col_width}}  "
            for col_name in self.agent_names:
                if row_name == col_name:
                    cell = "-"
                else:
                    m = self._get_matchup(row_name, col_name)
                    cell = f"{m.wins_a}/{m.draws}/{m.wins_b}" if m else "?"
                row += f"{cell:>{col_width}}  "
            lines.append(row)

        return "\n".join(lines)

    def cheese_table(self) -> str:
        """Format avg cheese matrix as string."""
        col_width = max(len(name) for name in self.agent_names)
        col_width = max(col_width, 10)

        lines = ["Average Cheese (row / col)", "=" * 30]

        # Header row
        header = " " * (col_width + 2)
        for name in self.agent_names:
            header += f"{name:>{col_width}}  "
        lines.append(header)

        # Data rows
        for row_name in self.agent_names:
            row = f"{row_name:<{col_width}}  "
            for col_name in self.agent_names:
                if row_name == col_name:
                    cell = "-"
                else:
                    m = self._get_matchup(row_name, col_name)
                    cell = f"{m.avg_cheese_a:.1f}/{m.avg_cheese_b:.1f}" if m else "?"
                row += f"{cell:>{col_width}}  "
            lines.append(row)

        return "\n".join(lines)

    def standings_table(self) -> str:
        """Format overall standings as string, sorted by points."""
        # Aggregate stats per agent
        stats: dict[str, dict[str, int | float]] = {
            name: {"wins": 0, "draws": 0, "losses": 0, "cheese": 0.0, "games": 0}
            for name in self.agent_names
        }

        for name in self.agent_names:
            for other in self.agent_names:
                if name == other:
                    continue
                m = self._get_matchup(name, other)
                if m:
                    stats[name]["wins"] += m.wins_a
                    stats[name]["draws"] += m.draws
                    stats[name]["losses"] += m.wins_b
                    stats[name]["cheese"] += m.avg_cheese_a * (m.wins_a + m.draws + m.wins_b)
                    stats[name]["games"] += m.wins_a + m.draws + m.wins_b

        # Calculate points (chess-style: 1 for win, 0.5 for draw) and avg cheese
        for name in self.agent_names:
            s = stats[name]
            s["points"] = s["wins"] + s["draws"] * 0.5
            s["avg_cheese"] = s["cheese"] / s["games"] if s["games"] > 0 else 0.0

        # Sort by points descending, then by wins
        ranked = sorted(
            self.agent_names,
            key=lambda n: (stats[n]["points"], stats[n]["wins"]),
            reverse=True,
        )

        # Build table
        lines = ["Standings", "=" * 60]
        header = f"{'Rank':<6}{'Agent':<15}{'W':>6}{'D':>6}{'L':>6}{'Pts':>6}{'AvgCheese':>10}"
        lines.append(header)
        lines.append("-" * 60)

        for i, name in enumerate(ranked, 1):
            s = stats[name]
            medal = " 🏆" if i == 1 else ""
            lines.append(
                f"{i:<6}{name + medal:<15}{s['wins']:>6}{s['draws']:>6}{s['losses']:>6}"
                f"{s['points']:>6.1f}{s['avg_cheese']:>10.1f}"
            )

        return "\n".join(lines)


@dataclass
class _GameTask:
    """Internal task for parallel game execution."""

    agent_a_name: str
    agent_b_name: str
    agent_a: Agent
    agent_b: Agent
    game_idx: int
    seed: int
    swap_sides: bool


@dataclass
class _GameResult:
    """Internal result from a single game."""

    agent_a_name: str
    agent_b_name: str
    winner: int  # 0=draw, 1=agent_a won, 2=agent_b won
    cheese_a: float
    cheese_b: float


def _run_single_game(
    task: _GameTask,
    game_params: GameParams,
) -> _GameResult:
    """Run a single game and return result."""
    if task.swap_sides:
        p1_agent, p2_agent = task.agent_b, task.agent_a
    else:
        p1_agent, p2_agent = task.agent_a, task.agent_b

    result = play_game(
        p1_agent,
        p2_agent,
        seed=task.seed,
        width=game_params.width,
        height=game_params.height,
        cheese_count=game_params.cheese_count,
        max_turns=game_params.max_turns,
        wall_density=game_params.wall_density,
        mud_density=game_params.mud_density,
    )

    # Convert to agent_a/agent_b perspective
    if task.swap_sides:
        # agent_a was P2, agent_b was P1
        cheese_a = result.p2_score
        cheese_b = result.p1_score
        if result.winner == 1:
            winner = 2  # agent_b won (was P1)
        elif result.winner == 2:
            winner = 1  # agent_a won (was P2)
        else:
            winner = 0
    else:
        cheese_a = result.p1_score
        cheese_b = result.p2_score
        winner = result.winner

    return _GameResult(
        agent_a_name=task.agent_a_name,
        agent_b_name=task.agent_b_name,
        winner=winner,
        cheese_a=cheese_a,
        cheese_b=cheese_b,
    )


def run_tournament(config: TournamentConfig, *, verbose: bool = True) -> TournamentResult:
    """Execute round-robin tournament.

    Args:
        config: Tournament configuration.
        verbose: Print progress if True.

    Returns:
        TournamentResult with all matchup results.
    """
    # Build agent dict: config.agents + Random + Greedy
    agents: dict[str, Agent] = {"Random": RandomAgent(), "Greedy": GreedyAgent()}
    for name, mcts_config in config.agents.items():
        agents[name] = MCTSAgent(mcts_config)

    agent_names = list(agents.keys())

    # Generate all matchup pairs
    matchup_pairs = list(itertools.combinations(agent_names, 2))

    # Build all game tasks
    tasks: list[_GameTask] = []
    for agent_a_name, agent_b_name in matchup_pairs:
        for game_idx in range(config.games_per_matchup):
            # Alternate sides for fairness
            swap_sides = game_idx % 2 == 1
            # Seed for reproducibility
            seed = hash((agent_a_name, agent_b_name, game_idx)) % (2**31)
            tasks.append(
                _GameTask(
                    agent_a_name=agent_a_name,
                    agent_b_name=agent_b_name,
                    agent_a=agents[agent_a_name],
                    agent_b=agents[agent_b_name],
                    game_idx=game_idx,
                    seed=seed,
                    swap_sides=swap_sides,
                )
            )

    if verbose:
        print(f"Running {len(tasks)} games ({len(matchup_pairs)} matchups)")
        print(f"Agents: {', '.join(agent_names)}")
        print()

    # Run games in parallel
    results: list[_GameResult] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_run_single_game, task, config.game): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if verbose and completed % 10 == 0:
                print(f"Completed {completed}/{len(tasks)} games")

    if verbose:
        print(f"Completed {completed}/{len(tasks)} games")
        print()

    # Aggregate results by matchup
    matchup_results: dict[tuple[str, str], MatchupResult] = {}

    for agent_a_name, agent_b_name in matchup_pairs:
        # Collect all games for this matchup
        games = [
            r for r in results if r.agent_a_name == agent_a_name and r.agent_b_name == agent_b_name
        ]

        wins_a = sum(1 for g in games if g.winner == 1)
        wins_b = sum(1 for g in games if g.winner == 2)
        draws = sum(1 for g in games if g.winner == 0)
        avg_cheese_a = sum(g.cheese_a for g in games) / len(games) if games else 0
        avg_cheese_b = sum(g.cheese_b for g in games) / len(games) if games else 0

        matchup_results[(agent_a_name, agent_b_name)] = MatchupResult(
            agent_a=agent_a_name,
            agent_b=agent_b_name,
            wins_a=wins_a,
            draws=draws,
            wins_b=wins_b,
            avg_cheese_a=avg_cheese_a,
            avg_cheese_b=avg_cheese_b,
        )

    return TournamentResult(
        matchups=list(matchup_results.values()),
        agent_names=agent_names,
    )
