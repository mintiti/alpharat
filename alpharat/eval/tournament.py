"""Round-robin tournament for comparing agents."""

from __future__ import annotations

import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from alpharat.ai.config import AgentConfig  # noqa: TC001
from alpharat.config.base import StrictBaseModel
from alpharat.config.game import GameConfig  # noqa: TC001
from alpharat.eval.game import play_game

# Games per batch - balances parallelism vs agent loading overhead
_BATCH_SIZE = 10


class TournamentConfig(StrictBaseModel):
    """Round-robin tournament configuration.

    Example YAML:
        name: baseline_tournament  # Required: benchmark name

        agents:
          random:
            variant: random
          greedy:
            variant: greedy
          mcts_baseline:
            variant: mcts
            simulations: 200
          mcts_with_nn:
            variant: mcts
            simulations: 200
            checkpoint: checkpoints/best_model.pt

        games_per_matchup: 50
        game:
          width: 5
          height: 5
          max_turns: 30
          cheese:
            count: 5
        workers: 4
        device: mps
    """

    name: str  # Required: human-chosen benchmark name
    agents: dict[str, AgentConfig]
    games_per_matchup: int
    game: GameConfig
    workers: int = 4
    device: str = "cpu"


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

    def standings(self) -> list[dict[str, str | int | float]]:
        """Return standings as list of dicts (for JSON serialization)."""
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

        for name in self.agent_names:
            s = stats[name]
            s["points"] = s["wins"] + s["draws"] * 0.5
            s["avg_cheese"] = s["cheese"] / s["games"] if s["games"] > 0 else 0.0

        ranked = sorted(
            self.agent_names,
            key=lambda n: (stats[n]["points"], stats[n]["wins"]),
            reverse=True,
        )

        return [
            {
                "rank": i,
                "agent": name,
                "wins": stats[name]["wins"],
                "draws": stats[name]["draws"],
                "losses": stats[name]["losses"],
                "points": stats[name]["points"],
                "avg_cheese": stats[name]["avg_cheese"],
            }
            for i, name in enumerate(ranked, 1)
        ]

    def wdl_matrix(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """Return W/D/L matrix as nested dict: agent -> opponent -> (wins, draws, losses)."""
        result: dict[str, dict[str, tuple[int, int, int]]] = {}
        for name in self.agent_names:
            result[name] = {}
            for other in self.agent_names:
                if name == other:
                    continue
                m = self._get_matchup(name, other)
                if m:
                    result[name][other] = (m.wins_a, m.draws, m.wins_b)
        return result

    def cheese_matrix(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Return cheese matrix: agent -> opponent -> (scored, conceded)."""
        result: dict[str, dict[str, tuple[float, float]]] = {}
        for name in self.agent_names:
            result[name] = {}
            for other in self.agent_names:
                if name == other:
                    continue
                m = self._get_matchup(name, other)
                if m:
                    result[name][other] = (m.avg_cheese_a, m.avg_cheese_b)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize results to a plain dict for JSON persistence.

        Returns:
            Dict with standings, wdl_matrix, and cheese_stats.
        """
        return {
            "standings": self.standings(),
            "wdl_matrix": {
                agent: {
                    opp: {"wins": wdl[0], "draws": wdl[1], "losses": wdl[2]}
                    for opp, wdl in opps.items()
                }
                for agent, opps in self.wdl_matrix().items()
            },
            "cheese_stats": {
                agent: {
                    opp: {"scored": cheese[0], "conceded": cheese[1]}
                    for opp, cheese in opps.items()
                }
                for agent, opps in self.cheese_matrix().items()
            },
        }


@dataclass
class _GameSpec:
    """Specification for a single game within a matchup batch."""

    game_idx: int
    seed: int
    swap_sides: bool


@dataclass
class _MatchupBatch:
    """All games for one matchup, with agent configs for worker to build."""

    agent_a_name: str
    agent_b_name: str
    agent_a_config: AgentConfig
    agent_b_config: AgentConfig
    games: list[_GameSpec]
    game_config: GameConfig
    device: str = "cpu"


@dataclass
class _GameResult:
    """Internal result from a single game."""

    agent_a_name: str
    agent_b_name: str
    winner: int  # 0=draw, 1=agent_a won, 2=agent_b won
    cheese_a: float
    cheese_b: float


def _run_matchup_batch(batch: _MatchupBatch) -> list[_GameResult]:
    """Run all games in a matchup batch. Called in worker process."""
    # Build agents from config (only the 2 needed for this matchup)
    agent_a = batch.agent_a_config.build(device=batch.device)
    agent_b = batch.agent_b_config.build(device=batch.device)

    # Build engine config once per batch, stamp out games cheaply
    engine_cfg = batch.game_config.to_engine_config()

    results: list[_GameResult] = []
    for game in batch.games:
        if game.swap_sides:
            p1_agent, p2_agent = agent_b, agent_a
        else:
            p1_agent, p2_agent = agent_a, agent_b

        pyrat_game = engine_cfg.create(seed=game.seed)
        result = play_game(p1_agent, p2_agent, pyrat_game)

        # Convert to agent_a/agent_b perspective
        if game.swap_sides:
            cheese_a = result.p2_score
            cheese_b = result.p1_score
            if result.winner == 1:
                winner = 2
            elif result.winner == 2:
                winner = 1
            else:
                winner = 0
        else:
            cheese_a = result.p1_score
            cheese_b = result.p2_score
            winner = result.winner

        results.append(
            _GameResult(
                agent_a_name=batch.agent_a_name,
                agent_b_name=batch.agent_b_name,
                winner=winner,
                cheese_a=cheese_a,
                cheese_b=cheese_b,
            )
        )

    return results


def run_tournament(config: TournamentConfig, *, verbose: bool = True) -> TournamentResult:
    """Execute round-robin tournament.

    Args:
        config: Tournament configuration.
        verbose: Print progress if True.

    Returns:
        TournamentResult with all matchup results.
    """
    agent_names = list(config.agents.keys())
    matchup_pairs = list(itertools.combinations(agent_names, 2))

    # Build batches - chunk each matchup into smaller batches for better parallelism
    batches: list[_MatchupBatch] = []
    total_games = 0

    for agent_a_name, agent_b_name in matchup_pairs:
        # Generate all game specs for this matchup
        all_games: list[_GameSpec] = []
        for game_idx in range(config.games_per_matchup):
            swap_sides = game_idx % 2 == 1
            seed = hash((agent_a_name, agent_b_name, game_idx)) % (2**31)
            all_games.append(_GameSpec(game_idx=game_idx, seed=seed, swap_sides=swap_sides))

        # Split into chunks of _BATCH_SIZE
        for i in range(0, len(all_games), _BATCH_SIZE):
            chunk = all_games[i : i + _BATCH_SIZE]
            batches.append(
                _MatchupBatch(
                    agent_a_name=agent_a_name,
                    agent_b_name=agent_b_name,
                    agent_a_config=config.agents[agent_a_name],
                    agent_b_config=config.agents[agent_b_name],
                    games=chunk,
                    game_config=config.game,
                    device=config.device,
                )
            )
            total_games += len(chunk)

    if verbose:
        print(
            f"Running {total_games} games ({len(matchup_pairs)} matchups, {len(batches)} batches)"
        )  # noqa: E501
        print(f"Agents: {', '.join(agent_names)}")
        print(f"Workers: {config.workers}")
        print()

    # Run batches in parallel using multiprocessing
    # Use spawn context for CUDA compatibility (fork doesn't work with CUDA)
    all_results: list[_GameResult] = []
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=config.workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_matchup_batch, batch) for batch in batches]

        # Collect results as they complete with progress bar
        for future in tqdm(
            as_completed(futures),
            total=len(batches),
            desc="Batches",
            disable=not verbose,
            unit="batch",
        ):
            all_results.extend(future.result())

    # Aggregate results by matchup
    final_matchups: dict[tuple[str, str], MatchupResult] = {}

    for agent_a_name, agent_b_name in matchup_pairs:
        matchup_games = [
            r
            for r in all_results
            if r.agent_a_name == agent_a_name and r.agent_b_name == agent_b_name
        ]

        wins_a = sum(1 for g in matchup_games if g.winner == 1)
        wins_b = sum(1 for g in matchup_games if g.winner == 2)
        draws = sum(1 for g in matchup_games if g.winner == 0)
        avg_cheese_a = (
            sum(g.cheese_a for g in matchup_games) / len(matchup_games) if matchup_games else 0
        )
        avg_cheese_b = (
            sum(g.cheese_b for g in matchup_games) / len(matchup_games) if matchup_games else 0
        )

        final_matchups[(agent_a_name, agent_b_name)] = MatchupResult(
            agent_a=agent_a_name,
            agent_b=agent_b_name,
            wins_a=wins_a,
            draws=draws,
            wins_b=wins_b,
            avg_cheese_a=avg_cheese_a,
            avg_cheese_b=avg_cheese_b,
        )

    return TournamentResult(
        matchups=list(final_matchups.values()),
        agent_names=agent_names,
    )
