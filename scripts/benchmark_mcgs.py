"""Head-to-head: MCGS vs MCTS at various sim budgets.

Standalone script — not a permanent entry point. Run with:
    uv run python scripts/benchmark_mcgs.py
    uv run python scripts/benchmark_mcgs.py --sims 100 500 1000 --games 50
"""

from __future__ import annotations

import argparse
import itertools

from alpharat.config.game import CheeseConfig, GameConfig
from alpharat.eval.elo import compute_elo, from_tournament_result
from alpharat.eval.game import play_game
from alpharat.eval.tournament import MatchupResult, TournamentResult
from alpharat.mcts.config import RustMCGSConfig, RustMCTSConfig


def build_agents(sim_budgets: list[int]) -> dict[str, object]:
    """Build MCTS and MCGS agents at each sim budget."""
    agents: dict[str, object] = {}
    for sims in sim_budgets:
        mcts_cfg = RustMCTSConfig(simulations=sims).for_evaluation()
        mcgs_cfg = RustMCGSConfig(simulations=sims).for_evaluation()
        agents[f"mcts_{sims}"] = mcts_cfg.build_agent()
        agents[f"mcgs_{sims}"] = mcgs_cfg.build_agent()
    return agents


def run_matchup(
    agent_a_name: str,
    agent_a: object,
    agent_b_name: str,
    agent_b: object,
    game_config: GameConfig,
    games_per_matchup: int,
) -> MatchupResult:
    """Run a single matchup between two agents."""
    from alpharat.ai.base import Agent

    assert isinstance(agent_a, Agent)
    assert isinstance(agent_b, Agent)

    engine_cfg = game_config.to_engine_config()
    wins_a = 0
    wins_b = 0
    draws = 0
    total_cheese_a = 0.0
    total_cheese_b = 0.0

    for game_idx in range(games_per_matchup):
        swap = game_idx % 2 == 1
        seed = hash((agent_a_name, agent_b_name, game_idx)) % (2**31)
        game = engine_cfg.create(seed=seed)

        if swap:
            result = play_game(agent_b, agent_a, game)
            cheese_a, cheese_b = result.p2_score, result.p1_score
            if result.winner == 1:
                winner = 2
            elif result.winner == 2:
                winner = 1
            else:
                winner = 0
        else:
            result = play_game(agent_a, agent_b, game)
            cheese_a, cheese_b = result.p1_score, result.p2_score
            winner = result.winner

        if winner == 1:
            wins_a += 1
        elif winner == 2:
            wins_b += 1
        else:
            draws += 1
        total_cheese_a += cheese_a
        total_cheese_b += cheese_b

    n = games_per_matchup
    return MatchupResult(
        agent_a=agent_a_name,
        agent_b=agent_b_name,
        wins_a=wins_a,
        draws=draws,
        wins_b=wins_b,
        avg_cheese_a=total_cheese_a / n,
        avg_cheese_b=total_cheese_b / n,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCGS vs MCTS benchmark")
    parser.add_argument(
        "--sims",
        nargs="+",
        type=int,
        default=[100, 500],
        help="Simulation budgets to test",
    )
    parser.add_argument("--games", type=int, default=50, help="Games per matchup")
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--cheese", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args()

    game_config = GameConfig(
        width=args.width,
        height=args.height,
        cheese=CheeseConfig(count=args.cheese),
        max_turns=args.max_turns,
    )

    print(f"Game: {args.width}x{args.height}, {args.cheese} cheese, {args.max_turns} turns")
    print(f"Sim budgets: {args.sims}")
    print(f"Games per matchup: {args.games}")
    print()

    agents = build_agents(args.sims)
    agent_names = list(agents.keys())
    pairs = list(itertools.combinations(agent_names, 2))

    print(f"Agents: {', '.join(agent_names)}")
    print(f"Matchups: {len(pairs)}")
    print()

    matchups: list[MatchupResult] = []
    for i, (a_name, b_name) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {a_name} vs {b_name} ...", end=" ", flush=True)
        result = run_matchup(
            a_name, agents[a_name], b_name, agents[b_name], game_config, args.games
        )
        print(f"{result.wins_a}W/{result.draws}D/{result.wins_b}L")
        matchups.append(result)

    tournament = TournamentResult(matchups=matchups, agent_names=agent_names)

    print()
    print(tournament.standings_table())
    print()
    print(tournament.wdl_table())
    print()
    print(tournament.cheese_table())

    # Elo — anchor on the lowest-sim MCTS agent
    anchor = f"mcts_{args.sims[0]}"
    try:
        records = from_tournament_result(tournament)
        elo = compute_elo(records, anchor=anchor, anchor_elo=1000, compute_uncertainty=True)
        print()
        print(elo.format_table())
    except ValueError as e:
        print(f"\nElo computation failed: {e}")


if __name__ == "__main__":
    main()
