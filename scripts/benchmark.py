#!/usr/bin/env python3
"""Benchmark agents against each other.

Usage:
    uv run python scripts/benchmark.py                     # MCTS(500) vs Random
    uv run python scripts/benchmark.py --agent1-sims 1000  # MCTS(1000) vs Random
    uv run python scripts/benchmark.py --agent2 mcts --agent2-sims 200  # MCTS vs MCTS
    uv run python scripts/benchmark.py --n-games 50 --width 5 --height 5
"""

import argparse

from alpharat.ai import MCTSAgent, RandomAgent
from alpharat.eval import evaluate


def create_agent(agent_type: str, n_sims: int) -> MCTSAgent | RandomAgent:
    """Create an agent by type."""
    if agent_type == "mcts":
        return MCTSAgent(n_sims=n_sims)
    elif agent_type == "random":
        return RandomAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyRat agents")

    # Agent config
    parser.add_argument("--agent1", default="mcts", choices=["mcts", "random"])
    parser.add_argument("--agent1-sims", type=int, default=500)
    parser.add_argument("--agent2", default="random", choices=["mcts", "random"])
    parser.add_argument("--agent2-sims", type=int, default=200)

    # Game config
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--cheese", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=100)

    args = parser.parse_args()

    agent1 = create_agent(args.agent1, args.agent1_sims)
    agent2 = create_agent(args.agent2, args.agent2_sims)

    print(f"{agent1.name} vs {agent2.name}")
    print(f"Map: {args.width}x{args.height}, {args.cheese} cheese, {args.max_turns} max turns")
    print()

    result = evaluate(
        agent1,
        agent2,
        n_games=args.n_games,
        width=args.width,
        height=args.height,
        cheese_count=args.cheese,
        max_turns=args.max_turns,
    )

    print()
    print(result.summary(agent1.name, agent2.name))


if __name__ == "__main__":
    main()
