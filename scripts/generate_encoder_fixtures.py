"""Generate encoder test fixtures for cross-language parity testing.

Creates JSON fixtures that the Rust parity test loads to verify FlatEncoder
produces byte-identical output to Python's FlatObservationBuilder.

Usage:
    uv run python scripts/generate_encoder_fixtures.py

Fixture format:
    The JSON stores the initial game config (width, height, walls, mud, cheese,
    p1_pos, p2_pos, max_turns) plus an optional `moves` list. The Rust test
    constructs the game from the config, replays the moves, then encodes.
    The `expected` array is what the Python encoder produces for the final state.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates, Direction, Mud, Wall

from alpharat.data.maze import build_maze_array
from alpharat.nn.builders.flat import FlatObservationBuilder
from alpharat.nn.extraction import from_pyrat_game

FIXTURE_DIR = Path("crates/alpharat-sampling/tests/fixtures")


def make_fixture(
    name: str,
    game: object,
    width: int,
    height: int,
    *,
    walls: list[dict] | None = None,
    mud: list[dict] | None = None,
    cheese: list[dict] | None = None,
    p1_pos: dict | None = None,
    p2_pos: dict | None = None,
    max_turns: int = 100,
    moves: list[list[int]] | None = None,
    description: str = "",
) -> dict:
    """Build one fixture: run the Python encoder and save config + expected output.

    Args:
        game: PyRat game in its final state (after any moves).
        width, height: Maze dimensions.
        walls, mud, cheese: Initial game config for reconstruction.
        p1_pos, p2_pos: Initial player positions for reconstruction.
        max_turns: Game max turns.
        moves: List of [p1_direction, p2_direction] pairs to replay.
            Direction values: UP=0, RIGHT=1, DOWN=2, LEFT=3, STAY=4.
        description: Human-readable description.
    """
    maze = build_maze_array(game, width, height)
    obs_input = from_pyrat_game(game, maze, max_turns)
    builder = FlatObservationBuilder(width, height)
    expected = builder.build(obs_input)

    return {
        "name": name,
        "description": description,
        "width": width,
        "height": height,
        "max_turns": max_turns,
        "walls": walls or [],
        "mud": mud or [],
        "cheese": cheese or [],
        "p1_pos": p1_pos or {"x": 0, "y": 0},
        "p2_pos": p2_pos or {"x": width - 1, "y": height - 1},
        "moves": moves or [],
        "expected": expected.tolist(),
    }


def scenario_open_5x5() -> dict:
    """Open 5x5, players at corners, one cheese at center."""
    cheese = [Coordinates(2, 2)]
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese(cheese)
        .build()
    )
    return make_fixture(
        "open_5x5",
        game,
        5,
        5,
        cheese=[{"x": 2, "y": 2}],
        p1_pos={"x": 0, "y": 0},
        p2_pos={"x": 4, "y": 4},
        description="Open 5x5 maze, players at corners, one cheese at center",
    )


def scenario_wall_5x5() -> dict:
    """5x5 with wall between (2,2) and (2,3)."""
    cheese = [Coordinates(0, 0)]
    wall = Wall(Coordinates(2, 2), Coordinates(2, 3))
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese(cheese)
        .with_walls([wall])
        .build()
    )
    return make_fixture(
        "wall_5x5",
        game,
        5,
        5,
        walls=[{"pos1": {"x": 2, "y": 2}, "pos2": {"x": 2, "y": 3}}],
        cheese=[{"x": 0, "y": 0}],
        p1_pos={"x": 0, "y": 0},
        p2_pos={"x": 4, "y": 4},
        description="5x5 with wall between (2,2) and (2,3)",
    )


def scenario_mud_5x5() -> dict:
    """5x5 with mud cost 3 between (2,2) and (2,3)."""
    cheese = [Coordinates(0, 0)]
    mud = Mud(Coordinates(2, 2), Coordinates(2, 3), value=3)
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese(cheese)
        .with_mud([mud])
        .build()
    )
    return make_fixture(
        "mud_5x5",
        game,
        5,
        5,
        mud=[{"pos1": {"x": 2, "y": 2}, "pos2": {"x": 2, "y": 3}, "value": 3}],
        cheese=[{"x": 0, "y": 0}],
        p1_pos={"x": 0, "y": 0},
        p2_pos={"x": 4, "y": 4},
        description="5x5 with mud cost 3 between (2,2) and (2,3)",
    )


def scenario_midgame_5x5() -> dict:
    """5x5 after a move: non-zero scores, turn > 0."""
    cheese = [Coordinates(0, 0), Coordinates(4, 4)]
    mud = Mud(Coordinates(2, 2), Coordinates(2, 3), value=3)
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(1, 0))
        .with_player2_pos(Coordinates(3, 4))
        .with_cheese(cheese)
        .with_mud([mud])
        .build()
    )
    # P1 moves LEFT to (0,0) collecting cheese, P2 moves RIGHT to (4,4) collecting cheese
    game.make_move(Direction.LEFT, Direction.RIGHT)

    return make_fixture(
        "midgame_5x5",
        game,
        5,
        5,
        mud=[{"pos1": {"x": 2, "y": 2}, "pos2": {"x": 2, "y": 3}, "value": 3}],
        cheese=[{"x": 0, "y": 0}, {"x": 4, "y": 4}],
        p1_pos={"x": 1, "y": 0},
        p2_pos={"x": 3, "y": 4},
        max_turns=100,
        moves=[[Direction.LEFT, Direction.RIGHT]],  # [3, 1]
        description="5x5 after one move: both players collected cheese, turn=1",
    )


def scenario_mud_stuck_5x5() -> dict:
    """5x5 where P1 stepped into mud, creating nonzero mud_timer."""
    cheese = [Coordinates(4, 0)]
    mud = Mud(Coordinates(1, 0), Coordinates(2, 0), value=3)
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(1, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese(cheese)
        .with_mud([mud])
        .build()
    )
    # P1 moves RIGHT through mud to (2,0), gets mud_timer=3. P2 stays.
    game.make_move(Direction.RIGHT, Direction.STAY)

    return make_fixture(
        "mud_stuck_5x5",
        game,
        5,
        5,
        mud=[{"pos1": {"x": 1, "y": 0}, "pos2": {"x": 2, "y": 0}, "value": 3}],
        cheese=[{"x": 4, "y": 0}],
        p1_pos={"x": 1, "y": 0},
        p2_pos={"x": 4, "y": 4},
        moves=[[Direction.RIGHT, Direction.STAY]],
        description="5x5 where P1 stepped into mud, nonzero mud_timer",
    )


def scenario_asymmetric_scores_5x5() -> dict:
    """5x5 where P1 collected cheese (score=1) but P2 hasn't (score=0)."""
    cheese = [Coordinates(1, 0), Coordinates(2, 2)]
    game = (
        GameConfigBuilder(5, 5)
        .with_max_turns(100)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(4, 4))
        .with_cheese(cheese)
        .build()
    )
    # P1 moves RIGHT to (1,0) collecting cheese. P2 stays (no cheese at (4,4)).
    game.make_move(Direction.RIGHT, Direction.STAY)

    return make_fixture(
        "asymmetric_scores_5x5",
        game,
        5,
        5,
        cheese=[{"x": 1, "y": 0}, {"x": 2, "y": 2}],
        p1_pos={"x": 0, "y": 0},
        p2_pos={"x": 4, "y": 4},
        moves=[[Direction.RIGHT, Direction.STAY]],
        description="5x5 where P1 collected cheese (score=1) but P2 hasn't (score=0)",
    )


def scenario_nonsquare_7x5() -> dict:
    """Non-square 7x5 with mixed features."""
    cheese = [Coordinates(3, 2), Coordinates(6, 0)]
    wall = Wall(Coordinates(1, 1), Coordinates(1, 2))
    mud = Mud(Coordinates(4, 3), Coordinates(4, 4), value=2)
    game = (
        GameConfigBuilder(7, 5)
        .with_max_turns(80)
        .with_player1_pos(Coordinates(0, 0))
        .with_player2_pos(Coordinates(6, 4))
        .with_cheese(cheese)
        .with_walls([wall])
        .with_mud([mud])
        .build()
    )
    return make_fixture(
        "nonsquare_7x5",
        game,
        7,
        5,
        walls=[{"pos1": {"x": 1, "y": 1}, "pos2": {"x": 1, "y": 2}}],
        mud=[{"pos1": {"x": 4, "y": 3}, "pos2": {"x": 4, "y": 4}, "value": 2}],
        cheese=[{"x": 3, "y": 2}, {"x": 6, "y": 0}],
        p1_pos={"x": 0, "y": 0},
        p2_pos={"x": 6, "y": 4},
        max_turns=80,
        description="Non-square 7x5 with wall, mud, two cheeses",
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        scenario_open_5x5,
        scenario_wall_5x5,
        scenario_mud_5x5,
        scenario_midgame_5x5,
        scenario_mud_stuck_5x5,
        scenario_asymmetric_scores_5x5,
        scenario_nonsquare_7x5,
    ]

    for scenario_fn in scenarios:
        fixture = scenario_fn()
        path = FIXTURE_DIR / f"{fixture['name']}.json"
        with open(path, "w") as f:
            json.dump(fixture, f, indent=2)
        n_vals = len(fixture["expected"])
        moves = fixture.get("moves", [])
        print(f"  {path.name}: {n_vals} values, {len(moves)} moves")

    print(f"\nGenerated {len(scenarios)} fixtures in {FIXTURE_DIR}/")


if __name__ == "__main__":
    main()
