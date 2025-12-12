---
name: pyrat-game-api
description: PyRat game instance methods and their return types. Applies when querying game state, topology, positions, or scores. Methods return properly typed objects from pyrat_engine.core.types.
---

# PyRat Game API

## What to Know

The `PyRat` game instance (from `pyrat_engine.core.game`) exposes methods that return properly typed objects.

**Topology queries (return typed objects):**
- `game.wall_entries()` → `list[Wall]`
- `game.mud_entries()` → `list[Mud]`
- `game.cheese_positions()` → `list[Coordinates]`
- `game.get_valid_moves(position)` → `list[Direction]`

**Position properties (return Coordinates):**
- `game.player1_position` → `Coordinates`
- `game.player2_position` → `Coordinates`

**Score/turn properties:**
- `game.player1_score`, `game.player2_score` → `float`
- `game.player1_mud_turns`, `game.player2_mud_turns` → `int`
- `game.turn` → `int`
- `game.max_turns` → `int`

**Move/unmove (for MCTS tree navigation):**
- `game.make_move(p1_action, p2_action)` → `MoveUndo`
- `game.unmake_move(undo: MoveUndo)` → None

## When This Applies

- Querying game state for observations
- Building maze representations from game topology
- Writing game simulation code
- Mocking PyRat games in tests

## What to Do Differently

**Expect typed return values:**
```python
# These return real typed objects, not raw dicts/tuples
for wall in game.wall_entries():
    # wall.pos1 and wall.pos2 are Coordinates
    print(wall.pos1.x, wall.pos1.y)

for cheese in game.cheese_positions():
    # cheese is a Coordinates object
    x, y = cheese.x, cheese.y
```

**When mocking, match the real return types:**
```python
class FakeGame:
    def cheese_positions(self) -> list[Coordinates]:
        return [Coordinates(x, y) for x, y in self._cheese]

    def wall_entries(self) -> list[Wall]:
        return [Wall(p1, p2) for p1, p2 in self._walls]
```
