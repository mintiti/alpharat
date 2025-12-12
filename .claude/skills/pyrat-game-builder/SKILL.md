---
name: pyrat-game-builder
description: Use GameConfigBuilder for custom PyRat game creation. Fluent API for setting dimensions, positions, cheese, walls, mud. Use when creating test games or custom scenarios.
---

# PyRat GameConfigBuilder

## What to Know

`GameConfigBuilder` provides a fluent API for creating custom PyRat games, cleaner than the static factory methods.

**Import:**
```python
from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates, Wall, Mud
```

**Fluent methods:**
- `GameConfigBuilder(width, height)` - start with dimensions
- `.with_max_turns(n)` - set turn limit
- `.with_player1_pos(Coordinates)` - set P1 starting position
- `.with_player2_pos(Coordinates)` - set P2 starting position
- `.with_cheese([Coordinates, ...])` - set cheese positions
- `.with_walls([Wall, ...])` - add walls
- `.with_mud([Mud, ...])` - add mud
- `.build()` - create the PyRat game

## When This Applies

- Creating games for unit tests
- Setting up specific scenarios for debugging
- Creating custom training environments
- Need more control than presets offer

## What to Do Differently

**Use builder instead of factory methods:**
```python
# Good - fluent and readable
game = (GameConfigBuilder(5, 5)
    .with_max_turns(100)
    .with_player1_pos(Coordinates(0, 0))
    .with_player2_pos(Coordinates(4, 4))
    .with_cheese([Coordinates(2, 2), Coordinates(2, 3)])
    .build())

# Also fine for simple cases - presets
game = PyRat.create_preset('small')
```

**Available presets** (for reference):
- `tiny` (11x9), `small` (15x11), `default`, `large` (31x21), `huge` (41x31)
- `empty` - no walls/mud
- `asymmetric` - non-symmetric maze

**Note:** `create_from_walls` requires symmetric wall placement for fairness. Builder doesn't have this constraint if you don't care about fairness.
