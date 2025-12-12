---
name: pyrat-coordinates
description: Use pyrat_engine Coordinates type for positions. Applies when working with player positions, cheese positions, or position calculations. Coordinates supports subtraction returning tuple.
---

# PyRat Coordinates Type

## What to Know

The `Coordinates` class from `pyrat_engine.core.types` is the canonical type for positions in PyRat.

**Key API:**
- `Coordinates(x, y)` - constructor takes x, y integers
- `.x`, `.y` - access components
- **Subtraction returns tuple**: `coord2 - coord1` returns `(dx, dy)` tuple, not a Coordinates object

## When This Applies

- Working with player positions (`game.player1_position`, `game.player2_position`)
- Working with cheese positions (`game.cheese_positions()` returns list of Coordinates)
- Computing direction deltas between adjacent cells
- Writing tests that involve positions

## What to Do Differently

**Import the real type:**
```python
from pyrat_engine.core.types import Coordinates
```

**Use subtraction for direction calculation:**
```python
# Coordinates subtraction returns (dx, dy) tuple
delta = to_pos - from_pos  # Returns tuple like (1, 0) or (0, -1)

# Map delta to direction
DELTA_TO_DIRECTION = {
    (0, -1): Direction.UP,
    (1, 0): Direction.RIGHT,
    (0, 1): Direction.DOWN,
    (-1, 0): Direction.LEFT,
}
direction = DELTA_TO_DIRECTION[delta]
```

**In tests, use real Coordinates:**
```python
# Good - use real type
self.player1_position = Coordinates(1, 1)
cheese = [Coordinates(x, y) for x, y in positions]

# Avoid - don't create mock coordinate classes
class FakeCoordinates:  # Don't do this
    ...
```
