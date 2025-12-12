---
name: pyrat-direction
description: Use pyrat_engine Direction enum for movement directions. Applies when indexing arrays by direction, working with valid moves, or maze topology. Direction enum values work as numpy array indices.
---

# PyRat Direction Enum

## What to Know

The `Direction` enum from `pyrat_engine.core.types` represents movement directions.

**Enum values (and their int values):**
- `Direction.UP = 0`
- `Direction.RIGHT = 1`
- `Direction.DOWN = 2`
- `Direction.LEFT = 3`
- `Direction.STAY = 4`

**Key insight:** Direction enum values can be used directly as numpy array indices.

## When This Applies

- Indexing maze arrays by direction
- Working with `game.get_valid_moves(position)` which returns list of Direction
- Computing opposite directions
- Storing per-direction data in arrays

## What to Do Differently

**Import the enum:**
```python
from pyrat_engine.core.types import Direction
```

**Use enum as array index (not raw ints):**
```python
# Good - use Direction enum as index
maze[y, x, Direction.UP] = -1
maze[y, x, Direction.LEFT] = -1

# Avoid - raw integers are unclear
maze[y, x, 0] = -1  # What direction is 0?
```

**Opposite direction formula:**
```python
def opposite_direction(d: Direction) -> Direction:
    # UP<->DOWN (0<->2), LEFT<->RIGHT (1<->3)
    return Direction((d + 2) % 4)
```

**Note:** STAY (4) doesn't have an opposite - it's not a movement direction.
