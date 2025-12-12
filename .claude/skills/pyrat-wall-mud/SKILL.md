---
name: pyrat-wall-mud
description: Use pyrat_engine Wall and Mud types for maze topology. Applies when working with walls, mud costs, maze building, or writing tests involving maze features. Use real types, not mocks.
---

# PyRat Wall and Mud Types

## What to Know

`Wall` and `Mud` from `pyrat_engine.core.types` represent maze topology features between adjacent cells.

**Wall API:**
- `Wall(pos1, pos2)` - constructor takes two positions (Coordinates or tuples)
- `.pos1`, `.pos2` - the two adjacent cells separated by the wall

**Mud API:**
- `Mud(pos1, pos2, value)` - constructor takes two positions and traversal cost
- `.pos1`, `.pos2` - the two adjacent cells connected by mud
- `.value` - integer cost (turns to traverse)

Both types represent bidirectional features between adjacent cells.

## When This Applies

- Querying game topology via `game.wall_entries()` and `game.mud_entries()`
- Building maze representations
- Writing tests that involve walls or mud
- Mocking game objects that need to return topology data

## What to Do Differently

**Import real types:**
```python
from pyrat_engine.core.types import Coordinates, Wall, Mud
```

**In tests, use real Wall/Mud (not mocks):**
```python
# Good - use real types
def wall_entries(self) -> list[Wall]:
    return [Wall(p1, p2) for p1, p2 in self._walls]

def mud_entries(self) -> list[Mud]:
    return [Mud(p1, p2, value) for p1, p2, value in self._muds]

# Avoid - don't create fake wall/mud classes
class FakeWall:  # Don't do this
    def __init__(self, p1, p2): ...
```

**Wall/Mud constructors accept tuples:**
```python
# Both work - Coordinates or tuple
Wall(Coordinates(1, 1), Coordinates(1, 2))
Wall((1, 1), (1, 2))  # Tuples converted internally
```
