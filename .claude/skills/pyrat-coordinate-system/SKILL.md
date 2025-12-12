---
name: pyrat-coordinate-system
description: PyRat uses Y-up coordinate system where UP increases Y. Origin (0,0) at bottom-left. Critical when working with positions, movement, or converting between coordinate systems.
---

# PyRat Coordinate System

## What to Know

PyRat uses a **Y-up** coordinate system, opposite to typical screen/image coordinates:

- **UP** movement **increases** Y (goes to `y+1`)
- **DOWN** movement **decreases** Y (goes to `y-1`)
- **Origin (0,0)** is at **bottom-left**
- **X increases rightward** (standard)

```
     y
     ^
     |
(0,2)|     .     .     .
(0,1)|     .     .     .
(0,0)+----->---------------> x
          (1,0) (2,0) (3,0)
```

## When This Applies

- Converting between PyRat positions and array indices
- Understanding `get_observation().movement_matrix` shape `[W, H, 4]`
- Working with Direction enum and position updates
- Rendering or visualizing game state

## What to Do Differently

**Direction deltas:**
```python
# UP increases Y, DOWN decreases Y
DIRECTION_DELTAS = {
    Direction.UP: (0, +1),    # Not (0, -1)!
    Direction.RIGHT: (+1, 0),
    Direction.DOWN: (0, -1),  # Not (0, +1)!
    Direction.LEFT: (-1, 0),
}
```

**Edge boundaries:**
- `y = 0` is BOTTOM edge (can't move DOWN)
- `y = height-1` is TOP edge (can't move UP)
- `x = 0` is LEFT edge
- `x = width-1` is RIGHT edge

**Array indexing caution:**
The engine's `movement_matrix` uses `[x, y, direction]` indexing, not `[y, x, direction]`:
```python
obs = game.get_observation(is_player_one=True)
# Shape is [W, H, 4] not [H, W, 4]
can_move_up = obs.movement_matrix[x, y, Direction.UP] >= 0
```
