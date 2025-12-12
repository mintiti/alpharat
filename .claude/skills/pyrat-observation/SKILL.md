---
name: pyrat-observation
description: Use PyRat.get_observation() for player-relative state with pre-built numpy matrices. Returns PyGameObservation with movement_matrix and cheese_matrix. Use when building neural network inputs.
---

# PyRat Game Observation API

## What to Know

`PyRat.get_observation(is_player_one)` returns a `PyGameObservation` with player-relative state and **pre-built numpy matrices**.

**Types:**
- `game`: `PyRat` (Rust-backed class from `pyrat_engine.core.game`)
- `observation`: `PyGameObservation` (Rust-backed class)

**Key attributes:**
```python
obs: PyGameObservation = game.get_observation(is_player_one=True)

# Positions (player-relative view)
obs.player_position      # tuple - your position
obs.opponent_position    # tuple - opponent position

# Scores and state
obs.player_score         # float
obs.opponent_score       # float
obs.player_mud_turns     # int - your remaining stuck turns
obs.opponent_mud_turns   # int
obs.current_turn         # int
obs.max_turns            # int

# Pre-built matrices (numpy arrays)
obs.cheese_matrix        # uint8[W, H] - 1 where cheese exists
obs.movement_matrix      # int8[W, H, 4] - traversal info per direction
```

## Movement Matrix Values

- `-1` = wall (blocked)
- `0` = normal passage (1 turn to traverse)
- `>=2` = mud cost (stuck for N turns)

**Note:** Value `1` does not appear. Normal passages are `0`, not `1`. Mud values must be `>=2`.

## When This Applies

- Building neural network input tensors
- Need player-relative view (symmetric for both players)
- Want pre-computed maze matrices

## What to Do Differently

**Matrix indexing is [X, Y, direction]:**
```python
# Shape is [W, H, 4] - note X before Y
obs = game.get_observation(is_player_one=True)
can_move = obs.movement_matrix[x, y, Direction.UP] >= 0
has_cheese = obs.cheese_matrix[x, y] == 1
```
