# CLAUDE.md

Guidance for coding agents working in this repository.

## Design Philosophy

This repo was built around the idea that experimentation should be fast to iterate on.

The pain with most ML pipelines: everything is coupled. Change training params? Re-sample everything. Try a different architecture? Start from scratch. Hours wasted redoing work you already did.

Here, each stage — sampling, sharding, training — produces reusable artifacts. Same samples, different architectures. Same checkpoint, different MCTS params. You only redo what actually changed.

Reproducibility falls out naturally: when artifacts are saved and stages are decoupled, you always know exactly what went into each run.

## Project Overview

**alpharat** is an MCTS implementation for simultaneous two-player games, targeting PyRat (a maze game where both players move at the same time). Uses scalar value heads (like LC0/KataGo) for value estimation and visit-proportional policies for action selection.

Python 3.11+, strict mypy, `uv` for package management.

## Development Commands

```bash
# Setup
uv sync                              # Install dependencies
uv sync --extra train                # PyTorch (CUDA on Linux, CPU on macOS)
uv pip install torch --torch-backend=cpu --reinstall  # Force CPU-only
uv run pre-commit install            # Install hooks

# Testing
uv run pytest                        # All tests with coverage
uv run pytest tests/mcts/test_node.py  # Specific file
uv run pytest --no-cov               # Skip coverage

# Code quality
uv run ruff format .                 # Format
uv run ruff check --fix .            # Lint
uv run mypy alpharat                 # Type check
uv run pre-commit run --all-files    # All hooks
```

## Project Structure

```
alpharat/
├── mcts/           # Core MCTS: nodes, tree, search, value backup
├── data/           # Data pipeline: sampling, recording, sharding
├── nn/             # Neural network: observation builders, targets, models
├── ai/             # Agents: MCTS, random, greedy
├── eval/           # Evaluation: game execution, tournaments
└── experiments/    # Experiment management: ExperimentManager, manifest

scripts/            # Entry points: sample.py, train.py, iterate.py, benchmark.py, manifest.py
configs/            # YAML config templates for sampling, training, evaluation
tests/              # Mirrors alpharat/ structure
experiments/        # Data folder (NOT in git): batches, shards, runs, benchmarks
```

### alpharat/mcts/

| File | Purpose |
|------|---------|
| `node.py` | `MCTSNode` — outcome-indexed storage for O(1) backup |
| `tree.py` | `MCTSTree` — efficient navigation via `make_move/unmake_move` |
| `decoupled_puct.py` | `DecoupledPUCTSearch` — each player picks via PUCT formula, returns visit-proportional policy |
| `reduction.py` | Boundary translation between 5-action and outcome-indexed space |
| `numba_ops.py` | JIT-compiled hot paths: backup, PUCT scores, marginal Q |

### alpharat/data/

| File | Purpose |
|------|---------|
| `types.py` | `PositionData`, `GameData` — data structures for recorded games |
| `recorder.py` | `GameRecorder` — saves games as .npz during self-play |
| `sampling.py` | `run_sampling()` — multi-worker self-play with MCTS agents |
| `batch.py` | Batch organization and metadata |
| `sharding.py` | `prepare_training_set()` — loads batches, shuffles globally, writes shards |
| `loader.py` | `load_game_data()` — reconstruct GameData from .npz |

### alpharat/nn/

**Core types and utilities:**

| File | Purpose |
|------|---------|
| `types.py` | `ObservationInput`, `TargetBundle` — source-agnostic data types |
| `extraction.py` | Build `ObservationInput` from game arrays or live PyRat |
| `targets.py` | `build_targets()` — visit-proportional policies + scalar value targets |
| `streaming.py` | `StreamingDataset` — memory-efficient PyTorch dataset |

**Training configuration (`config.py`):** `TrainConfig` — Pydantic discriminated unions

**Training infrastructure (`training/`):**

| File | Purpose |
|------|---------|
| `loop.py` | `run_training()` — generic training loop for any architecture |
| `base.py` | `BaseModelConfig`, `BaseOptimConfig`, `DataConfig` — base classes |
| `protocols.py` | `TrainableModel`, `LossFunction`, `AugmentationStrategy` protocols |
| `keys.py` | `ModelOutput`, `LossKey`, `BatchKey` — type-safe dict keys |

**Architectures (`architectures/{mlp,symmetric,local_value}/`):**

Each architecture folder contains `config.py` (ModelConfig, OptimConfig) and `loss.py`.

| Architecture | Description |
|------|---------|
| `mlp` | Flat observation → shared trunk → policy/value heads |
| `symmetric` | Structural P1/P2 symmetry, no augmentation needed |
| `local_value` | Per-cell ownership values + scalar value heads |

**Models (`models/`):** `PyRatMLP`, `SymmetricMLP`, `LocalValueMLP`

**Builders (`builders/`):** `FlatObservationBuilder` — 1D encoding for MLPs

### alpharat/ai/

| File | Purpose |
|------|---------|
| `base.py` | `Agent` ABC — `get_move(game, player) -> int` |
| `mcts_agent.py` | `MCTSAgent` — uses MCTS search, samples from visit-proportional policy |
| `random_agent.py` | `RandomAgent` — baseline |
| `greedy_agent.py` | `GreedyAgent` — moves toward closest cheese |

### alpharat/eval/

| File | Purpose |
|------|---------|
| `game.py` | `play_game()` — execute single game between agents |
| `runner.py` | `evaluate()` — run N games, compute stats |
| `tournament.py` | `run_tournament()` — round-robin with thread pool |
| `benchmark.py` | `build_benchmark_config()` — standard benchmark against baselines |
| `elo.py` | `compute_elo()` — Elo rating computation from tournament results |

### alpharat/experiments/

| File | Purpose |
|------|---------|
| `manager.py` | `ExperimentManager` — central API for managing artifacts |
| `schema.py` | Pydantic schemas for manifest entries |
| `paths.py` | Path constants and helpers |
| `templates.py` | Notes and CLAUDE.md templates |

---

## Experiments Folder

The `experiments/` folder (NOT in git) stores all experiment artifacts with automatic lineage tracking.

```
experiments/
├── manifest.yaml           # Central index of all artifacts
├── batches/{group}/{uuid}/ # Raw game recordings from sampling
├── shards/{group}/{uuid}/  # Processed train/val splits
├── runs/{name}/            # Training runs with checkpoints
└── benchmarks/{name}/      # Tournament results
```

### Using ExperimentManager

```python
from alpharat.experiments import ExperimentManager

exp = ExperimentManager()

# Sampling
batch_dir = exp.create_batch(
    group="uniform_5x5",
    mcts_config=mcts_config,
    game=game_config,
)

# Training
run_dir = exp.create_run(
    name="bimatrix_mlp_v1",
    config=train_config.model_dump(),
    source_shards="shard_uuid",
)

# Query
exp.list_batches()
exp.list_runs()
exp.get_run_path("bimatrix_mlp_v1")
```

### Querying Artifacts

Use `alpharat-manifest` to see what's in the experiments folder:

```bash
alpharat-manifest batches   # List all batches
alpharat-manifest shards    # List shards with lineage
alpharat-manifest runs      # List training runs
```

Example output:
```
GROUP                UUID       CREATED           SIZE   SIMS   PARENT
-------------------------------------------------------------------------------------
iteration_0          d84065d8   2026-01-15 18:05  5x5    50     -
iteration_1          cbb835c6   2026-01-15 18:08  5x5    30     runs/mlp_v1
```

---

## Action Equivalence — The Subtle Part

This is the most non-trivial aspect of the codebase. Multiple actions can lead to the same outcome (hitting walls, being stuck in mud), and this affects everything from MCTS statistics to neural network training.

### The Problem

In PyRat, if a player tries to move UP but there's a wall, they stay in place. So UP and STAY produce identical game states. If we treat them as different actions, MCTS explores redundant branches and Nash equilibrium computation becomes degenerate (multiple equivalent equilibria).

### How It's Handled: Outcome-Indexed Architecture

The key insight: store statistics indexed by **unique outcomes**, not raw actions. When UP is blocked, UP and STAY produce the same outcome — so we only need one entry.

**1. Detection (`tree.py:_compute_effective_actions`)**

At each position, we query `game.get_valid_moves(position)` to build the effective action mapping:
- Valid moves → map to themselves
- Blocked moves (walls/edges) → map to STAY (action 4)
- Mud → forces all actions to STAY for that player

```python
# Example: player against north wall
p1_effective = [4, 1, 2, 3, 4]  # UP(0) blocked → STAY(4)
p1_outcomes = [1, 2, 3, 4]      # 4 unique outcomes (action 0 collapsed into 4)
```

**2. Reduced Storage (`node.py`)**

Nodes store statistics in outcome-indexed space:
- `prior_p1_reduced[n1]`, `prior_p2_reduced[n2]` — priors over unique outcomes
- `_v1`, `_v2` — scalar value estimates (running averages)
- `_visits[n1, n2]` — visit counts per action pair
- Children keyed by outcome index pairs `(i, j)`, not action pairs

```python
# With 4 unique outcomes for P1 and 5 for P2, visit matrix is [4, 5] not [5, 5]
```

**3. O(1) Backup (`numba_ops.py:backup_node`)**

Backup directly indexes the reduced matrix — no need to update multiple equivalent pairs:

```python
idx1 = node._p1_action_to_idx[action_p1]  # Action → outcome index
idx2 = node._p2_action_to_idx[action_p2]
# Single update at [idx1, idx2]
```

**4. Boundary Translation (`reduction.py`)**

The algorithm operates in outcome space. Boundaries handle translation:
- **Input**: NN predictions `[5]`, `[5]` → reduced `[n1]`, `[n2]`
- **Output**: Visit-proportional strategies `[n1]`, `[n2]` → expanded `[5]`, `[5]`

```python
# reduce_prior(): sums probabilities of equivalent actions
# expand_prior(): copies outcome probs to canonical actions only (blocked → 0)
```

### Flow Through Training

**Recording (`recorder.py`):**
- Saves visit-proportional policies (which have 0 for blocked actions)
- Saves scalar `value_p1, value_p2` — expected remaining cheese for each player

**Targets (`targets.py`):**
- Passes through the policies as-is — no additional processing needed
- The 0s for blocked actions are already baked in
- Value targets are `final_score - current_score` (remaining cheese to collect)

**NN Training:**
- The NN sees policy targets where blocked actions have probability 0
- This creates an **implicit auxiliary task**: the NN learns maze topology
- The NN doesn't know which actions are blocked — it learns this from the target structure

### What the NN Must Learn

The NN outputs:
- `policy_p1[5]`, `policy_p2[5]` — should assign ~0 to blocked actions
- `value_p1`, `value_p2` — scalar estimates of remaining cheese each player will collect

This isn't hard-masked; it's learned behavior. The NN must infer from the observation (maze layout, positions) which actions are valid.

### At Inference (Without NN)

When `predict_fn` is None, MCTS uses "smart uniform" priors (`tree.py:_smart_uniform_prior`):

```python
# Only assign probability to unique effective actions
# If effective=[4,1,2,3,4], only actions 1,2,3,4 get probability
# Action 0 gets 0 (it's a duplicate of 4)
```

This is more efficient than naive uniform — doesn't waste exploration on redundant moves.

### At Inference (With NN)

The NN provides priors. MCTS expects:
- Policy predictions should assign low probability to blocked actions (learned from training)
- The tree maintains equivalence structure regardless of what the NN outputs

Even if the NN makes mistakes on blocked actions, the tree's child sharing and statistics updates preserve correctness.

---

## Other Architecture Notes

### Scalar Value Heads

Each node stores scalar value estimates `v1, v2` representing expected remaining cheese for each player. This is simpler than bimatrix approaches and follows modern AlphaZero-style implementations (LC0, KataGo).

### make_move / unmake_move

Nodes don't store full game states. The tree owns one PyRat simulator and navigates by making/unmaking moves. Efficient for deep trees.

### Decoupled PUCT Selection

The search uses decoupled PUCT (`decoupled_puct.py`): each player selects actions independently via the PUCT formula, maximizing Q + exploration bonus. The final policy is visit-proportional (actions selected proportional to visit counts).

### Value Formulation

Values store **expected remaining cheese** for each player:
- `v1` = expected cheese P1 will collect from this position
- `v2` = expected cheese P2 will collect from this position

PyRat is approximately **constant-sum** (not zero-sum): P1 + P2 ≈ remaining_cheese. Exact under infinite horizon; approximate under turn limits (wasted moves reduce total collection).

---

## Code Style

### Type Annotations (Strict)

```python
from __future__ import annotations
from typing import Any
import numpy as np

# Arrays - no shape annotations needed
def compute_nash(payout: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

# Unions with |
parent: MCTSNode | None = None

# Any for external types
game_state: Any
```

### Ruff Rules
Line length 100. Enabled: pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, comprehensions, simplify, type-checking.

---

## Dependencies

**Core:** numpy, numba, pydantic, pyyaml, pyrat-engine (Rust-backed)

**Training (optional):** torch, tensorboard

**Visualization:** optuna, plotly, pandas, matplotlib

**pyrat-engine imports:**
```python
from pyrat_engine.core.game import PyRat, MoveUndo
from pyrat_engine.core.types import Coordinates, Direction, Wall, Mud
```

---

## pyrat_engine Types — Use Them, Don't Reinvent Them

The pyrat_engine types are the **source of truth** for coordinates, directions, and maze topology. Don't reimplement their behavior — use them directly.

### Coordinate System (Y-up)

PyRat uses a **Y-up** coordinate system (opposite of screen/image coords):
- Origin `(0, 0)` at **bottom-left**
- **UP increases Y**, DOWN decreases Y
- X increases to the right

### Coordinates

```python
from pyrat_engine.core.types import Coordinates

pos = Coordinates(x=2, y=3)
neighbor = pos.get_neighbor(Direction.UP)  # Coordinates(2, 4)

# Subtraction returns a TUPLE (dx, dy), not Coordinates
delta = pos2 - pos1  # → tuple[int, int]
dx, dy = pos2 - pos1  # Can unpack directly

# Other operations
pos.manhattan_distance(other)  # int
pos.is_adjacent_to(other)      # bool
pos[0], pos[1]                 # x, y via indexing
```

### Direction (IntEnum)

Direction values ARE integers — use them as array indices:

```python
from pyrat_engine.core.types import Direction

# Direction.UP=0, RIGHT=1, DOWN=2, LEFT=3, STAY=4
maze[y, x, Direction.UP] = -1       # Direct indexing
action_probs[Direction.RIGHT] = 0.5 # Works in numpy arrays

# Construct from int
Direction(action_int)  # int → Direction
```

### Wall and Mud

Use real types for maze features, not mocks:

```python
from pyrat_engine.core.types import Wall, Mud, Coordinates

# Construction (tuples or Coordinates both work)
Wall((1, 1), (1, 2))
Wall(Coordinates(1, 1), Coordinates(1, 2))

Mud((1, 1), (1, 2), value=3)  # value = turns to traverse

# Query from game
for wall in game.wall_entries():
    print(wall.pos1, wall.pos2)
    wall.blocks_movement(from_pos, to_pos)  # bool

for mud in game.mud_entries():
    print(mud.pos1, mud.pos2, mud.value)
```

### Getting Deltas the Right Way

If you need direction→delta mappings, derive them from the engine:

```python
# Derive from Coordinates arithmetic — don't hardcode
origin = Coordinates(0, 0)
DIRECTION_DELTAS = {
    d: origin.get_neighbor(d) - origin
    for d in Direction
}
# Result: {UP: (0,1), RIGHT: (1,0), DOWN: (0,-1), LEFT: (-1,0), STAY: (0,0)}

# Or go the other way: delta → direction
DELTA_TO_DIRECTION = {v: k for k, v in DIRECTION_DELTAS.items() if k != Direction.STAY}
```

**Don't hardcode** `{Direction.UP: (0, -1)}` — that's wrong and will cause bugs.

### PyRat Game API

```python
from pyrat_engine.core.game import PyRat, MoveUndo

# Topology queries
game.wall_entries()              # → list[Wall]
game.mud_entries()               # → list[Mud]
game.cheese_positions()          # → list[Coordinates]
game.get_valid_moves(position)   # → list[int] (0-3: UP, RIGHT, DOWN, LEFT)

# Position properties
game.player1_position            # → Coordinates
game.player2_position            # → Coordinates

# Score/turn properties
game.player1_score, game.player2_score      # float
game.player1_mud_turns, game.player2_mud_turns  # int
game.turn, game.max_turns                   # int

# Move/unmove (for MCTS tree navigation)
undo = game.make_move(p1_action, p2_action)  # → MoveUndo
game.unmake_move(undo)                       # → None
```

### GameConfigBuilder

Fluent API for creating custom games (useful in tests):

```python
from pyrat_engine.core import GameConfigBuilder
from pyrat_engine.core.types import Coordinates, Wall, Mud

game = (GameConfigBuilder(5, 5)
    .with_max_turns(100)
    .with_player1_pos(Coordinates(0, 0))
    .with_player2_pos(Coordinates(4, 4))
    .with_cheese([Coordinates(2, 2), Coordinates(2, 3)])
    .with_walls([Wall((1, 1), (1, 2))])
    .with_mud([Mud((2, 2), (2, 3), value=3)])
    .build())

# Or use presets: tiny (11x9), small (15x11), default, large (31x21), huge (41x31)
game = PyRat.create_preset('small')
```

### get_observation()

Player-relative view with pre-built numpy matrices:

```python
obs = game.get_observation(is_player_one=True)

# Positions (player-relative) — Coordinates objects, indexable like tuples
obs.player_position, obs.opponent_position  # Coordinates

# Scores and state
obs.player_score, obs.opponent_score        # float
obs.player_mud_turns, obs.opponent_mud_turns  # int
obs.current_turn, obs.max_turns             # int

# Pre-built matrices
obs.cheese_matrix     # uint8[W, H] — 1 where cheese exists
obs.movement_matrix   # int8[W, H, 4] — traversal info per direction
```

**Movement matrix values:**
- `-1` = wall (blocked)
- `0` = normal passage (1 turn)
- `≥2` = mud cost (stuck for N turns)

**Indexing is [x, y, direction]** (not [y, x]):
```python
can_move_up = obs.movement_matrix[x, y, Direction.UP] >= 0
has_cheese = obs.cheese_matrix[x, y] == 1
```

---

## Development Tips

1. **Equivalence is handled at boundaries** — the core algorithm works in outcome-indexed space. `reduction.py` handles translation to/from 5-action space. You rarely need to think about equivalence directly.

2. **Node statistics are reduced** — visit matrix is `[n1, n2]` where n = unique outcomes. Use `node.action_visits` if you need the expanded `[5, 5]` view. Values `v1, v2` are scalar running averages.

3. **Don't copy game state** — use `make_move/unmake_move` pattern.

4. **Pre-commit is mandatory** — don't use `--no-verify`. If pip issues, set `PIP_INDEX_URL=https://pypi.org/simple/`.

5. **The NN learns blocked actions implicitly** — training targets have 0 for blocked actions. The NN figures out which actions are blocked from the maze layout in the observation.

---

## Experiment Workflow

This repo is for ML experimentation, not just code.

### Self-Play Loop

The AlphaZero iteration: sample games → train NN → use NN to sample better games → repeat.

### Auto-Iteration (Recommended)

Use `alpharat-iterate` to run the full loop automatically:

```bash
# Run forever (Ctrl+C to stop)
alpharat-iterate configs/iterate.yaml --prefix sym_5x5

# Run exactly 3 iterations
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --iterations 3

# Start with an existing checkpoint
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --start-checkpoint path/to/model.pt

# Resume from iteration 2 (skips completed phases)
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --start-iteration 2

# Benchmark every 2nd iteration (or --no-benchmark to skip)
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --benchmark-every 2
```

The script chains: **Sample → Shard → Train → (Benchmark) → repeat**, with automatic lineage tracking via ExperimentManager. Each phase checks for existing artifacts and skips if already done.

**Artifact naming convention:**

| Artifact | Pattern | Example |
|----------|---------|---------|
| Batch | `{prefix}_iter{N}` | `sym_5x5_iter0` |
| Shard | `{prefix}_iter{N}_shards` | `sym_5x5_iter0_shards` |
| Run | `{prefix}_iter{N}` | `sym_5x5_iter0` |
| Benchmark | `{prefix}_iter{N}_benchmark` | `sym_5x5_iter0_benchmark` |

### Manual Iteration

For more control, run each step separately:

**First iteration (no NN yet):**
```bash
# 1. Sample games with pure MCTS (uniform priors)
alpharat-sample configs/sample.yaml --group iteration_0

# 2. Convert games to training shards
alpharat-prepare-shards --architecture mlp --group iter0_shards --batches iteration_0

# 3. Train NN and benchmark against baselines
alpharat-train-and-benchmark configs/train.yaml --name mlp_v1 --shards iter0_shards/UUID --games 50
```

**Subsequent iterations (with NN):**
```bash
# 1. Sample using trained NN as MCTS prior
alpharat-sample configs/sample.yaml --group iteration_1 \
    --checkpoint experiments/runs/mlp_v1/checkpoints/best_model.pt

# 2. Create shards from new games
alpharat-prepare-shards --architecture mlp --group iter1_shards --batches iteration_1

# 3. Train on new data
alpharat-train configs/train.yaml --name mlp_v2 --shards iter1_shards/UUID

# 4. Benchmark against previous iteration
alpharat-benchmark configs/tournament.yaml
```

**Scripts:**
- `iterate.py` — full auto-iteration loop (recommended for most experiments)
- `train_and_benchmark.py` — single iteration: trains then auto-benchmarks
- `train.py` + `benchmark.py` — separate scripts for fine-grained control

### Experiment Commands

The `/exp:*` commands support the experiment lifecycle:

| Command | When to use |
|---------|-------------|
| `/exp:plan` | Before — crystallize hypothesis, draft log entry |
| `/exp:iterate` | During — set up next iteration from checkpoint |
| `/exp:learn` | After — capture results, update log |
| `/exp:compare` | Anytime — compare runs side-by-side |
| `/exp:status` | Anytime — quick dashboard of where we are |

**Context files in `experiments/`:**
- `LOG.md` — the official record: roadmap, decisions, experiment entries with results
- `IDEAS.md` — parking lot for fuzzy thinking, unstructured ideas

The commands pull from these files automatically. When helping with experiments, read them first.

---

## Game Configuration Scaling

When scaling to larger grids, maintain consistent cheese density (~20%, or 1 cheese per 5 cells):

| Grid | Cells | Cheese | Density |
|------|-------|--------|---------|
| 5×5  | 25    | 5      | 20%     |
| 7×7  | 49    | 10     | 20.4%   |
| 9×9  | 81    | 16     | 19.8%   |

**Max turns**: Scale roughly with grid size. 5×5 uses 30 turns; 7×7 might use ~50.

**Data organization**: Separate folders per grid size (`batches/5x5/`, `batches/7x7/`). Models are dimension-specific — a 5×5 checkpoint won't work on 7×7.

---

## Design Docs

The `.mt/` folder contains detailed design documentation:
- `project-architecture.md` — three-script philosophy
- `mcts-implementation.md` — node design decisions
- `game-specification.md` — PyRat rules
- `design-exploration.md` — alternatives considered
