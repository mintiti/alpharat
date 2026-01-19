# CLAUDE.md

Guidance for coding agents working in this repository.

## Design Philosophy

This repo was built around the idea that experimentation should be fast to iterate on.

The pain with most ML pipelines: everything is coupled. Change training params? Re-sample everything. Try a different architecture? Start from scratch. Hours wasted redoing work you already did.

Here, each stage — sampling, sharding, training — produces reusable artifacts. Same samples, different architectures. Same checkpoint, different MCTS params. You only redo what actually changed.

Reproducibility falls out naturally: when artifacts are saved and stages are decoupled, you always know exactly what went into each run.

## Project Overview

**alpharat** is a game-theoretic MCTS implementation for simultaneous two-player games, targeting PyRat (a maze game where both players move at the same time). Standard MCTS doesn't work for simultaneous moves — this project uses payout matrices and Nash equilibrium computation instead of single Q-values.

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
├── mcts/           # Core MCTS: nodes, tree, search, Nash equilibrium
├── data/           # Data pipeline: sampling, recording, sharding
├── nn/             # Neural network: observation builders, targets, models
├── ai/             # Agents: MCTS, random, greedy
├── eval/           # Evaluation: game execution, tournaments
└── experiments/    # Experiment management: ExperimentManager, manifest

scripts/            # Entry points: sample.py, train.py, benchmark.py, manifest.py
configs/            # YAML config templates for sampling, training, evaluation
tests/              # Mirrors alpharat/ structure
experiments/        # Data folder (NOT in git): batches, shards, runs, benchmarks
```

### alpharat/mcts/

| File | Purpose |
|------|---------|
| `node.py` | `MCTSNode` — payout matrices `[5,5]` instead of Q-values, action equivalence |
| `tree.py` | `MCTSTree` — efficient navigation via `make_move/unmake_move` |
| `decoupled_puct.py` | `DecoupledPUCTSearch` — each player picks via PUCT formula, returns Nash at root |
| `equivalence.py` | Action equivalence utilities (walls/edges/mud → same outcome) |
| `nash.py` | Nash equilibrium computation via nashpy |

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
| `targets.py` | `build_targets()` — Nash policies + value targets |
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
| `mlp` | Flat observation → shared trunk → policy/payout heads |
| `symmetric` | Structural P1/P2 symmetry, no augmentation needed |
| `local_value` | Per-cell ownership values + global payout |

**Models (`models/`):** `PyRatMLP`, `SymmetricMLP`, `LocalValueMLP`

**Builders (`builders/`):** `FlatObservationBuilder` — 1D encoding for MLPs

### alpharat/ai/

| File | Purpose |
|------|---------|
| `base.py` | `Agent` ABC — `get_move(game, player) -> int` |
| `mcts_agent.py` | `MCTSAgent` — uses MCTS search, samples from Nash |
| `random_agent.py` | `RandomAgent` — baseline |
| `greedy_agent.py` | `GreedyAgent` — moves toward closest cheese |

### alpharat/eval/

| File | Purpose |
|------|---------|
| `game.py` | `play_game()` — execute single game between agents |
| `runner.py` | `evaluate()` — run N games, compute stats |
| `tournament.py` | `run_tournament()` — round-robin with thread pool |

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
    game_params=game_params,
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

Use `manifest.py` to see what's in the experiments folder:

```bash
uv run python scripts/manifest.py batches   # List all batches
uv run python scripts/manifest.py shards    # List shards with lineage
uv run python scripts/manifest.py runs      # List training runs
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

### How It's Handled

**1. Detection (`tree.py:_compute_effective_actions`)**

At each position, we query `game.get_valid_moves(position)` to see which directions are actually valid. Each of the 5 actions maps to an "effective action":
- Valid moves → map to themselves
- Blocked moves (walls/edges) → map to STAY (action 4)
- Mud → forces all actions to STAY for that player

```python
# Example: player against north wall
p1_effective = [4, 1, 2, 3, 4]  # UP(0) blocked → STAY(4)
```

**2. Child Sharing (`tree.py:make_move_from`)**

When expanding, we look up children by *effective* action pair, not raw actions. Multiple raw action pairs that map to the same effective pair share a single child node:

```python
# Both (UP, LEFT) and (STAY, LEFT) lead to same child if UP is blocked
effective_pair = (node.p1_effective[action_p1], node.p2_effective[action_p2])
if effective_pair in node.children:
    child = node.children[effective_pair]  # Reuse existing child
```

**3. Statistics Consistency (`node.py:backup`)**

During backup, we update *all* equivalent action pairs together, not just the one played. If P1 played action 0 (UP) but it was blocked:

```python
p1_equiv = [0, 4]  # Both UP and STAY
p2_equiv = [2]     # Just DOWN
# Updates the rectangular region covering all (p1_equiv × p2_equiv) pairs
```

This maintains the invariant: **equivalent action pairs have identical payout matrix values**.

**4. Nash Computation (`equivalence.py:reduce_and_expand_nash`)**

For Nash equilibrium, we:
1. **Reduce** the payout matrix to only effective actions (removes duplicate rows/cols)
2. **Compute** Nash on the smaller matrix (unique equilibrium)
3. **Expand** strategies back to full 5-action space (non-effective get 0 probability)

```python
# If P1 has effective=[4,1,2,3,4], reduced matrix is 4×5 (only actions 1,2,3,4)
# After Nash, expand: strategy[0] = 0.0 (blocked), strategy[1:5] from Nash
```

### Flow Through Training

**Recording (`recorder.py`):**
- Saves post-MCTS Nash policies (which have 0 for blocked actions)
- Saves the 5×5 payout matrix (with equivalence structure: blocked rows equal STAY rows)

**Targets (`targets.py`):**
- Passes through the Nash policies as-is — no additional processing needed
- The 0s for blocked actions are already baked in

**NN Training:**
- The NN sees policy targets where blocked actions have probability 0
- This creates an **implicit auxiliary task**: the NN learns maze topology
- The NN doesn't know which actions are blocked — it learns this from the target structure

### What the NN Must Learn

The NN outputs:
- `policy_p1[5]`, `policy_p2[5]` — should assign ~0 to blocked actions
- `payout_matrix[5,5]` — should reflect equivalence structure

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

### Payout Matrices, Not Q-Values

Standard MCTS stores one value per action. For simultaneous games, we need the expected value for each *pair* of actions — a 5×5 payout matrix per node.

### make_move / unmake_move

Nodes don't store full game states. The tree owns one PyRat simulator and navigates by making/unmaking moves. Efficient for deep trees.

### Decoupled PUCT Selection

The search uses decoupled PUCT (`decoupled_puct.py`): each player selects actions independently via the PUCT formula, maximizing Q + exploration bonus. After search completes, Nash equilibrium is computed at the root for the final policy.

### Value Formulation

Payout matrices store **absolute cheese gains** for each player separately:
- `payout[0, i, j]` = expected cheese P1 collects from this position
- `payout[1, i, j]` = expected cheese P2 collects from this position

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

**Core:** numpy, nashpy, pydantic, pyyaml, pyrat-engine (Rust-backed)

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

1. **Action equivalence is everywhere** — when modifying MCTS logic, ensure equivalent actions share statistics. If you're touching `backup()`, `make_move_from()`, or Nash computation, think through the equivalence implications.

2. **Node statistics are [5,5]** — payout matrix and visit counts are per action pair, not per action.

3. **Don't copy game state** — use `make_move/unmake_move` pattern.

4. **Pre-commit is mandatory** — don't use `--no-verify`. If pip issues, set `PIP_INDEX_URL=https://pypi.org/simple/`.

5. **The NN learns blocked actions implicitly** — training targets have 0 for blocked actions. The NN figures out which actions are blocked from the maze layout in the observation.

---

## Experiment Workflow

This repo is for ML experimentation, not just code.

### Self-Play Loop

The AlphaZero iteration: sample games → train NN → use NN to sample better games → repeat.

**First iteration (no NN yet):**
```bash
# 1. Sample games with pure MCTS (uniform priors)
uv run python scripts/sample.py configs/sample.yaml --group iteration_0

# 2. Convert games to training shards
uv run python scripts/prepare_shards.py --architecture mlp --group iter0_shards --batches iteration_0

# 3. Train NN and benchmark against baselines
uv run python scripts/train_and_benchmark.py configs/train.yaml --name mlp_v1 --shards iter0_shards/UUID --games 50
```

**Subsequent iterations (with NN):**
```bash
# 1. Sample using trained NN as MCTS prior
uv run python scripts/sample.py configs/sample_with_nn.yaml --group iteration_1 \
    --checkpoint experiments/runs/mlp_v1/checkpoints/best_model.pt

# 2. Create shards from new games
uv run python scripts/prepare_shards.py --architecture mlp --group iter1_shards --batches iteration_1

# 3. Train on new data
uv run python scripts/train.py configs/train.yaml --name mlp_v2 --shards iter1_shards/UUID

# 4. Benchmark against previous iteration
uv run python scripts/benchmark.py configs/tournament.yaml
```

**Scripts:**
- `train_and_benchmark.py` — convenience script: trains then auto-benchmarks vs Random/Greedy/MCTS
- `train.py` + `benchmark.py` — separate scripts for more control (custom tournament configs)

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

## Design Docs

The `.mt/` folder contains detailed design documentation:
- `project-architecture.md` — three-script philosophy
- `mcts-implementation.md` — node design decisions
- `game-specification.md` — PyRat rules
- `design-exploration.md` — alternatives considered
