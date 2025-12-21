# CLAUDE.md

Guidance for coding agents working in this repository.

## Project Overview

**alpharat** is a game-theoretic MCTS implementation for simultaneous two-player games, targeting PyRat (a maze game where both players move at the same time). Standard MCTS doesn't work for simultaneous moves — this project uses payout matrices and Nash equilibrium computation instead of single Q-values.

Python 3.11+, strict mypy, `uv` for package management.

## Development Commands

```bash
# Setup
uv sync                              # Install dependencies
uv sync --extra train                # Include PyTorch for training
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
└── eval/           # Evaluation: game execution, tournaments

scripts/            # Entry points: sample.py, train_mcts.py
configs/            # YAML configs for sampling, evaluation, benchmarks
tests/              # Mirrors alpharat/ structure
```

### alpharat/mcts/

| File | Purpose |
|------|---------|
| `node.py` | `MCTSNode` — payout matrices `[5,5]` instead of Q-values, action equivalence |
| `tree.py` | `MCTSTree` — efficient navigation via `make_move/unmake_move` |
| `search.py` | `MCTSSearch` — prior-based sampling, returns Nash equilibrium at root |
| `decoupled_puct.py` | Alternative selection: each player picks via PUCT formula independently |
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

| File | Purpose |
|------|---------|
| `types.py` | `ObservationInput`, `TargetBundle` — source-agnostic data types |
| `extraction.py` | Build `ObservationInput` from game arrays or live PyRat |
| `targets.py` | `build_targets()` — Nash policies + value targets |
| `streaming.py` | `StreamingDataset` — memory-efficient PyTorch dataset |
| `builders/flat.py` | `FlatObservationBuilder` — 1D encoding for MLPs |
| `models/mlp.py` | `PyRatMLP` — shared trunk + policy/payout heads |

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

### Prior Sampling vs Decoupled PUCT

Two selection strategies:
- **Prior sampling** (`search.py`): Sample actions from NN policy priors during search
- **Decoupled PUCT** (`decoupled_puct.py`): Each player selects via PUCT formula independently

Both compute Nash equilibrium at the root after search completes.

### Zero-Sum Formulation

All values from Player 1's perspective: `score_p1 - score_p2`. Value of +2 means P1 ahead by 2 cheese.

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
from pyrat_engine.core.types import Direction
```

---

## Development Tips

1. **Action equivalence is everywhere** — when modifying MCTS logic, ensure equivalent actions share statistics. If you're touching `backup()`, `make_move_from()`, or Nash computation, think through the equivalence implications.

2. **Node statistics are [5,5]** — payout matrix and visit counts are per action pair, not per action.

3. **Don't copy game state** — use `make_move/unmake_move` pattern.

4. **Pre-commit is mandatory** — don't use `--no-verify`. If pip issues, set `PIP_INDEX_URL=https://pypi.org/simple/`.

5. **The NN learns blocked actions implicitly** — training targets have 0 for blocked actions. The NN figures out which actions are blocked from the maze layout in the observation.

---

## Design Docs

The `.mt/` folder contains detailed design documentation:
- `project-architecture.md` — three-script philosophy
- `mcts-implementation.md` — node design decisions
- `game-specification.md` — PyRat rules
- `design-exploration.md` — alternatives considered
