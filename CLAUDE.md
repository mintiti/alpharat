# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**alpharat** is an experimental AlphaZero adaptation implementing game-theoretic Monte Carlo Tree Search (MCTS) for simultaneous two-player games, specifically targeting the PyRat game. The project uses Python 3.11+ and employs strict type checking with mypy.

### PyRat Game Context

PyRat is a two-player simultaneous-move maze game where players (Rat and Python) compete to collect cheese. Key game mechanics:
- **Simultaneous moves**: Both players submit actions at the same time
- **Mud delays**: Movement through mud cells takes N turns, during which the player is stuck
- **Scoring**: Collect cheese for points (1 point normal, 0.5 each if collected simultaneously)
- **Zero-sum formulation**: The implementation tracks score differential (score_p1 - score_p2)
- **Actions**: UP, DOWN, LEFT, RIGHT, STAY (encoded as integers 0-4)

The simultaneous-move nature is why standard MCTS approaches don't work directly - this project implements a game-theoretic MCTS that uses Nash equilibrium computation.

## Development Commands

**Note**: This project uses `uv` for package management. Use `uv run` to execute commands in the virtual environment.

### Environment Setup
```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/mcts/test_node.py

# Run specific test
uv run pytest tests/mcts/test_node.py::test_backup_no_mud

# Run without coverage
uv run pytest --no-cov
```

### Code Quality
```bash
# Format code with ruff
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check --fix .

# Type checking
uv run mypy alpharat

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

## Architecture

### Three-Script Philosophy

The project is designed around **three independent scripts** that communicate only through persistent data:

1. **Sample Creation Script** (future): Runs MCTS agents to play games, saves game data to disk
2. **Training Script** (future): Loads saved games, trains neural networks, saves model checkpoints
3. **Evaluation Script** (future): Loads checkpoints, evaluates performance, determines best models

**Why this matters:**
- Full control over each phase (sampling, training, evaluation)
- Can reuse existing data with new model architectures
- No dependency on monolithic frameworks (avoiding RLlib, prebaked pipelines)
- Each script can be distributed independently

### Core MCTS Implementation

The MCTS implementation is adapted for simultaneous-move games and lives in `alpharat/mcts/`:

#### Node Structure (`alpharat/mcts/node.py`)

**Key difference from standard MCTS**: Uses **payout matrices** instead of single Q-values.

- **Payout matrix**: Shape `[num_p1_actions, num_p2_actions]` - tracks expected values for each action pair
- **Action visits**: Shape `[num_p1_actions, num_p2_actions]` - visit counts per action pair
- **Neural network priors**: Separate policy priors for each player (`prior_policy_p1`, `prior_policy_p2`)
- **Mud state handling**: When players are stuck in mud, updates vectorized regions:
  - P1 stuck: Updates entire column (all P1 actions → same outcome)
  - P2 stuck: Updates entire row (all P2 actions → same outcome)
  - Both stuck: Updates entire matrix

The `backup()` method uses incremental mean update: `Q_new = Q_old + (G - Q_old) / (n + 1)`

#### Tree Management (`alpharat/mcts/tree.py`)

**Key efficiency technique**: Uses PyRat's `make_move/unmake_move` pattern instead of storing full game states.

- Owns a PyRat game instance (the "simulator")
- Nodes store only `MoveUndo` objects, not full game state
- `_navigate_to()` efficiently moves simulator between nodes by:
  1. Finding common ancestor
  2. Unmaking moves back to ancestor
  3. Remaking moves forward to target
- `make_move_from()` handles both navigation and child creation
- `backup()` propagates discounted values up the tree with gamma factor

#### Nash Equilibrium (`alpharat/mcts/nash.py`)

Uses `nashpy` library for computing Nash equilibria of zero-sum games:

- `compute_nash_equilibrium()`: Takes payout matrix, returns mixed strategies for both players
- `compute_nash_value()`: Computes expected value of a strategy profile
- `select_action_from_strategy()`: Samples actions from mixed strategy with temperature control

**Current approach** (from design docs): Prior-based sampling during search, Nash equilibrium computation only at root after search completes.

### External Dependencies

- **pyrat-engine**: The PyRat game simulator (`git+https://github.com/mintiti/pyrat-rust.git@main#subdirectory=engine`)
  - Rust-backed Python module
  - Provides `PyRat` game class with `make_move/unmake_move`
  - `Direction` enum for actions
  - Access game state via `game.scores`, `game.state`, etc.
- **nashpy**: Nash equilibrium computation
- **numpy**: Matrix operations for payout matrices

## Code Style and Type Checking

### Ruff Configuration
- Line length: 100 characters
- Enabled rules: pycodestyle (E/W), pyflakes (F), isort (I), pep8-naming (N), pyupgrade (UP), bugbear (B), comprehensions (C4), simplify (SIM), type-checking (TCH)

### Mypy Configuration
- **Strict mode enabled**: All functions must have complete type annotations
- Required: `disallow_untyped_defs`, `disallow_incomplete_defs`
- Use `from __future__ import annotations` for forward references
- Type `Any` from `typing` module for game state (external PyRat types)

### Key Type Patterns
```python
from __future__ import annotations
from typing import Any
import numpy as np

# Use np.ndarray for arrays (mypy doesn't require shape annotations)
def compute_nash_equilibrium(payout_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

# Use | for unions (Python 3.11+)
parent: MCTSNode | None = None

# Use Any for external game state
game_state: Any
```

## Testing Approach

Tests live in `tests/` mirroring the `alpharat/` structure. Current coverage focus:
- Node backup mechanics with and without mud (`test_node.py`)
- Nash equilibrium computation edge cases (`test_nash.py`)
- Tree navigation and move making (`test_tree.py`)

Coverage reports generated in `htmlcov/` directory.

## Design Documentation

The `.mt/` folder contains personal design documentation:
- **project-architecture.md**: Three-script philosophy and workflow
- **mcts-implementation.md**: Detailed MCTS node design decisions
- **game-specification.md**: Complete PyRat game rules
- **design-exploration.md**: Open questions and alternative approaches

**Important design decisions from these docs:**
- Using NN prior policies during backup instead of recomputing Nash equilibrium (for stability)
- Zero-sum formulation: All values from Player 1's perspective (score_p1 - score_p2)
- Separation of concerns: Node handles statistics, Nash computation is external
- Future: Policy targets will be Nash equilibrium of post-MCTS payout matrix

## Current Implementation Status

Implemented:
- ✓ MCTSNode with payout matrices and mud handling
- ✓ MCTSTree with efficient game state navigation
- ✓ Nash equilibrium computation utilities
- ✓ Comprehensive unit tests for core components

Not yet implemented:
- ⧗ MCTS search algorithm (selection, expansion, simulation)
- ⧗ Neural network architecture
- ⧗ Training script
- ⧗ Sampling script
- ⧗ Evaluation script
- ⧗ PUCT-style selection for simultaneous moves

## Development Tips

1. **When modifying MCTS logic**: Consider mud state edge cases - they affect backup, selection, and potentially Nash computation
2. **When adding new node statistics**: Ensure they're properly shaped `[num_actions_p1, num_actions_p2]`
3. **When working with PyRat**: Remember it uses `make_move/unmake_move` - don't copy game state unnecessarily
4. **When implementing selection**: The current plan is prior-based sampling (see `.mt/design-exploration.md` for alternatives)
5. **All code must pass**: `uv run ruff check`, `uv run ruff format`, `uv run mypy`, and `uv run pytest` before committing (enforced by pre-commit hooks)
