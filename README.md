# alpharat

Experimental AlphaZero-style MCTS for simultaneous two-player games.

Standard MCTS assumes players take turns. PyRat (the target game here) has both players moving at the same time, which breaks that assumption. This project tries using payout matrices instead of single Q-values, and computing Nash equilibrium instead of just picking the best action.

Still figuring out if it actually works.

## Getting started

```bash
uv sync
uv sync --extra train  # PyTorch: CUDA on Linux, CPU on macOS

# CPU-only on Linux (if needed)
uv pip install torch --torch-backend=cpu --reinstall

# Run tests
uv run pytest

# Generate training data
uv run python scripts/sample.py configs/sample.yaml
```

## What's here

- `alpharat/mcts/` — Tree search with 5×5 payout matrices per node, handles action equivalence, computes Nash at root
- `alpharat/data/` — Self-play sampling, game recording, training set sharding
- `alpharat/nn/` — Observation encoding, training targets, MLP model
- `alpharat/ai/` — Agents (MCTS, random, greedy baselines)
- `alpharat/eval/` — Running games and tournaments

## The approach

When both players move at once, you can't just maximize — the opponent is choosing too. So each node stores expected values for all 25 action pairs, and action selection uses Nash equilibrium.

There's also a subtlety where multiple actions can be equivalent (hitting a wall = staying put). The tree shares branches for those and reduces the matrix before computing Nash.

See [CLAUDE.md](CLAUDE.md) for implementation details.

## Status

**5×5 validation complete.** Self-play loop works — each iteration produces a stronger model.

| Milestone | Status |
|-----------|--------|
| MCTS beats Random/Greedy | ✅ |
| NN learns from MCTS | ✅ |
| MCTS+NN beats pure MCTS | ✅ |
| Self-play iteration improves models | ✅ |
| Walls / mud / larger grids | Not yet tested |

Current best: 1106 Elo (MCTS+NN after 2 iterations), undefeated in 300 games.

See [.mt/experiment-log.md](.mt/experiment-log.md) for full results and roadmap.

## License

MIT
