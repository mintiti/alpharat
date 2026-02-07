# alpharat

Experimental AlphaZero-style MCTS for simultaneous two-player games.

Standard MCTS assumes players take turns. PyRat (the target game here) has both players moving at the same time, which breaks that assumption. This project uses scalar value heads (like LC0/KataGo) with decoupled PUCT selection and visit-proportional policies.

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
alpharat-sample configs/sample.yaml
```

## Self-play workflow

The AlphaZero loop: sample games → train NN → use NN to sample better games → repeat.

### First iteration (no NN yet)

```bash
# 1. Sample games with pure MCTS (uniform priors)
alpharat-sample configs/sample.yaml --group uniform_5x5

# 2. Convert games to training shards
alpharat-prepare-shards --group 5x5_v1 --batches uniform_5x5

# 3. Train NN and benchmark against baselines (Random, Greedy, MCTS)
#    Edit configs/train.yaml: set name and data paths, then:
alpharat-train-and-benchmark configs/train.yaml --name mlp_v1 --games 50
```

### Subsequent iterations (with NN)

```bash
# 1. Sample using trained NN as MCTS prior
#    Edit configs/sample_with_nn.yaml: set checkpoint path, then:
alpharat-sample configs/sample_with_nn.yaml --group nn_guided_v1

# 2. Create shards from new games
alpharat-prepare-shards --group 5x5_v2 --batches nn_guided_v1

# 3. Train on new data (optionally resuming from previous checkpoint)
alpharat-train configs/train.yaml --name mlp_v2

# 4. Benchmark against previous iteration
#    Edit tournament config to include previous checkpoint, then:
alpharat-benchmark configs/tournament.yaml
```

### CLI commands

| Command | Purpose |
|---------|---------|
| `alpharat-sample` | Generate self-play games with MCTS (with or without NN prior) |
| `alpharat-prepare-shards` | Convert game batches to shuffled train/val shards |
| `alpharat-train` | Train NN on shards |
| `alpharat-benchmark` | Run tournament between agents (custom matchups) |
| `alpharat-train-and-benchmark` | Convenience: train + auto-benchmark vs baselines |
| `alpharat-manifest` | Query artifacts: list batches, shards, runs with lineage |

### Where things go

```
experiments/
├── batches/{group}/{uuid}/   # Raw game recordings from sampling
├── shards/{group}/{uuid}/    # Processed train/val splits
├── runs/{name}/              # Training runs with checkpoints
└── benchmarks/{name}/        # Tournament results
```

The `experiments/manifest.yaml` tracks lineage (which shards came from which batches, etc.).

Quick scan of what exists:
```bash
alpharat-manifest batches  # See all batch groups
alpharat-manifest shards   # See shards + which batches they came from
alpharat-manifest runs     # See training runs + which shards they used
```

## What's here

- `alpharat/mcts/` — Tree search with scalar value heads, handles action equivalence, visit-proportional policies
- `alpharat/data/` — Self-play sampling, game recording, training set sharding
- `alpharat/nn/` — Observation encoding, training targets, MLP model
- `alpharat/ai/` — Agents (MCTS, random, greedy baselines)
- `alpharat/eval/` — Running games and tournaments

## The approach

When both players move at once, you can't just maximize — the opponent is choosing too. Each node stores scalar value estimates (expected remaining cheese per player), and action selection uses decoupled PUCT where each player independently picks via exploration bonus. The final policy is visit-proportional.

There's also a subtlety where multiple actions can be equivalent (hitting a wall = staying put). The tree shares branches for those and reduces the visit/Q matrices accordingly.

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

See [experiments/LOG.md](experiments/LOG.md) for full results and roadmap.

## License

MIT
