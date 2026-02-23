# alpharat

Experimental AlphaZero-style MCTS for simultaneous two-player games.

Standard MCTS assumes players take turns. PyRat (the target game here) has both players moving at the same time, which breaks that assumption. This project uses scalar value heads (like LC0/KataGo) with decoupled PUCT selection and visit-proportional policies.

Still figuring out if it actually works.

## Getting started

```bash
uv sync                # Requires Rust toolchain (builds the Rust crates)
uv sync --extra train  # PyTorch: CUDA on Linux, CPU on macOS

# CPU-only on Linux (if needed)
uv pip install torch --torch-backend=cpu --reinstall

# Run tests
uv run pytest
```

For GPU inference setup (CUDA, TensorRT, CoreML), see the [Inference Backends](CLAUDE.md#inference-backends) section in CLAUDE.md.

## Self-play workflow

The AlphaZero loop: sample games → train NN → use NN to sample better games → repeat.

### Auto-iteration (recommended)

```bash
# Run the full loop: sample → shard → train → benchmark → repeat
alpharat-iterate configs/iterate.yaml --prefix sym_5x5

# With GPU inference
alpharat-iterate configs/iterate.yaml --prefix sym_5x5 --device cuda
```

### Manual steps

```bash
# 1. Sample games with Rust self-play (fast, supports GPU)
alpharat-rust-sample configs/iterate.yaml --group iteration_0 --num-games 1000 --device cuda

# 2. Convert games to training shards
alpharat-prepare-shards --architecture mlp --group iter0_shards --batches iteration_0

# 3. Train NN and benchmark against baselines
alpharat-train-and-benchmark configs/train.yaml --name mlp_v1 --shards iter0_shards/UUID --games 50

# 4. Next iteration — sample with the trained NN
alpharat-rust-sample configs/iterate.yaml --group iteration_1 --num-games 1000 --device cuda \
    --checkpoint experiments/runs/mlp_v1/checkpoints/best_model.pt
```

Legacy Python sampling (`alpharat-sample`) is still available but slower.

### CLI commands

| Command | Purpose |
|---------|---------|
| `alpharat-iterate` | Auto-iteration loop: sample → shard → train → benchmark → repeat |
| `alpharat-rust-sample` | Rust self-play sampling with GPU inference (`--device cuda/tensorrt/coreml`) |
| `alpharat-sample` | Legacy Python self-play (slower, use `alpharat-rust-sample` instead) |
| `alpharat-prepare-shards` | Convert game batches to shuffled train/val shards |
| `alpharat-train` | Train NN on shards |
| `alpharat-benchmark` | Run tournament between agents (custom matchups) |
| `alpharat-train-and-benchmark` | Convenience: train + auto-benchmark vs baselines |
| `alpharat-export-onnx` | Export PyTorch checkpoint to ONNX for Rust inference |
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

## Rust MCTS backend

There's a Rust implementation of the same MCTS algorithm. Same interface, faster search.

`uv sync` builds it automatically (requires a Rust toolchain). To use it, set `backend: rust` in your MCTS config or override via CLI:

```bash
# Use the tuned Rust config for sampling
alpharat-sample configs/sample.yaml mcts=7x7_rust_tuned --group my_batch

# Or for iteration
alpharat-iterate configs/iterate.yaml --prefix rust_7x7 mcts=7x7_rust_tuned
```

Pre-tuned Rust configs in `configs/mcts/`: `7x7_rust_tuned`, `7x7_rust_fast`, `7x7_rust_strong` (from an Optuna sweep, 3956 trials).

In Python:

```python
from alpharat.mcts.config import RustMCTSConfig

config = RustMCTSConfig(simulations=1897, c_puct=0.512)
searcher = config.build_searcher()               # uniform priors
searcher = config.build_searcher(checkpoint=...)  # NN-guided
```

Both backends implement the `Searcher` protocol — consumers don't need to know which one they're using.

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
