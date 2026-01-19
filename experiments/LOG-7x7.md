# Experiment Log: 7x7 Grid (No Walls, No Mud)

Tracking experiments for 7x7 PyRat games. Separate from 5x5 to keep things organized.

---

## 2026-01-19: Initial 7x7 Training

### Setup

**Sampling:**
- Grid: 7x7 (49 cells)
- Cheese: 10 (~20% density, matching 5x5 ratio)
- Max turns: 50
- MCTS: 554 sims, c_puct=8.34, force_k=0.88
- Games: 50,000 → 884,328 positions

**Sampling results:**
- W/D/L: 6718/36749/6533 (73% draws)
- Cheese collected: 98.2%
- Avg game length: 17.7 turns

**Training:**
- Architecture: SymmetricMLP (291,102 params)
- hidden_dim=256, dropout=0.1
- lr=1e-3, batch_size=4096
- Losses: policy (1.0), value (1.0), nash (1.0), constant_sum (1.0)

### Results

**Training curve:**
- Best val loss: 2.2515 at epoch 84
- Train loss at epoch 200: 2.10
- Clear overfitting after ~epoch 30 (val plateaus, train keeps dropping)

**Benchmark (WRONG SETTINGS — 5 cheese, 30 turns instead of 10 cheese, 50 turns):**
| Agent | Elo |
|-------|-----|
| greedy | 1000 (anchor) |
| mcts | 716 |
| mcts+nn | 606 |
| nn | 498 |
| random | 31 |

NN **hurts** MCTS (-110 Elo). But this benchmark used wrong game settings — need to re-run.

### Issues Found

1. **Benchmark game mismatch**: `train_and_benchmark.py` hardcoded `cheese_count=5, max_turns=30` regardless of training data. Fixed to read from checkpoint config (but checkpoint doesn't store game params yet).

2. **Overfitting**: Val loss plateaus early while train loss keeps dropping. Need to try:
   - Earlier stopping (epoch 30-50)
   - More dropout
   - Less hidden_dim
   - More data

### Next Steps

1. ~~Re-run benchmark with correct settings (10 cheese, 50 turns) using `configs/benchmark_7x7.yaml`~~ ✓
2. Try training with more regularization
3. ~~Store game params in checkpoint for automatic benchmark matching~~ ✓

---

## 2026-01-19: Fixed Train/Test Mismatch

### The Bug

NN was performing **worse than random** in benchmarks. Investigation revealed:

- **Training data**: Generated with `wall_density=0.0, mud_density=0.0` (no walls)
- **Benchmark games**: Used PyRat's default wall density (non-zero)

The NN had never seen walls during training. When evaluated on mazes with walls, it output garbage (99.999% LEFT regardless of position).

### The Fix

Added `wall_density: 0.0, mud_density: 0.0` to:
- `configs/benchmark_7x7.yaml`
- `configs/train_7x7.yaml`

### Benchmark Results (Correct Settings)

| Rank | Agent | Elo | W | D | L | Avg Cheese |
|------|-------|-----|---|---|---|------------|
| 1 | mcts+nn | 1003.5 | 86 | 108 | 6 | 5.4 |
| 2 | greedy | 1000.0 | 92 | 98 | 10 | 5.4 |
| 3 | nn | 897.9 | 61 | 98 | 41 | 5.0 |
| 4 | mcts | 894.3 | 55 | 108 | 37 | 5.0 |
| 5 | random | 75.8 | 0 | 0 | 200 | 1.1 |

**Key observations:**
- **mcts+nn is now best** — +3.5 Elo over greedy anchor
- **nn ≈ mcts** — pure NN roughly matches 200-sim MCTS (897.9 vs 894.3)
- NN helps MCTS (+109 Elo over raw mcts)
- High draw rate (36/50 typical) — symmetric starting positions lead to symmetric play

### Head-to-Head

```
                random      greedy        mcts          nn     mcts+nn
greedy          50/0/0           -     14/36/0     22/26/2      6/36/8
mcts+nn         50/0/0      8/36/6     14/36/0     14/36/0           -
nn              50/0/0     2/26/22      9/36/5           -     0/36/14
```

- mcts+nn beats greedy 8-6 (36 draws)
- mcts+nn beats mcts 14-0 (36 draws) — NN priors clearly help
- mcts+nn beats nn 14-0 (36 draws) — search still matters

### Lessons Learned

> ⚠️ **Critical**: NNs are brittle to wall/mud distribution mismatch. A model trained on wall-free mazes outputs garbage on walled mazes (and vice versa). Always ensure sampling config matches evaluation config for `wall_density` and `mud_density`.

1. **Always match train/test distributions** — wall density, mud density, cheese count, max turns all matter
2. **Check sampling metadata** when debugging — the answer was in `batches/7x7/.../metadata.json`
3. **Observations should be comparable** — when NN fails catastrophically, compare live obs stats with training obs stats
