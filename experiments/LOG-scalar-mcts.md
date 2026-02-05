# Experiment Log: Scalar Value Heads

Tracking experiments for scalar value MCTS — replacing the full 5×5 payout matrix with scalar value estimates per player.

**Branch:** `worktree/scalar-value-heads`

---

## Background

### Payout Matrix vs Scalar Value

**Current approach (payout matrix):**
- NN predicts `[2, 5, 5]` payout matrix — expected cheese for each player under each action pair
- MCTS backs up the full matrix, computes Nash equilibrium at root
- 50 output values per position

**Scalar value approach:**
- NN predicts `[2]` scalar values — expected total cheese for P1 and P2 from this position
- MCTS backs up scalars, uses policies directly (no Nash computation at root)
- 2 output values per position

### Motivation

- Simpler architecture, fewer parameters in value head
- Faster inference (no Nash equilibrium computation)
- Question: Does the game-theoretic structure of payout matrices actually help?

---

## Optuna Sweep

### Setup

- Grid: 7×7 open maze (no walls, no mud)
- Cheese: 10, max turns: 50
- Opponent: Greedy baseline
- Games per config: 100
- Objectives: maximize win rate, minimize n_sims (Pareto front)

### Parameters

| Param | Range | Scale |
|-------|-------|-------|
| n_sims | 200–1200 | log |
| c_puct | 0.5–16.0 | log |
| force_k | 0.01–64.0 | log |

### Results

*To be filled as sweep completes.*

---

## Log

### 2026-01-31: Started Optuna Sweep

Initial sweep to find good PUCT parameters for scalar value MCTS against Greedy.

Seeded with configs from previous payout matrix sweep:
- n_sims=605, c_puct=0.89, force_k=0.04 (39.5% win rate on payout matrix)
- n_sims=507, c_puct=0.95, force_k=0.06 (37.5%)
- n_sims=422, c_puct=0.62, force_k=0.27 (32.5%)

**Issue encountered:** SQLite database locking with `n_jobs > 1`. Optuna's parallel workers hit concurrent write contention on the `.db` file. Fixed by switching to `JournalStorage` which handles concurrent writes properly.

### 2026-02-03: Self-Play Validation (7×7)

**Goal:** Validate that scalar value MCTS matches or exceeds the visits-based payout matrix approach.

**Type:** Iteration (first full test of new architecture)

**Baseline (from visits experiments on payout matrix branch):**

| Experiment | mcts+nn vs greedy (W/D/L) | mcts+nn Elo | nn Elo |
|------------|---------------------------|-------------|--------|
| visits_iter1 | 6/40/4 | 1004 | 630 |
| visits_iter2 | 7/40/3 | 1059 | 1024 |
| visits_matrixloss_iter2 | 4/44/2 | 1054 | 979 |

**Setup:**

| Component | Config |
|-----------|--------|
| Game | 7×7 open, 10 cheese, 50 turns |
| MCTS | 593 sims, c_puct=1.17, force_k=0.012 |
| Model | SymmetricMLP (hidden=256, dropout=0.1) |
| Training | policy_weight=1.0, value_weight=1.0, lr=1e-3 |
| Data | 50k games/iteration, 2 iterations |

**Command:**
```bash
alpharat-iterate configs/iterate.yaml \
  --prefix scalar_7x7 \
  --iterations 2 \
  game=7x7_open \
  mcts=7x7_scalar_tuned \
  model=symmetric \
  iteration.games=50000
```

**Success criteria:**

| Metric | iter1 target | iter2 target | Rationale |
|--------|--------------|--------------|-----------|
| mcts+nn vs greedy W/D/L | ≥6 wins, ≤5 losses | ≥6 wins, ≤3 losses | Match visits baseline |
| Losses to greedy | ≤6 | ≤3 | Robustness — visits got down to 2-3 losses |
| iter1→iter2 mcts+nn Elo gain | — | > +20 | visits showed +55, should see improvement |
| nn Elo improvement | — | > iter1 nn | Self-play loop producing learning signal |

**What could go wrong:**
- Scalar backup doesn't give MCTS enough signal (payout matrices had richer structure)
- Training fails to converge
- No improvement across iterations (like Nash-based approach stalled)

---

### 2026-02-05: Iter0 Results

**Initial run (100 epochs):** mcts+nn underperformed greedy (988.5 vs 1000 Elo, 4/42/4 vs greedy).

**Investigation findings:**
- Training metrics were actually good: 62% train accuracy (vs 59% for visits_iter1), lower loss
- No overfitting issues — scalar had smallest train/val gap

**Extended training (300 epochs):** Improved significantly.

| Metric | 100 epochs | 300 epochs |
|--------|------------|------------|
| mcts+nn Elo | 988.5 | **1013.8** |
| mcts+nn vs greedy | 4/42/4 | **8/38/4** |
| nn Elo | 881.8 | 909.8 |

**Comparison to payout matrix iter1 baselines:**

| Experiment | mcts+nn vs greedy (W/D/L) | Notes |
|------------|---------------------------|-------|
| **scalar_7x7_300ep** | **8/38/4** | ✓ Matches best iter1 |
| 7x7_matrixloss_v1_run2 | 7/40/3 | |
| 7x7_visits_iter1 | 6/40/4 | |
| 7x7_matrixloss_nocsum_run2 | 2/47/1 | Most draws, fewest losses |

**Conclusion:** Scalar value heads match payout matrix performance at iter1. The simpler architecture (2 outputs vs 50) works. Needed 300 epochs vs 100.

**Next:** Continue to iter1 with NN-guided sampling to test if self-play loop works.

---

## Open Questions

1. Does scalar value MCTS match payout matrix MCTS at the same sim budget?
2. How do optimal PUCT parameters differ between the two approaches?
3. Does training a NN on scalar targets work as well as payout matrix targets?
