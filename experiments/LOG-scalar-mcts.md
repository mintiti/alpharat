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

### 2026-02-09: Dirichlet Noise for Data Diversity

**Goal:** Does adding Dirichlet noise to MCTS root priors during sampling reduce value function overfitting and produce a stronger checkpoint?

**Hypothesis:** The value head overfits because self-play positions lack diversity — symmetric play, same openings, same cheese paths. Dirichlet noise at the root forces MCTS to explore suboptimal moves sometimes, producing more varied positions. This should reduce the train/val gap on value loss and possibly improve mcts+nn Elo through better value estimates.

**Setup:**

| | Baseline | Dirichlet |
|--|----------|-----------|
| MCTS | 475 sims, c_puct=0.529, force_k=0.017, fpu=0.264 | Same + Dirichlet noise at root |
| Game | 7×7 open, 10 cheese, 50 turns | Same |
| Model | SymmetricMLP (hidden=256, dropout=0.1) | Same |
| Data | 50k games, 1 iteration (no NN prior) | Same |
| Training | policy=1.0, value=1.0, 300 epochs | Same |

**Dirichlet params (AlphaZero-style):**
- `noise_alpha`: TBD (AlphaZero used 0.03 for Go, 0.3 for chess — fewer actions → larger alpha)
- `noise_epsilon`: 0.25 (standard — 75% prior, 25% noise)
- Applied at root only, during sampling only

**Baseline:** `scalar_baseline_7x7` — one iteration, no noise

**Success criteria:**
- Confirm if: val value loss lower (less overfitting), mcts+nn Elo ≥ baseline
- Deny if: Elo drops or val loss unchanged

**Metrics to compare:**
- Train vs val value loss curves (the overfitting signal)
- mcts+nn vs greedy W/D/L
- Head-to-head: Dirichlet checkpoint vs baseline checkpoint
- Sampling stats: draw rate, avg game length, cheese collected (diversity proxies)

**Type:** hypothesis_test

**Commands:**
```bash
# Step 1: Baseline (no noise)
alpharat-iterate configs/iterate_7x7.yaml \
  --prefix scalar_baseline_7x7 \
  --iterations 1

# Step 2: Implement Dirichlet noise in MCTS root (code change)

# Step 3: Dirichlet iteration (config TBD after implementation)
alpharat-iterate configs/iterate_7x7.yaml \
  --prefix scalar_dirichlet_7x7 \
  --iterations 1
```

#### Baseline Results (iter0, no noise, 300 epochs)

| Rank | Agent | W | D | L | Pts | AvgCheese | Elo |
|------|-------|---|---|---|-----|-----------|-----|
| 1 | mcts+nn | 91 | 106 | 3 | 144.0 | 5.4 | 1020 |
| 2 | greedy | 88 | 104 | 8 | 140.0 | 5.4 | 1000 |
| 3 | mcts | 58 | 102 | 40 | 109.0 | 5.0 | 897 |
| 4 | nn | 59 | 92 | 49 | 105.0 | 4.9 | 882 |
| 5 | random | 0 | 4 | 196 | 2.0 | 1.4 | 216 |

**mcts+nn vs greedy:** 5/43/2 — strong, only 2 losses. +20 Elo over greedy anchor.

**Head-to-head highlights:**
- mcts+nn dominates nn (20/30/0) — MCTS search adds a lot on top of the NN
- greedy vs mcts (16/33/1) — greedy still beats pure MCTS most of the time
- nn barely edges mcts (8/33/9 vs 9/33/8) — they're close without search

**Note:** These results use the new MCTS params from Optuna (475 sims, c_puct=0.529, force_k=0.017, fpu_reduction=0.264) which differ from the earlier iter0 (593 sims, c_puct=1.17, force_k=0.012). Direct Elo comparison to the earlier iter0 (1013.8) isn't meaningful — different params, different tournament.

**Next:** Run Dirichlet noise variant with same setup to test diversity hypothesis.

#### Dirichlet Results (iter0, alpha=2.0, epsilon=0.25, 300 epochs)

| Rank | Agent | W | D | L | Pts | AvgCheese | Elo |
|------|-------|---|---|---|-----|-----------|-----|
| 1 | greedy | 101 | 95 | 4 | 148.5 | 5.4 | 1000 |
| 2 | mcts+nn | 89 | 100 | 11 | 139.0 | 5.4 | 972 |
| 3 | mcts | 62 | 93 | 45 | 108.5 | 5.0 | 864 |
| 4 | nn | 54 | 95 | 51 | 101.5 | 4.8 | 839 |
| 5 | random | 1 | 3 | 196 | 2.5 | 1.3 | 208 |

**mcts+nn vs greedy:** 3/38/9 — worse than baseline (5/43/2). 9 losses vs 2.

**Dirichlet alpha:** 2.0 (following AlphaZero's ~10/N_actions scaling: 0.03 for Go/361, 0.3 for chess/30, 2.0 for PyRat/5).

#### Comparison: Baseline vs Dirichlet

| Agent | Baseline Elo | Dirichlet Elo | Delta |
|-------|-------------|---------------|-------|
| mcts+nn | 1020 | 972 | **-48** |
| greedy | 1000 | 1000 | (anchor) |
| mcts | 897 | 864 | -33 |
| nn | 882 | 839 | -43 |

**Conclusion:** Dirichlet noise hurt at iter0. Hypothesis denied for this setting.

**Observations:**
- Training curves were much smoother with Dirichlet (loss, accuracy — all monotonic, less epoch-to-epoch variance). More diverse data reduces batch variance, but the NN converges to a weaker checkpoint.
- The NN seems to learn a "blurry" policy — averaging over noisy and clean play — rather than a sharp one. Smoother learning ≠ better learning.
- Both nn and mcts+nn dropped by similar amounts (~43-48 Elo), suggesting the damage is in the learned policy/value quality, not in how MCTS uses it.

**Caveat:** This is iter0 — uniform priors, no NN. Dirichlet noise is designed to perturb a *learned* prior to maintain exploration. Adding noise on top of uniform priors is noise-on-noise. The real test would be iter1+ where there's a sharp prior worth diversifying. But the iter0 signal is negative enough that this isn't a priority.

**Next:** Move forward with the baseline (no noise) for iter1. Dirichlet might be worth revisiting once we have multi-iteration runs where overfitting becomes the bottleneck.

---

## Open Questions

1. Does scalar value MCTS match payout matrix MCTS at the same sim budget?
2. How do optimal PUCT parameters differ between the two approaches?
3. Does training a NN on scalar targets work as well as payout matrix targets?
