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

---

## 2026-01-20: Iteration 2 — Self-Play with NN Priors

### Setup

**Sampling:**
- Used iter1 checkpoint as MCTS prior
- Games: 50,000 → 584,185 positions (vs 884k in iter1)
- W/D/L: 2126/45657/2217 (91.3% draws — up from 73%)
- Cheese collected: 99.9%
- Avg game length: 11.7 turns (vs 17.7 in iter1)

Shorter games + higher draw rate = NN-guided MCTS plays more symmetrically.

**Training:**
- Same architecture (SymmetricMLP, 291k params)
- Resumed from iter1 checkpoint
- Trained 100 epochs on iter2 data

### Results

| Agent | Elo | Notes |
|-------|-----|-------|
| mcts+nn | 1026.2 | +0.7 vs iter1 |
| mcts+nn-prev | 1025.0 | iter1 checkpoint |
| greedy | 1000.0 | anchor |
| nn | 973.8 | +93.3 vs iter1 |
| mcts | 884.2 | -10.1 vs iter1 |
| nn-prev | 880.5 | iter1 checkpoint |
| random | 169.5 | |

**Head-to-head (W/D/L):**
```
              mcts+nn  mcts+nn-prev  greedy    nn
mcts+nn           -       5/39/6     8/38/4   11/38/1
mcts+nn-prev   6/39/5        -       6/37/7    9/36/5
nn             1/38/11   5/36/9    3/40/7       -
```

### Observations

1. **NN improved a lot** (+93 Elo). The self-play loop is working — learning from stronger data makes a stronger NN.

2. **MCTS+NN barely moved** (+1 Elo). The combined agent didn't improve despite the NN improving.

3. **Search contributes less:**
   - Iter1: mcts+nn - nn ≈ 125 Elo gap
   - Iter2: mcts+nn - nn ≈ 52 Elo gap

   Search is adding half as much value now.

4. **Many Nash equilibrium failures** during benchmark. Lots of "No Nash equilibrium found, falling back to prior" warnings.

5. **91% draw rate** — games are very symmetric. Both agents learned to play the "correct" response, leading to mirror matches.

### Interpretation

The NN keeps learning (policy improves), but search is hitting a ceiling.

**Leading hypothesis: NN payout matrix predictions are garbage.**

The NN outputs two things: policy priors and payout matrices. The policy is clearly improving (raw NN went +93 Elo). But the payout matrix might not be learning anything useful. If the predicted payout matrices are nonsense:
- Nash computation on garbage → garbage mixed strategy
- MCTS backup uses garbage values → tree statistics are unreliable
- Search can't improve on the prior because it's reasoning over bad payouts

This would explain why:
- Raw NN improves (policy head learns)
- MCTS+NN plateaus (payout head doesn't learn, search can't help)
- Nash failures increase (degenerate/nonsense matrices)

### Diagnostic Signal

Compared NN predictions vs MCTS-computed targets:

| Target | Correlation | Explained Variance |
|--------|-------------|-------------------|
| Value (scalar) | 0.99 | 0.97 |
| Payout matrix (5×5) | 0.10 | <<< -1 |

The NN learns **value** perfectly but **payout matrix** is garbage.

Why: MCTS only explores a subset of action pairs deeply. Explored pairs have good signal, unexplored pairs are noise/zeros in training data. The NN can't learn what it never sees.

### Next Steps (prioritized)

1. **Confidence-weighted payout loss** — Learn full 5×5 matrices, but weight by how confident we are in each entry. MCTS gives signal for all pairs (not just played), it's just noisier than value from actual rollout. Weight by visit counts or similar.

2. **Spatial architectures** — GNN/CNN/Transformer that learns game-relevant spatial features. Construct payout matrix from (H, W, feat_size) encoding rather than flat MLP. Auxiliary task: predict which cheese gets taken by whom. (See IDEAS.md for details)

---

## 2026-01-20: Payout Matrix Quality — Experiment Plan

### Problem

Iter2 diagnostic showed:
- Value correlation: 0.99 (NN learns value)
- Payout matrix correlation: 0.10 (NN doesn't learn payout)

MCTS only explores 1-2 action pairs deeply. Other pairs have garbage values. NN can't learn what it never sees.

### Approach

Two-part fix:

**1. Forced playout pruning (KataGo-style, adapted for simultaneous games)**

After search, subtract forced visits from the payout matrix before recording:
- Compute marginal adjustments per player using PUCT constraint
- Distribute to pairs: `V'[i,j] = max(0, V[i,j] - Δ₁[i]·π₂[j] - Δ₂[j]·π₁[i])`
- This removes "artificial" exploration from training targets

**2. Matrix loss**

Add MSE loss on full payout matrix (NN vs MCTS estimates), alongside existing outcome loss. Start with weight=0.1.

### Experiment Sequence

| Exp | What | Success Criteria |
|-----|------|------------------|
| A | Pruning only | Payout corr > 0.10, Elo stable |
| B | Pruning + matrix loss | Payout corr > 0.5, MCTS+NN Elo > 1026 |
| C | Iteration 3 with fixes | Continued improvement |

### Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Payout correlation | 0.10 | > 0.5 |
| Value correlation | 0.99 | ~0.99 |
| MCTS+NN Elo | 1026 | > 1050 |

### Implementation Status

**Completed:**
- `alpharat/mcts/selection.py` — Added `compute_pruning_adjustment()` and `prune_visit_counts()`
- `alpharat/mcts/decoupled_puct.py` — `_make_result()` now prunes forced visits when `force_k > 0`
- `alpharat/nn/architectures/symmetric/loss.py` — Added matrix loss with `matrix_loss_weight` config
- Tests added for pruning functions

**Usage:**
```yaml
# In train config (symmetric architecture)
optim:
  architecture: symmetric
  matrix_loss_weight: 0.1  # New: MSE on full payout matrix
```

Pruning is automatic when `force_k > 0` in MCTS config.

---

## 2026-01-20: Matrix Loss Experiments

### Setup

Tested matrix loss with ground-truth patching: MCTS payout matrix as target, but the played cell is replaced with actual game outcome (unbiased).

Two variants trained on iter1 data:

| Run | matrix_loss_weight | constant_sum_weight |
|-----|-------------------|---------------------|
| matrixloss_v1 | 0.1 | 1.0 |
| matrixloss_nocsum | 1.0 | 0.0 |

### Results

**Individual benchmarks (vs greedy/mcts/random):**

| Agent | mcts+nn Elo | nn Elo | Best val loss |
|-------|-------------|--------|---------------|
| iter1 baseline | 1003.5 | 897.9 | 2.2515 |
| iter2 baseline | 1026.2 | 973.8 | — |
| matrixloss_v1 | 988.5 | 836.8 | 2.7237 |
| matrixloss_nocsum | 1029.9 | 868.8 | 2.6390 |

**Head-to-head vs greedy (mcts+nn W/D/L):**

| Variant | W | D | L |
|---------|---|---|---|
| iter1 baseline | 6 | 40 | 4 |
| matrixloss_v1 | — | — | — |
| matrixloss_nocsum | 2 | 47 | 1 |

matrixloss_nocsum: only 1 loss, 47 draws — very robust against greedy.

**Head-to-head tournament (all 4 + greedy):**

| Agent | Elo |
|-------|-----|
| iter2 | 1029.7 |
| matrixloss_nocsum | 1020.0 |
| iter1 | 1020.0 |
| matrixloss_v1 | 1010.4 |
| greedy | 1000.0 |

All within error bars (±21.7). ~80% draw rate in every matchup.

### Diagnostic: Payout Quality

For matrixloss_nocsum:

| Metric | iter2 baseline | matrixloss_nocsum |
|--------|----------------|-------------------|
| Value correlation | 0.99 | 0.92 |
| Value exp var | 0.97 | 0.85 |
| Payout correlation | 0.10 | **0.61** |
| Payout exp var | << 0 | ~0 |

**Interpretation:**
- Payout correlation improved 6x (0.10 → 0.61) — NN is learning payout structure
- But explained variance still ~0 — scale is wrong (predicting direction but not magnitude)
- Value got worse (0.99 → 0.92) — matrix loss pulls capacity away from value
- constant_sum interferes with matrix loss — matrixloss_v1 (with csum) performed worst

**Open question:** High correlation + zero exp var suggests scale mismatch. Options:
1. Normalize payout targets
2. Confidence-weight by visit counts
3. Separate heads for value vs full matrix
4. Accept trade-off — 0.61 correlation might suffice for Nash

### Next Step

Try iteration 2 with matrixloss_nocsum checkpoint to see if better payout representations improve long-term convergence.

---

## 2026-01-21: matrixloss_nocsum Iteration 2

### Setup

**Sampling:**
- Used matrixloss_nocsum checkpoint as MCTS prior
- Games: 50,000 → 533,677 positions (shorter than iter2's 584k)
- Same game params: 7x7, 10 cheese, 50 turns, no walls/mud

**Training:**
- Resumed from matrixloss_nocsum_run2 checkpoint
- Same loss config: `matrix_loss_weight=1.0`, `constant_sum_weight=0.0`

### Results

| Agent | Elo | vs iter2 baseline |
|-------|-----|-------------------|
| mcts+nn | 1035.9 | +9.7 |
| mcts+nn-prev | 1017.0 | — |
| greedy | 1000.0 | — |
| nn | 980.8 | +7.0 |
| mcts | 923.5 | — |
| nn-prev | 919.9 | — |

**Head-to-head vs parent (mcts+nn vs mcts+nn-prev):**
- iter2 baseline: 5/39/6 (basically even)
- matrixloss_nocsum_iter2: 5/44/1 (fewer losses, more draws)

### Direct Comparison: iter2 vs matrixloss_nocsum_iter2

100 games per matchup, direct head-to-head:

| Agent | Elo |
|-------|-----|
| iter2 | 1037.0 |
| matrixloss_nocsum_iter2 | 1027.3 |
| iter2_nn | 1026.6 |
| matrixloss_nocsum_iter2_nn | 1014.8 |
| greedy | 1000.0 |

**Head-to-head:** iter2 vs matrixloss_nocsum_iter2 = 6/89/5 (basically even, ~90% draws)

Inconclusive — matrix loss doesn't help, but doesn't clearly hurt either.

### Overfitting Analysis

Both matrixloss_nocsum runs show clear overfitting on nash/indiff and matrix losses:

**matrixloss_nocsum_run2:**
| Loss | Train | Val min | Val end |
|------|-------|---------|---------|
| nash | 0.10 → 0.05 | 0.075 @ ep 28 | 0.12 |
| matrix | 0.73 → 0.36 | 0.37 @ ep 48 | 0.42 |

**matrixloss_nocsum_iter2:**
| Loss | Train | Val min | Val end |
|------|-------|---------|---------|
| nash | 0.14 → 0.05 | 0.17 @ ep 42 | 0.23 |
| matrix | 0.51 → 0.18 | 0.31 @ ep 69 | 0.41 |

Train keeps dropping, val has U-shaped curve. The model memorizes MCTS payout estimates rather than learning generalizable structure.

### Conclusion

Matrix loss with weight=1.0 causes overfitting without clear Elo benefit. The payout correlation improved (0.10 → 0.61) but that didn't translate to better play.

**Decision:** Drop matrix loss for now. Continue self-play loop with baseline config (no matrix loss, with constant_sum).

---

## 2026-01-21: Visits Policy Targets

### Hypothesis

Nash equilibrium is a sharp function on noisy payout estimates. If MCTS payout matrices are noisy (and they are — see payout correlation of 0.10), computing Nash on them produces overconfident policy targets. This could cause exploration collapse.

Alternative: **marginalize visit counts** to get policy targets. Instead of Nash's sharp mixed strategy, use the softer distribution of what MCTS actually explored.

```python
# Nash: sharp equilibrium on payout matrix
policy_p1, policy_p2 = compute_nash(payout_matrix)

# Visits: softer, preserves MCTS uncertainty
visits_p1 = visit_counts.sum(axis=1)  # marginalize over opponent
visits_p2 = visit_counts.sum(axis=0)  # marginalize over player
policy_p1 = visits_p1 / visits_p1.sum()
policy_p2 = visits_p2 / visits_p2.sum()
```

### Implementation

Added `--policy-target {nash,visits}` to `alpharat-prepare-shards`. Created `alpharat/nn/target_strategies.py` with:
- `PolicyTargetStrategy` protocol
- `NashPolicyStrategy` (default, existing behavior)
- `VisitPolicyStrategy` (new)

Design note: CLI has default (`nash`), but internal functions require explicit strategy. Keeps default policy decisions at the user-facing boundary, not leaking into library code.

### Results

**7x7_visits_iter1** — trained on same iter1 batch data, but with visits policy targets:

| Agent | Elo | vs nash baseline |
|-------|-----|------------------|
| greedy | 1000.0 | anchor |
| mcts+nn | 1004.0 | +0.5 |
| mcts | 893.5 | -0.8 |
| nn | 629.5 | **-268.4** |
| random | 8.5 | — |

**Comparison:**

| Metric | nash (iter1) | visits (visits_iter1) |
|--------|--------------|----------------------|
| mcts+nn Elo | 1003.5 | 1004.0 |
| nn Elo | 897.9 | 629.5 |
| mcts+nn - nn gap | 105.6 | 374.5 |

### Interpretation

1. **mcts+nn is equivalent** — no difference in combined agent strength. The NN + search combo performs identically regardless of policy target strategy.

2. **Pure nn is much weaker** — 268 Elo drop. Visits targets produce softer, less decisive policies. The NN learns to output uncertain distributions that need MCTS to sharpen into good play.

3. **Larger search gap** — mcts+nn - nn increased from 106 to 375 Elo. This means MCTS contributes more when NN policies are soft.

4. **Trade-off**: Nash targets → stronger nn standalone, weaker signal about MCTS exploration structure. Visits targets → weaker nn standalone, but maybe richer signal for MCTS to use.

### Open Questions

- Does visits help at iter2+? Nash may have been fine at iter1 because MCTS+random-priors explores broadly. At iter2, NN priors focus exploration — does visits preserve more useful diversity?

- Would combining visits targets with matrix loss help? The softer policies might complement explicit payout learning.

### Next Steps

1. **iter2 with visits** — sample with visits_iter1 checkpoint, shard with visits, train, benchmark
2. **matrix loss + no csum variant** — try visits targets combined with `matrix_loss_weight=1.0, constant_sum_weight=0.0` to see if the approaches complement each other

Configs ready:
- `configs/sample_7x7_visits_iter2.yaml`
- `configs/train_7x7_visits_iter2.yaml`

---

## 2026-01-22: Visits + Matrix Loss (No Constant Sum)

### Setup

Tested visits policy targets combined with matrix loss, trained on same iter1 data as visits_iter1:

| Config | visits_iter1 | visits_nocsum |
|--------|--------------|---------------|
| policy targets | visits | visits |
| constant_sum_weight | 1.0 | 0.0 |
| matrix_loss_weight | 0.0 | 1.0 |

### Results

**Benchmark:**

| Agent | visits_iter1 | visits_nocsum | Δ |
|-------|--------------|---------------|---|
| mcts+nn | 1004.0 | 1016.4 | +12 |
| nn | 629.5 | 440.7 | -189 |

mcts+nn slightly better (+12 Elo, within error bars). Pure nn dropped another 189 points.

**Head-to-head vs greedy (mcts+nn W/D/L):**

| Variant | W | D | L |
|---------|---|---|---|
| visits_iter1 | 6 | 40 | 4 |
| visits_nocsum | 4 | 43 | 3 |

Fewer losses (4→3), more draws (40→43). Slightly more robust against greedy.

**Training metrics:**

| Metric | visits_iter1 | visits_nocsum | Interpretation |
|--------|--------------|---------------|----------------|
| Top-1 accuracy | 58% | 56% | Less confident |
| Top-2 accuracy | 73% | 71% | Less confident |
| Policy entropy | 1.41 | 1.52 | Softer distributions |
| Value correlation | 0.95 | 0.93 | Slight drop |
| Value exp var | 0.90 | 0.85 | Slight drop |
| Indiff loss | 0.0023 | 0.113 | Much harder to satisfy |
| Policy loss | 1.41 | 1.46 | Higher |

### Interpretation: Matrix Loss as Implicit Regularization

Matrix loss forces internal consistency between payout predictions and policies. The NN can't output a sharp policy without a payout matrix that justifies it via Nash equilibrium. Since actual payouts are noisy, the NN backs off to softer policies.

Evidence:
- **Higher entropy** (1.41 → 1.52) = more uncertainty expressed
- **Lower accuracy** = less confident predictions
- **Higher indiff loss** (0.0023 → 0.113) = Nash indifference condition harder to satisfy when constrained by payout structure

The key insight: matrix loss doesn't just teach payout structure — it regularizes policies by requiring consistency. Sharp policies need sharp payouts to justify them, but learning sharp payouts from noisy MCTS data is hard.

Result: softer, less accurate policies that express *useful* uncertainty for MCTS to exploit. Pure nn worse, mcts+nn slightly better.

### Pattern Summary

| Variant | nn Elo | mcts+nn Elo | Policy character |
|---------|--------|-------------|------------------|
| nash targets (iter1) | 898 | 1004 | Sharp |
| visits targets | 630 | 1004 | Softer |
| visits + matrix loss | 441 | 1016 | Softest |

Each step makes the standalone nn weaker but the combined agent stays flat or improves slightly. The NN learns to express uncertainty that MCTS can use, rather than committing to potentially wrong sharp policies.

### Next Steps

Continue to iter2 with visits_nocsum config to see if the trend holds with NN-guided sampling.

---
*Implementation details in IDEAS.md under "2026-01-20: Forced Playout Pruning — Implementation"*

---

## 2026-01-23: Visits Policy Targets — Iteration 2

### Setup

Continued the visits-based policy target experiments to iteration 2. Two variants:

| Variant | Parent | Policy targets | constant_sum | matrix_loss |
|---------|--------|----------------|--------------|-------------|
| visits_iter2 | visits_iter1 | visits | 1.0 | 0.0 |
| visits_matrixloss_iter2 | visits_matrixloss | visits | 0.0 | 1.0 |

**Sampling:** Both used their respective iter1 checkpoints as MCTS priors for NN-guided self-play.

### Results

**visits_iter2:**

| Agent | Elo | vs iter1 |
|-------|-----|----------|
| mcts+nn | 1059 | **+55** |
| nn | 1024 | **+394** |
| greedy | 1000 | anchor |
| mcts+nn-prev | 991 | — |
| nn-prev | 959 | — |
| mcts | 933 | — |

**Head-to-head (mcts+nn vs mcts+nn-prev):** 7/40/3 — clear improvement

**visits_matrixloss_iter2:**

| Agent | Elo | vs iter1 |
|-------|-----|----------|
| mcts+nn | 1054 | **+38** |
| mcts+nn-prev | 1025 | — |
| greedy | 1000 | anchor |
| nn | 979 | **+538** |
| nn-prev | 416 | — |
| mcts | 904 | — |

**Head-to-head (mcts+nn vs mcts+nn-prev):** 4/44/2 — improvement, more draws

### Observations

1. **Self-play loop working** — Both variants showed substantial gains. This is the first time we've seen mcts+nn improve meaningfully across iterations (previous Nash-based iter2 stalled at +1 Elo).

2. **Raw NN now competitive** — visits nn=1024, matrixloss nn=979. Both above greedy. At iter1, visits nn was 630 and matrixloss nn was 441.

3. **Search gap collapsed for visits:**
   - iter1: mcts+nn - nn = 374 Elo (massive)
   - iter2: mcts+nn - nn = 35 Elo (small)

   The NN got so good that search adds little. For matrixloss: 575 → 75 Elo (still meaningful gap).

4. **Policy character preserved across iterations:**
   - visits nn-prev=959 (sharp, strong standalone)
   - matrixloss nn-prev=416 (soft, needs search)

   matrixloss continues to produce softer policies that express uncertainty for MCTS to use.

5. **Convergence** — The gap between variants narrowed. At iter1: mcts+nn difference was 12 Elo. At iter2: 5 Elo. Both approaches are converging to similar combined agent strength.

### Interpretation

**Visits targets break the plateau.** The Nash-based self-play loop stalled at iter2 (mcts+nn went +1 Elo). The visits-based loop didn't — both variants showed meaningful mcts+nn improvement (+55 and +38 Elo).

Why? Nash sharpens noisy payout estimates into overconfident policies. When the NN learns these, it focuses exploration on the "best" actions, starving other action pairs for data. Visits targets preserve the exploration signal — the NN learns "MCTS explored these actions this much," not "Nash says this is optimal."

### Conclusion

**Visits-based policy targets work better for self-play iteration** than Nash targets. Both variants showed the self-play loop producing real gains, which we hadn't achieved with Nash.

**Continuing with visits + matrix_loss (no constant_sum).** Both variants are within error bars on Elo, so this is a judgment call. Intuition:
- Softer policies might preserve more exploration headroom
- Matrix loss directly supervises payout structure — feels like better foundations for MCTS reasoning
- Search gap still meaningful (75 Elo vs 35) — suggests more room to grow

Not a strong signal either way. Could revisit if iter3 stalls.

### Next Steps

1. **Iteration 3** — Sample with visits_matrixloss_iter2 checkpoint, continue the loop
2. **Watch for divergence** — If the variants start separating at iter3+, that's signal

---

**Note (2026-01-24):** Rebased onto main, picking up fix #46 (correct marginal visit counts for equivalent actions in PUCT). This changes how visits-based policy targets are computed when blocked actions exist. Results from iter3 onwards may not be directly comparable to iter1/iter2.

---

## 2026-01-31: CNN Spatial Encoder — Iteration 0

### Motivation

Test whether spatial inductive bias helps on 7x7. The SymmetricMLP struggled to learn payout matrices (0.10 correlation at iter1). A CNN that processes spatial features might learn better representations.

### Architecture

**PyRatCNN** — CNN trunk with DeepSet heads for structural P1/P2 symmetry:
- Spatial tensor: 5 channels (maze adjacency 4 + cheese 1)
- Player positions NOT in spatial tensor (preserves symmetry) — used only for indexing
- ResNet trunk: stem conv + 1 residual block, 64 channels
- DeepSet combination: extract features at player positions, sum-aggregate
- Shared policy and payout heads (weight sharing guarantees symmetry)

Config: `hidden_channels=64, num_blocks=1, player_dim=32, hidden_dim=64, dropout=0.1`

### Training

- Data: Same iter1 shards as SymmetricMLP (`7x7/a590d934`, ~796k positions)
- Policy targets: Nash (not visits)
- 100 epochs, batch_size=4096, lr=1e-3

TensorBoard metrics looked much better than SymmetricMLP — faster convergence, smaller train/val gap.

### Results

**Elo ratings:**

| Agent | Elo |
|-------|-----|
| mcts+nn | 1012.0 ± 24.6 |
| greedy | 1000.0 (anchor) |
| mcts | 917.3 ± 24.7 |
| nn | 901.4 ± 24.9 |
| random | 85.2 |

**Head-to-head comparison (W/D/L):**

| Benchmark | mcts+nn vs greedy | mcts+nn vs mcts | mcts+nn vs nn |
|-----------|-------------------|-----------------|---------------|
| iter2 (Nash, SymmetricMLP) | 8/38/4 | 18/29/3 | 11/38/1 |
| visits_iter2 | 9/40/1 | 20/29/1 | 12/35/3 |
| visits_matrixloss_iter2 | 13/35/2 | 12/38/0 | 17/33/0 |
| **CNN (iter0)** | 7/39/4 | 16/33/1 | 13/36/1 |

### Observations

1. **CNN iter0 ≈ SymmetricMLP iter2** — Matching Nash-based iter2 performance (1012 vs 1026 Elo, within error bars) without any self-play iteration. Head-to-heads are nearly identical.

2. **Same pattern as SymmetricMLP iter1** — nn Elo (901) matches iter1 baseline (898). The search gap (mcts+nn - nn = 111 Elo) is similar to iter1 (106 Elo).

3. **visits_iter2 still strongest** — The visits-based policy targets after 2 iterations remain the best performers (1059 Elo). CNN hasn't caught up yet, but it's only iter0.

4. **Training metrics better** — TensorBoard showed faster convergence and better generalization. Need to check if payout correlation improved (the key weakness of SymmetricMLP).

### Open Questions

1. **Does CNN learn better payout representations?** Run the payout quality diagnostic to check correlation/explained variance.

2. **Will CNN benefit from self-play?** Nash-based SymmetricMLP stalled at iter2 (+1 Elo). visits-based broke through. What about CNN?

3. **CNN + visits?** The CNN architecture might combine well with visits policy targets — spatial features for better priors, soft targets for exploration.

### Next Steps

1. **Diagnostic** — Check payout correlation for CNN vs SymmetricMLP
2. **Iteration 1** — Sample with CNN checkpoint, train iter1, benchmark
3. **Consider CNN + visits** — If iter1 stalls, try visits policy targets

---

## 2026-02-02: CNN + Visits Policy Targets

### Motivation

The original CNN iter0 used Nash policy targets with no nash_weight loss. Comparing to the best SymmetricMLP experiments (which used visits targets + nash_weight=1.0), we needed a fair comparison.

Tested three CNN configurations to isolate what helps:

| Config | Policy targets | nash_weight | matrix_loss_weight |
|--------|----------------|-------------|--------------------|
| cnn_7x7_v1 (iter0) | Nash | 0.0 | 0.0 |
| cnn_7x7_v1_matrixloss | Nash | 0.0 | 1.0 |
| cnn_7x7_visits | **visits** | **1.0** | 0.0 |

### Results

| Config | mcts+nn Elo | nn Elo | vs greedy (W/D/L) |
|--------|-------------|--------|-------------------|
| CNN iter0 (Nash) | 1012 | 901 | 7/39/4 |
| CNN + matrix_loss | 997 | 905 | 2/45/3 |
| **CNN + visits** | **1016** | **666** | **8/40/2** |

### Observations

1. **Matrix loss hurt performance** — Adding matrix_loss to CNN made it worse (1012 → 997 Elo, lost to greedy 2-3). The spatial inductive bias might already provide enough structure.

2. **Visits targets work** — CNN + visits achieves best mcts+nn Elo (1016) and beats greedy 8-2. Matches the pattern from SymmetricMLP experiments.

3. **Same nn-collapse pattern** — nn alone dropped from 901 to 666 Elo with visits targets. Exactly what we saw with SymmetricMLP (898 → 630). The softer policies need MCTS to be useful.

4. **Overfitting visible** — Validation losses showed U-shaped curves. Train keeps dropping, val plateaus and rises. The model is memorizing rather than generalizing.

### Comparison to Best SymmetricMLP

| Metric | SymmetricMLP visits_iter2 | CNN visits (iter0) |
|--------|---------------------------|-------------------|
| mcts+nn Elo | 1059 | 1016 |
| nn Elo | 1024 | 666 |
| hidden_dim | 256 | 64 |
| iterations | 2 | 0 |

CNN at iter0 is ~40 Elo behind SymmetricMLP at iter2, but with:
- 4x smaller hidden dimension
- No self-play iteration yet
- Matching loss config now

### Next Steps

1. **Iteration 1** — Sample with CNN visits checkpoint, train on NN-guided data
2. **Address overfitting** — More dropout? Early stopping? More data?
3. **Check if CNN catches up** — SymmetricMLP gained ~55 Elo from iter1→iter2 with visits
