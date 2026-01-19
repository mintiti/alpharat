# Experiment Log

## Goal

Experiment with AlphaZero + transformer architectures on PyRat.

## Current Phase

**Phase 3: Self-play iteration** — done for 5×5, ready to add complexity

## Roadmap

### Phase 1: MCTS Core ✅
- [x] MCTS vs Random
- [x] MCTS vs Greedy

### Phase 2: Training Pipeline ✅
- [x] MLP learns from MCTS
- [x] MCTS+NN beats pure MCTS

### Phase 3: Self-Play Loop ✅
- [x] Generate data with MCTS+NN → train → new model beats old
- [x] Two iterations completed, both showed improvement

### Phase 4: Complexity Scaling (Next)
- [ ] Walls — tests action equivalence at scale
- [ ] Mud — delayed movement, different strategy
- [ ] Larger grids

### Phase 5: Architecture
- [ ] Transformer / attention
- [ ] Spatial processing for maze
- [ ] D4 augmentation

---

## 5×5 Validation Summary

**Domain:** 5×5 grid, no walls, no mud, 5 cheese, 30 max turns

Simplified domain to test the approach before adding complexity.

### Results

| What | Outcome |
|------|---------|
| MCTS vs Random | ~75% win rate |
| MCTS vs Greedy | Consistent wins |
| MCTS+NN vs MCTS | 22-1 head-to-head |
| Self-play iteration | +121 Elo (raw NN), +18 Elo (MCTS+NN) per iteration |

### Iteration 2 Standings

| Agent | Elo |
|-------|-----|
| mcts+nn | 1106 |
| mcts+nn-prev | 1088 |
| nn | 1076 |
| greedy | 1000 (anchor) |
| nn-prev | 955 |
| mcts | 938 |

Raw NN after iteration 2 nearly matches MCTS+NN from iteration 1.

### Not Yet Tested

- Walls (action equivalence code exists, not tested at scale)
- Mud (changes strategy significantly)
- Larger grids (longer horizons)

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid size | 5x5 | Small enough for fast iteration, large enough for decisions |
| Walls | None (Phase 4) | Shorter horizons, easier credit assignment for early experiments |
| Cheeses | 5 | Enough for allocation decisions, not too many configs |
| Max turns | 30 | Rough upper bound for 5x5 collection |
| Beat threshold | >50% @ 99% CI | Statistically confident improvement over baseline |

## Log

### 2024-12-11: MCTS vs Random (Phase 1)

**Config:** 5x5, no walls, 5 cheese, 30 turns, 200 games per matchup, 20 sims

**Results:**
| Agent | W | D | L | Pts | AvgCheese |
|-------|---|---|---|-----|-----------|
| puct_20 | 162 | 161 | 77 | 242.5 | 1.3 |
| prior_sampling_20 | 139 | 171 | 90 | 224.5 | 1.2 |
| Random | 53 | 160 | 187 | 133.0 | 0.7 |

**vs Random specifically:**
- prior_sampling: 87W/85D/28L
- puct: 100W/75D/25L

**Conclusion:** Both MCTS variants beat Random decisively (~75-80% win rate excluding draws). Sanity check passed. PUCT slightly better than prior sampling at 20 sims.

**Next:** Test against Greedy baseline.

### 2024-12-11: MCTS vs Random (10 sims)

**Config:** Same as above, but 10 sims instead of 20

**Results:**
| Agent | W | D | L | Pts | AvgCheese |
|-------|---|---|---|-----|-----------|
| prior_sampling_10 | 139 | 163 | 98 | 220.5 | 1.0 |
| puct_10 | 130 | 179 | 91 | 219.5 | 1.1 |
| Random | 62 | 196 | 142 | 160.0 | 0.8 |

**vs Random specifically:**
- prior_sampling: 76W/90D/34L
- puct: 66W/106D/28L

**Conclusion:** Even 10 sims beats Random clearly. Prior sampling and PUCT essentially tied against each other at this sim count. More draws than 20 sims — games less decisive with fewer sims.

### 2024-12-12: MCTS vs Greedy (Phase 1 complete)

**Config:** 5x5, no walls, 5 cheese, 30 turns, 200 games per matchup

Tested three PUCT configs (from Optuna sweep) against Greedy:

| Config | Sims | c_puct | vs Greedy (W/D/L) | Win% (excl. draws) |
|--------|------|--------|-------------------|-------------------|
| puct_200 | 200 | 4.73 | 47/138/15 | 76% |
| puct_959 | 959 | 6.11 | 43/126/31 | 58% |
| puct_278 | 278 | 3.98 | 35/137/28 | 56% |

**Observations:**
- All PUCT configs beat Greedy consistently
- High draw rate (60-70%) — games are close
- Average cheese nearly identical (2.4-2.6 for both sides)
- More sims doesn't guarantee better results — c_puct tuning matters

**Conclusion:** Phase 1 complete. MCTS beats both Random (decisively) and Greedy (consistently). Ready for Phase 2.

### 2024-12-25: Training Pipeline (Phase 2)

Built end-to-end training pipeline: MCTS self-play → data sharding → NN training.

**Architecture improvements that stabilized training:**

1. **BatchNorm** after each trunk layer — normalizes activations, smoother gradients
2. **Proper weight initialization:**
   - Trunk: Kaiming normal (accounts for ReLU)
   - Policy heads: small std=0.01 → softmax starts near-uniform
   - Payout head: small std=0.01 → predictions start near zero

**Full-batch training:** At 5x5 grid size, observations are small enough (181 dims) that the entire dataset fits in GPU memory. Single forward/backward per epoch is faster than mini-batch iteration due to reduced kernel launch overhead.

**Training config:** Config-based with Pydantic validation. Two loss variants:
- `mcts`: supervise full 5x5 payout matrix (learns MCTS estimates)
- `dqn`: supervise only played action pair (learns from outcomes)

**Status:** Training loss converges well. Haven't tested trained model in actual games yet.

**Next:** Evaluate trained MLP against MCTS/Greedy/Random to see if it learned anything useful.

### 2024-12-25: Cheese Ownership Targets — Data Pipeline

Implemented KataGo-inspired cheese ownership targets in the data pipeline. NN architecture changes deferred.

**Encoding (`CheeseOutcome` IntEnum in `alpharat/data/types.py`):**
```python
P1_WIN = 0       # P1 collected alone
SIMULTANEOUS = 1 # Both collected at same time
UNCOLLECTED = 2  # Game ended, nobody collected
P2_WIN = 3       # P2 collected alone
```

**Data flow:**
1. `recorder.py` computes `cheese_outcomes[H,W]` at game end by tracking when each cheese disappears and checking who was there
2. Saved to game `.npz` files
3. `sharding.py` broadcasts game-level outcomes to each position in shards as `int8[N,H,W]`

**Storage decision:** Class indices `int8[N,H,W]` instead of one-hot `float32[N,H,W,4]`:
- 4x smaller shards
- Works directly with `CrossEntropyLoss(input, target)` where target is class indices
- Mask gradients for non-cheese cells using `initial_cheese` mask:
  ```python
  loss = F.cross_entropy(logits, targets, reduction='none')  # [B,H,W]
  loss = (loss * cheese_mask).sum() / cheese_mask.sum()
  ```

**Files modified:**
- `alpharat/data/types.py` — added `CheeseOutcome` enum, `cheese_outcomes` field to `GameData`
- `alpharat/data/recorder.py` — compute outcomes in `_finalize()`
- `alpharat/data/loader.py` — load from npz
- `alpharat/nn/types.py` — added to `TargetBundle`
- `alpharat/nn/targets.py` — include in `build_targets()`
- `alpharat/data/sharding.py` — thread through pipeline

**TODO (NN architecture):**
- Add cheese ownership head to model
- Augmentation: swap P1_WIN↔P2_WIN (0↔3) when flipping perspective
- Add cheese ownership loss to training

## Open Questions

**Value learns slower than policy** — policy loss drops faster. Could be normal (policy is "easier" — just imitate MCTS), but worth investigating.

Ideas to explore:

1. **Cheese ownership targets (KataGo-inspired) — DATA PIPELINE DONE, NN PENDING**

   KataGo's key insight: when the final target is a sum of subevents, predicting those subevents gives sharper, more localized gradients. The NN gets direct feedback on *which part* it mispredicted, not just "you were wrong overall."

   In PyRat, final score = Σ(cheese outcomes). Each cheese is a mini-match from P1's perspective:
   - Win: P1 collects it alone (+1)
   - Simultaneous: both collect at same time (+0.5 each)
   - Uncollected: game ends, nobody got it (0)
   - Loss: P2 collects it alone (-1 relative)

   ✅ Data pipeline: `cheese_outcomes[H,W]` recorded per game, stored as class indices (0-3) in shards.

   ⏳ NN architecture: Add cheese ownership head, augmentation (swap 0↔3), training loss.

   **For later (PDF/CDF ideas from KataGo):**
   - Instead of just W/D/L classification, could predict probability distributions over timing/score
   - CDF loss: pushes probability mass near the true outcome
   - But W/D/L per cheese is simpler and captures the main idea

2. **Fast/slow rollouts (KataGo-inspired):** Policy needs many sims for quality targets, but that means fewer games → fewer value samples (roughly 1 per game). Idea: most moves use "fast" search (few sims), but periodically do "slow" search (many sims) for policy targets. Get policy quality on slow steps, value diversity from more games. More effort.

3. **Structured MLP:** Current MLP treats all 181 input dims uniformly. Could be more thoughtful — e.g., process maze topology separately, then combine with cheese/position info. Inductive bias that maze structure matters differently than cheese locations.

### 2024-12-27: MCTS+NN Benchmark (Phase 2 validation)

**Goal:** Test trained NN in actual games — both standalone and as MCTS prior.

**Infrastructure built:**

1. **Unified agent config system** — Pydantic discriminated union with `AgentConfigBase`:
   ```python
   class MCTSAgentConfig(AgentConfigBase):
       simulations: int = 200
       c_puct: float = 4.73
       checkpoint: str | None = None  # Optional NN priors
       def build(self, device) -> Agent: ...
   ```
   Config builds the agent. Tournament/benchmark code just calls `config.build()`.

2. **MCTSAgent with optional NN** — unified path for MCTS with/without NN priors:
   - `simulations > 0`: MCTS search, NN provides prior policies
   - `simulations = 0`: skip MCTS entirely, return raw NN policy head

3. **`train_and_benchmark.py`** — chains training with evaluation in one command:
   ```bash
   alpharat-train-and-benchmark configs/train.yaml --games 50
   ```

**Benchmark results (50 games per matchup, 200 MCTS sims):**

| Agent | W | D | L | Pts | AvgCheese |
|-------|---|---|---|-----|-----------|
| mcts+nn 🏆 | 118 | 74 | 8 | 155.0 | 2.8 |
| mcts | 77 | 78 | 45 | 116.0 | 2.6 |
| greedy | 72 | 87 | 41 | 115.5 | 2.6 |
| nn | 75 | 77 | 48 | 113.5 | 2.5 |
| random | 0 | 0 | 200 | 0.0 | 0.4 |

**Key head-to-head (W/D/L from row's perspective):**

| Matchup | W/D/L |
|---------|-------|
| mcts+nn vs mcts | 22/27/1 |
| mcts+nn vs nn | 26/21/3 |
| mcts+nn vs greedy | 20/26/4 |
| mcts vs nn | 16/23/11 |

**Observations:**

- **MCTS+NN dominates** — only 8 losses total across 200 games. 22-1 vs pure MCTS.
- **NN priors make search much better** — mcts+nn vs mcts is 22/27/1, not close
- **Pure NN ≈ Greedy ≈ MCTS** — middle tier all clustered (113.5-116 pts)
- **Random crushed** — 0 wins, 0 draws, 200 losses

**Conclusion:** Phase 2 validated — NN learns useful priors from MCTS self-play. MCTS+NN > MCTS > NN ≈ Greedy >> Random.

**Next:** Could try cheese ownership targets (KataGo-inspired) to see if value prediction improves further.

### 2026-01-06: Visit Filtering for Nash Computation

**Problem:** With biased NN priors, MCTS exploration becomes sparse. Most visits go to one action pair (e.g., 550/555 sims to a single pair). Payoff values for other pairs are just NN predictions — never updated by rollouts. Nash solver computes equilibrium on garbage data → "No Nash equilibrium found" warnings → falls back to uniform.

**Fix:** Filter actions with < 5 visits before Nash computation. Threshold = "at least one expected visit per opponent action" — the structural minimum for a row/column to contain any search information.

**Implementation:** `alpharat/mcts/nash.py` — new flow: reduce by effective actions → filter by visits → Nash → expand to full 5 actions.

**Benchmark (same checkpoint: dropout=0.1 NN trained on bimatrix data):**

| Config | mcts+nn Elo | vs greedy (W/D/L) |
|--------|-------------|-------------------|
| Before (sparse matrices) | 1033 | 16/20/14 |
| After (visit filtering) | 1069 | 11/39/0 |

**Full standings after fix:**

| Rank | Agent | Elo |
|------|-------|-----|
| 1 | mcts+nn | 1069 |
| 2 | greedy | 1000 (anchor) |
| 3 | nn | 970 |
| 4 | mcts | 957 |
| 5 | random | 207 |

**Observations:**
- +36 Elo improvement for mcts+nn
- Eliminates losses to greedy entirely (was 14 losses, now 0)
- More draws (39 vs 20) — games are closer, fewer blunders from bad Nash policies
- Pure NN unchanged (~950-970) — fix only affects MCTS+NN path

**Conclusion:** Computing Nash on explored submatrix is more reliable than including unreliable payoff estimates. Quick fix that complements the "forced playouts" idea (which would prevent sparse matrices upstream).

### 2026-01-06: Self-Play Iteration 2

**Setup:** Generated 50k games using mcts+nn (the checkpoint from iteration 1), then trained on that data resuming from the same checkpoint.

**Benchmark results (50 games per matchup):**

| Rank | Agent | Pts | Elo |
|------|-------|-----|-----|
| 1 | mcts+nn | 207.5 | 1106 |
| 2 | mcts+nn-prev | 200.0 | 1088 |
| 3 | nn | 195.0 | 1076 |
| 4 | greedy | 164.5 | 1000 (anchor) |
| 5 | nn-prev | 144.5 | 955 |
| 6 | mcts | 138.0 | 938 |
| 7 | random | 0.5 | 147 |

**Key head-to-heads:**
- mcts+nn vs mcts+nn-prev: 1/49/0 (1 win, 49 draws, 0 losses)
- mcts+nn vs nn-prev: 30/20/0
- nn vs nn-prev: 16/34/0

**Elo gains from iteration 1 → 2:**
- Raw NN: 955 → 1076 (+121 Elo)
- MCTS+NN: 1088 → 1106 (+18 Elo)

**Observations:**
- Massive improvement in raw NN quality — now nearly matches previous iteration's MCTS+NN
- MCTS+NN undefeated (0 losses across 300 games)
- Self-play loop is working: better NN → better training data → even better NN

**Conclusion:** AlphaZero-style self-play validated. One iteration of 50k games produced significant gains, especially for raw NN inference. The network is learning generalizable game understanding, not just memorizing.

### 2026-01-07: Architecture Comparison — SymmetricMLP vs MLP (Iteration 1)

**Goal:** Test whether SymmetricMLP (DeepSet architecture with structural P1/P2 symmetry) improves over standard MLP.

**Setup:** Both models trained on same data (bimatrix iteration 1). 50 games per matchup.

**Results:**

| Rank | Agent | Elo | StdErr |
|------|-------|-----|--------|
| 1 | mcts+nn_mlp | 1070 | ±20 |
| 2 | mcts+nn_symmetric | 1059 | ±20 |
| 3 | greedy | 1000 (anchor) | — |
| 4 | nn_symmetric | 944 | ±20 |
| 5 | mcts | 932 | ±20 |
| 6 | nn_mlp | 921 | ±20 |

**Key head-to-heads:**

| Matchup | W/D/L |
|---------|-------|
| mcts+nn_mlp vs mcts+nn_symmetric | 0/50/0 |
| nn_symmetric vs nn_mlp | 10/38/2 |

**Observations:**

- **MCTS+NN: No difference.** 50 pure draws — architectures are equivalent when combined with search.
- **Raw NN: Symmetric wins.** +23 Elo, 10-2 head-to-head. Structural symmetry helps standalone inference.
- MCTS is robust to prior quality — search self-corrects. Raw inference has no safety net.

**Conclusion:** Architecture choice doesn't matter for MCTS+NN (the strong config). For fast inference without search, SymmetricMLP is better. Need to test whether this advantage persists through self-play iterations.

**Next:** Train SymmetricMLP through iteration 2 for fair comparison with MLP iteration 2.

### 2026-01-07: SymmetricMLP with Softplus Payout Head

**Change:** Added softplus activation to payout head output, enforcing non-negative predictions. Cheese scores can't be negative — this is domain structure, not something to learn.

**Setup:** Same data (bimatrix iteration 1), same config as previous SymmetricMLP run. 50 games per matchup.

**Results:**

| Rank | Agent | Elo | StdErr |
|------|-------|-----|--------|
| 1 | mcts+nn | 1069 | ±25 |
| 2 | greedy | 1000 (anchor) | — |
| 3 | mcts | 961 | ±24 |
| 4 | nn | 926 | ±25 |

**Key head-to-heads:**

| Matchup | W/D/L |
|---------|-------|
| mcts+nn vs greedy | 10/40/0 |
| mcts+nn vs mcts | 12/38/0 |
| mcts+nn vs nn | 22/28/0 |

**Comparison to SymmetricMLP without softplus:**

| Agent | Before | After | Δ |
|-------|--------|-------|---|
| mcts+nn | 1059 | 1069 | +10 |
| nn | 944 | 926 | -18 |

**Observations:**
- mcts+nn still undefeated (0 losses across 200 games)
- Deltas within error bars — no significant change
- Softplus doesn't hurt, may help slightly for MCTS+NN
- Raw NN slightly worse, but high variance

**Conclusion:** Non-negativity constraint is neutral to slightly positive. Worth keeping as it's structurally correct. The payout head now outputs valid cheese scores by construction.

### 2026-01-07: Nash Consistency Loss

**Goal:** Add game-theoretic constraint forcing payout matrix predictions to be consistent with policy predictions. If the policy says "play action 0 with high probability," the payout matrix should show action 0 is actually optimal.

**The Loss:**

Two components enforcing Nash equilibrium conditions:

1. **Indifference:** Actions in support (π > threshold) should have equal expected utility against opponent's strategy.
2. **No profitable deviation:** Actions outside support should not have higher expected payoff than equilibrium value.

```python
# For P1: expected payoff per action against P2's strategy
exp1 = pred_payout @ pi2  # [batch, 5]
val1 = (pi1 * exp1).sum()  # equilibrium value

# Indifference: in-support actions should equal val1
indiff_loss = (support_mask * (exp1 - val1)**2).mean()

# No deviation: out-of-support actions shouldn't beat val1
dev_loss = (outside_mask * relu(exp1 - val1)**2).mean()
```

**Two modes:**
- `nash_mode: target` — uses MCTS Nash policies (have exact 0s for blocked/unexplored actions)
- `nash_mode: predicted` — uses NN's own softmax outputs (self-consistency regularization)

**Setup:** SymmetricMLP with softplus, same bimatrix data. `nash_weight: 1.0`. 50 games per matchup.

**Results with `nash_mode: target`:**

| Rank | Agent | Elo |
|------|-------|-----|
| 1 | mcts+nn | 1069 |
| 2 | greedy | 1000 |
| 3 | mcts | 933 |
| 4 | nn | 915 |

**Results with `nash_mode: predicted`:**

| Rank | Agent | Elo |
|------|-------|-----|
| 1 | mcts+nn | 1074 |
| 2 | greedy | 1000 |
| 3 | mcts | 966 |
| 4 | nn | 936 |

**Comparison vs SymmetricMLP+softplus baseline (no Nash loss):**

| Agent | Baseline | + Nash (predicted) | Δ |
|-------|----------|-------------------|---|
| mcts+nn | 1069 | 1074 | +5 |
| nn | 926 | 936 | +10 |

**Target vs predicted mode:**

| Agent | Target | Predicted | Δ |
|-------|--------|-----------|---|
| mcts+nn | 1069 | 1074 | +5 |
| nn | 915 | 936 | +21 |

**Observations:**
- Both modes show improvement over baseline, `predicted` mode slightly better
- Gains are within noise (+5-10 Elo) but consistently positive
- Self-consistency regularization (predicted mode) helps — forcing policy and payout heads to agree
- mcts+nn still undefeated

**Conclusion:** Nash consistency loss adds a structurally correct game-theoretic constraint. Empirically neutral to slightly positive. Default config now uses `nash_mode: predicted, nash_weight: 1.0`.

*Follow-up: 500 epochs showed no improvement over 100 — training converges quickly on this data size.*

### 2026-01-07: Constant-Sum Regularization

**Goal:** Add loss encouraging P1 + P2 payouts to sum to total cheese collected. PyRat is approximately constant-sum — this encodes that structure.

**The Loss:**
```python
sum_payout = pred_payout[:, 0] + pred_payout[:, 1]  # [batch, 5, 5]
total_collected = p1_value + p2_value  # [batch]
loss = MSE(sum_payout, total_collected)  # all 25 pairs → same sum
```

**Setup:** SymmetricMLP + softplus + Nash (predicted mode) + constant_sum_weight=1.0. Same bimatrix data.

**Results:**

| Rank | Agent | Elo |
|------|-------|-----|
| 1 | mcts+nn | 1076 |
| 2 | greedy | 1000 |
| 3 | nn | 970 |
| 4 | mcts | 949 |

**Comparison vs Nash-only baseline:**

| Agent | Nash only | + Constant-sum | Δ |
|-------|-----------|----------------|---|
| mcts+nn | 1074 | 1076 | +2 |
| nn | 936 | 970 | +34 |

**Observations:**
- Raw NN: +34 Elo — significant improvement
- mcts+nn: unchanged (still undefeated, only 1 loss)

**Conclusion:** Constant-sum regularization helps raw NN inference substantially. The constraint is structurally correct and provides additional supervision signal for the payout matrix (all 25 pairs get gradient, not just the played pair).

### 2026-01-07: Sigmoid × Remaining Cheese (Failed Experiment)

**Goal:** Enforce structural upper bound on payout predictions. Each player can score at most `remaining_cheese` from any position, so bound outputs to `[0, remaining_cheese]` by construction.

**The Change:**
```python
# Before: softplus (just non-negative)
payout = F.softplus(raw)

# After: sigmoid scaled by remaining cheese
remaining_cheese = cheese_mask.sum()  # from observation
payout = torch.sigmoid(raw) * remaining_cheese
```

**Results (SymmetricMLP + Nash + constant-sum baseline → + sigmoid bound):**

| Agent | Baseline | + Sigmoid bound | Δ |
|-------|----------|-----------------|---|
| nn | 970 | 926 | -44 |
| mcts+nn | 1076 | ~1070 | -6 |

**Hypothesis for why it hurt:** Gradients scale with `remaining_cheese`. Endgame positions (1-2 cheese left) get 5× weaker gradients than opening positions (5 cheese). The network under-learns endgame patterns.

**Attempted fix:** Per-sample gradient correction — divide loss by `remaining_cheese` to normalize gradient magnitude across game phases.

**Results with gradient correction:**

| Agent | Sigmoid | + Gradient correction | Δ |
|-------|---------|----------------------|---|
| nn | 926 | 913 | -13 |

Made it worse, not better. The hypothesis was wrong, or the correction introduced other problems.

**Conclusion:** Reverted to softplus. The sigmoid × remaining_cheese idea doesn't work — either the bound isn't helping, or the learning dynamics are too disrupted. Softplus (non-negative only, no upper bound) remains the right choice.

**Lesson:** Not all structurally correct constraints improve learning. The bound is mathematically valid but complicates optimization enough to hurt performance.

### 2026-01-08: MCTS Tree Reuse Between Turns (Inconclusive)

**Branch:** `feature/tree-reuse`

**Goal:** Preserve MCTS tree between turns — after both players move, advance root to the child node and keep accumulated statistics for next search.

**Implementation:**
- `MCTSAgent.reuse_tree` flag enables/disables
- `observe_move(a1, a2)` called after each turn to advance root
- `tree.advance_root()` navigates simulator, promotes child to root
- Fixed `_get_path_from_root()` to stop at current root (not original root)

**Benchmark: Head-to-head (reuse vs fresh MCTS, same sim budget)**

With NN checkpoint (50 games each, tuned c_puct=8.34, force_k=0.88):

| Sims | Reuse W | Fresh W | Draw | Reuse Win% |
|------|---------|---------|------|------------|
| 50 | 17 | 22 | 11 | 43.6% |
| 100 | 13 | 23 | 14 | 36.1% |
| 200 | 14 | 21 | 15 | 40.0% |
| 500 | 16 | 14 | 20 | 53.3% |
| 1000 | 18 | 20 | 12 | 47.4% |

**Observations:**
- Reuse often *worse* than fresh, which shouldn't be possible if implemented correctly
- High variance across sim counts — no consistent pattern
- Code review found no bugs — fresh tree path unchanged from main
- Simulator sync verified correct

**Hypothesis:** Unknown. Same position = same observation = same priors, so stale priors theory doesn't hold.

**Conclusion:** Inconclusive. Tree reuse doesn't provide clear benefit, and the "sometimes worse" anomaly remains unexplained. Not worth further debugging — complexity not justified by results.

### 2026-01-15: Centroid Nash — 2-Iteration Test

**Branch:** `feature/centroid-equilibria`

**Problem:** Multiple equilibria exist (~18% of positions). Need to pick one policy to train on.

**Change:** Use centroid (arithmetic mean per player) instead of random selection. Game-theoretically valid for constant-sum games.

**Plan:** Run 2 self-play iterations on both branches, compare final Elo. Dig into diagnostics only if results differ.

**Results (2-iteration test completed):**

Iteration 1: Pure MCTS sampling (no NN), 50k games → train from scratch.

Iteration 2: MCTS+NN sampling using iter1 checkpoint, 50k games → finetune from iter1.

| Rank | Agent | Elo |
|------|-------|-----|
| 1 | mcts+nn | 1080 |
| 2 | mcts+nn-prev | 1061 |
| 3 | nn | 1048 |
| 4 | greedy | 1000 (anchor) |
| 5 | nn-prev | 942 |
| 6 | mcts | 938 |

**Key head-to-heads:**
- mcts+nn vs mcts+nn-prev: 1/48/1 (essentially tied)
- mcts+nn vs nn-prev: 21/29/0
- nn vs nn-prev: 16/31/3

**Elo gains from iteration 1 → 2:**
- Raw NN: 942 → 1048 (+106 Elo)
- MCTS+NN: 1061 → 1080 (+19 Elo)

**Comparison to main branch iteration 2:**

| Agent | Main branch | Centroid | Δ |
|-------|-------------|----------|---|
| mcts+nn | 1106 | 1080 | -26 |
| nn | 1076 | 1048 | -28 |

Centroid results are slightly weaker than the main branch iteration 2 logged earlier. Reason unknown.

**Bug found during testing:** Pydantic silently ignored unknown fields. Config used `checkpoint:` but code expected `resume_from:`. Both "from scratch" and "finetune" runs trained identically (same seed → same weights). Fixed by adding `extra="forbid"` to TrainConfig.
