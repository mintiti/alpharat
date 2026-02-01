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

---

## Open Questions

1. Does scalar value MCTS match payout matrix MCTS at the same sim budget?
2. How do optimal PUCT parameters differ between the two approaches?
3. Does training a NN on scalar targets work as well as payout matrix targets?
