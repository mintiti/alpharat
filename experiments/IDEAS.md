# Ideas

Append-only parking lot. Dump ideas here, refine them in conversation when ready to work.

---

## 2024-12-27: Infrastructure

**Checkpoint/Experiment Dashboard**
Need something to track checkpoints, training curves, benchmark results, which checkpoint came from what config. Like Optuna dashboard but for the whole project. Would reduce cognitive load — don't have to remember where things are.

**Elo Calculation Module** ✅ DONE
Implemented in `alpharat/eval/elo.py`. Bradley-Terry MLE, anchored to greedy at 1000, optional uncertainty via Hessian. Integrated into train_and_benchmark scripts.

**Refactor training duplication**
`training.py` and `local_value_training.py` are 80% the same. Factor out common parts.

**Benchmark Result Persistence**
Currently we re-run all matchups every benchmark (random vs greedy, etc.) even when nothing changed. Want to cache results and only re-run what's needed.

Open questions:
- What defines "same matchup"? Agent config + game params + checkpoint hash?
- How to handle checkpoint-based agents? Key on path or content hash?
- Storage: JSON file, SQLite, per-checkpoint folder?
- How does this interact with Elo pool? Persistent pool of reference agents?
- Should cached results expire? Or trust them forever for deterministic agents?

Related to the Checkpoint Dashboard idea — this is part of the broader "track what we've done" problem.

---

## 2024-12-27: Architecture

**Spatial processing for maze**
Current MLP flattens everything — spatial info gets lost. Ideas:
- Small CNN (2-3 conv layers) for maze topology
- Cells as tokens with attention
- GNN where maze is naturally a graph

Question: Better performance for fewer FLOPs?

---

## Earlier Ideas

### ✅ DONE: Player Swap Augmentation
Implemented in `alpharat/nn/augmentation.py`.

### ✅ DONE: Cheese Ownership Prediction
Implemented as LocalValueMLP with 4-class ownership head.

### ✅ DONE: Shared Trunk, Separate Heads
Standard pattern, already in PyRatMLP.

### ✅ DONE: Nash Consistency Loss
Implemented in training. Forces payout matrix to be consistent with policy predictions (indifference in support, no profitable deviation outside). See `experiment-log.md` 2026-01-07.

### ✅ DONE: Constant-Sum Regularization
Implemented as soft MSE loss: `MSE(P1+P2, total_collected)`. PyRat is approximately constant-sum under infinite horizon; truncation makes it general-sum. Enforced as regularization, not hard constraint. See `experiment-log.md` 2026-01-07.

### ✅ DONE: Visit Filtering for Nash
Filter actions with < 5 visits before Nash computation. Fixes sparse matrix issues from biased NN priors. See `experiment-log.md` 2026-01-06.

---

## Pending: Augmentation

**Board Rotation/Reflection**
D4 group symmetries for 8× augmentation. Needs:
- Rotate maze array, positions, cheese mask
- Permute policy actions (UP→RIGHT→DOWN→LEFT under 90° CW)

---

## Pending: Value Targets

**Score Distribution (KataGo-style)**
Predict full P(final_score = k) instead of E[score]. Captures uncertainty, can derive win probability.

**Variance / Uncertainty Estimation**
Predict E[value] and Var[value]. MCTS could use uncertainty for exploration.

---

## Pending: Training

**Self-Play Iteration**
AlphaZero loop: train → use as prior → sample → repeat. Need evaluation to check improvement.

**Curriculum Learning**
Simple games first, increase complexity. Hypothesis: easier to learn basics before contested scenarios.

**Data Mixing**
Mix model-generated data with uniform-prior data. Prevents collapse if learned prior is too narrow.

---

## Pending: Evaluation

**Cheese Efficiency**
Turns to collect cheese. Tighter play = fewer turns.

**Policy Entropy**
Track over training. Should decrease but not collapse to deterministic.

---

## 2025-01-02: Forced Playouts for Degenerate Matrices

**Context**: P1 win bias (~70%) caused by sparse payout matrices → Nash picks arbitrary first equilibrium. Current fix: return uniform for degenerate matrices. But this trains NN on "fake uniform" targets.

**Better approach (from KataGo)**: Force exploration so matrices aren't sparse in the first place.

```
n_forced(c) = k * P(c) * sqrt(total_visits)
```

For simultaneous games, adapt to action pairs.

**Open question**: Should forced playouts be per joint action (i,j) or per single-player action?
- Joint: `n_forced(i,j) = k * P1(i) * P2(j) * sqrt(N)`
- Single: Force each player's actions independently, let pairs emerge

Don't have an answer yet. Joint seems more direct but might be expensive (25 pairs). Single might naturally fill in the matrix through independent exploration.

**Status**: `force_k` config parameter exists. Visit filtering (done) complements this — handles sparse matrices when they still occur.

---

## 2025-01-03: Averaging Pure Equilibria vs Uniform Fallback

**Problem with current uniform fallback**: The zero row/column check is too aggressive.

Example: If P1's reduced matrix has rows [0,0,0], [0,0,0.5], [0,0,1.0] — action 2 is clearly best, but zero-row check triggers uniform. Wrong.

**Better theoretical approach**: When multiple Nash equilibria exist, average the pure equilibria.

- Fully degenerate (constant matrix): average = uniform (same as now)
- Partially degenerate: average weights actions by how often they appear in equilibria
- Unique equilibrium: just use it

This is the centroid of the equilibrium polytope — principled, not arbitrary.

**Relationship to forced playouts**: These ideas complement each other:
- Forced playouts → less sparse matrix → fewer degenerate cases
- Averaging equilibria → correct handling when degeneracy still occurs

With forced playouts, we'd hit the averaging path less often, but when we do, it gives the "right" answer.

---

## 2026-01-04: Asymmetric Game Initialization

**Context**: Currently both players start in opposite corners, symmetric setup. Most games end in draws (~70-80%). Value function might not get strong learning signal from symmetric positions.

**Idea**: Randomize starting positions to create asymmetric games. One player closer to more cheese, different distances, etc.

**Potential benefits**:
- More decisive games (fewer draws) → stronger value signal
- More diverse training positions
- Tests whether the model generalizes beyond symmetric starts

**Open questions**:
- How asymmetric? Fully random positions, or controlled imbalance?
- Does this help or just add noise?
- Need to test empirically

---

## 2026-01-04: Optuna Sweep Improvements & Optimization Loop

**Elo Pool for Sweep Evaluation**

Current sweep uses win rate vs Greedy. Once MCTS is strong, this saturates. Better: maintain a fixed opponent pool (Greedy, Random, previous NN checkpoints) with precomputed matchups between pool members. Compute Elo for each trial config against the pool.

Benefits:
- Richer signal as system improves
- Answers "is this config better than our previous best?"
- Can add new opponents as training progresses

**Configurable Opponent**

Make the sweep script accept an opponent type (Greedy, or path to NN checkpoint). Allows testing new MCTS configs against NN-guided MCTS.

**Score Margin as Secondary Metric**

When win rates are similar, use average score margin as tiebreaker. Captures "how decisively does this config win?"

**The Optimization Loop**

Hypothesis: MCTS params and NN training can be optimized mostly independently because improving either improves the other downstream.

Proposed loop:
1. Tune MCTS vs Greedy (no NN) → best config
2. Sample data with best MCTS config
3. Tune NN training hyperparams → best NN
4. Re-tune MCTS with NN as prior, opponent = previous best
5. Iterate

Caveat: optimal c_puct/force_k shift when prior quality changes. Re-tuning MCTS after NN improves is important.

---

## 2026-01-05: Asymmetric Game Sampling Experiment

**Hypothesis**: More diverse training data (from asymmetric starting positions) might produce a better first-generation NN, even if the games are "less fair."

**Experiment idea**: Sample games with randomized starting positions (not symmetric corners), train NN, compare to current symmetric-start trained NN.

Not perfectly scientific (many confounds), but would give directional signal on whether data diversity matters more than game "purity" for early NN training.

---

## 2026-01-06: Structural Action Equivalence in NN Architecture

**The problem**: Action equivalence (UP blocked by wall → same as STAY) is currently learned implicitly. The NN sees 0-probability targets for blocked actions and figures it out. But this is learned, not structural.

**The insight**: An action's representation should come from its *effect* (destination cell), not its *label*. Two actions leading to the same cell should have identical representations by construction.

### Destination-Based Architecture

**Core idea**: The trunk produces per-cell embeddings. Actions are just indices into destination cells.

```
observation → trunk → cell_embeds [H*W, hidden]

# For each action, look up where it leads
destinations = adj[position]  # [5] precomputed from maze
action_embeds = cell_embeds[destinations]  # [5, hidden]
```

If UP and STAY both lead to the same cell, they get the same embedding. Structural equivalence.

**The adjacency tensor**: `adj[cell, action] = destination_cell`. Precomputed once per maze (or passed as input during training). 25×5 = 125 values for 5×5.

### Policies and Payouts from Cell Embeddings

**Policy**: Score each destination, softmax over actions.
```python
p1_action_embeds = cell_embeds[p1_dests]  # [batch, 5, hidden]
p1_logits = policy_head(p1_action_embeds)  # [batch, 5]
```
Equivalent actions → same embedding → same logit → same probability.

**Payout**: Pairwise interaction between destination embeddings.
```python
payout = einsum('bih,bjh->bij', p1_action_embeds, p2_action_embeds)
```
Equivalent actions → identical rows/columns in payout matrix.

### Player Symmetry (DeepSet-style)

Same operations for both players, swapped arguments:

```python
# Same cell embeddings, different destinations
p1_action_embeds = cell_embeds[p1_dests]
p2_action_embeds = cell_embeds[p2_dests]

# Same policy head
p1_logits = policy_head(p1_action_embeds, context)
p2_logits = policy_head(p2_action_embeds, context)

# Same payout function, swapped roles
payout_p1 = compute_payout(p1_action_embeds, p2_action_embeds, context)
payout_p2 = compute_payout(p2_action_embeds, p1_action_embeds, context)
```

Swap players → swap action embeds → outputs swap. Structural.

### Players as Maze Features

Instead of separate player encoders, encode player positions as cell features:

```python
cell_features[cell] = [
    wall_up, wall_right, wall_down, wall_left,  # topology
    has_cheese,
    p1_here,  # 1 if P1 at this cell
    p2_here,  # 1 if P2 at this cell
]
cell_embeds = trunk(cell_features)  # [H*W, hidden]
```

Players are just features on cells. No special treatment.

### The Abstraction

| Component | Role |
|-----------|------|
| **Trunk** | `observation → cell_embeds [H*W, hidden]` — all learning here |
| **Heads** | `cell_embeds + indices → policy, payout` — mostly structural (gather + einsum) |

The trunk could be CNN, GNN, transformer, MLP — doesn't matter. As long as it outputs cell embeddings, the rest is fixed.

### What's Structural vs Learned

| Structural (by construction) | Learned (in trunk) |
|------------------------------|-------------------|
| Action equivalence | Cell embedding quality |
| Player symmetry | What makes a cell "good" |
| Payout matrix symmetries | Cheese value, position advantage |

### GPU-Friendly Implementation

All fixed-size tensors:
```python
adj = precompute_adjacency(maze)           # [H*W, 5], done once
cell_embeds = trunk(cell_features)         # [H*W, hidden]
destinations = adj[positions]              # [batch, 5]
action_embeds = cell_embeds[destinations]  # [batch, 5, hidden] — gather
payout = einsum('bih,bjh->bij', ...)       # [batch, 5, 5]
```

No ragged dimensions. The "variable number of effective actions" is implicit — duplicate destinations → duplicate embeddings → identical outputs.

### Open Questions

- **Trunk architecture**: CNN on cell features? How many layers?
- **Head complexity**: Pure einsum, or add learned projections? Context term?
- **Mud handling**: Player in mud → all destinations = current cell. Works automatically.
- **Does this actually help?** Need to implement and test.

---

## 2026-01-09: D4 Equivariance via Graph Formulation

**The friction**: Current maze encoding uses `maze[i, j, direction]` for UP/RIGHT/DOWN/LEFT. Rotating the maze requires swapping channels in a specific way — this fights against symmetry.

**Possible fix**: Encode as a graph instead.
- Nodes = cells (features: player_here, cheese_here, etc.)
- Edges = adjacent pairs (feature: traversal_cost — 1=free, 2=mud, inf=wall)

Connectivity becomes an edge property, not a directional channel. Rotating the maze just permutes nodes and edges together.

**Architecture sketch**:
- Transformer/GNN over nodes, edge features as attention bias
- Policy = softmax over (source, target) edge scores instead of 4 directions
- Edge score = f(z_src, z_tgt, edge_feat)

**Why this might help**: D4 symmetry falls out naturally — no channel swapping, no special equivariant layers. Attention is already permutation-equivariant.

**Why it might not matter**: Standard arch + data augmentation might be just as good. Elegant structure that's wrong is worse than no structure. Would need to compare empirically.

**If we try this**: Start with simple baseline, measure if it actually struggles with symmetry, then add structure if there's evidence it helps.

---

## 2026-01-07: Connecting Cell Embeddings, Cheese Ownership, and Payout Structure

Loosely connected ideas around how spatial representations could inform the payout matrix. All intuition — no evidence yet.

### Local Cheese Ownership as Representation Shaping

The existing LocalValueMLP predicts per-cheese ownership (who gets each cheese). Thinking of this not just as an auxiliary task, but as something that shapes what cell embeddings learn to encode.

If cells learn "the cheese here is likely P1's" or "this is contested," then that information is baked into the spatial representation. Downstream things (like payout prediction) might benefit from embeddings that already "know" about cheese value.

Intuition: cheese ownership loss as a **representation prior**, not just a side prediction.

### Actions as Transitions

From the structural architecture idea: actions map to destination cells, so action embeddings can be destination cell embeddings.

But an action is really "from here → to there." Maybe action embeddings should encode the transition, not just the endpoint?

```python
p1_ego = cell_embeds[p1_pos]  # where I am
p1_dests = cell_embeds[p1_destinations]  # where I'd end up
p1_actions = some_combination(p1_ego, p1_dests)  # the transition
```

The combination could be additive, concat+project, difference, multiplicative — unclear which is right. The intuition is that "moving toward my cheese" vs "moving away" should produce different action embeddings, and that requires knowing the starting point.

### Payout from Crossing Action Embeddings

If both players have action embeddings (5 each), payout matrix could be their interaction:

```python
payout = einsum('bid,bjd->bij', p1_actions, p2_actions)
```

If the cell embeddings encode cheese ownership (from aux loss), and actions encode transitions, then this bilinear might naturally capture "what happens when P1 goes here while P2 goes there."

Feels like it could work, but also feels like a lot of "if"s stacked together.

### Decoupling Player Representation from Spatial Representation

One axis worth thinking about: should player representation be separate from cell representation?

**Coupled (player = cell they're on):**
```python
p1_ego = cell_embeds[p1_pos]
```
Simple. The player "is" their position. Works if the spatial encoder is rich enough that the cell embedding at your position already captures what you need.

**Decoupled (player has own representation):**
Player token that queries or attends to the spatial grid. Lives "outside" the grid but informed by it.

Why decouple?
- Player has state beyond position (mud, score)
- Player might need a "view" of the whole maze, not just their cell
- Same maze looks different to each player (my cheese vs your cheese)

Why not?
- Position is most of what matters in PyRat
- Deep enough trunk might make cell_embeds[pos] rich enough
- Simpler

Intuition says start coupled, decouple if there's a reason. But unclear.

### The Rough Shape

```
maze → spatial encoder → cell_embeds (H×W × d)
                              │
                              ├── (optional) cheese ownership aux loss
                              │
                              ├── p1_ego from cell_embeds[p1_pos] (or separate token?)
                              ├── p2_ego from cell_embeds[p2_pos]
                              │
                              ├── p1_actions = f(p1_ego, cell_embeds[p1_dests])
                              ├── p2_actions = f(p2_ego, cell_embeds[p2_dests])
                              │
                              └── payout = bilinear(p1_actions, p2_actions)
```

Lots of choices at each step. No clarity on which matter.

---

## 2026-01-09: Payout Matrix Quality & Training Signal (Refined)

### The Problem: A Feedback Loop

It's not just "exploration is bad" — it's a self-reinforcing cycle:

```
peaked Nash targets → peaked NN → peaked priors → only one pair explored → peaked Nash targets...
```

**How we got here:**
1. Smart uniform priors create well-explored payout matrices initially
2. Nash equilibrium often returns low-entropy (peaked) strategies
3. NN learns peaked policies from these targets
4. When NN is used as prior, only the "best" action pair gets explored
5. Other action pairs are starved for data (DQN-style: only played pair gets true signal)
6. NN predicts non-played pairs poorly (defaults to high values like the main pair)
7. Nash consistency + constant-sum losses don't fully correct this
8. Feeds back into MCTS with garbage predictions for most cells

### The Bias-Variance Framing

Different data sources have different properties:

| Source | Bias | Variance | What it is |
|--------|------|----------|------------|
| Game outcome (played pair) | Unbiased | High | Monte Carlo estimate |
| MCTS payout (other pairs) | Biased | Lower | n-step bootstrap estimates |

Current approaches:
- **"DQN loss"** (learn from game outcome only): Unbiased but starves non-played pairs
- **"Learn full MCTS matrix"**: Garbage in → garbage out

The insight: MCTS *does* explore other pairs during search. That data exists. It's biased (n-step estimates with variable n, bootstrapped from NN), but it's information. The question is how much to trust it.

### Hypotheses to Test

**H1: Confidence-weighted learning**
Weight the loss by visit count: `loss_weight[i,j] = f(visits[i,j])`. Low visits → low weight → don't trust. This uses information we already have about cell reliability.

**H2: Return multiple equilibria**
If there are multiple Nash equilibria, why return only one? Put all of them in the training batch. More diverse targets might help the NN generalize. *But first need to know*: how often are there genuinely distinct equilibria vs. just relabelings?

**H3: Add exploration during sampling**
Break the feedback loop at the data generation step. Force exploration of non-peak actions so the NN sees diverse data.

### What We Need to Know First

Before designing interventions, we're missing empirical facts:

1. **Payout matrix structure**: What does a typical matrix look like? How peaked is the visit distribution? How many cells have meaningful data?

2. **Equilibrium multiplicity**: How often are there multiple *distinct* Nash equilibria? (Issue #24 tooling would help here)

3. **NN prediction quality by cell**: For trained checkpoints, does prediction error correlate with visit count? (If yes → confidence weighting makes sense)

### Evaluation: Oracle Payout Matrix

**Problem**: How do we know if our training approach is working?

**Proposed solution**: Build an **exhaustive payout validation set**.

For N validation positions, compute the *true* payout for every valid action pair:
- Run full MCTS rollouts to terminal (no bootstrapping)
- This gives unbiased estimates of the full 5×5 matrix
- Compare NN predictions against this ground truth
- Measure correlation and explained variance

**Feasibility estimate:**
- 100 positions × 15 unique pairs × 50 rollouts = 75,000 simulations
- If rollout ≈ 50ms → ~1 hour (expensive but doable once)

**If too expensive, fallbacks:**
- Sparse ground truth (sample 5 pairs per position instead of all 15)
- Fewer rollouts (accept higher variance)
- Late-game positions only (fewer turns to terminal)

**Next step**: Run a timing test. Pick 10 positions, compute full-coverage payout, see how long it takes. Then decide if ideal solution is feasible.

### Related Ideas (Preserved)

**Nash Algorithm Findings:**

| Algorithm | Speed | Robustness | Bias |
|-----------|-------|------------|------|
| Support enumeration | ~1-30ms | Fails on degenerate | Unbiased |
| Lemke-Howson | ~0.1ms | Robust | Biased (P1 69% win rate) |

Conclusion: Fix payout quality → degenerate cases rare → support_enumeration works fine.

**PUCT Scaling Issue:**
Q values are raw cheese counts. Early game Q ∈ [0, 5], late game Q ∈ [0, 0.3]. Fixed c_puct means exploration term's relative importance changes. Might need Q normalization.

**Action Equivalence:**
Equivalent actions should have identical payout values. Currently not enforced — could average equivalent rows/columns for better estimates.

### Concrete Next Steps

1. **Diagnostic tooling** (Issue #24): Build tools to analyze payout matrices, equilibria structure, visit distributions
2. **Timing test**: How long does exhaustive payout computation take?
3. **Empirical probing**: Once we have data, answer the "what we need to know" questions
4. **Design intervention**: Based on what we learn, pick H1/H2/H3 or combination

---

## 2026-01-11: Handling Multiple Nash Equilibria — Two Approaches

**Context**: Diagnostic analysis of 2000 games with uniform priors revealed:
- 65.5% unique equilibrium (good)
- 17.7% multiple equilibria (ambiguous training signal)
- 16.7% constant matrix (no information)

Multiple equilibria are mostly endgame (66% with 1 cheese, 24% with 2 cheese). Of the multiple equilibria cases:
- 45.5% have one player indifferent (other player's choice is fixed across all equilibria)
- 54.5% have both players with choices ("both vary")

Of "both vary" cases: 66% all-pure equilibria, 34% have mixed equilibria.

**Key insight**: PyRat is approximately constant-sum → **interchangeability** holds. Each player can independently choose any strategy from their equilibrium strategy set. This allows decomposing the problem per-player.

### Approach 1: Centroid (Simple)

For each player independently, average their equilibrium strategies:

```python
p1_policy = np.mean([eq[0] for eq in equilibria], axis=0)
p2_policy = np.mean([eq[1] for eq in equilibria], axis=0)
```

**Properties:**
- Always in convex hull (guaranteed valid)
- Same formula for pure and mixed cases
- Closed form, fast
- Interpretation: "max entropy over choice of which equilibrium" — treat all equilibria as equally valid

For all-pure equilibria: centroid = uniform over support ✓

**Limitation**: Not the true max-entropy strategy in the convex hull. Just the centroid of the vertices.

### Approach 2: Max Entropy in Convex Hull (Principled)

For each player, find the strategy with maximum entropy within their convex hull of equilibrium strategies:

```
max H(σ_p)  subject to  σ_p ∈ conv({eq[p] for eq in equilibria})
```

**Properties:**
- Truly maximum exploration while staying game-theoretically valid
- Same formula for pure and mixed
- Requires convex optimization (scipy/cvxpy)

For all-pure equilibria: max entropy = uniform over support (same as centroid) ✓
For mixed equilibria: might differ from centroid

**Implementation sketch:**
```python
from scipy.optimize import minimize

def max_entropy_in_hull(strategies):
    strategies = np.array(strategies)
    n_eq = len(strategies)

    def neg_entropy(alpha):
        sigma = alpha @ strategies
        sigma_safe = np.clip(sigma, 1e-10, 1)
        return np.sum(sigma_safe * np.log(sigma_safe))

    constraints = {'type': 'eq', 'fun': lambda a: np.sum(a) - 1}
    bounds = [(0, 1) for _ in range(n_eq)]
    alpha0 = np.ones(n_eq) / n_eq

    result = minimize(neg_entropy, alpha0, bounds=bounds, constraints=constraints)
    return result.x @ strategies
```

### Recommendation

Start with **centroid** — it's simple, fast, and handles 100% of cases correctly (always valid). For the 34% of "both vary" cases with mixed equilibria, centroid might not be max-entropy, but it's a reasonable approximation.

If we find evidence that the approximation matters (NN struggling in specific cases), upgrade to the full optimization.

### What to do with constant matrices

Constant matrices (16.7% of positions, 95% with 1 cheese remaining) provide no training signal — MCTS found no payoff differentiation. Options:
1. **Filter from training** — don't train on these positions
2. **Use uniform** — current behavior, but teaches arbitrary policy
3. **Domain heuristic** — "move toward cheese" even when MCTS says indifferent

Leaning toward (1) — these are mostly decided endgames where the NN shouldn't need to learn anything

---

## 2026-01-19: KataGo Forced Playouts — How They Filter

**Context**: We implemented forced playouts (`force_k` parameter) to ensure exploration, but we should look at how KataGo handles them in the policy target.

**What we need to investigate**: KataGo doesn't just use forced playouts for exploration — they also filter them out of the final policy target. The intuition is that forced visits are exploration noise, not signal about move quality. If MCTS was forced to visit a move, its visit count doesn't reflect true preference.

**Question for our implementation**: Should we exclude forced visits when computing Nash targets? Current flow:
1. MCTS explores with forced playouts
2. We compute Nash from full payout matrix
3. Record Nash policy as target

Potential change:
1. MCTS explores with forced playouts
2. Filter out "forced" visits from payout matrix (how?)
3. Compute Nash on filtered matrix
4. Record that as target

**Open question**: How do we know which visits were "forced"? Our current forced playout implementation doesn't track this. Would need to either:
- Track forced vs organic visits separately
- Reconstruct from threshold: if visits < threshold at end, they were all forced

**Next step**: Read KataGo paper/code to understand their exact mechanism

---

## 2026-01-20: Forced Playout Pruning — Implementation Details

### Context

From 7x7 iter2: payout matrix correlation is 0.10 (garbage). Root cause: MCTS explores 1-2 pairs deeply, rest are noise. Forced playouts help exploration but pollute training targets.

### KataGo's Approach (Single-Player)

From the paper: "we identify the child c* with the most playouts, and then from each other child c, we subtract up to n_forced playouts so long as it does not cause PUCT(c) >= PUCT(c*)"

**PUCT formula:**
```
PUCT(a) = Q(a) + c · P(a) · √N / (1 + N(a))
```

**Inverse (solve for N given target PUCT):**
```
N'_min(a) = c · P(a) · √N / (PUCT* - Q(a)) - 1
```

This is "the weight that would make this action's PUCT equal to the best action's PUCT."

### Adaptation for Simultaneous Games

We have decoupled selection — each player selects independently. Adapt by:

1. Compute marginal adjustments per player:
```
For P1:
    i* = argmax_i M₁[i]
    PUCT* = Q₁[i*] + c · π₁[i*] · √N / (1 + M₁[i*])

    For i ≠ i*:
        if Q₁[i] ≥ PUCT*:
            Δ₁[i] = 0  # genuinely good
        else:
            N'_min = c · π₁[i] · √N / (PUCT* - Q₁[i]) - 1
            Δ₁[i] = max(0, M₁[i] - N'_min)

    Δ₁[i*] = 0
```

2. Distribute to pairs (additive):
```
V'[i,j] = max(0, V[i,j] - Δ₁[i] · π₂[j] - Δ₂[j] · π₁[i])
```

**Intuition:** When P1 was forced to action i, P2 was selecting via PUCT (initially ~proportional to π₂). So forced visits to pairs (i, ·) are spread roughly as π₂.

### Why Additive Form

The forced visits for pair (i,j) come from:
- P1 being forced to i (distributed across j by P2's prior)
- P2 being forced to j (distributed across i by P1's prior)

These are approximately independent contributions. Additive form is the natural decomposition.

Could over-subtract if both players were heavily forced to the same pair — but that's rare, and aggressive pruning is probably fine.

### Fractional Visits

After pruning, V'[i,j] can be fractional. That's fine:
- For Nash computation: weight pairs by V'
- For training: these become loss weights

### Edge Cases

- **Q(a) ≥ PUCT***: Action is genuinely good. Don't prune.
- **All visits forced**: Could result in empty matrix. Fall back to existing min_visits filter.
- **Best action for one player, worst for other**: Each player's adjustment is independent.

### Implementation Location

- `alpharat/mcts/selection.py`: `compute_pruning_adjustment()` and `prune_visit_counts()`
- `alpharat/mcts/decoupled_puct.py`: `_prune_forced_visits()` method, called from `_make_result()`

Pruning is automatic when `force_k > 0` in MCTS config.

---

## 2026-01-21: Re-tune MCTS After Pruning Changes

**Context**: Forced playout pruning changed how visits are recorded. The old c_puct and force_k values were tuned for the previous behavior.

**Problem**: Optimal MCTS parameters likely shifted. With pruning, we might want more aggressive forcing (since it gets cleaned up) or different c_puct.

**Approach**: Re-run Optuna sweep on MCTS params with the new pruning logic.

---

## 2026-01-21: Enforce Action Equivalence in NN Payout Output

**Context**: Blocked actions map to STAY (e.g., effective[UP] = STAY when UP is blocked). Equivalent actions should have identical payout values, but the NN doesn't naturally satisfy this.

**Idea**: When MCTS receives the NN's payout prediction, immediately average equivalent rows/columns before using it in the tree.

```python
# In MCTSTree or MCTSNode, after getting NN prediction:
payout = equalize_equivalent_actions(payout, p1_effective, p2_effective)
```

**Why here**: Clean intervention point. NN outputs whatever it outputs, we enforce the constraint before it affects search. No architecture or training changes needed.

**Implementation location**: `MCTSTree._get_nn_prediction()` or `MCTSNode` initialization.

---

## 2026-01-21: Learn from Marginal Visits (AlphaZero-style)

**Context**: Current approach: learn full 5×5 payout matrix → compute Nash → use Nash as policy target.

**The problem with Nash-from-noisy-payouts:**

Nash equilibrium is a *sharp* function. It takes a payout matrix and outputs a potentially peaked strategy — high confidence about which actions are optimal. But our payout matrices are approximations (MCTS estimates with limited rollouts, not ground truth). Applying a sharp function to noisy inputs produces overconfident outputs.

The feedback loop:
1. MCTS produces approximate payout matrix
2. Nash computes peaked strategy from it (throws away uncertainty)
3. NN learns to reproduce that confidence
4. Next iteration: confident NN priors → exploration concentrates on "best" actions
5. Other action pairs starved for data → worse payout estimates for them
6. Repeat — exploration collapses, convergence suffers

**Why visit distributions help:**

Visit counts naturally preserve uncertainty. When MCTS isn't sure, it explores multiple actions — the visit distribution reflects that. Learning from visits keeps the "I'm not sure" signal that Nash throws away.

This is the standard approach (AlphaZero, KataGo). The payout matrix still guides MCTS selection during search. We're not bypassing it — we're just not asking the NN to learn a sharpened version of it.

```python
# Instead of Nash equilibrium:
policy_target_p1 = marginal_visits_p1 / total_visits  # [5]
policy_target_p2 = marginal_visits_p2 / total_visits  # [5]
```

**Trade-offs:**

| Aspect | Nash targets | Visit targets |
|--------|--------------|---------------|
| Confidence | High (sharp) | Calibrated to search uncertainty |
| Exploration | Tends to collapse | Preserved |
| Game-theoretic soundness | Exact (given true payouts) | Approximate |
| Exploitability | Guaranteed unexploitable* | Might be exploitable |

*Only if payout matrix is accurate — which it isn't.

**Hypothesis**: For self-play with approximate payouts, visit targets lead to better long-term convergence because they maintain exploration. The game-theoretic guarantee of Nash is only as good as the payout estimates it's computed from.

---

## 2026-01-21: MCCFR for Simultaneous Games (Big Guns)

**Context**: Current MCTS + Nash approach is a bit ad-hoc. For proper game-theoretic reasoning in simultaneous/imperfect-info games, MCCFR (Monte Carlo Counterfactual Regret Minimization) is the standard.

**What it is**: Instead of Q-values or payout matrices, track regret per action. Policy converges to Nash equilibrium over iterations. Used in poker AI (Libratus, Pluribus).

**Why consider it**:
- Theoretically sound for simultaneous games
- Handles mixed strategies naturally
- Proven at scale

**Why not yet**:
- Big implementation effort
- Different algorithm entirely (not MCTS)
- Current approach might be "good enough" for PyRat

---

## 2026-01-21: Modular Policy Derivation — Acting vs Learning

**Context**: Currently both acting (during self-play) and learning (NN targets) use the same policy: Nash equilibrium from the payout matrix. These should be independently configurable.

### The Decomposition

```
              SearchResult
                   │
       ┌───────────┴───────────┐
       │                       │
  action_visits            payout_matrix
       │                       │
       ▼                       ▼
 marginalize()           reduce_and_expand_nash()
       │                       │
       ▼                       ▼
  visit_policy              nash_policy

         Either can be used for:
         • Acting (sample from it during self-play)
         • Learning (target for NN training)
```

**Two independent choices:**
1. **Acting policy**: How does the agent pick moves? (Nash vs marginal visits)
2. **Learning target**: What does the NN learn to predict? (Nash vs marginal visits)

Current: both use Nash. But these could be mixed — e.g., act from visits, learn from Nash (or vice versa).

### Why Marginal Visits?

AlphaZero uses visit counts directly as policy targets. Visits naturally reflect MCTS's uncertainty — when the search isn't sure, it explores multiple actions, and the visit distribution captures that.

Nash, by contrast, is a sharp function. It takes noisy payout estimates and outputs confident strategies, throwing away the uncertainty signal. This can cause exploration to collapse over self-play iterations (see "Learn from Marginal Visits" section above).

For simultaneous games, marginalize the visit matrix:
```python
policy_p1 = visits.sum(axis=1)  # sum over P2's actions
policy_p1 /= policy_p1.sum()

policy_p2 = visits.sum(axis=0)  # sum over P1's actions
policy_p2 /= policy_p2.sum()
```

**Key point**: The payout matrix still guides MCTS selection during search. We're not removing it — we're just using visits (which preserve uncertainty) as the learning target instead of Nash (which sharpens it).

### KataGo Reference

KataGo separates search params from action selection params in the same config struct:

```cpp
// searchparams.h
// Search exploration
double cpuctExploration;
...

// Action selection (post-search)
double chosenMoveTemperature;
double chosenMoveSubtract;
double chosenMovePrune;
```

The pattern: search produces visit counts, then `chosenMove*` params control how visits → action. See `.mt/reference_code/KataGo/cpp/search/searchresults.cpp:572-596` for `getChosenMoveLoc()`.

### Proposed Implementation

Add to `DecoupledPUCTConfig`:

```python
class DecoupledPUCTConfig(StrictBaseModel):
    # Exploration (existing)
    simulations: int
    c_puct: float
    ...

    # Action selection (new)
    action_method: Literal["nash", "visits"] = "nash"
    action_temperature: float = 1.0
```

Default `"nash"` preserves current behavior — existing configs work unchanged.

For learning targets, add similar option in training config or `targets.py`.

### Dependency: Config Infrastructure Branch

**Wait for `refactor/config-infrastructure` to merge first.** That branch:
- Adds `StrictBaseModel` with `extra='forbid'` (catches config typos)
- Adds Hydra-based config loading with validation
- Already migrates `DecoupledPUCTConfig` to `StrictBaseModel`

Adding action selection config on that foundation is cleaner.

### Implementation Steps (Post-Merge)

**Part 1 — Learning target (easy)**:
- `visit_counts` already saved in recordings
- Add utility: `visits_to_marginal_policy(visits, axis) -> policy`
- Add flag in `targets.py` or training config to switch between Nash and visits
- Test: check marginal policies match expected values

**Part 2 — Acting (more plumbing)**:
- Add `action_method` and `action_temperature` to `DecoupledPUCTConfig`
- Modify `MCTSAgent.get_move()` to use config to derive policy from `SearchResult`
- Could compute marginal in agent from `result.action_visits` rather than using `result.policy_*`
- Game loop stays unchanged — just calls `agent.get_move()`

### Open Questions

- Should `SearchResult.policy_*` always be Nash, with acting policy computed by consumer? Or should SearchResult not have pre-computed policies at all?
- For acting from visits: use temperature like AlphaZero (`visits^(1/temp)`)? Or just proportional?
- When to use which? Current hypothesis: **visits for learning** (preserves uncertainty, better long-term convergence), acting policy TBD (could be either).
