# MCGS Algorithm Formalization

**Status:** Draft
**Parent:** `mcgs-brief.md`

This document specifies the MCGS algorithm adapted for simultaneous two-player games with intermediate rewards. No payout matrices — purely visit-distribution based.

---

## Overview

Standard MCGS (KataGo-style) for alternating games:
- One player moves per turn
- Single Q value per node
- Single policy, single PUCT selection

Our adaptation for simultaneous moves:
- Both players move simultaneously
- Two Q values per node (Q_p1, Q_p2)
- Two policies, decoupled PUCT (each player selects independently)
- Intermediate rewards (cheese collection) on transitions

---

## Node State

Each node stores:

**Frozen after evaluation (read-only, thread-safe):**

| Field | Type | Description |
|-------|------|-------------|
| `is_evaluated` | bool | Whether NN has evaluated this node. |
| `U_p1`, `U_p2` | scalar | NN initial value estimates. Frozen after first evaluation. |
| `prior_p1[n1]` | array | NN policy for player 1, reduced to unique outcomes. |
| `prior_p2[n2]` | array | NN policy for player 2, reduced to unique outcomes. |

**Outcome indexing (frozen after creation):**

| Field | Type | Description |
|-------|------|-------------|
| `n1`, `n2` | int | Number of unique outcomes per player. |
| `p1_outcomes[n1]` | array | Outcome index → action (Direction value). |
| `p2_outcomes[n2]` | array | Outcome index → action (Direction value). |
| `p1_action_to_idx[5]` | array | Action → outcome index. |
| `p2_action_to_idx[5]` | array | Action → outcome index. |

**Mutable (written during backup, read during selection):**

| Field | Type | Description |
|-------|------|-------------|
| `Q_p1`, `Q_p2` | scalar | Expected value for each player. Recomputed idempotently. Future: atomic. |
| `virtual_losses` | int | 0 for single-threaded. Future: atomic, used during parallel selection. |

**Edge data — stored as arrays `[n1, n2]`:**

| Field | Type | Description |
|-------|------|-------------|
| `edge_visits[n1, n2]` | int array | Per-pair visit counts. The fundamental statistic. |
| `edge_r_p1[n1, n2]` | float array | Immediate reward for player 1 on this transition. |
| `edge_r_p2[n1, n2]` | float array | Immediate reward for player 2 on this transition. |
| `edge_children[n1][n2]` | Node array | Child pointers (None if unexpanded). May be shared via transposition. |

**Why arrays, not dicts:** The `(n1, n2)` dimensions are small (typically 3–5) and fixed at creation. Arrays give cache-friendly access, numpy marginal sums, and a clear path to C extensions with atomic operations for future threading.

**Rewards are per-edge, not per-child.** The same child can be reached from different parents with different rewards (different cheese configurations at the parent).

**Threading notes:** Single-threaded for now. The structure is designed so that the threading path is:
- Make `Q_p1`, `Q_p2`, `virtual_losses`, and `edge_visits` atomic.
- Add virtual loss increment at selection entry, decrement at backup.
- Transposition table needs concurrent insert/lookup (e.g. sharded locks).
- Per-thread playout path tracking (already implicit in the recursive call stack).

---

## Derived Quantities

From edge visits, we derive per-player marginal quantities:

```
total_visits = Σ_{i,j} edge_visits[i,j]

marginal_visits_p1[i] = Σ_j edge_visits[i,j]
marginal_visits_p2[j] = Σ_i edge_visits[i,j]
```

**Mapping to KataGo:** In alternating-move MCGS, one action = one edge = one child, so `edgeVisits` is a scalar per edge. In simultaneous moves, one player-action maps to a *bundle* of edges (one per opponent response). The per-player marginal is the analog of KataGo's `edgeVisits`:

| KataGo (alternating) | Ours (simultaneous) |
|---|---|
| `edgeVisits(action)` | `marginal_visits_p1[i] = Σ_j edge_visits[i,j]` |
| `child.Q` (one child per action) | `marginal_Q_p1[i]` (weighted over opponent responses) |
| PUCT uses `edgeVisits` | PUCT uses `marginal_visits_p1[i]` |

---

## Q Recomputation (Idempotent)

When a node is visited, recompute Q from children:

```
Q_p(n) = (U_p(n) + Σ_{i,j} edge_visits[i,j] * (edge_r_p[i,j] + child_Q_p(i,j)))
         / (1 + total_visits)
```

Where:
- `U_p(n)` is the NN's initial estimate (weight = 1, like a virtual visit)
- The sum is only over edges with `edge_visits[i,j] > 0`
- For visited edges, `child_Q_p(i,j) = edge_children[i][j].Q_p` — always available (see invariant below)

**Invariant:** When `edge_visits[i,j] > 0`, the child exists and is evaluated. This holds because: expansion creates the child, the playout recurses into it, and if it's new it gets evaluated on first visit. If it's a transposition, it was already evaluated from a prior path.

**FPU is not used in backup.** Unvisited edges have `edge_visits = 0` and contribute nothing to the sum. The formula only considers what we've actually observed.

When KataGo's uncertainty weighting is set to uniform (all weights = 1), their Q formula simplifies to exactly this. The `edgeVisits / childVisits` scaling in KataGo becomes a no-op: `child.weightSum * edgeVisits / childVisits = childVisits * edgeVisits / childVisits = edgeVisits`.

---

## First Play Urgency (FPU)

Unexplored actions need a Q estimate for PUCT selection. Rather than 0 (pessimistic), use the parent's current Q minus an offset:

```
fpu_Q_p(n) = Q_p(n) - fpu_offset
```

Where `fpu_offset` is a small positive constant (start with ~0.1, tune empirically).

**Why not 0:** A pessimistic 0 makes unexplored actions look terrible, forcing PUCT to rely solely on the exploration term. FPU gives a warmer estimate: "this unexplored action is probably roughly as good as what I've seen, minus a penalty for uncertainty."

**Where it's used:** Only in marginal Q computation during PUCT selection (for actions where `marginal_visits_p1[i] == 0`). Not used in backup.

**KataGo comparison:** KataGo uses `parentUtility - fpuReductionMax * sqrt(visitedPolicyMass)` — adjusting based on how much of the prior has been explored. Our simpler constant offset is a reasonable starting point.

---

## PUCT Selection

Each player selects independently via decoupled PUCT.

### Step 1: Opponent Weighting for Marginal Q

For computing player 1's marginal Q at action `i`, we need to weight over player 2's responses. The key question: what weights?

**Option A (Prior-based):**
```
weight[j] = prior_p2[j]    (always defined, no cold-start)
```

**Option B (Visit-based):**
```
weight[j] = marginal_visits_p2[j] / total_visits    (if total > 0, else prior)
```

**Option C (Blended):**
```
weight[j] ∝ prior_p2[j] + marginal_visits_p2[j]    (normalized)
```

**Option D (Conditional visit proportions) — preferred:**
```
When marginal_visits_p1[i] > 0:
    weight[j] = edge_visits[i,j] / marginal_visits_p1[i]

When marginal_visits_p1[i] == 0:
    Use FPU directly (no weighting needed, see Step 2)
```

Option D is "what actually happened when I played action i" — the empirical opponent response distribution conditional on our action. It's the closest analog to KataGo, where `child.Q` inherently reflects the actual play below it. Unvisited opponent responses (within a visited action) naturally get weight 0, so we only average over what we've observed.

**Start with Option D.** Fall back to Option A if empirically problematic.

### Step 2: Marginal Q

Player 1 computes expected value for each of their outcomes:

```
If marginal_visits_p1[i] > 0:
    Q_p1[i] = Σ_j (edge_visits[i,j] / marginal_visits_p1[i])
              * (edge_r_p1[i,j] + edge_children[i][j].Q_p1)
              (sum only over j where edge_visits[i,j] > 0)

If marginal_visits_p1[i] == 0:
    Q_p1[i] = fpu_Q_p1 = Q_p1(node) - fpu_offset
```

Symmetrically for player 2.

### Step 3: PUCT Score

```
score_p1[i] = Q_p1[i] + c_puct * prior_p1[i] * sqrt(total_visits) / (1 + marginal_visits_p1[i])
```

Select: `outcome_p1 = argmax_i score_p1[i]`

Symmetrically for player 2.

### Step 4: Joint Selection

The selected outcome pair is `(outcome_p1, outcome_p2)`. This determines which edge to traverse.

---

## Core Algorithm

### Main Search Loop

```python
def search(root, game, num_simulations):
    """Run MCGS from root for num_simulations playouts."""
    for _ in range(num_simulations):
        playout(root, game.clone())

    # Return visit distribution as policy
    return get_visit_distribution(root)


def get_visit_distribution(node):
    """Extract marginal visit distributions for both players."""
    total = node.edge_visits.sum()
    if total == 0:
        return node.prior_p1, node.prior_p2

    π_p1 = node.edge_visits.sum(axis=1) / total  # marginal over j → [n1]
    π_p2 = node.edge_visits.sum(axis=0) / total  # marginal over i → [n2]

    return π_p1, π_p2
```

### Playout (Selection → Expansion → Evaluation → Backup)

```python
def playout(node, game):
    """Single MCGS playout. Returns after updating node's Q values."""

    # ---------- TERMINAL CHECK ----------
    if game.is_over():
        return

    # ---------- EVALUATION (first visit) ----------
    if not node.is_evaluated:
        evaluate_node(node, game)
        return  # Stop here on first visit (leaf evaluation)

    # ---------- SELECTION ----------
    o1 = select_outcome_p1(node)  # outcome index (0..n1-1)
    o2 = select_outcome_p2(node)  # outcome index (0..n2-1)

    # ---------- EXPANSION ----------
    if node.edge_children[o1][o2] is None:
        expand_edge(node, game, o1, o2)

    child = node.edge_children[o1][o2]

    # ---------- RECURSE ----------
    # Map outcome indices back to game actions for make_move
    a1 = node.p1_outcomes[o1]  # outcome idx → Direction value
    a2 = node.p2_outcomes[o2]
    undo = game.make_move(a1, a2)

    # NOTE: If child is already evaluated (transposition hit), this does NOT
    # stop at the leaf — it recurses into the child's subtree immediately.
    # This is the main benefit of MCGS: we skip NN eval and leverage
    # existing search. First visit through a new edge to a transposed child
    # can recurse arbitrarily deep.
    playout(child, game)

    game.unmake_move(undo)

    # ---------- BACKUP ----------
    node.edge_visits[o1, o2] += 1
    recompute_Q(node)
```

### Evaluation

```python
def evaluate_node(node, game):
    """First-time evaluation of a node via NN (or uniform prior if no NN)."""

    if nn is not None:
        policy_p1, policy_p2, value_p1, value_p2 = nn.predict(game)
    else:
        policy_p1 = uniform(node.n1)
        policy_p2 = uniform(node.n2)
        value_p1, value_p2 = 0.0, 0.0

    # Store initial estimates (frozen)
    node.prior_p1 = policy_p1  # Already reduced to outcome space
    node.prior_p2 = policy_p2
    node.U_p1 = value_p1
    node.U_p2 = value_p2

    # Initialize Q to NN estimate
    node.Q_p1 = value_p1
    node.Q_p2 = value_p2

    node.is_evaluated = True
```

### Selection (Decoupled PUCT)

```python
def select_outcome_p1(node):
    """Select player 1's outcome via PUCT."""

    total_visits = node.edge_visits.sum()
    marginal_visits = node.edge_visits.sum(axis=1)  # [n1]

    marginal_Q = compute_marginal_Q_p1(node)

    exploration = node.prior_p1 * sqrt(total_visits) / (1 + marginal_visits)
    scores = marginal_Q + c_puct * exploration

    return argmax(scores)


def select_outcome_p2(node):
    """Symmetric to p1."""
    # ... symmetric implementation ...


def compute_marginal_Q_p1(node):
    """Compute Q_p1[i] — expected value of player 1's action i.

    Uses conditional visit proportions (Option D): weight opponent
    responses by how often they were actually played against action i.
    """
    fpu = node.Q_p1 - fpu_offset
    marginal_Q = zeros(node.n1)

    for i in range(node.n1):
        marginal_i = node.edge_visits[i, :].sum()

        if marginal_i > 0:
            # Option D: weight by conditional visit proportions
            q_sum = 0.0
            for j in range(node.n2):
                v = node.edge_visits[i, j]
                if v > 0:
                    child = node.edge_children[i][j]
                    q_sum += v * (node.edge_r_p1[i, j] + child.Q_p1)
            marginal_Q[i] = q_sum / marginal_i
        else:
            # Never played action i: use FPU
            marginal_Q[i] = fpu

    return marginal_Q
```

### Expansion

```python
def expand_edge(node, game, o1, o2):
    """Expand edge (o1, o2): make move, observe reward, get/create child."""

    # Map outcome indices to game actions
    a1 = node.p1_outcomes[o1]
    a2 = node.p2_outcomes[o2]

    # Make the move and observe intermediate rewards
    undo = game.make_move(a1, a2)
    node.edge_r_p1[o1, o2] = undo.cheese_collected_p1
    node.edge_r_p2[o1, o2] = undo.cheese_collected_p2

    # Check transposition table
    state_key = compute_state_key(game)

    if state_key in transposition_table:
        child = transposition_table[state_key]
    else:
        child = Node(
            n1=len(game.effective_outcomes_p1()),
            n2=len(game.effective_outcomes_p2()),
        )
        transposition_table[state_key] = child

    node.edge_children[o1][o2] = child

    game.unmake_move(undo)


def compute_state_key(game):
    """Compute transposition table key from game state."""
    return (
        game.player1_position,
        game.player2_position,
        frozenset(game.cheese_positions()),
        game.player1_mud_turns,
        game.player2_mud_turns,
        game.turn,  # Include turn — no cycles possible (turn always increases)
    )
```

### Backup (Idempotent Q Recomputation)

```python
def recompute_Q(node):
    """Recompute Q from children. Idempotent — result depends only on current state.

    Only sums over visited edges (visits > 0). FPU is not used here — it's
    only for PUCT selection. The backup formula is purely empirical: what did
    we actually observe?
    """
    if not node.is_evaluated:
        return

    total_visits = node.edge_visits.sum()

    if total_visits == 0:
        node.Q_p1 = node.U_p1
        node.Q_p2 = node.U_p2
        return

    weighted_sum_p1 = 0.0
    weighted_sum_p2 = 0.0

    for i in range(node.n1):
        for j in range(node.n2):
            v = node.edge_visits[i, j]
            if v > 0:
                child = node.edge_children[i][j]
                weighted_sum_p1 += v * (node.edge_r_p1[i, j] + child.Q_p1)
                weighted_sum_p2 += v * (node.edge_r_p2[i, j] + child.Q_p2)

    node.Q_p1 = (node.U_p1 + weighted_sum_p1) / (1 + total_visits)
    node.Q_p2 = (node.U_p2 + weighted_sum_p2) / (1 + total_visits)
```

---

## Final Policy

After search completes, the policy IS the visit distribution:

```
π_p1[i] = marginal_visits_p1[i] / total_visits
π_p2[j] = marginal_visits_p2[j] / total_visits
```

To select a move:
- Sample from π (with temperature for exploration)
- Or take argmax (for best play)

The visit distribution is the learned policy.

---

## Transposition Table

**State key:** `(p1_pos, p2_pos, frozenset(cheese), p1_mud, p2_mud, turn)`

Including turn means we only merge same-turn transpositions. Cross-depth transpositions (same position at different turns) stay separate. This is the conservative choice — different turns may warrant different valuations due to remaining horizon.

**No cycles possible:** Turn always increases, so a playout path can never revisit a node. No cycle detection needed (unlike KataGo, which needs it for Go's ko/superko rules).

**Lookup:** Before creating a new child node, check if state_key exists. If yes, link to existing node.

**Zobrist hashing:** For performance, use incremental XOR-based hashing rather than recomputing the full key. Design TBD — optimization for later.

---

## Outcome Indexing Across Parents

Different parents may have different effective action mappings (due to different wall configurations at parent positions). But the child's state determines its own outcome indexing.

When parent A links to child C via outcome pair (i_A, j_A):
- The game state after (i_A, j_A) from A equals C's state
- C has its own outcome indexing based on its position
- The mapping is: parent's (i, j) → child state → child's outcome space

This works because:
1. Each parent stores its own edge data (visits, rewards) keyed by ITS outcome indices
2. The child node is shared, with its own Q values and outcome indexing
3. No confusion — the parent's edge maps to a specific child, regardless of how other parents reach that child

---

## Open Questions

1. **Opponent policy for marginal Q** — Option D (conditional visit proportions) is the current plan. May experiment with prior-based (Option A) if cold-start is problematic.

2. **FPU offset value** — Starting with a fixed constant (~0.1). KataGo uses a more sophisticated formula based on explored policy mass. Tune empirically.

3. **Short-circuit optimization** — When `child.total_visits >> edge_visits[i,j]`, skip recursion and just increment edge visits. The child's Q is already well-estimated from other parents' visits. KataGo makes this configurable. Defer to implementation phase.

4. **Staleness propagation** — Currently only update nodes on playout path. Other parents of updated children keep stale Q until revisited. KataGo confirms this is fine: PUCT guarantees eventual revisits, and idempotent Q self-corrects on next visit.

---

## Differences from KataGo

| Aspect | KataGo | Ours |
|--------|--------|------|
| Players | Alternating | Simultaneous |
| Q values | 1 per node | 2 per node (Q_p1, Q_p2) |
| Policies | 1 per node | 2 per node |
| PUCT | Single selection, `edgeVisits` | Decoupled, `marginal_visits_p[i] = Σ_j edge_visits[i,j]` |
| Edge Q | `child.Q` directly | Weighted over opponent responses (Option D) |
| Rewards | Terminal only | Intermediate (cheese per transition) |
| Edge storage | Per-action child pointer | `[n1, n2]` arrays (action pairs) |
| NN weights | Uncertainty-based | Uniform (weight = 1) |
| FPU | `parentQ - offset * sqrt(visitedPolicyMass)` | `parentQ - fpu_offset` (simpler) |
| Cycle detection | Yes (Go ko/superko) | No (turn in hash, always increases) |
| Final policy | Visit distribution | Visit distribution (same) |

---

## Next Steps

1. Walk through a concrete example (3×3 grid, few cheese) by hand
2. Verify the formulas produce sensible values
3. Identify edge cases (terminal states, no valid moves, etc.)
4. Implement in `alpharat/mcgs/`
