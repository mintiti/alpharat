# MCGS Algorithm Formalization

**Status:** Draft
**Parent:** `mcgs-brief.md`

This document specifies the MCGS algorithm adapted for simultaneous two-player games with intermediate rewards. No payout matrices, no Nash equilibrium — purely visit-distribution based.

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

| Field | Type | Description |
|-------|------|-------------|
| `Q_p1`, `Q_p2` | scalar | Expected value for each player from this position. Recomputed idempotently. |
| `U_p1`, `U_p2` | scalar | NN initial estimates. Frozen after first evaluation. |
| `prior_p1[n1]` | array | NN policy for player 1, reduced to unique outcomes. |
| `prior_p2[n2]` | array | NN policy for player 2, reduced to unique outcomes. |
| `edges[(i,j)]` | dict | Maps outcome pair to edge data. See below. |

**Edge data** for outcome pair (i, j):

| Field | Type | Description |
|-------|------|-------------|
| `child` | Node* | Pointer to child node (may be shared via transposition). |
| `visits` | int | Edge visits — how many times THIS parent selected this outcome pair. |
| `r_p1`, `r_p2` | scalar | Immediate reward (cheese collected) on this transition. |

**Note:** Rewards are per-edge, not per-child. The same child can be reached from different parents with different rewards.

---

## Derived Quantities

From edge visits, we derive:

```
total_visits = Σ_{i,j} edges[i,j].visits

marginal_visits_p1[i] = Σ_j edges[i,j].visits
marginal_visits_p2[j] = Σ_i edges[i,j].visits
```

---

## Q Recomputation (Idempotent)

When a node is visited, recompute Q from children:

```
Q_p(n) = (U_p(n) + Σ_{i,j} edges[i,j].visits * (edges[i,j].r_p + child_Q_p(i,j)))
         / (1 + total_visits)
```

Where:
- `U_p(n)` is the NN's initial estimate (weight = 1, like a virtual visit)
- `child_Q_p(i,j) = edges[i,j].child.Q_p` if child exists, else `0` (pessimistic)

**Pessimistic default:** Unexpanded edges contribute 0 to the Q sum. This encourages exploration — you have to visit a path to get credit for it.

---

## PUCT Selection

Each player selects independently via decoupled PUCT.

### Step 1: Opponent's Policy

For computing marginal Q, we need the opponent's policy. Options:

**Option A (Prior-based):**
```
π_2[j] = prior_p2[j]
```

**Option B (Visit-based):**
```
π_2[j] = marginal_visits_p2[j] / total_visits  (if total > 0, else prior)
```

**Option C (Blended):**
```
π_2[j] ∝ prior_p2[j] + marginal_visits_p2[j]  (normalized)
```

Start with Option A (prior-based) for simplicity. The prior is always defined and doesn't have cold-start issues.

### Step 2: Marginal Q

Player 1 computes expected value for each of their outcomes:

```
Q_p1[i] = Σ_j π_2[j] * (edges[i,j].r_p1 + child_Q_p1(i,j))
```

Where `child_Q_p1(i,j) = 0` if edge unexpanded (pessimistic).

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

## Playout Algorithm

```python
def playout(node, game):
    if game.is_over():
        return  # Terminal, nothing to update

    if node.visits == 0:
        # First visit: evaluate with NN
        node.U_p1, node.U_p2 = nn_evaluate_value(game)
        node.prior_p1, node.prior_p2 = nn_evaluate_policy(game)
        node.Q_p1, node.Q_p2 = node.U_p1, node.U_p2
        return

    # Select outcome pair via decoupled PUCT
    o1 = puct_select_p1(node)
    o2 = puct_select_p2(node)

    # Expand edge if needed
    if (o1, o2) not in node.edges:
        # Make move, observe reward
        undo, r_p1, r_p2 = game.make_move(o1, o2)

        # Check transposition table
        state_key = compute_state_key(game)
        if state_key in transposition_table:
            child = transposition_table[state_key]
        else:
            child = Node()
            transposition_table[state_key] = child

        node.edges[(o1, o2)] = Edge(child=child, visits=0, r_p1=r_p1, r_p2=r_p2)
    else:
        edge = node.edges[(o1, o2)]
        undo = game.make_move(o1, o2)  # Just for navigation

    edge = node.edges[(o1, o2)]

    # Optional: short-circuit if child has enough visits
    # if edge.child.visits <= edge.visits:
    playout(edge.child, game)

    # Increment edge visits
    edge.visits += 1

    # Recompute Q (idempotent)
    recompute_Q(node)

    # Undo move for navigation
    game.unmake_move(undo)
```

---

## Final Policy (No Nash)

After search completes, the policy IS the visit distribution:

```
π_p1[i] = marginal_visits_p1[i] / total_visits
π_p2[j] = marginal_visits_p2[j] / total_visits
```

To select a move:
- Sample from π (with temperature for exploration)
- Or take argmax (for best play)

No Nash equilibrium computation. The visit distribution is the learned policy.

---

## Transposition Table

**State key:** `(p1_pos, p2_pos, frozenset(cheese), p1_mud, p2_mud, turn)`

Including turn means we only merge same-turn transpositions. Cross-depth transpositions (same position at different turns) stay separate. This is the conservative choice — different turns may warrant different valuations due to remaining horizon.

**Lookup:** Before creating a new child node, check if state_key exists. If yes, link to existing node.

**Zobrist hashing:** For performance, use incremental XOR-based hashing rather than recomputing the full key. Design TBD.

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

1. **Opponent policy for marginal Q** — Prior vs visits vs blend. Needs experimentation.

2. **Short-circuit optimization** — Skip child playout when `child.visits > edge.visits`? KataGo makes this configurable.

3. **Staleness propagation** — Currently only update nodes on playout path. Could also propagate to other parents, but adds complexity.

4. **Virtual loss for parallelism** — Not addressed yet. Needed for multi-threaded search.

---

## Differences from KataGo

| Aspect | KataGo | Ours |
|--------|--------|------|
| Players | Alternating | Simultaneous |
| Q values | 1 per node | 2 per node (Q_p1, Q_p2) |
| Policies | 1 per node | 2 per node |
| PUCT | Single selection | Decoupled (independent per player) |
| Rewards | Terminal only | Intermediate (cheese per transition) |
| Final policy | Visit distribution | Visit distribution (same) |
| Nash | N/A | Removed (was in tree MCTS) |

---

## Next Steps

1. Walk through a concrete example (3×3 grid, few cheese) by hand
2. Verify the formulas produce sensible values
3. Identify edge cases (terminal states, no valid moves, etc.)
4. Implement in `alpharat/mcgs/`
