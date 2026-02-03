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
    total = sum(e.visits for e in node.edges.values())
    if total == 0:
        return node.prior_p1, node.prior_p2

    π_p1 = zeros(node.n1)
    π_p2 = zeros(node.n2)
    for (i, j), edge in node.edges.items():
        π_p1[i] += edge.visits
        π_p2[j] += edge.visits

    return π_p1 / total, π_p2 / total
```

### Playout (Selection → Expansion → Evaluation → Backup)

```python
def playout(node, game):
    """Single MCGS playout. Returns after updating node's Q values."""

    # ---------- TERMINAL CHECK ----------
    if game.is_over():
        # Terminal node: Q = actual game outcome
        # (Could also set Q = 0 if we only care about future rewards)
        return

    # ---------- EVALUATION (first visit) ----------
    if not node.is_evaluated:
        evaluate_node(node, game)
        return  # Stop here on first visit (leaf evaluation)

    # ---------- SELECTION ----------
    o1 = select_outcome_p1(node)
    o2 = select_outcome_p2(node)

    # ---------- EXPANSION ----------
    if (o1, o2) not in node.edges:
        expand_edge(node, game, o1, o2)

    edge = node.edges[(o1, o2)]

    # ---------- RECURSE ----------
    undo = game.make_move(o1, o2)
    playout(edge.child, game)
    game.unmake_move(undo)

    # ---------- BACKUP ----------
    edge.visits += 1
    recompute_Q(node)
```

### Evaluation

```python
def evaluate_node(node, game):
    """First-time evaluation of a node via NN (or uniform prior if no NN)."""

    # Get NN predictions
    if nn is not None:
        policy_p1, policy_p2, value_p1, value_p2 = nn.predict(game)
    else:
        # Uniform prior, zero value (pessimistic)
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

    total_visits = sum(e.visits for e in node.edges.values())

    # Compute marginal visits per outcome
    marginal_visits = zeros(node.n1)
    for (i, j), edge in node.edges.items():
        marginal_visits[i] += edge.visits

    # Compute marginal Q for each outcome
    marginal_Q = compute_marginal_Q_p1(node)

    # PUCT score
    exploration = node.prior_p1 * sqrt(total_visits) / (1 + marginal_visits)
    scores = marginal_Q + c_puct * exploration

    return argmax(scores)


def select_outcome_p2(node):
    """Select player 2's outcome via PUCT. Symmetric to p1."""
    # ... symmetric implementation ...


def compute_marginal_Q_p1(node):
    """Compute Q_p1[i] = E_{j~π_2}[r_p1[i,j] + child[i,j].Q_p1]"""

    # Opponent's policy (using prior for now)
    π_2 = node.prior_p2

    marginal_Q = zeros(node.n1)

    for i in range(node.n1):
        q_sum = 0.0
        for j in range(node.n2):
            if (i, j) in node.edges:
                edge = node.edges[(i, j)]
                child_Q = edge.child.Q_p1 if edge.child.is_evaluated else 0.0
                q_sum += π_2[j] * (edge.r_p1 + child_Q)
            else:
                # Unexpanded edge: pessimistic Q = 0
                q_sum += π_2[j] * 0.0
        marginal_Q[i] = q_sum

    return marginal_Q


def compute_marginal_Q_p2(node):
    """Symmetric to p1."""
    # ... symmetric implementation ...
```

### Expansion

```python
def expand_edge(node, game, o1, o2):
    """Expand edge (o1, o2): make move, observe reward, get/create child."""

    # Make the move and observe intermediate rewards
    undo = game.make_move(o1, o2)
    r_p1 = undo.cheese_collected_p1  # Immediate reward
    r_p2 = undo.cheese_collected_p2

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

    # Create edge (visits starts at 0, incremented after recursion)
    node.edges[(o1, o2)] = Edge(
        child=child,
        visits=0,
        r_p1=r_p1,
        r_p2=r_p2,
    )

    game.unmake_move(undo)


def compute_state_key(game):
    """Compute transposition table key from game state."""
    return (
        game.player1_position,
        game.player2_position,
        frozenset(game.cheese_positions()),
        game.player1_mud_turns,
        game.player2_mud_turns,
        game.turn,  # Include turn for same-depth-only transpositions
    )
```

### Backup (Idempotent Q Recomputation)

```python
def recompute_Q(node):
    """Recompute Q from children. Idempotent — result depends only on current state."""

    if not node.is_evaluated:
        return  # Nothing to do

    total_visits = sum(e.visits for e in node.edges.values())

    if total_visits == 0:
        # No children visited yet, Q = U (NN estimate)
        node.Q_p1 = node.U_p1
        node.Q_p2 = node.U_p2
        return

    # Weighted sum over visited edges
    weighted_sum_p1 = 0.0
    weighted_sum_p2 = 0.0

    for (i, j), edge in node.edges.items():
        if edge.visits > 0:
            child_Q_p1 = edge.child.Q_p1 if edge.child.is_evaluated else 0.0
            child_Q_p2 = edge.child.Q_p2 if edge.child.is_evaluated else 0.0

            weighted_sum_p1 += edge.visits * (edge.r_p1 + child_Q_p1)
            weighted_sum_p2 += edge.visits * (edge.r_p2 + child_Q_p2)

    # Q = weighted average of U (weight 1) and children (weight = visits)
    node.Q_p1 = (node.U_p1 + weighted_sum_p1) / (1 + total_visits)
    node.Q_p2 = (node.U_p2 + weighted_sum_p2) / (1 + total_visits)
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
