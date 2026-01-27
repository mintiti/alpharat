# MCGS — Monte Carlo Graph Search for PyRat

**Status:** Investigation phase
**Branch:** `mcgs-exploration`
**Created:** 2026-01-26

> **What this file is:** The living document for the MCGS work. Progress notes, ideas, questions we run into and their answers — all go here. Not a spec frozen in time; it evolves as we learn.

---

## What is MCGS?

Standard MCTS treats the search space as a tree: if two different move sequences reach the same position, they get separate nodes with separate statistics. MCGS (Monte Carlo Graph Search) detects these **transpositions** and merges them into a single shared node, turning the tree into a DAG.

The wins:
- **Fewer NN evaluations** — a transposition reuses the existing node's evaluation instead of querying the net again.
- **Information sharing** — a tactic discovered via one path immediately benefits all paths that reach that position.
- **Memory savings** — fewer nodes for the same effective search depth.

The cost:
- Hashing overhead per node expansion.
- More complex backup (Q must be recomputed, not just averaged).
- Implementation and debugging complexity.

## Why We Expect It Might Help in PyRat

PyRat is a grid where two players move simultaneously. Several properties suggest transpositions should be common:

1. **Move ordering doesn't matter** — simultaneous moves mean different orderings of the same moves can converge to the same cell pair. The MCTS tree searches both branches.
2. **Small state space** — on small grids, the number of reachable position pairs is bounded. Even accounting for cheese subsets, convergence seems likely.
3. **Symmetric positions** — both players often traverse the same corridors, creating convergent paths.
4. **Action equivalence already shows value** — our outcome-indexed architecture saves work by merging blocked moves within a node. MCGS is the next level: merging across nodes.

But these are hypotheses. We need to measure before building anything.

## Phase 1: Measure Transposition Rates

**Goal:** Determine empirically whether there are enough transpositions to justify implementation.

**Tool:** `scripts/measure_transpositions.py` (already exists)

**What to measure:**
- Duplicate ratio with and without turn in the state hash
- Depth distribution — where do transpositions concentrate?
- How rates change as we vary: grid size, cheese count, simulation count

**State hash components:**
`(p1_pos, p2_pos, frozenset(cheese), p1_mud, p2_mud, [turn])` — this is the Markov state.

**What do the numbers mean?**
We don't have predefined thresholds. What matters is the shape of the data:
- Is duplication negligible or substantial?
- Does it grow with simulation count? (If yes, MCGS matters more at higher compute budgets — exactly where we'd want it.)
- Does grid size amplify or reduce it? (Larger grids = more paths = possibly more transpositions, but also larger state space.)
- Does including turn in the hash kill most transpositions? (If so, we need to think about whether to include it.)

The measurement tells us whether the problem is real. If it's clearly negligible, we stop. If it's meaningful, we proceed. The data decides.

### Results (2026-01-27)

Sweep over sim counts on 7×7, 10 cheese, 50 max turns, 20 games per sim count. Raw data in `experiments/transposition_measurement/`.

| Sims | Dup% (+turn) | Dup% (-turn) | Avg Nodes/Search |
|------|-------------|-------------|-----------------|
| 50   | 10.0%       | 26.1%       | 50              |
| 200  | 34.6%       | 50.4%       | 194             |
| 554  | 49.1%       | 63.8%       | 524             |
| 1000 | 56.8%       | 69.5%       | 923             |
| 2000 | 64.6%       | 75.4%       | 1,783           |

**Key findings:**

1. **Transpositions scale hard with sim count.** 10% → 65% (+turn) across the sweep. The marginal gain decelerates but hasn't flattened by 2000 sims. At our production sim count (554), roughly half the tree is wasted on duplicate states.

2. **The +turn / -turn gap is entirely cross-depth.** In every depth breakdown across all games, `unique_with_turn == unique_no_turn` at every depth level. Within a fixed tree depth, turn = root_turn + depth, so turn adds no discriminating power. The extra ~15pp from dropping turn comes from the same position appearing at different depths (e.g., both players STAY, arriving at the same position one turn later). Whether MCGS should merge these is a design question — same board at different turns may warrant different valuations due to remaining horizon.

3. **Games get shorter with more sims** (631 → 288 searches across the sweep). Stronger play collects cheese faster, fewer wasted moves.

**Conclusion:** The problem is real. At 554 sims, MCGS could cut tree size roughly in half from within-depth sharing alone (the safe, same-turn transpositions). Proceed to Phase 2.

**Side note — NN evaluation cache:** Even without full MCGS, a simple cache keyed by state hash would avoid redundant NN forward passes for transposed positions. This is much cheaper to implement than full graph search and captures the "fewer NN evaluations" win directly. Tracked separately as a GitHub issue (#50).

### Beyond speed: search quality

The measurements above frame MCGS as a compute savings story — fewer NN calls for the same search. But there's a potentially stronger argument: **MCGS improves search quality at the same sim budget.**

Two effects worth investigating:

1. **Unique state coverage.** At 554 sims with walls, ~530 nodes per search but only ~140 unique states (74% wasted). MCGS would spend those same 554 sims exploring ~530 *unique* states — roughly 3.8× more coverage of the actual game tree. More of the state space understood per search.

2. **Effective search depth.** Duplicates consume expansion budget that could push the search deeper. With MCGS, the freed-up sims should reach greater depth, giving the agent better lookahead. The depth breakdowns in our data already show where duplicates concentrate — could quantify the expected depth gain.

Both are "the search understands the problem space better" rather than "the search runs faster." This matters for playing strength: at equal sim counts, MCGS should produce better policies, not just the same policies cheaper. Worth measuring in Phase 4 tournaments — compare not just sims/s but also win rates at equal sim budgets.

## Phase 2: Formalize the Algorithm

Before writing any code, work out the full algorithm on paper. The goal is a complete description of how every component works — selection, expansion, backup, transposition handling — adapted to our simultaneous-move, bimatrix setting.

This means:
- **Define the node state.** What does each node store? How do edge visits, payout matrices, and the NN utility estimate relate? Write the formulas.
- **Define the backup rule.** Extend the KataGo idempotent Q formula to bimatrix payouts with two independent policies. Prove (or at least convince ourselves) that it's sound for decoupled PUCT.
- **Define transposition handling.** When a node is reached via a new parent: what gets shared, what stays per-edge? How does outcome indexing interact with shared nodes?
- **Define navigation.** How does the simulator reach a shared node? Canonical parent? Stored state? Spell out the trade-offs and pick one.
- **Walk through examples.** Take a concrete 3x3 or 4x4 scenario, trace the algorithm by hand through a few playouts including a transposition. Make sure the formulas produce sensible values.

The output of this phase is a self-contained algorithm description that we're confident in. No code yet — just the math and the logic. If something doesn't work out during formalization (e.g. the bimatrix extension is unsound), we find out here rather than after writing 500 lines.

## Phase 3: Implement

Build a separate `alpharat/mcgs/` module. Completely independent from the current `alpharat/mcts/` — both usable interchangeably through the same `Agent` interface.

The decoupled PUCT logic (PUCT formula, marginal Q computation) should be shared or kept identical. The difference is in node creation, backup, and navigation.

## Phase 4: Validate

**Correctness:**
- Unit tests: transposition detection, backup correctness, PUCT with edge visits
- Property tests: MCGS payout matrix converges to same values as MCTS given enough sims
- Hand-traced examples from Phase 2 as regression tests

**Strength:**
Same simulation budget, does MCGS produce better play? Tournament of MCTS vs MCGS at equal sim counts, enough games for statistical significance. Test with and without NN priors.

If MCGS wins at equal sims, the feature ships. If it's neutral on strength but faster in sims/s, also worth it. If it's slower AND not stronger, abandon.

**Speed:**
Measure sims/s for MCTS vs MCGS. Without NN, the comparison is purely CPU overhead. With NN, savings from skipped evaluations should dominate.

---

## Technical Context

### The Core Insight (from KataGo)

The key to correct MCGS is distinguishing **edge visits** from **child visits**:

- **Edge visit** N(n,a): how many times PUCT at node n selected action a
- **Child visit** N(c): total visits to child node c from ALL parents

In a tree these are identical. In a DAG they diverge — a child can have more visits than any single parent's edge to it.

PUCT must use edge visits (the parent's local exploration budget), not child visits. And Q must be recomputed recursively, not averaged incrementally:

```
Q(n) = (1/N(n)) * (U(n) + sum over actions a: edge_visits(n,a) * Q(child(n,a)))
```

This is the **idempotent formulation** from KataGo. It's simpler, easier to reason about, and avoids the staleness correction hacks in Czech et al.

### Adaptation Challenges for Simultaneous Moves

Standard MCGS assumes alternating moves with a single PUCT selection. We have:
- Two players selecting independently via decoupled PUCT
- Outcome-indexed storage (reduced matrices)
- Bimatrix payouts (separate P1/P2 values, not a single Q)

**What stays the same:** Decoupled PUCT selection, Nash at root, action equivalence / outcome indexing.

**What needs work:**
- Extending the idempotent Q formula to bimatrix payouts with two independent policies
- Edge visits become per-outcome-pair, not per-action
- Navigation with multiple parents (canonical parent pointer is the likely approach)
- Turn number in state hash (include or not — measure both)

### Architecture

```
alpharat/mcgs/           # New graph-based MCGS
    node.py              # MCGSNode — multi-parent support
    graph.py             # MCGSGraph — transposition table
    search.py            # Graph search with decoupled PUCT
    state_key.py         # State hashing
```

---

## References

### Papers
- **Czech, Korus, Kersting (2020)** — *Monte-Carlo Graph Search for AlphaZero* — [arXiv:2012.11045](https://arxiv.org/abs/2012.11045)
  Original MCGS formulation. Demonstrates improvements in chess and crazyhouse. Uses incremental Q updates with staleness correction.

- **Grill et al. (2020)** — *Monte-Carlo Tree Search as Regularized Policy Optimization* — [arXiv:2007.12509](https://arxiv.org/abs/2007.12509)
  Theoretical foundation: MCTS visit distribution approximates regularized policy optimization. Basis for KataGo's cleaner derivation.

### Implementations
- **KataGo** — `reference_code/KataGo/docs/GraphSearch.md`
  Cleanest conceptual explanation. Key insight: idempotent Q updates, edge visits vs child visits. This is the formulation we should follow.

- **Leela Chess Zero** — `reference_code/lc0/src/search/dag_classic/`
  Production C++ implementation. Two-tier Node/LowNode architecture. Atomic parent counting, weak_ptr transposition table for GC.

### Key Takeaways
1. **Edge visits, not child visits, drive PUCT** — the fundamental correctness requirement
2. **Idempotent Q recomputation is simpler and equally sound** (KataGo) vs incremental with staleness correction (Czech et al., lc0)
3. **Stale Q values are okay for correctness** — PUCT guarantees every node is eventually revisited
4. **Short-circuit when child visits > edge visits** — optional efficiency improvement (KataGo makes it configurable)

---

## Open Questions

1. **Is the KataGo derivation sound for two independent policies?** The MCTS-as-policy-optimization framework assumes one agent optimizing one policy. We have two agents each optimizing independently. Need to verify the idempotent Q formula extends correctly to bimatrix payouts with decoupled selection.

2. **Payout matrix backup in shared nodes.** Each parent tracks its own edge visits and sees the same child payout matrix. The value propagated upward depends on which outcome pair the parent selected. This should work naturally, but needs formalization.

3. **Outcome indexing across parents.** Different parents might have different effective action mappings. But the shared child node has one set of effective actions determined by its own position. The mapping from parent outcome pair to child state is well-defined — just need to make sure the implementation handles this correctly.

4. **Zobrist hashing.** For performance, we'd want incremental XOR-based hashing rather than recomputing the full state hash at each expansion. Design needed for position/cheese/mud hash components.
