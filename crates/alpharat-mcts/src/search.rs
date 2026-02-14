use rand::Rng;

use crate::backend::Backend;
use crate::node::{HalfNode, Node, NodeArena, NodeIndex};
use crate::tree::{compute_rewards, find_or_extend_child, populate_node, MCTSTree};
use pyrat::{Direction, GameState};

/// Score assigned to forced-playout outcomes to guarantee selection.
const FORCED_PLAYOUT_SCORE: f32 = 1e20;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Search configuration — immutable, shareable across threads.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Exploration constant (PUCT).
    pub c_puct: f32,
    /// First-play urgency penalty.
    pub fpu_reduction: f32,
    /// Forced playout coefficient. 0 disables forced playouts.
    pub force_k: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            fpu_reduction: 0.2,
            force_k: 2.0,
        }
    }
}

/// A single step on the search path: (node_index, p1_outcome_idx, p2_outcome_idx).
pub type SearchPath = Vec<(NodeIndex, u8, u8)>;

// ---------------------------------------------------------------------------
// backup
// ---------------------------------------------------------------------------

/// Walk leaf→root, updating node values and edge Q along the path.
///
/// `path` entries are ancestors of `leaf_idx`, ordered root-first.
/// The last entry is the leaf's parent; the leaf itself is NOT in the path.
///
/// `g1, g2` are the leaf evaluation (NN value or terminal reward).
/// The leaf gets `update_value(g1, g2)` directly.
/// Each ancestor accumulates `q = edge_reward + child_value`.
pub fn backup(
    arena: &mut NodeArena,
    path: &[(NodeIndex, u8, u8)],
    leaf_idx: NodeIndex,
    g1: f32,
    g2: f32,
) {
    // Visit 1 on the leaf: NN eval or terminal value.
    arena[leaf_idx].update_value(g1, g2);

    debug_assert!(
        path.last().is_none_or(|&(parent_idx, _, _)| arena[leaf_idx].parent() == Some(parent_idx)),
        "backup: leaf's parent should match last path entry"
    );

    // Carry raw values upward — NOT running averages from child nodes.
    // This matches lc0's DoBackupUpdateSingleNode: each ancestor sees
    // q = edge_reward + propagated_value, where propagated_value chains
    // from the leaf evaluation, not from stale Welford averages.
    let mut v1 = g1;
    let mut v2 = g2;
    let mut child_idx = leaf_idx;

    for &(node_idx, a1, a2) in path.iter().rev() {
        let q1 = arena[child_idx].edge_r1() + v1;
        let q2 = arena[child_idx].edge_r2() + v2;

        let node = &mut arena[node_idx];
        node.update_value(q1, q2);
        node.p1.edge_mut(a1 as usize).update(q1);
        node.p2.edge_mut(a2 as usize).update(q2);

        v1 = q1;
        v2 = q2;
        child_idx = node_idx;
    }
}

// ---------------------------------------------------------------------------
// select_actions
// ---------------------------------------------------------------------------

/// Select an action pair (p1_outcome_idx, p2_outcome_idx) via decoupled PUCT.
///
/// Each player independently picks the outcome with the highest PUCT score.
/// Returns raw outcome indices (not 5-action space).
pub fn select_actions(
    node: &Node,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> (u8, u8) {
    let a1 = select_half(&node.p1, node.v1(), node.value_scale(), node.total_visits(), config, is_root, rng);
    let a2 = select_half(&node.p2, node.v2(), node.value_scale(), node.total_visits(), config, is_root, rng);
    (a1, a2)
}

/// PUCT selection for one player's half-node.
fn select_half(
    half: &HalfNode,
    node_value: f32,
    value_scale: f32,
    total_visits: u32,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = half.n_outcomes();
    debug_assert!(n > 0, "select_half: no outcomes");

    if n == 1 {
        return 0;
    }

    // FPU: compute visited prior mass for pessimistic default.
    let mut visited_prior_mass = 0.0f32;
    for i in 0..n {
        if half.edge(i).visits > 0 {
            visited_prior_mass += half.prior(i);
        }
    }

    debug_assert!(value_scale > 0.0, "value_scale must be positive, got {value_scale}");
    let fpu = node_value - config.fpu_reduction * value_scale * visited_prior_mass.sqrt();

    let sqrt_total = (total_visits as f32).sqrt();

    argmax_tiebreak(n, rng, |i| {
        let edge = half.edge(i);
        let prior = half.prior(i);

        let q = if edge.visits > 0 { edge.q } else { fpu };
        let q_norm = q / value_scale;

        let exploration =
            config.c_puct * prior * sqrt_total / (1.0 + edge.visits as f32 + edge.n_in_flight() as f32);
        let mut score = q_norm + exploration;

        // Forced playouts: at root, boost undervisited outcomes.
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * total_visits as f32).sqrt();
            if (edge.visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        score
    })
}

/// Argmax with reservoir-sampling tie-breaking.
fn argmax_tiebreak(n: usize, rng: &mut impl Rng, score_fn: impl Fn(usize) -> f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut tie_count = 0u32;

    for i in 0..n {
        let s = score_fn(i);
        if s > best_score {
            best_score = s;
            best_idx = i as u8;
            tie_count = 1;
        } else if (s - best_score).abs() < 1e-12 {
            tie_count += 1;
            // Reservoir sampling: replace with probability 1/tie_count.
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = i as u8;
            }
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// compute_pruned_visits
// ---------------------------------------------------------------------------

/// Forced-playout pruning: cap visits on low-Q outcomes.
///
/// Returns `[f32; 5]` padded array (only first `n` entries valid).
/// - Best outcome (most visited) keeps all visits.
/// - Outcomes with `q_norm >= puct_star` keep all visits.
/// - Others capped to `n_min = c_puct * prior * sqrt(N) / (puct_star - q_norm) - 1`.
pub fn compute_pruned_visits(
    q_norm: &[f32],
    prior: &[f32],
    visits: &[f32],
    n: usize,
    total_visits: u32,
    c_puct: f32,
) -> [f32; 5] {
    let mut result = [0.0f32; 5];

    if n <= 1 {
        if n == 1 {
            result[0] = visits[0];
        }
        return result;
    }

    // Find best outcome (most visited).
    let mut best_idx = 0;
    let mut best_visits = visits[0];
    for (i, &v) in visits.iter().enumerate().take(n).skip(1) {
        if v > best_visits {
            best_visits = v;
            best_idx = i;
        }
    }

    // PUCT* threshold: the PUCT score of the best outcome.
    let sqrt_total = (total_visits as f32).sqrt();
    let puct_star =
        q_norm[best_idx] + c_puct * prior[best_idx] * sqrt_total / (1.0 + visits[best_idx]);

    for i in 0..n {
        if i == best_idx || q_norm[i] >= puct_star {
            result[i] = visits[i];
        } else {
            let denom = puct_star - q_norm[i];
            if denom <= 0.0 {
                result[i] = visits[i];
            } else {
                let n_min = (c_puct * prior[i] * sqrt_total / denom - 1.0).max(0.0);
                result[i] = visits[i].min(n_min);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// SearchResult — public return type
// ---------------------------------------------------------------------------

/// Result of an MCTS search: policies and values for both players.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Policy in 5-action space, sums to 1, blocked actions = 0.
    pub policy_p1: [f32; 5],
    pub policy_p2: [f32; 5],
    /// Expected remaining cheese for each player.
    pub value_p1: f32,
    pub value_p2: f32,
    /// Root visit count after search.
    pub total_visits: u32,
}

// ---------------------------------------------------------------------------
// DescentOutcome — internal, what happened at the leaf
// ---------------------------------------------------------------------------

/// What happened at the leaf of a single PUCT descent.
enum DescentOutcome {
    /// Leaf needs NN evaluation.
    NeedsEval {
        path: SearchPath,
        leaf_idx: NodeIndex,
        game_state: GameState,
    },
    /// Leaf is terminal (game over).
    Terminal {
        path: SearchPath,
        leaf_idx: NodeIndex,
        /// Whether try_start_score_update was called on the leaf.
        leaf_claimed: bool,
    },
    /// Collision: another descent already claimed this unvisited leaf.
    Collision {
        path: SearchPath,
    },
}


// ---------------------------------------------------------------------------
// run_search — public API
// ---------------------------------------------------------------------------

/// Run MCTS search: N simulations with within-tree batching.
///
/// Returns policies (visit-proportional with forced playout pruning) and
/// values for both players.
pub fn run_search(
    tree: &mut MCTSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
) -> SearchResult {
    let mut remaining = n_sims;
    while remaining > 0 {
        let actual = remaining.min(batch_size);
        simulate_batch(tree, game, backend, config, actual, rng);
        remaining -= actual;
    }

    let root = tree.root();
    extract_result(tree.arena(), root, config, rng)
}

// ---------------------------------------------------------------------------
// simulate_batch — gather/eval/backup cycle
// ---------------------------------------------------------------------------

fn simulate_batch(
    tree: &mut MCTSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
) {
    let root = tree.root();

    // Gather phase: descend batch_size times.
    let mut outcomes = Vec::with_capacity(batch_size as usize);
    for _ in 0..batch_size {
        outcomes.push(descend(tree.arena_mut(), root, game, config, rng));
    }

    // Eval phase: collect game states from NeedsEval, batch evaluate.
    let needs_eval_refs: Vec<&GameState> = outcomes
        .iter()
        .filter_map(|o| match o {
            DescentOutcome::NeedsEval { game_state, .. } => Some(game_state),
            _ => None,
        })
        .collect();

    let eval_results = if needs_eval_refs.is_empty() {
        Vec::new()
    } else {
        backend.evaluate_batch(&needs_eval_refs)
    };

    // Populate + backup + cleanup phase.
    let mut eval_idx = 0;
    for outcome in outcomes {
        match outcome {
            DescentOutcome::NeedsEval {
                path,
                leaf_idx,
                ..
            } => {
                let eval = &eval_results[eval_idx];
                eval_idx += 1;

                populate_node(tree.arena_mut(), leaf_idx, Some(eval));
                backup(tree.arena_mut(), &path, leaf_idx, eval.value_p1, eval.value_p2);
                cleanup_descent(tree.arena_mut(), &path, Some(leaf_idx));
            }
            DescentOutcome::Terminal {
                path,
                leaf_idx,
                leaf_claimed,
            } => {
                // Mark as terminal if first visit.
                if tree.arena()[leaf_idx].total_visits() == 0 {
                    populate_node(tree.arena_mut(), leaf_idx, None);
                }
                backup(tree.arena_mut(), &path, leaf_idx, 0.0, 0.0);
                let claimed = if leaf_claimed { Some(leaf_idx) } else { None };
                cleanup_descent(tree.arena_mut(), &path, claimed);
            }
            DescentOutcome::Collision { path } => {
                cleanup_descent(tree.arena_mut(), &path, None);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// descend — single PUCT descent with virtual loss
// ---------------------------------------------------------------------------

fn descend(
    arena: &mut NodeArena,
    root: NodeIndex,
    game: &GameState,
    config: &SearchConfig,
    rng: &mut impl Rng,
) -> DescentOutcome {
    let mut current = root;
    let mut path: SearchPath = Vec::new();
    let mut game = game.clone();

    loop {
        let node = &arena[current];

        // Unvisited leaf — try to claim it.
        if node.total_visits() == 0 && !node.is_terminal() {
            if !arena[current].try_start_score_update() {
                return DescentOutcome::Collision { path };
            }
            // Leaf claimed. Game is at leaf position.
            if game.check_game_over() {
                return DescentOutcome::Terminal {
                    path,
                    leaf_idx: current,
                    leaf_claimed: true,
                };
            }
            return DescentOutcome::NeedsEval {
                path,
                leaf_idx: current,
                game_state: game,
            };
        }

        // Revisited terminal — no claim needed.
        if node.is_terminal() {
            return DescentOutcome::Terminal {
                path,
                leaf_idx: current,
                leaf_claimed: false,
            };
        }

        // Interior node: select actions via PUCT.
        let is_root = current == root;
        let (idx1, idx2) = select_actions(&arena[current], config, is_root, rng);

        // Add virtual loss on selected edges.
        arena[current].p1.edge_mut(idx1 as usize).add_virtual_loss();
        arena[current].p2.edge_mut(idx2 as usize).add_virtual_loss();

        // Record path step.
        path.push((current, idx1, idx2));

        // Convert outcome indices to actions.
        let a1 = arena[current].p1.outcome_action(idx1 as usize);
        let a2 = arena[current].p2.outcome_action(idx2 as usize);

        // Advance game state.
        let scores_before = (game.player1_score(), game.player2_score());
        let d1 = Direction::try_from(a1).expect("valid direction");
        let d2 = Direction::try_from(a2).expect("valid direction");
        let _undo = game.make_move(d1, d2);
        let (r1, r2) = compute_rewards(&game, scores_before);

        // Find or create child.
        let (child_idx, _is_new) =
            find_or_extend_child(arena, current, idx1, idx2, &game, r1, r2);

        current = child_idx;
    }
}

// ---------------------------------------------------------------------------
// cleanup_descent — revert virtual losses
// ---------------------------------------------------------------------------

fn cleanup_descent(
    arena: &mut NodeArena,
    path: &[(NodeIndex, u8, u8)],
    leaf_claimed: Option<NodeIndex>,
) {
    // Revert edge virtual losses along the path.
    for &(node_idx, a1, a2) in path {
        arena[node_idx]
            .p1
            .edge_mut(a1 as usize)
            .revert_virtual_loss();
        arena[node_idx]
            .p2
            .edge_mut(a2 as usize)
            .revert_virtual_loss();
    }

    // Cancel leaf claim if applicable.
    if let Some(leaf_idx) = leaf_claimed {
        arena[leaf_idx].cancel_score_update();
    }
}

// ---------------------------------------------------------------------------
// extract_result — policies and values from root
// ---------------------------------------------------------------------------

fn extract_result(
    arena: &NodeArena,
    root: NodeIndex,
    config: &SearchConfig,
    _rng: &mut impl Rng,
) -> SearchResult {
    let node = &arena[root];
    let total_visits = node.total_visits();

    let (policy_p1, value_p1) =
        extract_half(&node.p1, node.v1(), node.value_scale(), total_visits, config);
    let (policy_p2, value_p2) =
        extract_half(&node.p2, node.v2(), node.value_scale(), total_visits, config);

    SearchResult {
        policy_p1,
        policy_p2,
        value_p1,
        value_p2,
        total_visits,
    }
}

/// Extract policy and value for one player from root.
fn extract_half(
    half: &HalfNode,
    node_value: f32,
    value_scale: f32,
    total_visits: u32,
    config: &SearchConfig,
) -> ([f32; 5], f32) {
    let n = half.n_outcomes();

    if n == 0 {
        return ([0.0; 5], node_value);
    }

    // Compute FPU for unvisited outcomes.
    let mut visited_prior_mass = 0.0f32;
    for i in 0..n {
        if half.edge(i).visits > 0 {
            visited_prior_mass += half.prior(i);
        }
    }
    let fpu = node_value - config.fpu_reduction * value_scale * visited_prior_mass.sqrt();

    // Read Q and visits per outcome.
    let mut q = [0.0f32; 5];
    let mut raw_visits = [0.0f32; 5];
    let mut prior = [0.0f32; 5];
    let mut q_norm = [0.0f32; 5];

    for i in 0..n {
        let edge = half.edge(i);
        q[i] = if edge.visits > 0 { edge.q } else { fpu };
        raw_visits[i] = edge.visits as f32;
        prior[i] = half.prior(i);
        q_norm[i] = q[i] / value_scale;
    }

    // Compute pruned visits for policy.
    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, total_visits, config.c_puct);

    // Expand pruned visits to 5-action space.
    let mut policy = [0.0f32; 5];
    for (i, &pv) in pruned.iter().enumerate().take(n) {
        let action = half.outcome_action(i) as usize;
        policy[action] = pv;
    }

    // Normalize policy to sum=1.
    let policy_sum: f32 = policy.iter().sum();
    if policy_sum > 0.0 {
        for p in &mut policy {
            *p /= policy_sum;
        }
    } else {
        // Fallback to expanded prior.
        policy = half.expand_prior();
    }

    // Value = dot(q, raw_visits) / sum(raw_visits).
    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|i| q[i] * raw_visits[i]).sum();
        dot / visit_sum
    } else {
        node_value
    };

    (policy, value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ConstantValueBackend, SmartUniformBackend};
    use crate::node::{Node, NodeArena};
    use crate::test_util;
    use crate::tree::MCTSTree;
    use pyrat::Coordinates;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn default_config() -> SearchConfig {
        SearchConfig::default()
    }

    /// Build a simple open half-node with uniform priors.
    fn open_half(prior: [f32; 5]) -> HalfNode {
        HalfNode::new(prior, [0, 1, 2, 3, 4])
    }

    /// Allocate a node with open topology (5 outcomes per player) and given priors.
    fn alloc_open_node(arena: &mut NodeArena, p1_prior: [f32; 5], p2_prior: [f32; 5]) -> NodeIndex {
        let p1 = open_half(p1_prior);
        let p2 = open_half(p2_prior);
        arena.alloc(Node::new(p1, p2))
    }

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // ---- backup ----

    #[test]
    fn backup_single_level() {
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        // Wire child under root at outcome (0, 1).
        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 1));
        arena[child].set_edge_rewards(1.0, 0.5);
        arena[child].set_value_scale(5.0);
        arena[root].set_value_scale(5.0);

        let path = vec![(root, 0, 1)];
        backup(&mut arena, &path, child, 3.0, 2.0);

        // Leaf: v=(3.0, 2.0), visits=1
        assert_eq!(arena[child].total_visits(), 1);
        assert!((arena[child].v1() - 3.0).abs() < 1e-6);
        assert!((arena[child].v2() - 2.0).abs() < 1e-6);

        // Root: q1 = edge_r1 + child_v1 = 1.0 + 3.0 = 4.0
        //       q2 = edge_r2 + child_v2 = 0.5 + 2.0 = 2.5
        assert_eq!(arena[root].total_visits(), 1);
        assert!((arena[root].v1() - 4.0).abs() < 1e-6);
        assert!((arena[root].v2() - 2.5).abs() < 1e-6);

        // Edge visits: outcome 0 for P1, outcome 1 for P2
        assert_eq!(arena[root].p1.edge(0).visits, 1);
        assert!((arena[root].p1.edge(0).q - 4.0).abs() < 1e-6);
        assert_eq!(arena[root].p2.edge(1).visits, 1);
        assert!((arena[root].p2.edge(1).q - 2.5).abs() < 1e-6);
    }

    #[test]
    fn backup_two_level_q_chain() {
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let mid = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let leaf = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(mid));
        arena[mid].set_parent(Some(root), (0, 0));
        arena[mid].set_edge_rewards(1.0, 0.5);

        arena[mid].set_first_child(Some(leaf));
        arena[leaf].set_parent(Some(mid), (1, 2));
        arena[leaf].set_edge_rewards(0.5, 1.0);

        for idx in [root, mid, leaf] {
            arena[idx].set_value_scale(5.0);
        }

        let path = vec![(root, 0, 0), (mid, 1, 2)];
        backup(&mut arena, &path, leaf, 2.0, 3.0);

        // Leaf: v=(2.0, 3.0)
        assert!((arena[leaf].v1() - 2.0).abs() < 1e-6);
        assert!((arena[leaf].v2() - 3.0).abs() < 1e-6);

        // Mid: q1 = r_leaf(0.5) + v_leaf(2.0) = 2.5
        //      q2 = r_leaf(1.0) + v_leaf(3.0) = 4.0
        assert!((arena[mid].v1() - 2.5).abs() < 1e-6);
        assert!((arena[mid].v2() - 4.0).abs() < 1e-6);

        // Root: q1 = r_mid(1.0) + v_mid(2.5) = 3.5
        //       q2 = r_mid(0.5) + v_mid(4.0) = 4.5
        assert!((arena[root].v1() - 3.5).abs() < 1e-6);
        assert!((arena[root].v2() - 4.5).abs() < 1e-6);
    }

    #[test]
    fn backup_multiple_same_edge() {
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(0.0, 0.0);

        for idx in [root, child] {
            arena[idx].set_value_scale(5.0);
        }

        let path = vec![(root, 0, 0)];

        // 3 backups with different leaf values.
        backup(&mut arena, &path, child, 2.0, 1.0);
        backup(&mut arena, &path, child, 4.0, 3.0);
        backup(&mut arena, &path, child, 6.0, 5.0);

        // Child: mean of [2,4,6]=4.0 and [1,3,5]=3.0
        assert_eq!(arena[child].total_visits(), 3);
        assert!((arena[child].v1() - 4.0).abs() < 1e-5);
        assert!((arena[child].v2() - 3.0).abs() < 1e-5);

        // Root edge: q = edge_r(0) + raw_v propagated from leaf.
        // Each backup propagates the raw leaf value (not the running average).
        // Backup 1: v1=2.0 → q1=2.0
        // Backup 2: v1=4.0 → q1=4.0
        // Backup 3: v1=6.0 → q1=6.0
        // Edge Q = mean(2.0, 4.0, 6.0) = 4.0
        assert_eq!(arena[root].p1.edge(0).visits, 3);
        assert!((arena[root].p1.edge(0).q - 4.0).abs() < 1e-5);
        assert!((arena[root].p2.edge(0).q - 3.0).abs() < 1e-5);
    }

    #[test]
    fn backup_multiple_different_edges() {
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child_a = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child_b = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child_a));
        arena[child_a].set_parent(Some(root), (0, 0));
        arena[child_a].set_next_sibling(Some(child_b));
        arena[child_b].set_parent(Some(root), (1, 1));

        arena[child_a].set_edge_rewards(0.0, 0.0);
        arena[child_b].set_edge_rewards(0.0, 0.0);

        for idx in [root, child_a, child_b] {
            arena[idx].set_value_scale(5.0);
        }

        // Backup through child_a: leaf_v = (5.0, 1.0)
        backup(&mut arena, &[(root, 0, 0)], child_a, 5.0, 1.0);
        // Backup through child_b: leaf_v = (1.0, 5.0)
        backup(&mut arena, &[(root, 1, 1)], child_b, 1.0, 5.0);

        // Each edge visited once with independent Q.
        assert_eq!(arena[root].p1.edge(0).visits, 1);
        assert!((arena[root].p1.edge(0).q - 5.0).abs() < 1e-6);
        assert_eq!(arena[root].p1.edge(1).visits, 1);
        assert!((arena[root].p1.edge(1).q - 1.0).abs() < 1e-6);

        assert_eq!(arena[root].p2.edge(0).visits, 1);
        assert!((arena[root].p2.edge(0).q - 1.0).abs() < 1e-6);
        assert_eq!(arena[root].p2.edge(1).visits, 1);
        assert!((arena[root].p2.edge(1).q - 5.0).abs() < 1e-6);
    }

    #[test]
    fn backup_terminal_leaf() {
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (2, 3));
        arena[child].set_edge_rewards(0.0, 0.0);
        arena[child].set_terminal();

        for idx in [root, child] {
            arena[idx].set_value_scale(5.0);
        }

        // Terminal: g=(0, 0)
        backup(&mut arena, &[(root, 2, 3)], child, 0.0, 0.0);

        assert_eq!(arena[child].total_visits(), 1);
        assert!((arena[child].v1() - 0.0).abs() < 1e-6);
        assert_eq!(arena[root].total_visits(), 1);
        assert!((arena[root].v1() - 0.0).abs() < 1e-6);
        assert_eq!(arena[root].p1.edge(2).visits, 1);
    }

    #[test]
    fn backup_edge_visit_sum() {
        // After N backups through various edges, sum(edge.visits) == total_visits - 1
        // (the -1 is because update_value on the node counts the NN eval too,
        // but actually here each backup calls update_value once, so
        // total_visits == number of backups, and sum(edge.visits) == number of backups).
        // Let's verify: sum of p1 edge visits == total_visits for the root.
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        let mut children = Vec::new();
        for i in 0..3u8 {
            let c = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
            arena[c].set_edge_rewards(0.0, 0.0);
            arena[c].set_parent(Some(root), (i, 0));
            arena[c].set_value_scale(5.0);
            children.push(c);
        }
        // Wire linked list.
        arena[root].set_first_child(Some(children[2]));
        arena[children[2]].set_next_sibling(Some(children[1]));
        arena[children[1]].set_next_sibling(Some(children[0]));
        arena[root].set_value_scale(5.0);

        // 5 backups: 2 through edge 0, 2 through edge 1, 1 through edge 2
        backup(&mut arena, &[(root, 0, 0)], children[0], 1.0, 1.0);
        backup(&mut arena, &[(root, 0, 0)], children[0], 2.0, 2.0);
        backup(&mut arena, &[(root, 1, 0)], children[1], 3.0, 3.0);
        backup(&mut arena, &[(root, 1, 0)], children[1], 4.0, 4.0);
        backup(&mut arena, &[(root, 2, 0)], children[2], 5.0, 5.0);

        let edge_sum: u32 = (0..5).map(|i| arena[root].p1.edge(i).visits).sum();
        assert_eq!(edge_sum, arena[root].total_visits());
    }

    #[test]
    fn backup_value_mixing() {
        // Raw leaf values propagated upward: q = edge_r + raw_v (not running avg).
        // g=[2,4,6], edge_r=1.0 → Q values = [3, 5, 7] → root.v1 = mean(3,5,7) = 5.0
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(1.0, 0.0);

        for idx in [root, child] {
            arena[idx].set_value_scale(5.0);
        }

        backup(&mut arena, &[(root, 0, 0)], child, 2.0, 0.0);
        backup(&mut arena, &[(root, 0, 0)], child, 4.0, 0.0);
        backup(&mut arena, &[(root, 0, 0)], child, 6.0, 0.0);

        // Root Q: edge_r(1.0) + raw_v for each backup = [3.0, 5.0, 7.0]
        // mean = 5.0
        assert!(
            (arena[root].v1() - 5.0).abs() < 1e-5,
            "root v1={} expected=5.0",
            arena[root].v1(),
        );
    }

    #[test]
    fn backup_asymmetric_rewards() {
        // P1 collects cheese (r1=1.0), P2 doesn't (r2=0.0).
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(1.0, 0.0);

        for idx in [root, child] {
            arena[idx].set_value_scale(5.0);
        }

        backup(&mut arena, &[(root, 0, 0)], child, 2.0, 3.0);

        // Root q1 = 1.0 + 2.0 = 3.0, q2 = 0.0 + 3.0 = 3.0
        assert!((arena[root].v1() - 3.0).abs() < 1e-6);
        assert!((arena[root].v2() - 3.0).abs() < 1e-6);

        // Edge P1: q=3.0. Edge P2: q=3.0.
        assert!((arena[root].p1.edge(0).q - 3.0).abs() < 1e-6);
        assert!((arena[root].p2.edge(0).q - 3.0).abs() < 1e-6);
    }

    // ---- PUCT selection ----

    #[test]
    fn puct_monotonic_q() {
        // Higher Q → selected.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Give the node a visit so total_visits > 0.
        arena[node_idx].update_value(0.0, 0.0);

        // Set up edges: outcome 2 has much higher Q.
        for i in 0..5 {
            arena[node_idx].p1.edge_mut(i).update(1.0);
        }
        // Give outcome 2 extra high-Q updates.
        for _ in 0..10 {
            arena[node_idx].p1.edge_mut(2).update(10.0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);
        assert_eq!(a1, 2, "Should select highest-Q outcome");
    }

    #[test]
    fn puct_monotonic_prior() {
        // Higher prior → selected when all else equal.
        let mut arena = NodeArena::new();
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);
        arena[node_idx].update_value(0.0, 0.0);

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);
        assert_eq!(a1, 2, "Should select highest-prior outcome");
    }

    #[test]
    fn puct_exploration_positive() {
        // PUCT score > Q_norm when prior > 0 and there are visits.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Give multiple visits so sqrt(total_visits) > 0.
        for _ in 0..10 {
            arena[node_idx].update_value(2.0, 2.0);
        }

        // Visit outcome 0 a few times.
        for _ in 0..5 {
            arena[node_idx].p1.edge_mut(0).update(2.0);
        }

        let config = default_config();
        let edge = arena[node_idx].p1.edge(0);
        let vs = arena[node_idx].value_scale();
        let q_norm = edge.q / vs;
        let total = arena[node_idx].total_visits();
        let exploration = config.c_puct * 0.2 * (total as f32).sqrt() / (1.0 + edge.visits as f32);

        assert!(exploration > 0.0, "Exploration bonus should be positive");
        assert!(q_norm + exploration > q_norm, "PUCT > Q_norm");
    }

    #[test]
    fn puct_unvisited_selected() {
        // With enough visits on other outcomes, an unvisited one should be pulled by exploration.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Give many visits — FPU becomes relevant.
        for _ in 0..100 {
            arena[node_idx].update_value(2.0, 2.0);
        }

        // Visit outcomes 0-3 heavily but leave outcome 4 unvisited.
        for i in 0..4 {
            for _ in 0..25 {
                arena[node_idx].p1.edge_mut(i).update(2.0);
            }
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);
        // Outcome 4 has 0 visits → huge exploration bonus → selected.
        assert_eq!(a1, 4, "Unvisited outcome should be selected");
    }

    #[test]
    fn puct_fpu_pessimism() {
        // Unvisited outcome Q = v - fpu_reduction * value_scale * sqrt(visited_mass).
        // More visited mass → more pessimistic FPU → lower score for unvisited.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Set node value.
        arena[node_idx].update_value(3.0, 3.0);

        let config = default_config();
        let v = arena[node_idx].v1();
        let vs = arena[node_idx].value_scale();

        // Visit outcome 0 — now visited_prior_mass = 0.2.
        arena[node_idx].p1.edge_mut(0).update(3.0);

        let fpu_one = v - config.fpu_reduction * vs * (0.2f32).sqrt();

        // Visit outcomes 0 and 1 — visited_prior_mass = 0.4.
        arena[node_idx].p1.edge_mut(1).update(3.0);

        let fpu_two = v - config.fpu_reduction * vs * (0.4f32).sqrt();

        assert!(
            fpu_two < fpu_one,
            "More visited mass should give more pessimistic FPU: {} < {}",
            fpu_two,
            fpu_one
        );
    }

    #[test]
    fn puct_fpu_no_visits() {
        // total_visits=1 (just NN eval), no edges visited → visited_mass=0 → fpu=v.
        // All outcomes have same Q (=v), selection should be by prior.
        let mut arena = NodeArena::new();
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);
        arena[node_idx].update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);
        assert_eq!(a1, 2, "With no edge visits, should select by prior");
    }

    #[test]
    fn puct_forced_fires() {
        // At root, an undervisited outcome with prior > 0 gets force-boosted.
        let mut arena = NodeArena::new();
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Many visits, heavily on outcomes 0 and 1.
        for _ in 0..100 {
            arena[node_idx].update_value(2.0, 2.0);
        }
        for _ in 0..45 {
            arena[node_idx].p1.edge_mut(0).update(2.0);
        }
        for _ in 0..45 {
            arena[node_idx].p1.edge_mut(1).update(2.0);
        }
        // Outcome 2 has prior 0.1, visited once.
        arena[node_idx].p1.edge_mut(2).update(2.0);
        // Outcomes 3, 4 have prior 0.05, visited once each.
        arena[node_idx].p1.edge_mut(3).update(2.0);
        arena[node_idx].p1.edge_mut(4).update(2.0);

        // Threshold for outcome 2: sqrt(2.0 * 0.1 * 100) = sqrt(20) ≈ 4.47
        // Visits = 1 < 4.47 → forced
        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, true, &mut r);

        // One of the undervisited outcomes should be boosted.
        // Outcome 2 has highest prior among undervisited → highest threshold.
        // But all undervisited get 1e20, so tie-break is by RNG. Just check it's undervisited.
        assert!(
            arena[node_idx].p1.edge(a1 as usize).visits < 5,
            "Forced playout should select an undervisited outcome, got outcome {} with {} visits",
            a1,
            arena[node_idx].p1.edge(a1 as usize).visits
        );
    }

    #[test]
    fn puct_forced_not_at_nonroot() {
        // Same node shape as puct_forced_fires but is_root=false → no forced boost.
        // Give undervisited outcomes enough visits that exploration alone won't
        // beat the high-Q outcomes, so the only way they'd win is via forced playouts.
        let mut arena = NodeArena::new();
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        for _ in 0..1000 {
            arena[node_idx].update_value(2.0, 2.0);
        }
        for _ in 0..450 {
            arena[node_idx].p1.edge_mut(0).update(8.0);
        }
        for _ in 0..450 {
            arena[node_idx].p1.edge_mut(1).update(7.0);
        }
        for _ in 0..40 {
            arena[node_idx].p1.edge_mut(2).update(1.0);
        }
        for _ in 0..30 {
            arena[node_idx].p1.edge_mut(3).update(1.0);
        }
        for _ in 0..30 {
            arena[node_idx].p1.edge_mut(4).update(1.0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);

        // Without forced playouts, the high-Q outcome should win.
        assert!(
            a1 == 0 || a1 == 1,
            "Without forced playouts, should select a high-Q outcome, got {}",
            a1
        );
    }

    #[test]
    fn puct_value_scale() {
        // value_scale controls how much Q matters vs exploration.
        // Outcome 0 has slightly higher Q than outcome 1.
        // With small value_scale: Q difference is large relative to exploration → exploitation wins.
        // With large value_scale: Q difference shrinks → prior (equal here) + exploration dominate.
        let mut arena = NodeArena::new();
        let p1_prior = [0.1, 0.5, 0.1, 0.1, 0.2];
        let idx_small = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        let idx_large = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);

        arena[idx_small].set_value_scale(1.0);
        arena[idx_large].set_value_scale(100.0);

        // Same raw values and edge pattern.
        for idx in [idx_small, idx_large] {
            arena[idx].update_value(5.0, 5.0);
            // Outcome 0: high Q, low prior (0.1).
            arena[idx].p1.edge_mut(0).update(8.0);
            // Outcome 1: low Q, high prior (0.5).
            arena[idx].p1.edge_mut(1).update(6.0);
        }

        let config = default_config();
        let mut r = rng();

        // Small scale: (8-6)/1 = 2 gap in q_norm. Exploitation wins → outcome 0.
        let (a_small, _) = select_actions(&arena[idx_small], &config, false, &mut r);
        assert_eq!(a_small, 0, "Small value_scale: exploitation should select high-Q outcome");

        // Large scale: (8-6)/100 = 0.02 gap. Prior dominates → outcome 1 (prior 0.5).
        let mut r2 = rng();
        let (a_large, _) = select_actions(&arena[idx_large], &config, false, &mut r2);
        assert_eq!(a_large, 1, "Large value_scale: exploration should select high-prior outcome");
    }

    #[test]
    fn puct_decoupled() {
        // P2's state shouldn't affect P1's selection.
        let mut arena = NodeArena::new();
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];

        // Two nodes: same P1 state, different P2 priors.
        let idx1 = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        let idx2 = alloc_open_node(&mut arena, p1_prior, [0.05, 0.8, 0.05, 0.05, 0.05]);

        arena[idx1].set_value_scale(5.0);
        arena[idx2].set_value_scale(5.0);
        arena[idx1].update_value(2.0, 2.0);
        arena[idx2].update_value(2.0, 2.0);

        let config = default_config();
        let mut r1 = rng();
        let mut r2 = rng();
        let (a1_first, _) = select_actions(&arena[idx1], &config, false, &mut r1);
        let (a1_second, _) = select_actions(&arena[idx2], &config, false, &mut r2);

        assert_eq!(
            a1_first, a1_second,
            "P1 selection should be independent of P2 state"
        );
    }

    // ---- pruning ----

    #[test]
    fn prune_best_unchanged() {
        let q_norm = [0.5, 0.3, 0.8, 0.2, 0.1];
        let prior = [0.2; 5];
        let visits = [10.0, 5.0, 20.0, 3.0, 2.0];

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 5, 40, 1.5);
        // Best (most visited) is outcome 2 with 20 visits.
        assert!((result[2] - 20.0).abs() < 1e-6, "Best should keep all visits");
    }

    #[test]
    fn prune_high_q_unchanged() {
        // Outcome with Q >= PUCT* should keep all visits.
        // Best is outcome 2 (most visited). We set outcome 3's Q high enough
        // to exceed PUCT* so it's preserved unconditionally.
        let q_norm = [0.5, 0.3, 0.8, 0.95, 0.1];
        let prior = [0.2; 5];
        let visits = [10.0, 5.0, 20.0, 18.0, 2.0];

        let puct_star = 0.8 + 1.5 * 0.2 * (55.0f32).sqrt() / 21.0;
        assert!(
            q_norm[3] >= puct_star,
            "precondition: outcome 3 Q ({}) should >= PUCT* ({})",
            q_norm[3],
            puct_star
        );

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 5, 55, 1.5);

        assert!(
            (result[3] - visits[3]).abs() < 1e-6,
            "High-Q outcome should keep all visits"
        );
    }

    #[test]
    fn prune_low_q_capped() {
        // Low-Q outcome should have visits reduced.
        let q_norm = [0.8, 0.1, 0.0, 0.0, 0.0];
        let prior = [0.2; 5];
        let visits = [50.0, 20.0, 10.0, 10.0, 10.0];
        let total = 100u32;

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 5, total, 1.5);

        // Outcome 0 is best (50 visits).
        assert!((result[0] - 50.0).abs() < 1e-6);

        // Outcomes 2-4 have low Q, should be capped.
        for i in 2..5 {
            assert!(
                result[i] <= visits[i],
                "Outcome {} should be capped: {} <= {}",
                i,
                result[i],
                visits[i]
            );
        }
    }

    #[test]
    fn prune_leq_raw() {
        // pruned[i] <= visits[i] always.
        let q_norm = [0.5, 0.3, 0.8, 0.2, 0.1];
        let prior = [0.2; 5];
        let visits = [10.0, 5.0, 20.0, 3.0, 2.0];

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 5, 40, 1.5);

        for i in 0..5 {
            assert!(
                result[i] <= visits[i] + 1e-6,
                "Pruned visits should not exceed raw: outcome {} has {} > {}",
                i,
                result[i],
                visits[i]
            );
        }
    }

    #[test]
    fn prune_no_negative() {
        let q_norm = [0.9, 0.0, 0.0, 0.0, 0.0];
        let prior = [0.2; 5];
        let visits = [50.0, 1.0, 1.0, 1.0, 1.0];

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 5, 54, 1.5);

        for i in 0..5 {
            assert!(result[i] >= 0.0, "Outcome {} has negative visits: {}", i, result[i]);
        }
    }

    #[test]
    fn prune_single_passthrough() {
        let q_norm = [0.5];
        let prior = [1.0];
        let visits = [42.0];

        let result = compute_pruned_visits(&q_norm, &prior, &visits, 1, 42, 1.5);
        assert!((result[0] - 42.0).abs() < 1e-6, "Single outcome should pass through");
    }

    // ---- backup: edge cases ----

    #[test]
    fn backup_empty_path() {
        // Root-as-leaf: empty path means the root IS the leaf (e.g. terminal root).
        // Only the leaf gets update_value; no edge updates happen.
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[root].set_value_scale(5.0);

        backup(&mut arena, &[], root, 3.0, 2.0);

        assert_eq!(arena[root].total_visits(), 1);
        assert!((arena[root].v1() - 3.0).abs() < 1e-6);
        assert!((arena[root].v2() - 2.0).abs() < 1e-6);

        // No edges should have been touched.
        for i in 0..5 {
            assert_eq!(arena[root].p1.edge(i).visits, 0);
            assert_eq!(arena[root].p2.edge(i).visits, 0);
        }
    }

    #[test]
    fn backup_edge_visit_sum_with_nn_eval() {
        // When root gets an NN eval (update_value before any backup),
        // sum(p1.edge.visits) == root.total_visits - 1.
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[root].set_value_scale(5.0);

        // Simulate NN eval on root (visit 1, no edge updates).
        arena[root].update_value(2.0, 2.0);

        // Wire two children.
        let child_a = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child_b = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[root].set_first_child(Some(child_a));
        arena[child_a].set_parent(Some(root), (0, 0));
        arena[child_a].set_next_sibling(Some(child_b));
        arena[child_b].set_parent(Some(root), (1, 1));
        for c in [child_a, child_b] {
            arena[c].set_edge_rewards(0.0, 0.0);
            arena[c].set_value_scale(5.0);
        }

        // 5 backups through the two edges.
        for _ in 0..3 {
            backup(&mut arena, &[(root, 0, 0)], child_a, 1.0, 1.0);
        }
        for _ in 0..2 {
            backup(&mut arena, &[(root, 1, 1)], child_b, 1.0, 1.0);
        }

        let edge_sum: u32 = (0..5).map(|i| arena[root].p1.edge(i).visits).sum();
        // total_visits = 1 (NN eval) + 5 (backups) = 6, edge_sum = 5.
        assert_eq!(arena[root].total_visits(), 6);
        assert_eq!(edge_sum, arena[root].total_visits() - 1);
    }

    // ---- backup: value propagation correctness ----

    #[test]
    fn backup_multi_level_multi_backup() {
        // The killer regression test: root→mid→leaf with edge rewards.
        // Two backups through the same path. The bug reads child.v1() (running
        // average) instead of propagating the raw leaf value upward, causing
        // stale averages to bleed into parent values.
        //
        // edge_r_mid=1.0, edge_r_leaf=0.5
        // Backup 1: g=10 → leaf=10, mid=10.5, root=11.5
        // Backup 2: g=6  → leaf=8, mid=8.5 (bug: 9.5), root=9.5 (bug: 11.0)
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let mid = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let leaf = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(mid));
        arena[mid].set_parent(Some(root), (0, 0));
        arena[mid].set_edge_rewards(1.0, 1.0);

        arena[mid].set_first_child(Some(leaf));
        arena[leaf].set_parent(Some(mid), (0, 0));
        arena[leaf].set_edge_rewards(0.5, 0.5);

        for idx in [root, mid, leaf] {
            arena[idx].set_value_scale(15.0);
        }

        let path = vec![(root, 0, 0), (mid, 0, 0)];

        // Backup 1: g=10
        backup(&mut arena, &path, leaf, 10.0, 10.0);
        assert!((arena[leaf].v1() - 10.0).abs() < 1e-5);
        // mid: q1 = edge_r_leaf(0.5) + raw_v(10.0) = 10.5
        assert!((arena[mid].v1() - 10.5).abs() < 1e-5);
        // root: q1 = edge_r_mid(1.0) + propagated_v(10.5) = 11.5
        assert!((arena[root].v1() - 11.5).abs() < 1e-5);

        // Backup 2: g=6
        backup(&mut arena, &path, leaf, 6.0, 6.0);
        // leaf: mean(10, 6) = 8.0
        assert!((arena[leaf].v1() - 8.0).abs() < 1e-5);
        // mid: q1 = 0.5 + 6.0 = 6.5. mean(10.5, 6.5) = 8.5
        assert!(
            (arena[mid].v1() - 8.5).abs() < 1e-5,
            "mid.v1={} expected=8.5 (bug would give 9.5)",
            arena[mid].v1()
        );
        // root: q1 = 1.0 + 6.5 = 7.5. mean(11.5, 7.5) = 9.5
        assert!(
            (arena[root].v1() - 9.5).abs() < 1e-5,
            "root.v1={} expected=9.5 (bug would give 11.0)",
            arena[root].v1()
        );
    }

    #[test]
    fn backup_same_edge_raw_propagation() {
        // Edge Q = mean of raw Q values, not stale running averages.
        // 3 backups g=[10,4,7], edge_r=2.0 → edge Q = mean(12, 6, 9) = 9.0
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(2.0, 2.0);

        for idx in [root, child] {
            arena[idx].set_value_scale(15.0);
        }

        let path = vec![(root, 0, 0)];
        backup(&mut arena, &path, child, 10.0, 10.0);
        backup(&mut arena, &path, child, 4.0, 4.0);
        backup(&mut arena, &path, child, 7.0, 7.0);

        // Edge Q = mean(2+10, 2+4, 2+7) = mean(12, 6, 9) = 9.0
        assert!(
            (arena[root].p1.edge(0).q - 9.0).abs() < 1e-5,
            "edge Q={} expected=9.0",
            arena[root].p1.edge(0).q
        );
    }

    #[test]
    fn backup_three_level_reward_chain() {
        // Rewards stack across depth: root→A→B→leaf
        // edge_r = [1.0, 0.5, 0.25]
        //
        // Backup 1: g=4.0
        //   leaf=4.0, B=4.25, A=4.75, root=5.75
        // Backup 2: g=2.0
        //   leaf=3.0, B=mean(4.25, 2.25)=3.25, A=mean(4.75, 2.75)=3.75, root=mean(5.75, 3.75)=4.75
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let a = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let b = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let leaf = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(a));
        arena[a].set_parent(Some(root), (0, 0));
        arena[a].set_edge_rewards(1.0, 1.0);

        arena[a].set_first_child(Some(b));
        arena[b].set_parent(Some(a), (0, 0));
        arena[b].set_edge_rewards(0.5, 0.5);

        arena[b].set_first_child(Some(leaf));
        arena[leaf].set_parent(Some(b), (0, 0));
        arena[leaf].set_edge_rewards(0.25, 0.25);

        for idx in [root, a, b, leaf] {
            arena[idx].set_value_scale(10.0);
        }

        let path = vec![(root, 0, 0), (a, 0, 0), (b, 0, 0)];

        // Backup 1: g=4.0
        backup(&mut arena, &path, leaf, 4.0, 4.0);
        assert!((arena[leaf].v1() - 4.0).abs() < 1e-5);
        assert!((arena[b].v1() - 4.25).abs() < 1e-5);
        assert!((arena[a].v1() - 4.75).abs() < 1e-5);
        assert!((arena[root].v1() - 5.75).abs() < 1e-5);

        // Backup 2: g=2.0
        backup(&mut arena, &path, leaf, 2.0, 2.0);
        assert!((arena[leaf].v1() - 3.0).abs() < 1e-5);
        // B: mean(4.25, 0.25+2.0) = mean(4.25, 2.25) = 3.25
        assert!(
            (arena[b].v1() - 3.25).abs() < 1e-5,
            "B.v1={} expected=3.25",
            arena[b].v1()
        );
        // A: mean(4.75, 0.5+2.25) = mean(4.75, 2.75) = 3.75
        assert!(
            (arena[a].v1() - 3.75).abs() < 1e-5,
            "A.v1={} expected=3.75",
            arena[a].v1()
        );
        // root: mean(5.75, 1.0+2.75) = mean(5.75, 3.75) = 4.75
        assert!(
            (arena[root].v1() - 4.75).abs() < 1e-5,
            "root.v1={} expected=4.75",
            arena[root].v1()
        );
    }

    #[test]
    fn backup_p2_independent_propagation() {
        // Asymmetric P1/P2 edge rewards, 2 backups, verifies independent propagation.
        // P1 edge_r=2.0, P2 edge_r=0.5
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(2.0, 0.5);

        for idx in [root, child] {
            arena[idx].set_value_scale(10.0);
        }

        let path = vec![(root, 0, 0)];

        // Backup 1: g1=3, g2=7
        backup(&mut arena, &path, child, 3.0, 7.0);
        // root: q1 = 2.0 + 3.0 = 5.0, q2 = 0.5 + 7.0 = 7.5
        assert!((arena[root].v1() - 5.0).abs() < 1e-5);
        assert!((arena[root].v2() - 7.5).abs() < 1e-5);

        // Backup 2: g1=1, g2=5
        backup(&mut arena, &path, child, 1.0, 5.0);
        // root: q1 = 2.0 + 1.0 = 3.0. mean(5.0, 3.0) = 4.0
        // root: q2 = 0.5 + 5.0 = 5.5. mean(7.5, 5.5) = 6.5
        assert!(
            (arena[root].v1() - 4.0).abs() < 1e-5,
            "root.v1={} expected=4.0",
            arena[root].v1()
        );
        assert!(
            (arena[root].v2() - 6.5).abs() < 1e-5,
            "root.v2={} expected=6.5",
            arena[root].v2()
        );
    }

    // ---- PUCT selection: edge cases ----

    #[test]
    fn puct_single_outcome() {
        // Mud-stuck node: 1 outcome. select_actions should return (0, 0).
        let stuck = [4, 4, 4, 4, 4]; // all actions map to STAY
        let p1 = HalfNode::new([0.2; 5], stuck);
        let p2 = HalfNode::new([0.2; 5], stuck);
        assert_eq!(p1.n_outcomes(), 1);

        let mut arena = NodeArena::new();
        let idx = arena.alloc(Node::new(p1, p2));
        arena[idx].set_value_scale(5.0);
        arena[idx].update_value(1.0, 1.0);

        let config = default_config();
        let mut r = rng();
        let (a1, a2) = select_actions(&arena[idx], &config, false, &mut r);
        assert_eq!(a1, 0);
        assert_eq!(a2, 0);
    }

    // ---- Virtual loss: PUCT ----

    #[test]
    fn puct_virtual_loss_diversifies() {
        // Heavy virtual loss on best edge shifts selection to a different outcome.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);
        arena[node_idx].update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();

        // First selection with no virtual loss — pick baseline.
        let (baseline, _) = select_actions(&arena[node_idx], &config, false, &mut r);

        // Add heavy virtual loss on the baseline outcome.
        for _ in 0..100 {
            arena[node_idx].p1.edge_mut(baseline as usize).add_virtual_loss();
        }

        let mut r2 = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r2);
        assert_ne!(a1, baseline, "Virtual loss should shift selection away");
    }

    #[test]
    fn puct_no_virtual_loss_unchanged() {
        // When all n_in_flight == 0, behavior is identical to before.
        let mut arena = NodeArena::new();
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);
        arena[node_idx].update_value(2.0, 2.0);

        // Verify no edge has in-flight.
        for i in 0..5 {
            assert_eq!(arena[node_idx].p1.edge(i).n_in_flight(), 0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);
        // Same as puct_monotonic_prior: highest prior wins.
        assert_eq!(a1, 2);
    }

    #[test]
    fn puct_virtual_loss_unvisited_still_fpu() {
        // Unvisited edge with virtual loss gets FPU, not edge.q=0.
        // FPU uses edge.visits (real visits), not n_in_flight.
        let mut arena = NodeArena::new();
        let node_idx = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);
        arena[node_idx].update_value(5.0, 5.0);

        // Visit some edges with low Q so unvisited FPU-based edges dominate.
        for i in 0..3 {
            arena[node_idx].p1.edge_mut(i).update(1.0);
        }

        // Add virtual loss on unvisited edge 3. It should still get FPU (visits==0).
        arena[node_idx].p1.edge_mut(3).add_virtual_loss();

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, false, &mut r);

        // Edges 3 and 4 are both unvisited. Edge 3 has virtual loss reducing
        // its exploration bonus, so edge 4 should be preferred.
        assert_eq!(a1, 4, "Unvisited edge without virtual loss should beat one with");
    }

    #[test]
    fn virtual_loss_sum_invariant() {
        // R1: sum(edge.n_in_flight) == node.n_in_flight after N descents.
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        let config = default_config();

        // Simulate 3 descents: claim node, pick an action, add virtual loss.
        let mut r = rng();
        for _ in 0..3 {
            assert!(node.try_start_score_update());
            let a1 = select_half(
                &node.p1, node.v1(), node.value_scale(), node.total_visits(),
                &config, false, &mut r,
            );
            node.p1.edge_mut(a1 as usize).add_virtual_loss();
        }

        let edge_sum: u32 = (0..5).map(|i| node.p1.edge(i).n_in_flight()).sum();
        assert_eq!(edge_sum, node.n_in_flight());
    }

    #[test]
    fn virtual_loss_cleanup_invariant() {
        // R2: after full revert, all n_in_flight == 0.
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();

        // Track which edges got virtual loss.
        let mut vl_edges = Vec::new();
        for _ in 0..3 {
            assert!(node.try_start_score_update());
            let a1 = select_half(
                &node.p1, node.v1(), node.value_scale(), node.total_visits(),
                &config, false, &mut r,
            );
            node.p1.edge_mut(a1 as usize).add_virtual_loss();
            vl_edges.push(a1);
        }

        // Revert everything.
        for a in &vl_edges {
            node.p1.edge_mut(*a as usize).revert_virtual_loss();
            node.cancel_score_update();
        }

        assert_eq!(node.n_in_flight(), 0);
        for i in 0..5 {
            assert_eq!(node.p1.edge(i).n_in_flight(), 0);
        }
    }

    #[test]
    fn puct_multi_descent_diversification() {
        // 3 descents with virtual loss → 3 different outcomes.
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();
        let mut selected = Vec::new();

        for _ in 0..3 {
            let a1 = select_half(
                &node.p1, node.v1(), node.value_scale(), node.total_visits(),
                &config, false, &mut r,
            );
            node.p1.edge_mut(a1 as usize).add_virtual_loss();
            selected.push(a1);
        }

        // All 3 should be different outcomes.
        selected.sort();
        selected.dedup();
        assert_eq!(
            selected.len(), 3,
            "3 descents with virtual loss should diversify to 3 outcomes, got {:?}",
            selected
        );
    }

    #[test]
    fn puct_forced_playout_ignores_virtual_loss() {
        // Forced playout threshold uses real visits only (edge.visits), not n_in_flight.
        let mut arena = NodeArena::new();
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let node_idx = alloc_open_node(&mut arena, p1_prior, [0.2; 5]);
        arena[node_idx].set_value_scale(5.0);

        // Many visits, heavily on 0 and 1.
        for _ in 0..100 {
            arena[node_idx].update_value(2.0, 2.0);
        }
        for _ in 0..45 {
            arena[node_idx].p1.edge_mut(0).update(2.0);
        }
        for _ in 0..45 {
            arena[node_idx].p1.edge_mut(1).update(2.0);
        }
        arena[node_idx].p1.edge_mut(2).update(2.0);
        arena[node_idx].p1.edge_mut(3).update(2.0);
        arena[node_idx].p1.edge_mut(4).update(2.0);

        // Add large virtual loss on the undervisited outcomes.
        // Threshold for outcome 2: sqrt(2.0 * 0.1 * 100) ≈ 4.47. Visits=1 < 4.47 → forced.
        // Even with virtual loss, forced playouts should still fire (they check real visits).
        for _ in 0..50 {
            arena[node_idx].p1.edge_mut(2).add_virtual_loss();
            arena[node_idx].p1.edge_mut(3).add_virtual_loss();
            arena[node_idx].p1.edge_mut(4).add_virtual_loss();
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&arena[node_idx], &config, true, &mut r);

        // Forced playouts should still fire on undervisited outcomes.
        assert!(
            arena[node_idx].p1.edge(a1 as usize).visits < 5,
            "Forced playout should still select undervisited outcome despite virtual loss, got outcome {} with {} visits",
            a1,
            arena[node_idx].p1.edge(a1 as usize).visits
        );
    }

    // ---- integration: run_search with real GameState + SmartUniformBackend ----

    const BACKEND: SmartUniformBackend = SmartUniformBackend;

    fn search_rng() -> SmallRng {
        SmallRng::seed_from_u64(123)
    }

    // 1. root_evaluation: 1 sim populates and visits root
    #[test]
    fn search_root_evaluation() {
        let cheese = [Coordinates::new(2, 2), Coordinates::new(3, 3)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 1, 1, &mut r);

        assert_eq!(result.total_visits, 1);
        // SmartUniform values are 0 → root value should be 0.
        assert!((result.value_p1).abs() < 1e-6);
        assert!((result.value_p2).abs() < 1e-6);
        // Root priors should be set.
        let root = &tree.arena()[tree.root()];
        assert!(root.total_visits() >= 1);
    }

    // 2. first_expansion: 2 sims → root visits=2, one child with visits=1
    #[test]
    fn search_first_expansion() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 2, 1, &mut r);

        assert_eq!(result.total_visits, 2);

        // Root should have at least one child.
        let root = &tree.arena()[tree.root()];
        assert!(root.first_child().is_some());

        // Walk children: at least one with visits=1.
        let mut found_child_with_visit = false;
        let mut cur = root.first_child();
        while let Some(idx) = cur {
            if tree.arena()[idx].total_visits() == 1 {
                found_child_with_visit = true;
            }
            cur = tree.arena()[idx].next_sibling();
        }
        assert!(found_child_with_visit, "Should have a child with 1 visit");
    }

    // 3. invariants_after_50_sims: walk all nodes, check edge visit sum
    #[test]
    fn search_invariants_after_50_sims() {
        let cheese: Vec<_> = (0..5).map(|i| Coordinates::new(i, 0)).collect();
        let game = test_util::open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // batch_size=1 to avoid collisions → exact visit count.
        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 1, &mut r);
        assert_eq!(result.total_visits, 50);

        // Walk all nodes: for visited interior nodes,
        // sum(p1.edge.visits) == total_visits - 1 (the -1 is the NN eval).
        let arena = tree.arena();
        for i in 0..arena.len() {
            let idx = NodeIndex::from_raw(i as u32);
            let node = &arena[idx];
            if node.total_visits() == 0 {
                continue;
            }

            let p1_edge_sum: u32 = (0..node.p1.n_outcomes())
                .map(|j| node.p1.edge(j).visits)
                .sum();

            if node.first_child().is_some() {
                // Interior node with children: edge_sum == total_visits - 1
                assert_eq!(
                    p1_edge_sum,
                    node.total_visits() - 1,
                    "Node {i}: p1 edge visit sum mismatch"
                );
            }

            // No negative edge visits (can't happen with u32 but check Q is finite).
            for j in 0..node.p1.n_outcomes() {
                assert!(node.p1.edge(j).q.is_finite(), "Node {i}: P1 edge {j} Q is not finite");
            }
            for j in 0..node.p2.n_outcomes() {
                assert!(node.p2.edge(j).q.is_finite(), "Node {i}: P2 edge {j} Q is not finite");
            }
        }
    }

    // 4. corridor: linear maze, policy weights the real moves
    #[test]
    fn search_corridor() {
        let game = test_util::corridor_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 4, &mut r);

        // P1 at (0,0) in corridor: UP blocked by wall, DOWN blocked by edge.
        // Only RIGHT, LEFT, STAY are effective — but LEFT is blocked by edge too.
        // So real moves are RIGHT and STAY. Policy should put 0 on UP and DOWN.
        assert_eq!(result.policy_p1[0], 0.0, "UP should be 0 (wall)");
        assert_eq!(result.policy_p1[2], 0.0, "DOWN should be 0 (edge)");

        // RIGHT should have substantial weight (leads toward cheese).
        assert!(
            result.policy_p1[1] > 0.3,
            "RIGHT should have significant weight, got {}",
            result.policy_p1[1]
        );
    }

    // 5. adjacent_cheese: cheese 1 step away, policy prefers that direction
    #[test]
    fn search_adjacent_cheese() {
        let game = test_util::one_cheese_adjacent_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 100, 4, &mut r);

        // P1 at (0,0), cheese at (1,0). RIGHT collects it.
        // With enough sims, RIGHT should dominate.
        let right = result.policy_p1[1]; // Direction::Right = 1
        assert!(
            right > 0.5,
            "RIGHT should dominate toward adjacent cheese, got {}",
            right
        );
    }

    // 6. terminal_mid_tree: short game, terminal nodes get (0,0) backup
    #[test]
    fn search_terminal_mid_tree() {
        let game = test_util::short_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // batch_size=1 to avoid collisions.
        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 1, &mut r);

        // Should complete without panics. 3-turn game means terminals appear.
        assert_eq!(result.total_visits, 50);

        // Walk nodes: any terminal node should have is_terminal set.
        let arena = tree.arena();
        for i in 0..arena.len() {
            let idx = NodeIndex::from_raw(i as u32);
            let node = &arena[idx];
            if node.is_terminal() {
                // Terminal nodes should have 0 for edge visits (no children).
                assert!(node.first_child().is_none(), "Terminal node shouldn't have children");
            }
        }
    }

    // 7. terminal_root: already game-over
    #[test]
    fn search_terminal_root() {
        let game = test_util::terminal_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 10, 4, &mut r);

        // Root is terminal → values should be 0.
        assert!((result.value_p1).abs() < 1e-6, "Terminal root v1 should be 0, got {}", result.value_p1);
        assert!((result.value_p2).abs() < 1e-6, "Terminal root v2 should be 0, got {}", result.value_p2);
    }

    // 8. mud_position: P1 stuck, policy 100% STAY
    #[test]
    fn search_mud_position() {
        let game = test_util::mud_game_p1_stuck();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r);

        // P1 stuck in mud: only STAY is valid.
        assert!(
            (result.policy_p1[4] - 1.0).abs() < 1e-6,
            "Stuck P1 should have 100% STAY, got {:?}",
            result.policy_p1
        );
        for a in 0..4 {
            assert_eq!(result.policy_p1[a], 0.0, "Action {a} should be 0 for stuck P1");
        }
    }

    // 9. policy_sums_to_one
    #[test]
    fn search_policy_sums_to_one() {
        let games = [
            test_util::open_5x5_game(
                Coordinates::new(2, 2),
                Coordinates::new(2, 2),
                &[Coordinates::new(0, 0)],
            ),
            test_util::one_cheese_adjacent_game(),
            test_util::mud_game_p1_stuck(),
            test_util::corridor_game(),
        ];

        let config = default_config();
        for (gi, game) in games.iter().enumerate() {
            let mut tree = MCTSTree::new(game);
            let mut r = search_rng();
            let result = run_search(&mut tree, game, &BACKEND, &config, 30, 4, &mut r);

            let sum_p1: f32 = result.policy_p1.iter().sum();
            let sum_p2: f32 = result.policy_p2.iter().sum();
            assert!(
                (sum_p1 - 1.0).abs() < 1e-5,
                "Game {gi}: P1 policy sum = {sum_p1}"
            );
            assert!(
                (sum_p2 - 1.0).abs() < 1e-5,
                "Game {gi}: P2 policy sum = {sum_p2}"
            );
        }
    }

    // 10. blocked_actions_zero
    #[test]
    fn search_blocked_actions_zero() {
        // P1 at (0,0): DOWN and LEFT blocked by board edges.
        let game = test_util::open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 30, 4, &mut r);

        assert_eq!(result.policy_p1[2], 0.0, "DOWN should be 0 (board edge)");
        assert_eq!(result.policy_p1[3], 0.0, "LEFT should be 0 (board edge)");
    }

    // 11. value_bounded: 0 <= value <= remaining_cheese (approximately)
    #[test]
    fn search_value_bounded() {
        let cheese: Vec<_> = (0..5).map(|i| Coordinates::new(i, 0)).collect();
        let game = test_util::open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &cheese,
        );
        let remaining = game.cheese.remaining_cheese() as f32;
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 100, 4, &mut r);

        // Values should be non-negative and bounded by remaining cheese.
        // SmartUniform starts at 0, so values stay near 0. Allow some slack.
        assert!(
            result.value_p1 >= -0.1,
            "v1 should be roughly non-negative, got {}",
            result.value_p1
        );
        assert!(
            result.value_p1 <= remaining + 0.1,
            "v1 should be <= remaining cheese ({}), got {}",
            remaining,
            result.value_p1
        );
        assert!(
            result.value_p2 >= -0.1,
            "v2 should be roughly non-negative, got {}",
            result.value_p2
        );
        assert!(
            result.value_p2 <= remaining + 0.1,
            "v2 should be <= remaining cheese ({}), got {}",
            remaining,
            result.value_p2
        );
    }

    // 12. replay_correctness: children's effective actions match game after move
    #[test]
    fn search_replay_correctness() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let _ = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r);

        // For each child of root, replay the move and verify effective actions match.
        let root = tree.root();
        let arena = tree.arena();
        let root_node = &arena[root];

        let mut cur = root_node.first_child();
        while let Some(child_idx) = cur {
            let child = &arena[child_idx];
            let (oi, oj) = child.parent_outcome();

            // Convert outcome indices to actions.
            let a1 = root_node.p1.outcome_action(oi as usize);
            let a2 = root_node.p2.outcome_action(oj as usize);

            // Replay the move.
            let mut game_copy = game.clone();
            let d1 = Direction::try_from(a1).expect("valid");
            let d2 = Direction::try_from(a2).expect("valid");
            let _undo = game_copy.make_move(d1, d2);

            // Child's outcome count should match game at child position.
            let eff_p1 = game_copy.effective_actions_p1();
            let eff_p2 = game_copy.effective_actions_p2();

            // Count unique outcomes from effective actions.
            let count_unique = |eff: &[u8; 5]| -> usize {
                let mut seen = [false; 5];
                let mut n = 0;
                for &e in eff {
                    if !seen[e as usize] {
                        seen[e as usize] = true;
                        n += 1;
                    }
                }
                n
            };

            assert_eq!(
                child.p1.n_outcomes(),
                count_unique(&eff_p1),
                "P1 outcome count mismatch for child at ({oi}, {oj})"
            );
            assert_eq!(
                child.p2.n_outcomes(),
                count_unique(&eff_p2),
                "P2 outcome count mismatch for child at ({oi}, {oj})"
            );

            cur = child.next_sibling();
        }
    }

    // 13. n_in_flight_zero_after_search: all nodes and edges clean
    #[test]
    fn search_n_in_flight_zero_after_search() {
        let cheese = [Coordinates::new(2, 2), Coordinates::new(3, 3)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let _ = run_search(&mut tree, &game, &BACKEND, &config, 30, 4, &mut r);

        let arena = tree.arena();
        for i in 0..arena.len() {
            let idx = NodeIndex::from_raw(i as u32);
            let node = &arena[idx];

            assert_eq!(
                node.n_in_flight(),
                0,
                "Node {i}: n_in_flight should be 0, got {}",
                node.n_in_flight()
            );

            for j in 0..node.p1.n_outcomes() {
                assert_eq!(
                    node.p1.edge(j).n_in_flight(),
                    0,
                    "Node {i} P1 edge {j}: n_in_flight should be 0"
                );
            }
            for j in 0..node.p2.n_outcomes() {
                assert_eq!(
                    node.p2.edge(j).n_in_flight(),
                    0,
                    "Node {i} P2 edge {j}: n_in_flight should be 0"
                );
            }
        }
    }

    // 14. batch_size_larger_than_n_sims: only n_sims run
    #[test]
    fn search_batch_size_larger_than_n_sims() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // batch_size=10 but only n_sims=3. With batch_size > n_sims, we run one
        // batch of 3 descents. Some may collide (root is unvisited), so
        // total_visits <= 3. At least 1 should succeed (the first descent).
        let result = run_search(&mut tree, &game, &BACKEND, &config, 3, 10, &mut r);

        assert_eq!(
            result.total_visits, 1,
            "3 descents on unvisited root: first claims it, other 2 collide. Got {}",
            result.total_visits
        );
    }

    // ---- integration: ConstantValueBackend (non-zero leaf values) ----

    // 15. root values positive with constant v=(3,2)
    #[test]
    fn search_nonzero_backend_root_value() {
        let cheese = [Coordinates::new(2, 2), Coordinates::new(3, 3)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let backend = ConstantValueBackend {
            value_p1: 3.0,
            value_p2: 2.0,
        };
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 1, &mut r);

        assert_eq!(result.total_visits, 50);
        assert!(
            result.value_p1 > 2.0,
            "v1 should be >= leaf value 3.0 (edge rewards non-negative), got {}",
            result.value_p1
        );
        assert!(
            result.value_p2 > 1.0,
            "v2 should be >= leaf value 2.0 (edge rewards non-negative), got {}",
            result.value_p2
        );
    }

    // 16. visited edges have Q > 0 with constant v=(5,5)
    #[test]
    fn search_nonzero_backend_edge_q_positive() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let backend = ConstantValueBackend {
            value_p1: 5.0,
            value_p2: 5.0,
        };
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let _ = run_search(&mut tree, &game, &backend, &config, 30, 1, &mut r);

        let root = &tree.arena()[tree.root()];
        for i in 0..root.p1.n_outcomes() {
            let edge = root.p1.edge(i);
            if edge.visits > 0 {
                assert!(
                    edge.q > 2.0,
                    "P1 edge {i}: Q should be well above 0 with v=(5,5), got {}",
                    edge.q
                );
            }
        }
        for i in 0..root.p2.n_outcomes() {
            let edge = root.p2.edge(i);
            if edge.visits > 0 {
                assert!(
                    edge.q > 2.0,
                    "P2 edge {i}: Q should be well above 0 with v=(5,5), got {}",
                    edge.q
                );
            }
        }
    }

    // 17. structural invariants hold with non-zero values
    #[test]
    fn search_nonzero_backend_invariants() {
        let cheese: Vec<_> = (0..5).map(|i| Coordinates::new(i, 0)).collect();
        let game = test_util::open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &cheese,
        );
        let backend = ConstantValueBackend {
            value_p1: 3.0,
            value_p2: 2.0,
        };
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 1, &mut r);
        assert_eq!(result.total_visits, 50);

        let arena = tree.arena();
        for i in 0..arena.len() {
            let idx = NodeIndex::from_raw(i as u32);
            let node = &arena[idx];
            if node.total_visits() == 0 {
                continue;
            }

            // Edge visit sum invariant.
            let p1_edge_sum: u32 = (0..node.p1.n_outcomes())
                .map(|j| node.p1.edge(j).visits)
                .sum();

            if node.first_child().is_some() {
                assert_eq!(
                    p1_edge_sum,
                    node.total_visits() - 1,
                    "Node {i}: p1 edge visit sum mismatch"
                );
            }

            // n_in_flight should be 0 after search.
            assert_eq!(
                node.n_in_flight(),
                0,
                "Node {i}: n_in_flight should be 0"
            );

            // All Q values should be finite.
            for j in 0..node.p1.n_outcomes() {
                assert!(node.p1.edge(j).q.is_finite(), "Node {i}: P1 edge {j} Q not finite");
            }
            for j in 0..node.p2.n_outcomes() {
                assert!(node.p2.edge(j).q.is_finite(), "Node {i}: P2 edge {j} Q not finite");
            }
        }
    }
}
