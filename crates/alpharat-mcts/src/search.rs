use rand::Rng;

use crate::node::{HalfNode, Node, NodeArena, NodeIndex};

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
        path.last().map_or(true, |&(parent_idx, _, _)| arena[leaf_idx].parent() == Some(parent_idx)),
        "backup: leaf's parent should match last path entry"
    );

    let mut child_idx = leaf_idx;

    for &(node_idx, a1, a2) in path.iter().rev() {
        // Read child fields into locals before mutating parent.
        let child = &arena[child_idx];
        let child_r1 = child.edge_r1();
        let child_r2 = child.edge_r2();
        let child_v1 = child.v1();
        let child_v2 = child.v2();

        let q1 = child_r1 + child_v1;
        let q2 = child_r2 + child_v2;

        let node = &mut arena[node_idx];
        node.update_value(q1, q2);
        node.p1.edge_mut(a1 as usize).update(q1);
        node.p2.edge_mut(a2 as usize).update(q2);

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

        let exploration = config.c_puct * prior * sqrt_total / (1.0 + edge.visits as f32);
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeArena};
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

        // Root edge: q = 0 + child_v at each backup time
        // After backup 1: child_v=(2,1) → edge q=2.0
        // After backup 2: child_v=(3,2) → edge q=(2+3)/2=2.5
        // After backup 3: child_v=(4,3) → edge q=(2+3+4)/3=3.0
        assert_eq!(arena[root].p1.edge(0).visits, 3);
        // Edge Q is Welford average of [2.0, 3.0, 4.0] = 3.0
        assert!((arena[root].p1.edge(0).q - 3.0).abs() < 1e-5);
        assert!((arena[root].p2.edge(0).q - 2.0).abs() < 1e-5);
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
        // Node value after N backups: v1 * N ≈ sum of all q1 values that passed through.
        let mut arena = NodeArena::new();
        let root = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);
        let child = alloc_open_node(&mut arena, [0.2; 5], [0.2; 5]);

        arena[root].set_first_child(Some(child));
        arena[child].set_parent(Some(root), (0, 0));
        arena[child].set_edge_rewards(1.0, 0.0);

        for idx in [root, child] {
            arena[idx].set_value_scale(5.0);
        }

        let leaf_values = [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0)];
        let mut sum_q1 = 0.0f32;
        for (g1, g2) in &leaf_values {
            backup(&mut arena, &[(root, 0, 0)], child, *g1, *g2);
            // At backup time, child_v1 is the running average so far.
            // The q1 passed to root is edge_r1 + child_v1_at_this_point.
            sum_q1 += 1.0 + arena[child].v1();
        }

        let expected_mean = sum_q1 / 3.0;
        assert!(
            (arena[root].v1() - expected_mean).abs() < 1e-5,
            "root v1={} expected={}",
            arena[root].v1(),
            expected_mean
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
}
