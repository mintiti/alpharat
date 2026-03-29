use rand::Rng;
use rand_distr::Gamma;

use crate::backend::{Backend, BackendError};
use crate::node::{HalfNode, Node, NodePtr};
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
    /// Dirichlet noise mixing weight. 0.0 = disabled, typical: 0.25.
    pub noise_epsilon: f32,
    /// Total Dirichlet concentration (KataGo-style).
    /// Per-move alpha = concentration / n_outcomes.
    /// Default: 10.83 (KataGo's value, = 361 * 0.03).
    pub noise_concentration: f32,
    /// Collision budget scaling (LC0 pattern). The collision limit scales
    /// with tree size from `collision_limit_min` to `collision_limit_max`.
    pub collision_limit_min: u32,
    pub collision_limit_max: u32,
    /// Tree node count at which collision limit starts ramping.
    pub collision_scaling_start: u32,
    /// Tree node count at which collision limit reaches max.
    pub collision_scaling_end: u32,
    /// Power-law interpolation exponent.
    pub collision_scaling_power: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            fpu_reduction: 0.2,
            force_k: 2.0,
            noise_epsilon: 0.0,
            noise_concentration: 10.83,
            collision_limit_min: 1,
            collision_limit_max: 256,
            collision_scaling_start: 800,
            collision_scaling_end: 50_000,
            collision_scaling_power: 1.0,
        }
    }
}

/// A single step on the search path: (node_ptr, p1_outcome_idx, p2_outcome_idx).
pub type SearchPath = Vec<(NodePtr, u8, u8)>;

// ---------------------------------------------------------------------------
// backup
// ---------------------------------------------------------------------------

/// Walk leaf→root, updating node values and edge Q along the path.
///
/// `path` entries are ancestors of `leaf`, ordered root-first.
/// The last entry is the leaf's parent; the leaf itself is NOT in the path.
///
/// `g1, g2` are the leaf evaluation (NN value or terminal reward).
/// The leaf gets `update_value(g1, g2)` directly.
/// Each ancestor accumulates `q = edge_reward + child_value`.
pub fn backup(
    path: &[(NodePtr, u8, u8)],
    leaf: NodePtr,
    g1: f32,
    g2: f32,
) {
    // Visit 1 on the leaf: NN eval or terminal value.
    unsafe { leaf.as_mut() }.update_value(g1, g2);

    debug_assert!(
        path.last().is_none_or(|&(parent_ptr, _, _)| unsafe { leaf.as_ref() }.parent() == Some(parent_ptr)),
        "backup: leaf's parent should match last path entry"
    );

    // Carry raw values upward — NOT running averages from child nodes.
    // This matches lc0's DoBackupUpdateSingleNode: each ancestor sees
    // q = edge_reward + propagated_value, where propagated_value chains
    // from the leaf evaluation, not from stale Welford averages.
    let mut v1 = g1;
    let mut v2 = g2;
    let mut child_ptr = leaf;

    for &(node_ptr, a1, a2) in path.iter().rev() {
        let child = unsafe { child_ptr.as_ref() };
        let q1 = child.edge_r1() + v1;
        let q2 = child.edge_r2() + v2;

        let node = unsafe { node_ptr.as_mut() };
        node.update_value(q1, q2);
        node.p1.edge_mut(a1 as usize).update(q1);
        node.p2.edge_mut(a2 as usize).update(q2);

        v1 = q1;
        v2 = q2;
        child_ptr = node_ptr;
    }
}

// ---------------------------------------------------------------------------
// compute_fpu — first-play urgency
// ---------------------------------------------------------------------------

/// Compute FPU: pessimistic value for unvisited outcomes, scaled by the
/// fraction of prior mass already visited. Used in PUCT selection and policy
/// extraction. Matches LC0's GetFpu.
fn compute_fpu(half: &HalfNode, node_value: f32, value_scale: f32, fpu_reduction: f32) -> f32 {
    let mut visited_prior_mass = 0.0f32;
    for i in 0..half.n_outcomes() {
        if half.edge(i).visits > 0 {
            visited_prior_mass += half.prior(i);
        }
    }
    node_value - fpu_reduction * value_scale * visited_prior_mass.sqrt()
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
    let children_visits = node.children_visits();
    let a1 = select_half(&node.p1, node.v1(), node.value_scale(), children_visits, config, is_root, rng);
    let a2 = select_half(&node.p2, node.v2(), node.value_scale(), children_visits, config, is_root, rng);
    (a1, a2)
}

/// PUCT selection for one player's half-node.
fn select_half(
    half: &HalfNode,
    node_value: f32,
    value_scale: f32,
    children_visits: u32,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = half.n_outcomes();
    debug_assert!(n > 0, "select_half: no outcomes");

    if n == 1 {
        return 0;
    }

    debug_assert!(value_scale > 0.0, "value_scale must be positive, got {value_scale}");
    let fpu = compute_fpu(half, node_value, value_scale, config.fpu_reduction);

    let sqrt_total = (children_visits.max(1) as f32).sqrt();

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
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
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
    parent_visits: u32,
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
    let sqrt_total = (parent_visits.max(1) as f32).sqrt();
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
    /// Pruned visit counts in 5-action space.
    pub visit_counts_p1: [f32; 5],
    pub visit_counts_p2: [f32; 5],
    /// NN/uniform prior at root in 5-action space.
    pub prior_p1: [f32; 5],
    pub prior_p2: [f32; 5],
    /// Root visit count after search.
    pub total_visits: u32,
    /// Number of descents that required NN evaluation.
    pub nn_evals: u32,
    /// Number of descents that hit terminal nodes (free — no NN call).
    pub terminals: u32,
    /// Number of descents that collided (wasted — no backup).
    pub collisions: u32,
}

/// Per-batch counters from simulate_batch.
struct BatchStats {
    nn_evals: u32,
    terminals: u32,
    collisions: u32,
}

// ---------------------------------------------------------------------------
// NodeToProcess — LC0-style batch entry
// ---------------------------------------------------------------------------

/// What this batch entry represents.
enum NodeKind {
    /// Leaf needs NN evaluation (multivisit always 1).
    NeedsEval { game_state: GameState },
    /// Terminal node (game over). Backed up with (0, 0).
    Terminal,
}

/// A single entry from the batch gather phase. Matches LC0's NodeToProcess.
struct NodeToProcess {
    node: NodePtr,
    kind: NodeKind,
    multivisit: u32,
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
) -> Result<SearchResult, BackendError> {
    let mut remaining = n_sims;
    let mut total_nn_evals = 0u32;
    let mut total_terminals = 0u32;
    let mut total_collisions = 0u32;
    while remaining > 0 {
        let batch = simulate_batch(tree, game, backend, config, remaining.min(batch_size), rng)?;
        total_nn_evals += batch.nn_evals;
        total_terminals += batch.terminals;
        total_collisions += batch.collisions;
        let produced = batch.nn_evals + batch.terminals;
        remaining = remaining.saturating_sub(produced.max(1));
    }

    let root = tree.root();
    let mut result = extract_result(root, config, rng);
    result.nn_evals = total_nn_evals;
    result.terminals = total_terminals;
    result.collisions = total_collisions;
    Ok(result)
}

// ---------------------------------------------------------------------------
// apply_dirichlet_noise — root exploration noise
// ---------------------------------------------------------------------------

/// Mix Dirichlet noise into a HalfNode's outcome-indexed priors.
///
/// Uses KataGo's total-concentration approach: per-move alpha = concentration / n_outcomes.
/// No-op if n_outcomes <= 1 (only one possible outcome, noise is meaningless).
fn apply_dirichlet_noise(half: &mut HalfNode, epsilon: f32, concentration: f32, rng: &mut impl Rng) {
    let n = half.n_outcomes();
    if n <= 1 {
        return;
    }

    // Per-move alpha = total concentration / n_outcomes (KataGo-style)
    let alpha = (concentration / n as f32) as f64;

    // Sample Gamma(alpha, 1.0) per outcome, normalize → Dirichlet sample
    let gamma_dist = match Gamma::new(alpha, 1.0) {
        Ok(d) => d,
        Err(_) => return, // alpha <= 0 or invalid
    };

    let mut noise = [0.0f32; 5];
    let mut total = 0.0f32;
    for item in noise.iter_mut().take(n) {
        *item = rng.sample(gamma_dist) as f32;
        total += *item;
    }
    if total < f32::MIN_POSITIVE {
        return;
    }

    // Blend: prior = (1 - eps) * prior + eps * normalized_noise
    for (i, &noise_val) in noise.iter().enumerate().take(n) {
        half.set_prior_at(i, half.prior(i) * (1.0 - epsilon) + epsilon * noise_val / total);
    }
}

// ---------------------------------------------------------------------------
// calculate_collisions_left — LC0's tree-size-based collision budget
// ---------------------------------------------------------------------------

/// LC0's CalculateCollisionsLeft: power-law interpolation from min to max
/// based on tree node count.
fn calculate_collisions_left(tree_node_count: u32, config: &SearchConfig) -> u32 {
    if tree_node_count >= config.collision_scaling_end {
        return config.collision_limit_max;
    }
    if tree_node_count <= config.collision_scaling_start {
        return config.collision_limit_min;
    }
    let ratio = (tree_node_count - config.collision_scaling_start) as f32
        / (config.collision_scaling_end - config.collision_scaling_start) as f32;
    let scaled = config.collision_limit_min as f32
        + (config.collision_limit_max as f32 - config.collision_limit_min as f32)
            * ratio.powf(config.collision_scaling_power);
    (scaled.round() as u32).clamp(config.collision_limit_min, config.collision_limit_max)
}

// ---------------------------------------------------------------------------
// estimated_visits_to_change_best — LC0's batch allocation helper
// ---------------------------------------------------------------------------

/// For one player's half-node, compute how many more visits to the best
/// outcome before the second-best outcome overtakes it in PUCT score.
///
/// Returns (best_idx, visits_to_change). If only one outcome or the best
/// utility alone beats second-best, returns u32::MAX.
///
/// Port of LC0's `estimated_visits_to_change_best` logic (search.cc:1787-1799).
fn estimated_visits_to_change_best_half(
    half: &HalfNode,
    node_value: f32,
    value_scale: f32,
    children_visits: u32,
    config: &SearchConfig,
    is_root: bool,
    nstarted: &[u32; 5],
    rng: &mut impl Rng,
) -> (u8, u32) {
    let n = half.n_outcomes();
    if n <= 1 {
        return (0, u32::MAX);
    }

    let fpu = compute_fpu(half, node_value, value_scale, config.fpu_reduction);
    let sqrt_total = (children_visits.max(1) as f32).sqrt();
    let c_puct = config.c_puct;

    // Find best and second-best by PUCT score.
    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_utility = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;

    for i in 0..n {
        let edge = half.edge(i);
        let prior = half.prior(i);
        let q = if edge.visits > 0 { edge.q } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + nstarted[i] as f32);
        let mut score = q_norm + exploration;

        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (edge.visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_idx = i as u8;
            best_utility = q_norm;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    // Handle ties with reservoir sampling (same rng as select_half).
    let mut tie_count = 1u32;
    for i in 0..n {
        if i as u8 == best_idx {
            continue;
        }
        let edge = half.edge(i);
        let prior = half.prior(i);
        let q = if edge.visits > 0 { edge.q } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + nstarted[i] as f32);
        let mut score = q_norm + exploration;
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (edge.visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }
        if (score - best_score).abs() < 1e-12 {
            tie_count += 1;
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = i as u8;
                best_utility = q_norm;
            }
        }
    }

    if second_best_score <= f32::NEG_INFINITY {
        // Only one outcome was viable.
        return (best_idx, u32::MAX);
    }

    // LC0 formula: if utility alone beats second_best, never changes.
    if best_utility >= second_best_score {
        return (best_idx, u32::MAX);
    }

    // Solve: c_puct * prior * sqrt_total / (1 + nstarted + k) + utility = second_best
    // k = c_puct * prior * sqrt_total / (second_best - utility) - (1 + nstarted)
    let prior_best = half.prior(best_idx as usize);
    let n1 = nstarted[best_idx as usize] as f32 + 1.0;
    let denom = second_best_score - best_utility;
    if denom <= 0.0 {
        return (best_idx, u32::MAX);
    }
    let vtc = (c_puct * prior_best * sqrt_total / denom - n1 + 1.0).max(1.0);
    (best_idx, (vtc as u32).max(1))
}

// ---------------------------------------------------------------------------
// pick_nodes_to_extend — LC0-style batch allocation via tree traversal
// ---------------------------------------------------------------------------

/// Gather phase state for one level of the iterative tree traversal.
struct GatherLevel {
    node: NodePtr,
    /// Flat [a1 * 5 + a2] → allocated visits for that (a1, a2) child.
    vtp: [u32; 25],
    /// Next flat index to process.
    next_idx: usize,
    /// Last flat index with non-zero visits.
    last_idx: usize,
}

/// LC0's PickNodesToExtendTask adapted for 2-player. Distributes `batch_size`
/// visits through the tree in a single pass, producing NodeToProcess entries.
///
/// Returns (to_process, shared_collisions) where shared_collisions are
/// (node, multivisit) pairs to cancel after backup.
fn pick_nodes_to_extend(
    tree: &mut MCTSTree,
    game: &GameState,
    config: &SearchConfig,
    budget: u32,
    rng: &mut impl Rng,
) -> (Vec<NodeToProcess>, Vec<(NodePtr, u32)>) {
    let root = tree.root();
    let mut to_process: Vec<NodeToProcess> = Vec::with_capacity(budget as usize);
    let mut shared_collisions: Vec<(NodePtr, u32)> = Vec::new();
    let mut work_game = game.clone();
    let mut undos: Vec<_> = Vec::new();

    let cur_limit = budget;

    // Handle root: unvisited or terminal.
    let root_node = unsafe { root.as_ref() };
    if root_node.total_visits() == 0 || root_node.is_terminal() {
        if root_node.total_visits() == 0 && !root_node.is_terminal() {
            if unsafe { root.as_mut() }.try_start_score_update() {
                if work_game.check_game_over() {
                    populate_node(root, None);
                    to_process.push(NodeToProcess {
                        node: root,
                        kind: NodeKind::Terminal,
                        multivisit: 1,
                    });
                } else {
                    to_process.push(NodeToProcess {
                        node: root,
                        kind: NodeKind::NeedsEval { game_state: work_game.clone() },
                        multivisit: 1,
                    });
                }
                if cur_limit > 1 {
                    shared_collisions.push((root, cur_limit - 1));
                }
            } else {
                shared_collisions.push((root, cur_limit));
            }
        } else {
            // Terminal root: one productive visit + collisions (LC0 pattern).
            // Mirrors the unvisited non-terminal root path above.
            if root_node.total_visits() == 0 {
                populate_node(root, None);
            }
            if unsafe { root.as_mut() }.try_start_score_update() {
                to_process.push(NodeToProcess {
                    node: root,
                    kind: NodeKind::Terminal,
                    multivisit: 1,
                });
                if cur_limit > 1 {
                    shared_collisions.push((root, cur_limit - 1));
                }
            } else {
                shared_collisions.push((root, cur_limit));
            }
        }
        return (to_process, shared_collisions);
    }

    // Root is an interior node: increment n_in_flight for all visits.
    unsafe { root.as_mut() }.increment_n_in_flight(cur_limit);

    // Build first level: distribute cur_limit visits at root.
    let first_level = build_gather_level(root, cur_limit, config, true, rng);
    let mut levels: Vec<GatherLevel> = vec![first_level];

    while let Some(level) = levels.last_mut() {
        // Find next (a1, a2) pair with allocated visits.
        let mut found_child = false;
        while level.next_idx <= level.last_idx {
            let idx = level.next_idx;
            level.next_idx += 1;
            if level.vtp[idx] == 0 {
                continue;
            }
            let a1 = (idx / 5) as u8;
            let a2 = (idx % 5) as u8;
            let k = level.vtp[idx];

            // Convert outcome indices to canonical actions.
            let (act1, act2) = unsafe {
                let node = level.node.as_ref();
                (node.p1.outcome_action(a1 as usize), node.p2.outcome_action(a2 as usize))
            };
            let d1 = Direction::try_from(act1).expect("valid direction");
            let d2 = Direction::try_from(act2).expect("valid direction");
            let scores_before = (work_game.player1_score(), work_game.player2_score());
            let undo = work_game.make_move(d1, d2);
            let (r1, r2) = compute_rewards(&work_game, scores_before);

            let (child_ptr, is_new) = find_or_extend_child(level.node, a1, a2, &work_game, r1, r2);
            if is_new {
                tree.increment_node_count();
            }

            let child = unsafe { child_ptr.as_ref() };
            if child.total_visits() == 0 || child.is_terminal() {
                // Leaf or terminal.
                if unsafe { child_ptr.as_mut() }.try_start_score_update() {
                    if child.is_terminal() || work_game.check_game_over() {
                        // Terminal.
                        if child.total_visits() == 0 {
                            populate_node(child_ptr, None);
                        }
                        to_process.push(NodeToProcess {
                            node: child_ptr,
                            kind: NodeKind::Terminal,
                            multivisit: 1,
                        });
                        if k > 1 {
                            shared_collisions.push((child_ptr, k - 1));
                        }
                    } else {
                        // Needs NN eval.
                        to_process.push(NodeToProcess {
                            node: child_ptr,
                            kind: NodeKind::NeedsEval { game_state: work_game.clone() },
                            multivisit: 1,
                        });
                        if k > 1 {
                            shared_collisions.push((child_ptr, k - 1));
                        }
                    }
                } else {
                    // Collision: all k visits.
                    shared_collisions.push((child_ptr, k));
                }
                work_game.unmake_move(undo);
            } else {
                // Interior child: descend with k visits.
                // TryStartScoreUpdate for the initial claim, then add remaining.
                if unsafe { child_ptr.as_mut() }.try_start_score_update() {
                    // Claim succeeded. Add remaining k-1 visits.
                    if k > 1 {
                        unsafe { child_ptr.as_mut() }.increment_n_in_flight(k - 1);
                    }
                    undos.push(undo);
                    let child_level = build_gather_level(child_ptr, k, config, false, rng);
                    levels.push(child_level);
                    found_child = true;
                    break;
                } else {
                    // Collision on interior node: all k visits.
                    shared_collisions.push((child_ptr, k));
                    work_game.unmake_move(undo);
                }
            }
        }

        if !found_child {
            // All children processed, backtrack.
            levels.pop();
            if let Some(undo) = undos.pop() {
                work_game.unmake_move(undo);
            }
        }
    }

    (to_process, shared_collisions)
}

/// Build a GatherLevel: distribute `cur_limit` visits at `node` using
/// decoupled PUCT with LC0's estimated_visits_to_change_best.
fn build_gather_level(
    node: NodePtr,
    cur_limit: u32,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> GatherLevel {
    let node_ref = unsafe { node.as_ref() };
    let n1 = node_ref.p1.n_outcomes();
    let n2 = node_ref.p2.n_outcomes();
    let children_visits = node_ref.children_visits();
    let value_scale = node_ref.value_scale();
    let v1 = node_ref.v1();
    let v2 = node_ref.v2();

    // Initialize nstarted from current edge state.
    let mut ns_p1 = [0u32; 5];
    let mut ns_p2 = [0u32; 5];
    for i in 0..n1 {
        ns_p1[i] = node_ref.p1.edge(i).n_started();
    }
    for j in 0..n2 {
        ns_p2[j] = node_ref.p2.edge(j).n_started();
    }

    let mut vtp = [0u32; 25];
    let mut remaining = cur_limit;
    let mut last_idx = 0usize;

    while remaining > 0 {
        // Select best outcome for each player.
        let (best1, vtcb1) = estimated_visits_to_change_best_half(
            &node_ref.p1, v1, value_scale, children_visits + (cur_limit - remaining),
            config, is_root, &ns_p1, rng,
        );
        let (best2, vtcb2) = estimated_visits_to_change_best_half(
            &node_ref.p2, v2, value_scale, children_visits + (cur_limit - remaining),
            config, is_root, &ns_p2, rng,
        );

        let k = remaining.min(vtcb1).min(vtcb2).max(1);

        let flat = best1 as usize * 5 + best2 as usize;
        vtp[flat] += k;
        ns_p1[best1 as usize] += k;
        ns_p2[best2 as usize] += k;
        remaining -= k;
        if flat > last_idx && vtp[flat] > 0 {
            last_idx = flat;
        }
    }

    // Apply edge virtual loss: compute delta per edge and write back.
    let node_mut = unsafe { node.as_mut() };
    for i in 0..n1 {
        let delta = ns_p1[i] - node_ref.p1.edge(i).n_started();
        if delta > 0 {
            node_mut.p1.edge_mut(i).add_virtual_loss_multi(delta);
        }
    }
    for j in 0..n2 {
        let delta = ns_p2[j] - node_ref.p2.edge(j).n_started();
        if delta > 0 {
            node_mut.p2.edge_mut(j).add_virtual_loss_multi(delta);
        }
    }

    GatherLevel { node, vtp, next_idx: 0, last_idx }
}

// ---------------------------------------------------------------------------
// backup_and_finalize — LC0-style backup with n_in_flight cleanup
// ---------------------------------------------------------------------------

/// Walk from leaf to root, applying finalize_score_update (weighted Welford +
/// n_in_flight decrement) at each node and updating edge Q. Combines the old
/// `backup` + `cleanup_descent` into a single pass.
fn backup_and_finalize(leaf: NodePtr, g1: f32, g2: f32, multivisit: u32) {
    // Leaf: finalize.
    unsafe { leaf.as_mut() }.finalize_score_update(g1, g2, multivisit);

    let mut v1 = g1;
    let mut v2 = g2;
    let mut current = leaf;

    while let Some(parent_ptr) = unsafe { current.as_ref() }.parent() {
        let (a1, a2) = unsafe { current.as_ref() }.parent_outcome();
        let r1 = unsafe { current.as_ref() }.edge_r1();
        let r2 = unsafe { current.as_ref() }.edge_r2();
        let q1 = r1 + v1;
        let q2 = r2 + v2;

        let parent = unsafe { parent_ptr.as_mut() };
        parent.finalize_score_update(q1, q2, multivisit);
        parent.p1.edge_mut(a1 as usize).update_multivisit(q1, multivisit);
        parent.p2.edge_mut(a2 as usize).update_multivisit(q2, multivisit);
        parent.p1.edge_mut(a1 as usize).revert_virtual_loss_multi(multivisit);
        parent.p2.edge_mut(a2 as usize).revert_virtual_loss_multi(multivisit);

        v1 = q1;
        v2 = q2;
        current = parent_ptr;
    }
}

// ---------------------------------------------------------------------------
// cancel_shared_collisions — LC0's CancelSharedCollisions
// ---------------------------------------------------------------------------

/// Cancel virtual losses from collision entries. Walks from each collision
/// node's parent up to root, decrementing n_in_flight and edge VL.
fn cancel_shared_collisions(
    collisions: &[(NodePtr, u32)],
    root: NodePtr,
) {
    for &(collision_node, multivisit) in collisions {
        // The collision node itself: if it was claimed (n_in_flight > 0 from
        // TryStartScoreUpdate), cancel that claim. But for pure collisions
        // (TryStartScoreUpdate failed), n_in_flight was never incremented
        // on the collision node. We handle this by canceling on the node
        // only if it has n_in_flight contributed by this collision.
        //
        // LC0 pattern: walk from collision_node's PARENT up to root.
        // The collision node's n_in_flight was never incremented for collisions
        // (TryStartScoreUpdate failed). For Visit entries where k>1 generated
        // k-1 collisions, the node's n_in_flight includes the TryStartScoreUpdate
        // claim (1) which gets handled by backup_and_finalize for the Visit.
        let mut current = collision_node;
        while let Some(parent_ptr) = unsafe { current.as_ref() }.parent() {
            let (a1, a2) = unsafe { current.as_ref() }.parent_outcome();
            let parent = unsafe { parent_ptr.as_mut() };
            parent.cancel_score_update(multivisit);
            parent.p1.edge_mut(a1 as usize).revert_virtual_loss_multi(multivisit);
            parent.p2.edge_mut(a2 as usize).revert_virtual_loss_multi(multivisit);
            if parent_ptr == root {
                break;
            }
            current = parent_ptr;
        }
    }
}

// ---------------------------------------------------------------------------
// cancel_leaf_and_path — revert a gathered entry's bookkeeping
// ---------------------------------------------------------------------------

/// Revert n_in_flight and virtual loss for a gathered leaf entry.
/// Unlike `cancel_shared_collisions` (which starts from the collision node's
/// parent because try_start_score_update failed), this includes the leaf node
/// itself because try_start_score_update succeeded for gathered entries.
fn cancel_leaf_and_path(leaf: NodePtr, multivisit: u32) {
    unsafe { leaf.as_mut() }.cancel_score_update(multivisit);
    let mut current = leaf;
    while let Some(parent_ptr) = unsafe { current.as_ref() }.parent() {
        let (a1, a2) = unsafe { current.as_ref() }.parent_outcome();
        let parent = unsafe { parent_ptr.as_mut() };
        parent.cancel_score_update(multivisit);
        parent.p1.edge_mut(a1 as usize).revert_virtual_loss_multi(multivisit);
        parent.p2.edge_mut(a2 as usize).revert_virtual_loss_multi(multivisit);
        current = parent_ptr;
    }
}

// ---------------------------------------------------------------------------
// GatherCleanupGuard — RAII revert of VL/n_in_flight on early exit
// ---------------------------------------------------------------------------

/// Drop guard that reverts virtual loss and n_in_flight for gathered-but-not-
/// backed-up entries if simulate_batch exits early (e.g., backend error).
/// Call `disarm()` on the success path to skip cleanup.
struct GatherCleanupGuard<'a> {
    to_process: &'a [NodeToProcess],
    collisions: &'a [(NodePtr, u32)],
    root: NodePtr,
    armed: bool,
}

impl<'a> GatherCleanupGuard<'a> {
    fn new(
        to_process: &'a [NodeToProcess],
        collisions: &'a [(NodePtr, u32)],
        root: NodePtr,
    ) -> Self {
        Self {
            to_process,
            collisions,
            root,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for GatherCleanupGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        for entry in self.to_process {
            cancel_leaf_and_path(entry.node, entry.multivisit);
        }
        cancel_shared_collisions(self.collisions, self.root);
    }
}

// ---------------------------------------------------------------------------
// simulate_batch — LC0-style gather/eval/backup cycle
// ---------------------------------------------------------------------------

fn simulate_batch(
    tree: &mut MCTSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<BatchStats, BackendError> {
    let root = tree.root();
    let mut collisions_left = calculate_collisions_left(tree.node_count(), config) as i32;

    let mut all_to_process: Vec<NodeToProcess> = Vec::with_capacity(batch_size as usize);
    let mut all_collisions: Vec<(NodePtr, u32)> = Vec::new();
    let mut minibatch_size = 0u32;
    let mut terminals = 0u32;

    // ---- Outer Gather Loop (LC0's GatherMinibatch) ----
    // Each iteration picks budget = min(collisions_left, remaining_batch_slots)
    // visits. minibatch_size counts all non-collision entries (NeedsEval +
    // Terminal), matching LC0. Collision VLs persist through the loop.
    while minibatch_size < batch_size && collisions_left > 0 {
        let budget = (collisions_left as u32).min(batch_size - minibatch_size);
        let (to_process, shared_collisions) =
            pick_nodes_to_extend(tree, game, config, budget, rng);

        // Accumulate all entries (LC0 pattern: backup happens after eval, not during gather).
        for entry in to_process {
            if let NodeKind::Terminal = &entry.kind {
                terminals += entry.multivisit;
            }
            minibatch_size += 1;
            all_to_process.push(entry);
        }

        for &(_, mv) in &shared_collisions {
            collisions_left -= mv as i32;
        }
        all_collisions.extend(shared_collisions);
    }

    let nn_evals = all_to_process
        .iter()
        .filter(|e| matches!(e.kind, NodeKind::NeedsEval { .. }))
        .count() as u32;
    let total_collisions = all_collisions.iter().map(|&(_, mv)| mv).sum::<u32>();

    // Guard: if evaluate_batch fails, revert all gathered VL/n_in_flight.
    let mut cleanup_guard = GatherCleanupGuard::new(&all_to_process, &all_collisions, root);

    // ---- Eval Phase: batch NN evaluation ----
    let game_states: Vec<&GameState> = all_to_process
        .iter()
        .filter_map(|entry| match &entry.kind {
            NodeKind::NeedsEval { game_state } => Some(game_state),
            _ => None,
        })
        .collect();

    let eval_results = if game_states.is_empty() {
        Vec::new()
    } else {
        backend.evaluate_batch(&game_states)?
    };

    // ---- Backup Phase: NN eval results + terminal values (LC0 pattern) ----
    // All entries backed up together after eval, matching LC0's DoBackupUpdate.
    let mut eval_idx = 0;
    for entry in &all_to_process {
        match &entry.kind {
            NodeKind::NeedsEval { .. } => {
                let eval = &eval_results[eval_idx];
                eval_idx += 1;

                populate_node(entry.node, Some(eval));

                if entry.node == root && config.noise_epsilon > 0.0 {
                    let node = unsafe { entry.node.as_mut() };
                    apply_dirichlet_noise(
                        &mut node.p1,
                        config.noise_epsilon,
                        config.noise_concentration,
                        rng,
                    );
                    apply_dirichlet_noise(
                        &mut node.p2,
                        config.noise_epsilon,
                        config.noise_concentration,
                        rng,
                    );
                }

                backup_and_finalize(entry.node, eval.value_p1, eval.value_p2, entry.multivisit);
            }
            NodeKind::Terminal => {
                backup_and_finalize(entry.node, 0.0, 0.0, entry.multivisit);
            }
        }
    }

    // Success: disarm guard before normal collision cancellation.
    cleanup_guard.disarm();

    // ---- Cancel all accumulated shared collisions (LC0 pattern) ----
    // Virtual losses from collisions persisted through gather+backup.
    // Now revert them by walking from each collision node to root.
    cancel_shared_collisions(&all_collisions, root);

    Ok(BatchStats {
        nn_evals,
        terminals,
        collisions: total_collisions,
    })
}

// ---------------------------------------------------------------------------
// extract_result — policies and values from root
// ---------------------------------------------------------------------------

fn extract_result(
    root: NodePtr,
    config: &SearchConfig,
    _rng: &mut impl Rng,
) -> SearchResult {
    let node = unsafe { root.as_ref() };
    let total_visits = node.total_visits();
    let children_visits = node.children_visits();

    let (policy_p1, visit_counts_p1, value_p1) =
        extract_half(&node.p1, node.v1(), node.value_scale(), children_visits, config);
    let (policy_p2, visit_counts_p2, value_p2) =
        extract_half(&node.p2, node.v2(), node.value_scale(), children_visits, config);

    let prior_p1 = node.p1.expand_prior();
    let prior_p2 = node.p2.expand_prior();

    SearchResult {
        policy_p1,
        policy_p2,
        value_p1,
        value_p2,
        visit_counts_p1,
        visit_counts_p2,
        prior_p1,
        prior_p2,
        total_visits,
        // Filled in by run_search after accumulation.
        nn_evals: 0,
        terminals: 0,
        collisions: 0,
    }
}

/// Extract policy, visit counts, and value for one player from root.
///
/// Returns (policy, visit_counts, value) all in 5-action space.
fn extract_half(
    half: &HalfNode,
    node_value: f32,
    value_scale: f32,
    children_visits: u32,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32) {
    let n = half.n_outcomes();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], node_value);
    }

    let fpu = compute_fpu(half, node_value, value_scale, config.fpu_reduction);

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
    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, children_visits, config.c_puct);

    // Expand pruned visits to 5-action space.
    let mut visit_counts = [0.0f32; 5];
    for (i, &pv) in pruned.iter().enumerate().take(n) {
        let action = half.outcome_action(i) as usize;
        visit_counts[action] = pv;
    }

    // Normalize to get policy.
    let mut policy = visit_counts;
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

    (policy, visit_counts, value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ConstantValueBackend, SmartUniformBackend};
    use crate::node::{Node, NodePtr};
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

    /// Allocate a boxed node with open topology (5 outcomes per player) and given priors.
    fn make_open_node(p1_prior: [f32; 5], p2_prior: [f32; 5]) -> Box<Node> {
        let p1 = open_half(p1_prior);
        let p2 = open_half(p2_prior);
        Box::new(Node::new(p1, p2))
    }

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // ---- backup ----

    #[test]
    fn backup_single_level() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        // Configure child BEFORE moving it into root.
        child_box.set_parent(Some(root_ptr), (0, 1));
        child_box.set_edge_rewards(1.0, 0.5);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        // Wire: root owns child via first_child.
        root_box.first_child = Some(child_box);

        let path = vec![(root_ptr, 0u8, 1u8)];
        backup(&path, child_ptr, 3.0, 2.0);

        // Leaf: v=(3.0, 2.0), visits=1
        assert_eq!(unsafe { child_ptr.as_ref() }.total_visits(), 1);
        assert!((unsafe { child_ptr.as_ref() }.v1() - 3.0).abs() < 1e-6);
        assert!((unsafe { child_ptr.as_ref() }.v2() - 2.0).abs() < 1e-6);

        // Root: q1 = edge_r1 + child_v1 = 1.0 + 3.0 = 4.0
        //       q2 = edge_r2 + child_v2 = 0.5 + 2.0 = 2.5
        assert_eq!(unsafe { root_ptr.as_ref() }.total_visits(), 1);
        assert!((unsafe { root_ptr.as_ref() }.v1() - 4.0).abs() < 1e-6);
        assert!((unsafe { root_ptr.as_ref() }.v2() - 2.5).abs() < 1e-6);

        // Edge visits: outcome 0 for P1, outcome 1 for P2
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(0).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p1.edge(0).q - 4.0).abs() < 1e-6);
        assert_eq!(unsafe { root_ptr.as_ref() }.p2.edge(1).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p2.edge(1).q - 2.5).abs() < 1e-6);
    }

    #[test]
    fn backup_two_level_q_chain() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut mid_box = make_open_node([0.2; 5], [0.2; 5]);
        let mid_ptr = NodePtr::from_ref(&mid_box);
        let mut leaf_box = make_open_node([0.2; 5], [0.2; 5]);
        let leaf_ptr = NodePtr::from_ref(&leaf_box);

        // Configure leaf, then mid, then wire bottom-up.
        leaf_box.set_parent(Some(mid_ptr), (1, 2));
        leaf_box.set_edge_rewards(0.5, 1.0);
        leaf_box.set_value_scale(5.0);

        mid_box.set_parent(Some(root_ptr), (0, 0));
        mid_box.set_edge_rewards(1.0, 0.5);
        mid_box.set_value_scale(5.0);
        mid_box.first_child = Some(leaf_box);

        root_box.set_value_scale(5.0);
        root_box.first_child = Some(mid_box);

        let path = vec![(root_ptr, 0u8, 0u8), (mid_ptr, 1u8, 2u8)];
        backup(&path, leaf_ptr, 2.0, 3.0);

        // Leaf: v=(2.0, 3.0)
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 2.0).abs() < 1e-6);
        assert!((unsafe { leaf_ptr.as_ref() }.v2() - 3.0).abs() < 1e-6);

        // Mid: q1 = r_leaf(0.5) + v_leaf(2.0) = 2.5
        //      q2 = r_leaf(1.0) + v_leaf(3.0) = 4.0
        assert!((unsafe { mid_ptr.as_ref() }.v1() - 2.5).abs() < 1e-6);
        assert!((unsafe { mid_ptr.as_ref() }.v2() - 4.0).abs() < 1e-6);

        // Root: q1 = r_mid(1.0) + v_mid(2.5) = 3.5
        //       q2 = r_mid(0.5) + v_mid(4.0) = 4.5
        assert!((unsafe { root_ptr.as_ref() }.v1() - 3.5).abs() < 1e-6);
        assert!((unsafe { root_ptr.as_ref() }.v2() - 4.5).abs() < 1e-6);
    }

    #[test]
    fn backup_multiple_same_edge() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(0.0, 0.0);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        let path = vec![(root_ptr, 0u8, 0u8)];

        // 3 backups with different leaf values.
        backup(&path, child_ptr, 2.0, 1.0);
        backup(&path, child_ptr, 4.0, 3.0);
        backup(&path, child_ptr, 6.0, 5.0);

        // Child: mean of [2,4,6]=4.0 and [1,3,5]=3.0
        assert_eq!(unsafe { child_ptr.as_ref() }.total_visits(), 3);
        assert!((unsafe { child_ptr.as_ref() }.v1() - 4.0).abs() < 1e-5);
        assert!((unsafe { child_ptr.as_ref() }.v2() - 3.0).abs() < 1e-5);

        // Root edge: q = edge_r(0) + raw_v propagated from leaf.
        // Each backup propagates the raw leaf value (not the running average).
        // Backup 1: v1=2.0 → q1=2.0
        // Backup 2: v1=4.0 → q1=4.0
        // Backup 3: v1=6.0 → q1=6.0
        // Edge Q = mean(2.0, 4.0, 6.0) = 4.0
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(0).visits, 3);
        assert!((unsafe { root_ptr.as_ref() }.p1.edge(0).q - 4.0).abs() < 1e-5);
        assert!((unsafe { root_ptr.as_ref() }.p2.edge(0).q - 3.0).abs() < 1e-5);
    }

    #[test]
    fn backup_multiple_different_edges() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_a_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_a_ptr = NodePtr::from_ref(&child_a_box);
        let mut child_b_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_b_ptr = NodePtr::from_ref(&child_b_box);

        child_b_box.set_parent(Some(root_ptr), (1, 1));
        child_b_box.set_edge_rewards(0.0, 0.0);
        child_b_box.set_value_scale(5.0);

        child_a_box.set_parent(Some(root_ptr), (0, 0));
        child_a_box.set_edge_rewards(0.0, 0.0);
        child_a_box.set_value_scale(5.0);
        // child_a owns child_b via next_sibling.
        child_a_box.next_sibling = Some(child_b_box);

        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_a_box);

        // Backup through child_a: leaf_v = (5.0, 1.0)
        backup(&[(root_ptr, 0u8, 0u8)], child_a_ptr, 5.0, 1.0);
        // Backup through child_b: leaf_v = (1.0, 5.0)
        backup(&[(root_ptr, 1u8, 1u8)], child_b_ptr, 1.0, 5.0);

        // Each edge visited once with independent Q.
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(0).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p1.edge(0).q - 5.0).abs() < 1e-6);
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(1).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p1.edge(1).q - 1.0).abs() < 1e-6);

        assert_eq!(unsafe { root_ptr.as_ref() }.p2.edge(0).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p2.edge(0).q - 1.0).abs() < 1e-6);
        assert_eq!(unsafe { root_ptr.as_ref() }.p2.edge(1).visits, 1);
        assert!((unsafe { root_ptr.as_ref() }.p2.edge(1).q - 5.0).abs() < 1e-6);
    }

    #[test]
    fn backup_terminal_leaf() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (2, 3));
        child_box.set_edge_rewards(0.0, 0.0);
        child_box.set_terminal();
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        // Terminal: g=(0, 0)
        backup(&[(root_ptr, 2u8, 3u8)], child_ptr, 0.0, 0.0);

        assert_eq!(unsafe { child_ptr.as_ref() }.total_visits(), 1);
        assert!((unsafe { child_ptr.as_ref() }.v1() - 0.0).abs() < 1e-6);
        assert_eq!(unsafe { root_ptr.as_ref() }.total_visits(), 1);
        assert!((unsafe { root_ptr.as_ref() }.v1() - 0.0).abs() < 1e-6);
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(2).visits, 1);
    }

    #[test]
    fn backup_edge_visit_sum() {
        // After N backups through various edges, sum(edge.visits) == total_visits - 1
        // (the -1 is because update_value on the node counts the NN eval too,
        // but actually here each backup calls update_value once, so
        // total_visits == number of backups, and sum(edge.visits) == number of backups).
        // Let's verify: sum of p1 edge visits == total_visits for the root.
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);

        // Build 3 children, capture ptrs before moving.
        let mut c0_box = make_open_node([0.2; 5], [0.2; 5]);
        let c0_ptr = NodePtr::from_ref(&c0_box);
        let mut c1_box = make_open_node([0.2; 5], [0.2; 5]);
        let c1_ptr = NodePtr::from_ref(&c1_box);
        let mut c2_box = make_open_node([0.2; 5], [0.2; 5]);
        let c2_ptr = NodePtr::from_ref(&c2_box);

        c0_box.set_edge_rewards(0.0, 0.0);
        c0_box.set_parent(Some(root_ptr), (0, 0));
        c0_box.set_value_scale(5.0);

        c1_box.set_edge_rewards(0.0, 0.0);
        c1_box.set_parent(Some(root_ptr), (1, 0));
        c1_box.set_value_scale(5.0);

        c2_box.set_edge_rewards(0.0, 0.0);
        c2_box.set_parent(Some(root_ptr), (2, 0));
        c2_box.set_value_scale(5.0);

        // Wire linked list: root → c2 → c1 → c0
        c1_box.next_sibling = Some(c0_box);
        c2_box.next_sibling = Some(c1_box);
        root_box.first_child = Some(c2_box);
        root_box.set_value_scale(5.0);

        // 5 backups: 2 through edge 0, 2 through edge 1, 1 through edge 2
        backup(&[(root_ptr, 0u8, 0u8)], c0_ptr, 1.0, 1.0);
        backup(&[(root_ptr, 0u8, 0u8)], c0_ptr, 2.0, 2.0);
        backup(&[(root_ptr, 1u8, 0u8)], c1_ptr, 3.0, 3.0);
        backup(&[(root_ptr, 1u8, 0u8)], c1_ptr, 4.0, 4.0);
        backup(&[(root_ptr, 2u8, 0u8)], c2_ptr, 5.0, 5.0);

        let edge_sum: u32 = (0..5).map(|i| unsafe { root_ptr.as_ref() }.p1.edge(i).visits).sum();
        assert_eq!(edge_sum, unsafe { root_ptr.as_ref() }.total_visits());
    }

    #[test]
    fn backup_value_mixing() {
        // Raw leaf values propagated upward: q = edge_r + raw_v (not running avg).
        // g=[2,4,6], edge_r=1.0 → Q values = [3, 5, 7] → root.v1 = mean(3,5,7) = 5.0
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(1.0, 0.0);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        backup(&[(root_ptr, 0u8, 0u8)], child_ptr, 2.0, 0.0);
        backup(&[(root_ptr, 0u8, 0u8)], child_ptr, 4.0, 0.0);
        backup(&[(root_ptr, 0u8, 0u8)], child_ptr, 6.0, 0.0);

        // Root Q: edge_r(1.0) + raw_v for each backup = [3.0, 5.0, 7.0]
        // mean = 5.0
        assert!(
            (unsafe { root_ptr.as_ref() }.v1() - 5.0).abs() < 1e-5,
            "root v1={} expected=5.0",
            unsafe { root_ptr.as_ref() }.v1(),
        );
    }

    #[test]
    fn backup_asymmetric_rewards() {
        // P1 collects cheese (r1=1.0), P2 doesn't (r2=0.0).
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(1.0, 0.0);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        backup(&[(root_ptr, 0u8, 0u8)], child_ptr, 2.0, 3.0);

        // Root q1 = 1.0 + 2.0 = 3.0, q2 = 0.0 + 3.0 = 3.0
        assert!((unsafe { root_ptr.as_ref() }.v1() - 3.0).abs() < 1e-6);
        assert!((unsafe { root_ptr.as_ref() }.v2() - 3.0).abs() < 1e-6);

        // Edge P1: q=3.0. Edge P2: q=3.0.
        assert!((unsafe { root_ptr.as_ref() }.p1.edge(0).q - 3.0).abs() < 1e-6);
        assert!((unsafe { root_ptr.as_ref() }.p2.edge(0).q - 3.0).abs() < 1e-6);
    }

    // ---- PUCT selection ----

    #[test]
    fn puct_monotonic_q() {
        // Higher Q → selected.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);

        // Give the node a visit so total_visits > 0.
        node.update_value(0.0, 0.0);

        // Set up edges: outcome 2 has much higher Q.
        for i in 0..5 {
            node.p1.edge_mut(i).update(1.0);
        }
        // Give outcome 2 extra high-Q updates.
        for _ in 0..10 {
            node.p1.edge_mut(2).update(10.0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);
        assert_eq!(a1, 2, "Should select highest-Q outcome");
    }

    #[test]
    fn puct_monotonic_prior() {
        // Higher prior → selected when all else equal.
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);
        node.update_value(0.0, 0.0);

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);
        assert_eq!(a1, 2, "Should select highest-prior outcome");
    }

    #[test]
    fn puct_exploration_positive() {
        // PUCT score > Q_norm when prior > 0 and there are visits.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);

        // Give multiple visits so sqrt(total_visits) > 0.
        for _ in 0..10 {
            node.update_value(2.0, 2.0);
        }

        // Visit outcome 0 a few times.
        for _ in 0..5 {
            node.p1.edge_mut(0).update(2.0);
        }

        let config = default_config();
        let edge = node.p1.edge(0);
        let vs = node.value_scale();
        let q_norm = edge.q / vs;
        let cv = node.children_visits().max(1);
        let exploration = config.c_puct * 0.2 * (cv as f32).sqrt() / (1.0 + edge.visits as f32);

        assert!(exploration > 0.0, "Exploration bonus should be positive");
        assert!(q_norm + exploration > q_norm, "PUCT > Q_norm");
    }

    #[test]
    fn puct_unvisited_selected() {
        // With enough visits on other outcomes, an unvisited one should be pulled by exploration.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);

        // Give many visits — FPU becomes relevant.
        for _ in 0..100 {
            node.update_value(2.0, 2.0);
        }

        // Visit outcomes 0-3 heavily but leave outcome 4 unvisited.
        for i in 0..4 {
            for _ in 0..25 {
                node.p1.edge_mut(i).update(2.0);
            }
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);
        // Outcome 4 has 0 visits → huge exploration bonus → selected.
        assert_eq!(a1, 4, "Unvisited outcome should be selected");
    }

    #[test]
    fn puct_fpu_pessimism() {
        // Unvisited outcome Q = v - fpu_reduction * value_scale * sqrt(visited_mass).
        // More visited mass → more pessimistic FPU → lower score for unvisited.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);

        // Set node value.
        node.update_value(3.0, 3.0);

        let config = default_config();
        let v = node.v1();
        let vs = node.value_scale();

        // Visit outcome 0 — now visited_prior_mass = 0.2.
        node.p1.edge_mut(0).update(3.0);

        let fpu_one = v - config.fpu_reduction * vs * (0.2f32).sqrt();

        // Visit outcomes 0 and 1 — visited_prior_mass = 0.4.
        node.p1.edge_mut(1).update(3.0);

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
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);
        assert_eq!(a1, 2, "With no edge visits, should select by prior");
    }

    #[test]
    fn puct_forced_fires() {
        // At root, an undervisited outcome with prior > 0 gets force-boosted.
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);

        // Many visits, heavily on outcomes 0 and 1.
        for _ in 0..100 {
            node.update_value(2.0, 2.0);
        }
        for _ in 0..45 {
            node.p1.edge_mut(0).update(2.0);
        }
        for _ in 0..45 {
            node.p1.edge_mut(1).update(2.0);
        }
        // Outcome 2 has prior 0.1, visited once.
        node.p1.edge_mut(2).update(2.0);
        // Outcomes 3, 4 have prior 0.05, visited once each.
        node.p1.edge_mut(3).update(2.0);
        node.p1.edge_mut(4).update(2.0);

        // Threshold for outcome 2: sqrt(2.0 * 0.1 * 100) = sqrt(20) ≈ 4.47
        // Visits = 1 < 4.47 → forced
        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, true, &mut r);

        // One of the undervisited outcomes should be boosted.
        // Outcome 2 has highest prior among undervisited → highest threshold.
        // But all undervisited get 1e20, so tie-break is by RNG. Just check it's undervisited.
        assert!(
            node.p1.edge(a1 as usize).visits < 5,
            "Forced playout should select an undervisited outcome, got outcome {} with {} visits",
            a1,
            node.p1.edge(a1 as usize).visits
        );
    }

    #[test]
    fn puct_forced_not_at_nonroot() {
        // Same node shape as puct_forced_fires but is_root=false → no forced boost.
        // Give undervisited outcomes enough visits that exploration alone won't
        // beat the high-Q outcomes, so the only way they'd win is via forced playouts.
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);

        for _ in 0..1000 {
            node.update_value(2.0, 2.0);
        }
        for _ in 0..450 {
            node.p1.edge_mut(0).update(8.0);
        }
        for _ in 0..450 {
            node.p1.edge_mut(1).update(7.0);
        }
        for _ in 0..40 {
            node.p1.edge_mut(2).update(1.0);
        }
        for _ in 0..30 {
            node.p1.edge_mut(3).update(1.0);
        }
        for _ in 0..30 {
            node.p1.edge_mut(4).update(1.0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);

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
        let p1_prior = [0.1, 0.5, 0.1, 0.1, 0.2];
        let mut node_small = *make_open_node(p1_prior, [0.2; 5]);
        let mut node_large = *make_open_node(p1_prior, [0.2; 5]);

        node_small.set_value_scale(1.0);
        node_large.set_value_scale(100.0);

        // Same raw values and edge pattern.
        for node in [&mut node_small, &mut node_large] {
            node.update_value(5.0, 5.0);
            // Outcome 0: high Q, low prior (0.1).
            node.p1.edge_mut(0).update(8.0);
            // Outcome 1: low Q, high prior (0.5).
            node.p1.edge_mut(1).update(6.0);
        }

        let config = default_config();
        let mut r = rng();

        // Small scale: (8-6)/1 = 2 gap in q_norm. Exploitation wins → outcome 0.
        let (a_small, _) = select_actions(&node_small, &config, false, &mut r);
        assert_eq!(a_small, 0, "Small value_scale: exploitation should select high-Q outcome");

        // Large scale: (8-6)/100 = 0.02 gap. Prior dominates → outcome 1 (prior 0.5).
        let mut r2 = rng();
        let (a_large, _) = select_actions(&node_large, &config, false, &mut r2);
        assert_eq!(a_large, 1, "Large value_scale: exploration should select high-prior outcome");
    }

    #[test]
    fn puct_decoupled() {
        // P2's state shouldn't affect P1's selection.
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];

        // Two nodes: same P1 state, different P2 priors.
        let mut node1 = *make_open_node(p1_prior, [0.2; 5]);
        let mut node2 = *make_open_node(p1_prior, [0.05, 0.8, 0.05, 0.05, 0.05]);

        node1.set_value_scale(5.0);
        node2.set_value_scale(5.0);
        node1.update_value(2.0, 2.0);
        node2.update_value(2.0, 2.0);

        let config = default_config();
        let mut r1 = rng();
        let mut r2 = rng();
        let (a1_first, _) = select_actions(&node1, &config, false, &mut r1);
        let (a1_second, _) = select_actions(&node2, &config, false, &mut r2);

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
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        root_box.set_value_scale(5.0);

        backup(&[], root_ptr, 3.0, 2.0);

        assert_eq!(unsafe { root_ptr.as_ref() }.total_visits(), 1);
        assert!((unsafe { root_ptr.as_ref() }.v1() - 3.0).abs() < 1e-6);
        assert!((unsafe { root_ptr.as_ref() }.v2() - 2.0).abs() < 1e-6);

        // No edges should have been touched.
        for i in 0..5 {
            assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(i).visits, 0);
            assert_eq!(unsafe { root_ptr.as_ref() }.p2.edge(i).visits, 0);
        }
    }

    #[test]
    fn backup_edge_visit_sum_with_nn_eval() {
        // When root gets an NN eval (update_value before any backup),
        // sum(p1.edge.visits) == root.total_visits - 1.
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        root_box.set_value_scale(5.0);

        // Simulate NN eval on root (visit 1, no edge updates).
        root_box.update_value(2.0, 2.0);

        // Build two children, capturing ptrs before moving.
        let mut child_b_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_b_ptr = NodePtr::from_ref(&child_b_box);
        let mut child_a_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_a_ptr = NodePtr::from_ref(&child_a_box);

        child_b_box.set_parent(Some(root_ptr), (1, 1));
        child_b_box.set_edge_rewards(0.0, 0.0);
        child_b_box.set_value_scale(5.0);

        child_a_box.set_parent(Some(root_ptr), (0, 0));
        child_a_box.set_edge_rewards(0.0, 0.0);
        child_a_box.set_value_scale(5.0);
        child_a_box.next_sibling = Some(child_b_box);

        root_box.first_child = Some(child_a_box);

        // 5 backups through the two edges.
        for _ in 0..3 {
            backup(&[(root_ptr, 0u8, 0u8)], child_a_ptr, 1.0, 1.0);
        }
        for _ in 0..2 {
            backup(&[(root_ptr, 1u8, 1u8)], child_b_ptr, 1.0, 1.0);
        }

        let edge_sum: u32 = (0..5).map(|i| unsafe { root_ptr.as_ref() }.p1.edge(i).visits).sum();
        // total_visits = 1 (NN eval) + 5 (backups) = 6, edge_sum = 5.
        assert_eq!(unsafe { root_ptr.as_ref() }.total_visits(), 6);
        assert_eq!(edge_sum, unsafe { root_ptr.as_ref() }.total_visits() - 1);
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
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut mid_box = make_open_node([0.2; 5], [0.2; 5]);
        let mid_ptr = NodePtr::from_ref(&mid_box);
        let mut leaf_box = make_open_node([0.2; 5], [0.2; 5]);
        let leaf_ptr = NodePtr::from_ref(&leaf_box);

        leaf_box.set_parent(Some(mid_ptr), (0, 0));
        leaf_box.set_edge_rewards(0.5, 0.5);
        leaf_box.set_value_scale(15.0);

        mid_box.set_parent(Some(root_ptr), (0, 0));
        mid_box.set_edge_rewards(1.0, 1.0);
        mid_box.set_value_scale(15.0);
        mid_box.first_child = Some(leaf_box);

        root_box.set_value_scale(15.0);
        root_box.first_child = Some(mid_box);

        let path = vec![(root_ptr, 0u8, 0u8), (mid_ptr, 0u8, 0u8)];

        // Backup 1: g=10
        backup(&path, leaf_ptr, 10.0, 10.0);
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 10.0).abs() < 1e-5);
        // mid: q1 = edge_r_leaf(0.5) + raw_v(10.0) = 10.5
        assert!((unsafe { mid_ptr.as_ref() }.v1() - 10.5).abs() < 1e-5);
        // root: q1 = edge_r_mid(1.0) + propagated_v(10.5) = 11.5
        assert!((unsafe { root_ptr.as_ref() }.v1() - 11.5).abs() < 1e-5);

        // Backup 2: g=6
        backup(&path, leaf_ptr, 6.0, 6.0);
        // leaf: mean(10, 6) = 8.0
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 8.0).abs() < 1e-5);
        // mid: q1 = 0.5 + 6.0 = 6.5. mean(10.5, 6.5) = 8.5
        assert!(
            (unsafe { mid_ptr.as_ref() }.v1() - 8.5).abs() < 1e-5,
            "mid.v1={} expected=8.5 (bug would give 9.5)",
            unsafe { mid_ptr.as_ref() }.v1()
        );
        // root: q1 = 1.0 + 6.5 = 7.5. mean(11.5, 7.5) = 9.5
        assert!(
            (unsafe { root_ptr.as_ref() }.v1() - 9.5).abs() < 1e-5,
            "root.v1={} expected=9.5 (bug would give 11.0)",
            unsafe { root_ptr.as_ref() }.v1()
        );
    }

    #[test]
    fn backup_same_edge_raw_propagation() {
        // Edge Q = mean of raw Q values, not stale running averages.
        // 3 backups g=[10,4,7], edge_r=2.0 → edge Q = mean(12, 6, 9) = 9.0
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(2.0, 2.0);
        child_box.set_value_scale(15.0);
        root_box.set_value_scale(15.0);
        root_box.first_child = Some(child_box);

        let path = vec![(root_ptr, 0u8, 0u8)];
        backup(&path, child_ptr, 10.0, 10.0);
        backup(&path, child_ptr, 4.0, 4.0);
        backup(&path, child_ptr, 7.0, 7.0);

        // Edge Q = mean(2+10, 2+4, 2+7) = mean(12, 6, 9) = 9.0
        assert!(
            (unsafe { root_ptr.as_ref() }.p1.edge(0).q - 9.0).abs() < 1e-5,
            "edge Q={} expected=9.0",
            unsafe { root_ptr.as_ref() }.p1.edge(0).q
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
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut a_box = make_open_node([0.2; 5], [0.2; 5]);
        let a_ptr = NodePtr::from_ref(&a_box);
        let mut b_box = make_open_node([0.2; 5], [0.2; 5]);
        let b_ptr = NodePtr::from_ref(&b_box);
        let mut leaf_box = make_open_node([0.2; 5], [0.2; 5]);
        let leaf_ptr = NodePtr::from_ref(&leaf_box);

        // Wire bottom-up.
        leaf_box.set_parent(Some(b_ptr), (0, 0));
        leaf_box.set_edge_rewards(0.25, 0.25);
        leaf_box.set_value_scale(10.0);

        b_box.set_parent(Some(a_ptr), (0, 0));
        b_box.set_edge_rewards(0.5, 0.5);
        b_box.set_value_scale(10.0);
        b_box.first_child = Some(leaf_box);

        a_box.set_parent(Some(root_ptr), (0, 0));
        a_box.set_edge_rewards(1.0, 1.0);
        a_box.set_value_scale(10.0);
        a_box.first_child = Some(b_box);

        root_box.set_value_scale(10.0);
        root_box.first_child = Some(a_box);

        let path = vec![(root_ptr, 0u8, 0u8), (a_ptr, 0u8, 0u8), (b_ptr, 0u8, 0u8)];

        // Backup 1: g=4.0
        backup(&path, leaf_ptr, 4.0, 4.0);
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 4.0).abs() < 1e-5);
        assert!((unsafe { b_ptr.as_ref() }.v1() - 4.25).abs() < 1e-5);
        assert!((unsafe { a_ptr.as_ref() }.v1() - 4.75).abs() < 1e-5);
        assert!((unsafe { root_ptr.as_ref() }.v1() - 5.75).abs() < 1e-5);

        // Backup 2: g=2.0
        backup(&path, leaf_ptr, 2.0, 2.0);
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 3.0).abs() < 1e-5);
        // B: mean(4.25, 0.25+2.0) = mean(4.25, 2.25) = 3.25
        assert!(
            (unsafe { b_ptr.as_ref() }.v1() - 3.25).abs() < 1e-5,
            "B.v1={} expected=3.25",
            unsafe { b_ptr.as_ref() }.v1()
        );
        // A: mean(4.75, 0.5+2.25) = mean(4.75, 2.75) = 3.75
        assert!(
            (unsafe { a_ptr.as_ref() }.v1() - 3.75).abs() < 1e-5,
            "A.v1={} expected=3.75",
            unsafe { a_ptr.as_ref() }.v1()
        );
        // root: mean(5.75, 1.0+2.75) = mean(5.75, 3.75) = 4.75
        assert!(
            (unsafe { root_ptr.as_ref() }.v1() - 4.75).abs() < 1e-5,
            "root.v1={} expected=4.75",
            unsafe { root_ptr.as_ref() }.v1()
        );
    }

    #[test]
    fn backup_p2_independent_propagation() {
        // Asymmetric P1/P2 edge rewards, 2 backups, verifies independent propagation.
        // P1 edge_r=2.0, P2 edge_r=0.5
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(2.0, 0.5);
        child_box.set_value_scale(10.0);
        root_box.set_value_scale(10.0);
        root_box.first_child = Some(child_box);

        let path = vec![(root_ptr, 0u8, 0u8)];

        // Backup 1: g1=3, g2=7
        backup(&path, child_ptr, 3.0, 7.0);
        // root: q1 = 2.0 + 3.0 = 5.0, q2 = 0.5 + 7.0 = 7.5
        assert!((unsafe { root_ptr.as_ref() }.v1() - 5.0).abs() < 1e-5);
        assert!((unsafe { root_ptr.as_ref() }.v2() - 7.5).abs() < 1e-5);

        // Backup 2: g1=1, g2=5
        backup(&path, child_ptr, 1.0, 5.0);
        // root: q1 = 2.0 + 1.0 = 3.0. mean(5.0, 3.0) = 4.0
        // root: q2 = 0.5 + 5.0 = 5.5. mean(7.5, 5.5) = 6.5
        assert!(
            (unsafe { root_ptr.as_ref() }.v1() - 4.0).abs() < 1e-5,
            "root.v1={} expected=4.0",
            unsafe { root_ptr.as_ref() }.v1()
        );
        assert!(
            (unsafe { root_ptr.as_ref() }.v2() - 6.5).abs() < 1e-5,
            "root.v2={} expected=6.5",
            unsafe { root_ptr.as_ref() }.v2()
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

        let mut node = Node::new(p1, p2);
        node.set_value_scale(5.0);
        node.update_value(1.0, 1.0);

        let config = default_config();
        let mut r = rng();
        let (a1, a2) = select_actions(&node, &config, false, &mut r);
        assert_eq!(a1, 0);
        assert_eq!(a2, 0);
    }

    // ---- Virtual loss: PUCT ----

    #[test]
    fn puct_virtual_loss_diversifies() {
        // Heavy virtual loss on best edge shifts selection to a different outcome.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();

        // First selection with no virtual loss — pick baseline.
        let (baseline, _) = select_actions(&node, &config, false, &mut r);

        // Add heavy virtual loss on the baseline outcome.
        for _ in 0..100 {
            node.p1.edge_mut(baseline as usize).add_virtual_loss();
        }

        let mut r2 = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r2);
        assert_ne!(a1, baseline, "Virtual loss should shift selection away");
    }

    #[test]
    fn puct_no_virtual_loss_unchanged() {
        // When all n_in_flight == 0, behavior is identical to before.
        let p1_prior = [0.05, 0.05, 0.7, 0.1, 0.1];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);
        node.update_value(2.0, 2.0);

        // Verify no edge has in-flight.
        for i in 0..5 {
            assert_eq!(node.p1.edge(i).n_in_flight(), 0);
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);
        // Same as puct_monotonic_prior: highest prior wins.
        assert_eq!(a1, 2);
    }

    #[test]
    fn puct_virtual_loss_unvisited_still_fpu() {
        // Unvisited edge with virtual loss gets FPU, not edge.q=0.
        // FPU uses edge.visits (real visits), not n_in_flight.
        let mut node = *make_open_node([0.2; 5], [0.2; 5]);
        node.set_value_scale(5.0);
        node.update_value(5.0, 5.0);

        // Visit some edges with low Q so unvisited FPU-based edges dominate.
        for i in 0..3 {
            node.p1.edge_mut(i).update(1.0);
        }

        // Add virtual loss on unvisited edge 3. It should still get FPU (visits==0).
        node.p1.edge_mut(3).add_virtual_loss();

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, false, &mut r);

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
                &node.p1, node.v1(), node.value_scale(), node.children_visits(),
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
                &node.p1, node.v1(), node.value_scale(), node.children_visits(),
                &config, false, &mut r,
            );
            node.p1.edge_mut(a1 as usize).add_virtual_loss();
            vl_edges.push(a1);
        }

        // Revert everything.
        for a in &vl_edges {
            node.p1.edge_mut(*a as usize).revert_virtual_loss();
            node.cancel_score_update(1);
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
                &node.p1, node.v1(), node.value_scale(), node.children_visits(),
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
        let p1_prior = [0.4, 0.4, 0.1, 0.05, 0.05];
        let mut node = *make_open_node(p1_prior, [0.2; 5]);
        node.set_value_scale(5.0);

        // Many visits, heavily on 0 and 1.
        for _ in 0..100 {
            node.update_value(2.0, 2.0);
        }
        for _ in 0..45 {
            node.p1.edge_mut(0).update(2.0);
        }
        for _ in 0..45 {
            node.p1.edge_mut(1).update(2.0);
        }
        node.p1.edge_mut(2).update(2.0);
        node.p1.edge_mut(3).update(2.0);
        node.p1.edge_mut(4).update(2.0);

        // Add large virtual loss on the undervisited outcomes.
        // Threshold for outcome 2: sqrt(2.0 * 0.1 * 100) ≈ 4.47. Visits=1 < 4.47 → forced.
        // Even with virtual loss, forced playouts should still fire (they check real visits).
        for _ in 0..50 {
            node.p1.edge_mut(2).add_virtual_loss();
            node.p1.edge_mut(3).add_virtual_loss();
            node.p1.edge_mut(4).add_virtual_loss();
        }

        let config = default_config();
        let mut r = rng();
        let (a1, _) = select_actions(&node, &config, true, &mut r);

        // Forced playouts should still fire on undervisited outcomes.
        assert!(
            node.p1.edge(a1 as usize).visits < 5,
            "Forced playout should still select undervisited outcome despite virtual loss, got outcome {} with {} visits",
            a1,
            node.p1.edge(a1 as usize).visits
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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 1, 1, &mut r).unwrap();

        assert_eq!(result.total_visits, 1);
        // SmartUniform values are 0 → root value should be 0.
        assert!((result.value_p1).abs() < 1e-6);
        assert!((result.value_p2).abs() < 1e-6);
        // Root priors should be set.
        assert!(tree.root_node().total_visits() >= 1);
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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 2, 1, &mut r).unwrap();

        assert_eq!(result.total_visits, 2);

        // Root should have at least one child.
        let root = tree.root_node();
        assert!(root.first_child().is_some());

        // Walk children: at least one with visits=1.
        let mut found_child_with_visit = false;
        let mut cur = root.first_child();
        while let Some(child_ptr) = cur {
            if unsafe { child_ptr.as_ref() }.total_visits() == 1 {
                found_child_with_visit = true;
            }
            cur = unsafe { child_ptr.as_ref() }.next_sibling();
        }
        assert!(found_child_with_visit, "Should have a child with 1 visit");
    }

    /// Walk the entire subtree rooted at `ptr`, calling `f` on each node.
    fn walk_tree(ptr: NodePtr, f: &mut impl FnMut(&Node)) {
        let node = unsafe { ptr.as_ref() };
        f(node);
        let mut child = node.first_child();
        while let Some(c) = child {
            walk_tree(c, f);
            child = unsafe { c.as_ref() }.next_sibling();
        }
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
        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 1, &mut r).unwrap();
        assert_eq!(result.total_visits, 50);

        // Walk all nodes: for visited interior nodes,
        // sum(p1.edge.visits) == total_visits - 1 (the -1 is the NN eval).
        let mut node_index = 0usize;
        walk_tree(tree.root(), &mut |node| {
            let i = node_index;
            node_index += 1;
            if node.total_visits() == 0 {
                return;
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
        });
    }

    // 4. corridor: linear maze, policy weights the real moves
    #[test]
    fn search_corridor() {
        let game = test_util::corridor_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 4, &mut r).unwrap();

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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 100, 4, &mut r).unwrap();

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
        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 1, &mut r).unwrap();

        // Should complete without panics. 3-turn game means terminals appear.
        // OOO may add free terminal visits beyond n_sims.
        assert!(result.total_visits >= 50, "expected >= 50, got {}", result.total_visits);

        // Walk nodes: any terminal node should have is_terminal set.
        walk_tree(tree.root(), &mut |node| {
            if node.is_terminal() {
                // Terminal nodes should have 0 for edge visits (no children).
                assert!(node.first_child().is_none(), "Terminal node shouldn't have children");
            }
        });
    }

    // 7. terminal_root: already game-over
    #[test]
    fn search_terminal_root() {
        let game = test_util::terminal_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 10, 4, &mut r).unwrap();

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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r).unwrap();

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
            let result = run_search(&mut tree, game, &BACKEND, &config, 30, 4, &mut r).unwrap();

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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 30, 4, &mut r).unwrap();

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

        let result = run_search(&mut tree, &game, &BACKEND, &config, 100, 4, &mut r).unwrap();

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

        let _ = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r).unwrap();

        // For each child of root, replay the move and verify effective actions match.
        let root_node = tree.root_node();

        let mut cur = root_node.first_child();
        while let Some(child_ptr) = cur {
            let child = unsafe { child_ptr.as_ref() };
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

        let _ = run_search(&mut tree, &game, &BACKEND, &config, 30, 4, &mut r).unwrap();

        let mut node_index = 0usize;
        walk_tree(tree.root(), &mut |node| {
            let i = node_index;
            node_index += 1;

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
        });
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

        // batch_size=10 but only n_sims=3. With collision-budgeted gathering
        // (LC0 pattern), small trees get budget=1 per pick call. The outer
        // loop retries, so we produce ~3 actual visits.
        let result = run_search(&mut tree, &game, &BACKEND, &config, 3, 10, &mut r).unwrap();

        assert!(
            result.total_visits >= 3,
            "n_sims=3 should produce at least 3 visits, got {}",
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

        let result = run_search(&mut tree, &game, &backend, &config, 50, 1, &mut r).unwrap();

        // OOO may add free terminal visits beyond n_sims.
        assert!(result.total_visits >= 50, "expected >= 50, got {}", result.total_visits);
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

        let _ = run_search(&mut tree, &game, &backend, &config, 30, 1, &mut r).unwrap();

        let root = tree.root_node();
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

        let result = run_search(&mut tree, &game, &backend, &config, 50, 1, &mut r).unwrap();
        assert_eq!(result.total_visits, 50);

        let mut node_index = 0usize;
        walk_tree(tree.root(), &mut |node| {
            let i = node_index;
            node_index += 1;
            if node.total_visits() == 0 {
                return;
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
        });
    }

    // ---- Dirichlet noise ----

    #[test]
    fn noise_disabled_priors_unchanged() {
        // epsilon=0 → priors should be untouched.
        let prior_5 = [0.1, 0.3, 0.2, 0.15, 0.25];
        let effective = [0, 1, 2, 3, 4];
        let mut half = HalfNode::new(prior_5, effective);

        let original: Vec<f32> = (0..5).map(|i| half.prior(i)).collect();
        apply_dirichlet_noise(&mut half, 0.0, 10.83, &mut rng());

        for i in 0..5 {
            assert!(
                (half.prior(i) - original[i]).abs() < 1e-6,
                "Prior {i} changed with epsilon=0"
            );
        }
    }

    #[test]
    fn noise_enabled_priors_modified() {
        // epsilon>0 → priors should differ from original.
        let prior_5 = [0.2; 5];
        let effective = [0, 1, 2, 3, 4];
        let mut half = HalfNode::new(prior_5, effective);

        let original: Vec<f32> = (0..5).map(|i| half.prior(i)).collect();
        apply_dirichlet_noise(&mut half, 0.25, 10.83, &mut rng());

        let mut any_changed = false;
        for i in 0..5 {
            if (half.prior(i) - original[i]).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(any_changed, "At least one prior should change with noise");

        // Sum should still be close to 1.0.
        let total: f32 = (0..half.n_outcomes()).map(|i| half.prior(i)).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "Noisy priors should sum to ~1.0, got {}",
            total
        );
    }

    #[test]
    fn noise_deterministic_with_seed() {
        // Same seed → same noise.
        let prior_5 = [0.2; 5];
        let effective = [0, 1, 2, 3, 4];

        let mut half1 = HalfNode::new(prior_5, effective);
        let mut half2 = HalfNode::new(prior_5, effective);

        apply_dirichlet_noise(&mut half1, 0.25, 10.83, &mut SmallRng::seed_from_u64(99));
        apply_dirichlet_noise(&mut half2, 0.25, 10.83, &mut SmallRng::seed_from_u64(99));

        for i in 0..5 {
            assert!(
                (half1.prior(i) - half2.prior(i)).abs() < 1e-6,
                "Same seed should give same noise at outcome {}",
                i
            );
        }
    }

    #[test]
    fn noise_single_outcome_noop() {
        // n_outcomes=1 → noise is a no-op.
        let prior_5 = [0.2; 5];
        let effective = [4, 4, 4, 4, 4]; // all → STAY
        let mut half = HalfNode::new(prior_5, effective);
        assert_eq!(half.n_outcomes(), 1);

        let original = half.prior(0);
        apply_dirichlet_noise(&mut half, 0.25, 10.83, &mut rng());

        assert!(
            (half.prior(0) - original).abs() < 1e-6,
            "Single-outcome node priors should be unchanged"
        );
    }

    #[test]
    fn noise_search_integration() {
        // Run search with noise enabled, verify root priors differ from
        // SmartUniform baseline.
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );

        // Without noise.
        let mut tree_no_noise = MCTSTree::new(&game);
        let config_no_noise = default_config();
        let mut r1 = SmallRng::seed_from_u64(42);
        let result_no_noise = run_search(
            &mut tree_no_noise, &game, &BACKEND, &config_no_noise, 50, 1, &mut r1,
        ).unwrap();

        // With noise.
        let mut tree_noise = MCTSTree::new(&game);
        let config_noise = SearchConfig {
            noise_epsilon: 0.25,
            noise_concentration: 10.83,
            ..default_config()
        };
        let mut r2 = SmallRng::seed_from_u64(42);
        let result_noise = run_search(
            &mut tree_noise, &game, &BACKEND, &config_noise, 50, 1, &mut r2,
        ).unwrap();

        // Priors should differ because noise was injected.
        let mut any_prior_diff = false;
        for i in 0..5 {
            if (result_noise.prior_p1[i] - result_no_noise.prior_p1[i]).abs() > 1e-6 {
                any_prior_diff = true;
            }
        }
        assert!(any_prior_diff, "Root priors should differ with noise enabled");

        // Policies should still sum to 1.
        let sum: f32 = result_noise.policy_p1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Noisy policy should sum to ~1.0, got {}", sum);
    }

    // ---- LC0-style batch allocation tests ----

    // LC0-style batch allocation: in a short game, terminals absorb visits
    // from the batch allocation. Excess visits to already-claimed terminals
    // become collisions, so total_visits may be less than n_sims.
    #[test]
    fn batch_short_game_completes() {
        // short_game: 3 turns, 1 cheese. Terminals appear at depth 3.
        let game = test_util::short_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // Should complete without panics. Total visits may be less than 50
        // because collisions at terminals don't produce new visits.
        let result = run_search(&mut tree, &game, &BACKEND, &config, 50, 8, &mut r).unwrap();

        assert!(result.total_visits > 0, "Should have some visits");
        // n_in_flight should be zero after search.
        walk_tree(tree.root(), &mut |node| {
            assert_eq!(
                node.n_in_flight(), 0,
                "n_in_flight should be 0 after search, got {} on node with {} visits",
                node.n_in_flight(), node.total_visits()
            );
        });
    }

    // LC0-style: batch on unvisited root claims it once, rest become collisions.
    #[test]
    fn batch_unvisited_root_one_eval() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // batch_size=10 on unvisited root: pick_nodes_to_extend claims root
        // once (NeedsEval), remaining 9 become shared collisions.
        let result = run_search(&mut tree, &game, &BACKEND, &config, 1, 10, &mut r).unwrap();

        assert_eq!(
            result.total_visits, 1,
            "Should have exactly 1 visit (1 NN eval). Got {}",
            result.total_visits
        );
    }

    // OOO: with batch_size=1, each batch is exactly 1 descent — no retries
    // possible. Behavior is identical to the old fixed gather loop.
    #[test]
    fn ooo_no_change_batch_size_1() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 20, 1, &mut r).unwrap();

        assert_eq!(
            result.total_visits, 20,
            "batch_size=1: each batch is 1 descent, 20 sims = 20 visits. Got {}",
            result.total_visits
        );
    }

    // ---- error propagation ----

    /// Backend that always returns an error on evaluate_batch.
    struct FailingBackend;

    impl Backend for FailingBackend {
        fn evaluate(
            &self,
            _game: &GameState,
        ) -> Result<crate::backend::EvalResult, BackendError> {
            Err(BackendError::msg("intentional test failure"))
        }
    }

    #[test]
    fn failing_backend_propagates_through_run_search() {
        let game = test_util::open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &FailingBackend, &config, 10, 4, &mut r);
        assert!(result.is_err(), "run_search should propagate backend errors");
        assert!(
            result.unwrap_err().to_string().contains("intentional test failure"),
            "error message should be preserved"
        );
    }

    // ---- backup_and_finalize: direct tests ----

    #[test]
    fn backup_finalize_single_level() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 1));
        child_box.set_edge_rewards(1.0, 0.5);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        // Simulate the n_in_flight and edge VL that pick_nodes_to_extend would set.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(1);
        unsafe { child_ptr.as_mut() }.increment_n_in_flight(1);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(0).add_virtual_loss();
        unsafe { root_ptr.as_mut() }.p2.edge_mut(1).add_virtual_loss();

        backup_and_finalize(child_ptr, 3.0, 2.0, 1);

        // Values: same as backup_single_level.
        let child = unsafe { child_ptr.as_ref() };
        assert_eq!(child.total_visits(), 1);
        assert!((child.v1() - 3.0).abs() < 1e-6);
        assert!((child.v2() - 2.0).abs() < 1e-6);

        let root = unsafe { root_ptr.as_ref() };
        assert_eq!(root.total_visits(), 1);
        assert!((root.v1() - 4.0).abs() < 1e-6); // 1.0 + 3.0
        assert!((root.v2() - 2.5).abs() < 1e-6); // 0.5 + 2.0

        // n_in_flight and edge VL all cleaned up.
        assert_eq!(child.n_in_flight(), 0);
        assert_eq!(root.n_in_flight(), 0);
        assert_eq!(root.p1.edge(0).n_in_flight(), 0);
        assert_eq!(root.p2.edge(1).n_in_flight(), 0);
        assert_eq!(root.p1.edge(0).visits, 1);
        assert_eq!(root.p2.edge(1).visits, 1);
    }

    #[test]
    fn backup_finalize_two_level() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut mid_box = make_open_node([0.2; 5], [0.2; 5]);
        let mid_ptr = NodePtr::from_ref(&mid_box);
        let mut leaf_box = make_open_node([0.2; 5], [0.2; 5]);
        let leaf_ptr = NodePtr::from_ref(&leaf_box);

        leaf_box.set_parent(Some(mid_ptr), (1, 2));
        leaf_box.set_edge_rewards(0.5, 1.0);
        leaf_box.set_value_scale(5.0);
        mid_box.set_parent(Some(root_ptr), (0, 0));
        mid_box.set_edge_rewards(1.0, 0.5);
        mid_box.set_value_scale(5.0);
        mid_box.first_child = Some(leaf_box);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(mid_box);

        // Set n_in_flight chain and edge VL.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(1);
        unsafe { mid_ptr.as_mut() }.increment_n_in_flight(1);
        unsafe { leaf_ptr.as_mut() }.increment_n_in_flight(1);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(0).add_virtual_loss();
        unsafe { root_ptr.as_mut() }.p2.edge_mut(0).add_virtual_loss();
        unsafe { mid_ptr.as_mut() }.p1.edge_mut(1).add_virtual_loss();
        unsafe { mid_ptr.as_mut() }.p2.edge_mut(2).add_virtual_loss();

        backup_and_finalize(leaf_ptr, 2.0, 3.0, 1);

        // Values: same as backup_two_level_q_chain.
        assert!((unsafe { leaf_ptr.as_ref() }.v1() - 2.0).abs() < 1e-6);
        assert!((unsafe { mid_ptr.as_ref() }.v1() - 2.5).abs() < 1e-6); // 0.5 + 2.0
        assert!((unsafe { root_ptr.as_ref() }.v1() - 3.5).abs() < 1e-6); // 1.0 + 2.5

        // All n_in_flight and edge VL cleaned up.
        assert_eq!(unsafe { leaf_ptr.as_ref() }.n_in_flight(), 0);
        assert_eq!(unsafe { mid_ptr.as_ref() }.n_in_flight(), 0);
        assert_eq!(unsafe { root_ptr.as_ref() }.n_in_flight(), 0);
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(0).n_in_flight(), 0);
        assert_eq!(unsafe { mid_ptr.as_ref() }.p1.edge(1).n_in_flight(), 0);
    }

    #[test]
    fn backup_finalize_multivisit() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        child_box.set_edge_rewards(0.0, 0.0);
        child_box.set_value_scale(5.0);
        root_box.set_value_scale(5.0);
        root_box.first_child = Some(child_box);

        // 3 visits worth of n_in_flight and edge VL.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(3);
        unsafe { child_ptr.as_mut() }.increment_n_in_flight(3);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(0).add_virtual_loss_multi(3);
        unsafe { root_ptr.as_mut() }.p2.edge_mut(0).add_virtual_loss_multi(3);

        backup_and_finalize(child_ptr, 4.0, 2.0, 3);

        let child = unsafe { child_ptr.as_ref() };
        let root = unsafe { root_ptr.as_ref() };
        assert_eq!(child.total_visits(), 3);
        assert_eq!(root.total_visits(), 3);
        // All same value → result is that value.
        assert!((child.v1() - 4.0).abs() < 1e-6);
        assert!((root.v1() - 4.0).abs() < 1e-6); // edge_r=0 + 4.0
        // All cleaned up.
        assert_eq!(child.n_in_flight(), 0);
        assert_eq!(root.n_in_flight(), 0);
        assert_eq!(root.p1.edge(0).n_in_flight(), 0);
        assert_eq!(root.p2.edge(0).n_in_flight(), 0);
        assert_eq!(root.p1.edge(0).visits, 3);
    }

    // ---- build_gather_level: visit distribution ----

    #[test]
    fn pick_distribute_uniform_priors() {
        let mut node_box = make_open_node([0.2; 5], [0.2; 5]);
        let node_ptr = NodePtr::from_ref(&node_box);
        node_box.set_value_scale(5.0);
        node_box.update_value(2.0, 2.0); // 1 visit so it's "interior"

        let config = default_config();
        let mut r = rng();
        let level = build_gather_level(node_ptr, 10, &config, false, &mut r);

        // Total distributed = 10.
        let total: u32 = level.vtp.iter().sum();
        assert_eq!(total, 10, "vtp should sum to cur_limit");

        // Multiple pairs should get visits (uniform priors → diversification).
        let non_zero = level.vtp.iter().filter(|&&v| v > 0).count();
        assert!(
            non_zero > 1,
            "Uniform priors should distribute across multiple pairs, got {}",
            non_zero
        );

        // Edge VL written back: sum of p1 edge VL = 10, sum of p2 edge VL = 10.
        let node = unsafe { node_ptr.as_ref() };
        let p1_vl: u32 = (0..node.p1.n_outcomes()).map(|i| node.p1.edge(i).n_in_flight()).sum();
        let p2_vl: u32 = (0..node.p2.n_outcomes()).map(|j| node.p2.edge(j).n_in_flight()).sum();
        assert_eq!(p1_vl, 10, "P1 edge VL should sum to 10");
        assert_eq!(p2_vl, 10, "P2 edge VL should sum to 10");
    }

    #[test]
    fn pick_distribute_dominant_prior() {
        let p1_prior = [0.8, 0.05, 0.05, 0.05, 0.05];
        let mut node_box = make_open_node(p1_prior, [0.2; 5]);
        let node_ptr = NodePtr::from_ref(&node_box);
        node_box.set_value_scale(5.0);
        node_box.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();
        let level = build_gather_level(node_ptr, 20, &config, false, &mut r);

        // Outcome 0 for P1 should get the majority.
        let mut p1_visits = [0u32; 5];
        for a1 in 0..5 {
            for a2 in 0..5 {
                p1_visits[a1] += level.vtp[a1 * 5 + a2];
            }
        }
        assert!(
            p1_visits[0] > 10,
            "Dominant P1 prior should get majority of 20 visits, got {}",
            p1_visits[0]
        );
    }

    #[test]
    fn estimated_vtcb_single_outcome() {
        let stuck = [4, 4, 4, 4, 4];
        let half = HalfNode::new([0.2; 5], stuck);
        assert_eq!(half.n_outcomes(), 1);

        let config = default_config();
        let ns = [0u32; 5];
        let mut r = rng();
        let (best, vtcb) = estimated_visits_to_change_best_half(
            &half, 2.0, 5.0, 1, &config, false, &ns, &mut r,
        );
        assert_eq!(best, 0);
        assert_eq!(vtcb, u32::MAX);
    }

    #[test]
    fn estimated_vtcb_uniform_unvisited() {
        let half = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let config = default_config();
        let ns = [0u32; 5];
        let mut r = rng();
        let (_, vtcb) = estimated_visits_to_change_best_half(
            &half, 2.0, 5.0, 0, &config, false, &ns, &mut r,
        );
        // All tied → after 1 visit to best, the score changes.
        assert_eq!(vtcb, 1, "Uniform tied edges should change after 1 visit");
    }

    // ---- cancel_shared_collisions ----

    #[test]
    fn collision_cancel_depth1() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (2, 3));
        root_box.first_child = Some(child_box);

        // Simulate: 5 visits allocated through edge (2, 3), all collided.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(5);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(2).add_virtual_loss_multi(5);
        unsafe { root_ptr.as_mut() }.p2.edge_mut(3).add_virtual_loss_multi(5);

        cancel_shared_collisions(&[(child_ptr, 5)], root_ptr);

        let root = unsafe { root_ptr.as_ref() };
        assert_eq!(root.n_in_flight(), 0);
        assert_eq!(root.p1.edge(2).n_in_flight(), 0);
        assert_eq!(root.p2.edge(3).n_in_flight(), 0);
    }

    #[test]
    fn collision_cancel_depth2() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut mid_box = make_open_node([0.2; 5], [0.2; 5]);
        let mid_ptr = NodePtr::from_ref(&mid_box);
        let mut leaf_box = make_open_node([0.2; 5], [0.2; 5]);
        let leaf_ptr = NodePtr::from_ref(&leaf_box);

        leaf_box.set_parent(Some(mid_ptr), (1, 0));
        mid_box.set_parent(Some(root_ptr), (0, 0));
        mid_box.first_child = Some(leaf_box);
        root_box.first_child = Some(mid_box);

        // 3 visits through root→mid→leaf, all collided at leaf.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(3);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(0).add_virtual_loss_multi(3);
        unsafe { root_ptr.as_mut() }.p2.edge_mut(0).add_virtual_loss_multi(3);
        unsafe { mid_ptr.as_mut() }.increment_n_in_flight(3);
        unsafe { mid_ptr.as_mut() }.p1.edge_mut(1).add_virtual_loss_multi(3);
        unsafe { mid_ptr.as_mut() }.p2.edge_mut(0).add_virtual_loss_multi(3);

        cancel_shared_collisions(&[(leaf_ptr, 3)], root_ptr);

        assert_eq!(unsafe { root_ptr.as_ref() }.n_in_flight(), 0);
        assert_eq!(unsafe { mid_ptr.as_ref() }.n_in_flight(), 0);
        assert_eq!(unsafe { root_ptr.as_ref() }.p1.edge(0).n_in_flight(), 0);
        assert_eq!(unsafe { mid_ptr.as_ref() }.p1.edge(1).n_in_flight(), 0);
    }

    #[test]
    fn collision_cancel_partial() {
        let mut root_box = make_open_node([0.2; 5], [0.2; 5]);
        let root_ptr = NodePtr::from_ref(&root_box);
        let mut child_box = make_open_node([0.2; 5], [0.2; 5]);
        let child_ptr = NodePtr::from_ref(&child_box);

        child_box.set_parent(Some(root_ptr), (0, 0));
        root_box.first_child = Some(child_box);

        // 5 visits allocated, cancel only 2.
        unsafe { root_ptr.as_mut() }.increment_n_in_flight(5);
        unsafe { root_ptr.as_mut() }.p1.edge_mut(0).add_virtual_loss_multi(5);
        unsafe { root_ptr.as_mut() }.p2.edge_mut(0).add_virtual_loss_multi(5);

        cancel_shared_collisions(&[(child_ptr, 2)], root_ptr);

        let root = unsafe { root_ptr.as_ref() };
        assert_eq!(root.n_in_flight(), 3);
        assert_eq!(root.p1.edge(0).n_in_flight(), 3);
        assert_eq!(root.p2.edge(0).n_in_flight(), 3);
    }

    // ---- Shared-edge VL with multiple children ----

    #[test]
    fn edge_vl_multiple_children_same_p1() {
        // Skew P1 priors so outcome 0 dominates, uniform P2.
        // With enough visits, outcome 0 gets visits with multiple P2 actions.
        let p1_prior = [0.9, 0.025, 0.025, 0.025, 0.025];
        let mut node_box = make_open_node(p1_prior, [0.2; 5]);
        let node_ptr = NodePtr::from_ref(&node_box);
        node_box.set_value_scale(5.0);
        node_box.update_value(2.0, 2.0);

        let config = default_config();
        let mut r = rng();
        let level = build_gather_level(node_ptr, 15, &config, false, &mut r);

        // Compute marginals.
        let mut p1_marginal = [0u32; 5];
        let mut p2_marginal = [0u32; 5];
        for a1 in 0..5 {
            for a2 in 0..5 {
                let v = level.vtp[a1 * 5 + a2];
                p1_marginal[a1] += v;
                p2_marginal[a2] += v;
            }
        }

        let node = unsafe { node_ptr.as_ref() };
        // P1 edge VL should match marginals.
        for i in 0..5 {
            assert_eq!(
                node.p1.edge(i).n_in_flight(), p1_marginal[i],
                "P1 edge {} VL mismatch: got {}, expected {}",
                i, node.p1.edge(i).n_in_flight(), p1_marginal[i]
            );
        }
        // P2 edge VL should match marginals.
        for j in 0..5 {
            assert_eq!(
                node.p2.edge(j).n_in_flight(), p2_marginal[j],
                "P2 edge {} VL mismatch: got {}, expected {}",
                j, node.p2.edge(j).n_in_flight(), p2_marginal[j]
            );
        }
    }

    #[test]
    fn full_cycle_shared_edge_balance() {
        // Integration: batch_size=8, 20 sims on open maze. After search,
        // every node should have n_in_flight=0 and edge VL=0.
        let cheese: Vec<_> = (0..5).map(|i| Coordinates::new(i, 0)).collect();
        let game = test_util::open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &cheese,
        );
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let _result = run_search(&mut tree, &game, &BACKEND, &config, 20, 8, &mut r).unwrap();

        walk_tree(tree.root(), &mut |node| {
            assert_eq!(
                node.n_in_flight(), 0,
                "Node n_in_flight should be 0 after search, got {} (visits={})",
                node.n_in_flight(), node.total_visits()
            );
            for i in 0..node.p1.n_outcomes() {
                assert_eq!(
                    node.p1.edge(i).n_in_flight(), 0,
                    "P1 edge {} n_in_flight should be 0, got {}",
                    i, node.p1.edge(i).n_in_flight()
                );
            }
            for j in 0..node.p2.n_outcomes() {
                assert_eq!(
                    node.p2.edge(j).n_in_flight(), 0,
                    "P2 edge {} n_in_flight should be 0, got {}",
                    j, node.p2.edge(j).n_in_flight()
                );
            }
            // Interior nodes: edge visit sum == total_visits - 1.
            if node.total_visits() > 0 && node.first_child().is_some() {
                let p1_sum: u32 = (0..node.p1.n_outcomes())
                    .map(|i| node.p1.edge(i).visits)
                    .sum();
                assert_eq!(
                    p1_sum,
                    node.total_visits() - 1,
                    "P1 edge visit sum mismatch: {} != {} - 1",
                    p1_sum, node.total_visits()
                );
            }
        });
    }

    // ---- LC0-parity regression tests ----

    /// Assert all nodes and edges have n_in_flight == 0.
    fn assert_tree_clean(root: NodePtr) {
        let mut node_index = 0usize;
        walk_tree(root, &mut |node| {
            let i = node_index;
            node_index += 1;

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
        });
    }

    // Fix 1: terminal root produces exactly 1 visit per batch, not O(batch²).
    #[test]
    fn terminal_root_exact_accounting() {
        let game = test_util::terminal_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        let result = run_search(&mut tree, &game, &BACKEND, &config, 100, 16, &mut r).unwrap();

        assert_eq!(
            result.total_visits, 100,
            "Terminal root: expected exactly 100 visits, got {}",
            result.total_visits
        );
        assert_eq!(
            result.terminals, 100,
            "Terminal root: expected exactly 100 terminals, got {}",
            result.terminals
        );
        assert_eq!(
            result.nn_evals, 0,
            "Terminal root: expected 0 nn_evals, got {}",
            result.nn_evals
        );
        assert!(
            result.collisions > 0,
            "Terminal root: expected collisions > 0 (collision budget drains), got 0"
        );
    }

    // Fix 2: backend error on a warm tree leaves all n_in_flight and edge VL clean.
    #[test]
    fn backend_error_cleanup_warm_tree() {
        let game = test_util::short_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // Warm the tree: build depth so the failing batch exercises interior paths.
        let _ = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r).unwrap();

        // Now fail on a batch with the warm tree.
        let result = simulate_batch(&mut tree, &game, &FailingBackend, &config, 8, &mut r);
        assert!(result.is_err(), "FailingBackend should return an error");

        // Every node and edge must have n_in_flight == 0.
        assert_tree_clean(tree.root());
    }

    // Fix 3: deferred terminal backup — visit accounting proves terminals go
    // through the common backup phase exactly once.
    #[test]
    fn deferred_terminal_backup_visit_accounting() {
        let game = test_util::short_game();
        let mut tree = MCTSTree::new(&game);
        let config = default_config();
        let mut r = search_rng();

        // Warm the tree: terminals appear at depth 3 in short_game.
        let _ = run_search(&mut tree, &game, &BACKEND, &config, 20, 4, &mut r).unwrap();

        let pre_visits = unsafe { tree.root().as_ref() }.total_visits();

        let stats = simulate_batch(&mut tree, &game, &BACKEND, &config, 8, &mut r).unwrap();

        assert!(
            stats.terminals > 0,
            "Expected terminals > 0 on warm short_game tree, got 0"
        );

        let post_visits = unsafe { tree.root().as_ref() }.total_visits();
        assert_eq!(
            post_visits,
            pre_visits + stats.nn_evals + stats.terminals,
            "Visit accounting: post ({post_visits}) != pre ({pre_visits}) + nn ({}) + term ({})",
            stats.nn_evals,
            stats.terminals,
        );

        // All n_in_flight and edge VL must be clean after a successful batch.
        assert_tree_clean(tree.root());
    }
}
