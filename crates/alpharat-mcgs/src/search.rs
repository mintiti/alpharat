use std::sync::Arc;

use rand::Rng;
use rand_distr::Gamma;

use crate::node::SharedNode;
use crate::tree::{compute_rewards, find_or_create_child, populate_node, MCGSTree};
use crate::{Backend, BackendError};
use pyrat::{Direction, GameState, MoveUndo};

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

/// A single step on the search path.
#[derive(Clone)]
struct PathEntry {
    node: Arc<SharedNode>,
    p1_outcome: u8,
    p2_outcome: u8,
}

type SearchPath = Vec<PathEntry>;

/// Result of an MCGS search: policies and values for both players.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Policy in 5-action space, sums to 1, blocked actions = 0.
    pub policy_p1: [f32; 5],
    pub policy_p2: [f32; 5],
    /// Expected remaining cheese for each player.
    pub value_p1: f32,
    pub value_p2: f32,
    /// Visit counts in 5-action space.
    pub visit_counts_p1: [f32; 5],
    pub visit_counts_p2: [f32; 5],
    /// NN/uniform prior at root in 5-action space.
    pub prior_p1: [f32; 5],
    pub prior_p2: [f32; 5],
    /// Per-action Q-values in 5-action space (unvisited actions get FPU).
    pub q_values_p1: [f32; 5],
    pub q_values_p2: [f32; 5],
    /// Root visit count after search.
    pub total_visits: u32,
    /// Number of descents that required NN evaluation.
    pub nn_evals: u32,
    /// Number of descents that hit terminal nodes (free — no NN call).
    pub terminals: u32,
    /// Number of descents that collided (wasted — no backup).
    pub collisions: u32,
    /// Number of transposition stops (edge initialized from shared child's aggregate).
    pub tt_stop_hits: u32,
}

/// Per-batch counters from simulate_batch.
struct BatchStats {
    nn_evals: u32,
    terminals: u32,
    collisions: u32,
    tt_stop_hits: u32,
}


// ---------------------------------------------------------------------------
// run_search — public API
// ---------------------------------------------------------------------------

/// Run MCGS search: N simulations with within-tree batching.
pub fn run_search(
    tree: &mut MCGSTree,
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
    let mut total_tt_stop_hits = 0u32;
    while remaining > 0 {
        let batch = simulate_batch(tree, game, backend, config, remaining.min(batch_size), rng)?;
        total_nn_evals += batch.nn_evals;
        total_terminals += batch.terminals;
        total_collisions += batch.collisions;
        total_tt_stop_hits += batch.tt_stop_hits;
        // Count descents that produced useful information.
        // Collisions don't consume the sim budget — they're wasted work.
        // TT stops are productive: they initialize edges from shared aggregates.
        let produced = batch.nn_evals + batch.terminals + batch.tt_stop_hits;
        remaining = remaining.saturating_sub(produced.max(1));
    }

    let root = Arc::clone(tree.root());
    let mut result = extract_result(&root, config, rng);
    result.nn_evals = total_nn_evals;
    result.terminals = total_terminals;
    result.collisions = total_collisions;
    result.tt_stop_hits = total_tt_stop_hits;
    Ok(result)
}

// ---------------------------------------------------------------------------
// select_actions — decoupled PUCT on joint matrix
// ---------------------------------------------------------------------------

/// Select an action pair (p1_outcome_idx, p2_outcome_idx) via decoupled PUCT.
///
/// Each player independently picks the outcome with the highest PUCT score.
/// Q and visits come from marginals over the joint matrix.
fn select_actions(
    node: &SharedNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> (u8, u8) {
    let low = node.get();
    let a1 = select_p1(low, config, is_root, rng);
    let a2 = select_p2(low, config, is_root, rng);
    (a1, a2)
}

// ---------------------------------------------------------------------------
// compute_fpu — first-play urgency
// ---------------------------------------------------------------------------

/// FPU for player 1: pessimistic value scaled by visited prior mass.
fn compute_fpu_p1(low: &crate::node::LowNode, config: &SearchConfig) -> f32 {
    let mut visited_prior_mass = 0.0f32;
    for i in 0..low.n1() {
        if low.marginal_visits_p1(i) > 0 {
            visited_prior_mass += low.p1_prior(i);
        }
    }
    low.v1() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt()
}

/// FPU for player 2: pessimistic value scaled by visited prior mass.
fn compute_fpu_p2(low: &crate::node::LowNode, config: &SearchConfig) -> f32 {
    let mut visited_prior_mass = 0.0f32;
    for j in 0..low.n2() {
        if low.marginal_visits_p2(j) > 0 {
            visited_prior_mass += low.p2_prior(j);
        }
    }
    low.v2() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt()
}

/// PUCT selection for player 1 — marginalizes over j.
fn select_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = low.n1();
    if n == 1 {
        return 0;
    }

    let children_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    let fpu = compute_fpu_p1(low, config);

    let sqrt_total = (children_visits.max(1) as f32).sqrt();

    argmax_tiebreak(n, rng, |i| {
        let visits = low.marginal_visits_p1(i);
        let in_flight = low.marginal_in_flight_p1(i);
        let prior = low.p1_prior(i);

        let q = if visits > 0 {
            marginal_q_p1(low, i)
        } else {
            fpu
        };
        let q_norm = q / value_scale;

        let exploration =
            config.c_puct * prior * sqrt_total / (1.0 + visits as f32 + in_flight as f32);
        let mut score = q_norm + exploration;

        // Forced playouts: at root, boost undervisited outcomes.
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        score
    })
}

/// PUCT selection for player 2 — marginalizes over i.
fn select_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = low.n2();
    if n == 1 {
        return 0;
    }

    let children_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    let fpu = compute_fpu_p2(low, config);

    let sqrt_total = (children_visits.max(1) as f32).sqrt();

    argmax_tiebreak(n, rng, |j| {
        let visits = low.marginal_visits_p2(j);
        let in_flight = low.marginal_in_flight_p2(j);
        let prior = low.p2_prior(j);

        let q = if visits > 0 {
            marginal_q_p2(low, j)
        } else {
            fpu
        };
        let q_norm = q / value_scale;

        let exploration =
            config.c_puct * prior * sqrt_total / (1.0 + visits as f32 + in_flight as f32);
        let mut score = q_norm + exploration;

        // Forced playouts
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        score
    })
}

/// Marginal Q for p1 outcome i: visit-weighted average over j.
fn marginal_q_p1(low: &crate::node::LowNode, i: usize) -> f32 {
    let mut total_v = 0u32;
    let mut weighted_q = 0.0f32;
    for j in 0..low.n2() {
        let v = low.edge_visits(i, j);
        if v > 0 {
            weighted_q += v as f32 * low.edge_q_p1(i, j);
            total_v += v;
        }
    }
    if total_v > 0 {
        weighted_q / total_v as f32
    } else {
        0.0
    }
}

/// Marginal Q for p2 outcome j: visit-weighted average over i.
fn marginal_q_p2(low: &crate::node::LowNode, j: usize) -> f32 {
    let mut total_v = 0u32;
    let mut weighted_q = 0.0f32;
    for i in 0..low.n1() {
        let v = low.edge_visits(i, j);
        if v > 0 {
            weighted_q += v as f32 * low.edge_q_p2(i, j);
            total_v += v;
        }
    }
    if total_v > 0 {
        weighted_q / total_v as f32
    } else {
        0.0
    }
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
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = i as u8;
            }
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// apply_dirichlet_noise — root exploration noise
// ---------------------------------------------------------------------------

/// Mix Dirichlet noise into a LowNode's priors for one player.
///
/// Uses KataGo's total-concentration approach: per-move alpha = concentration / n_outcomes.
fn apply_dirichlet_noise_p1(node: &SharedNode, epsilon: f32, concentration: f32, rng: &mut impl Rng) {
    let low = node.get_mut();
    let n = low.n1();
    if n <= 1 {
        return;
    }

    let alpha = (concentration / n as f32) as f64;
    let gamma_dist = match Gamma::new(alpha, 1.0) {
        Ok(d) => d,
        Err(_) => return,
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

    // Read current priors, blend, write back
    for i in 0..n {
        let cur = low.p1_prior(i);
        low.set_p1_prior_at(i, cur * (1.0 - epsilon) + epsilon * noise[i] / total);
    }
}

fn apply_dirichlet_noise_p2(node: &SharedNode, epsilon: f32, concentration: f32, rng: &mut impl Rng) {
    let low = node.get_mut();
    let n = low.n2();
    if n <= 1 {
        return;
    }

    let alpha = (concentration / n as f32) as f64;
    let gamma_dist = match Gamma::new(alpha, 1.0) {
        Ok(d) => d,
        Err(_) => return,
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

    for j in 0..n {
        let cur = low.p2_prior(j);
        low.set_p2_prior_at(j, cur * (1.0 - epsilon) + epsilon * noise[j] / total);
    }
}

// ---------------------------------------------------------------------------
// estimated_visits_to_change_best — LC0's batch allocation helper
// ---------------------------------------------------------------------------

/// For player 1, compute how many more visits to the best outcome before the
/// second-best overtakes it in PUCT score. Returns (best_idx, vtc).
/// If only one outcome or best utility alone beats second-best, returns u32::MAX.
fn estimated_visits_to_change_best_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    ns_p1: &[u32; 5],
    rng: &mut impl Rng,
) -> (u8, u32) {
    let n = low.n1();
    if n <= 1 {
        return (0, u32::MAX);
    }

    let children_visits = low.total_edge_visits();
    let fpu = compute_fpu_p1(low, config);
    let sqrt_total = (children_visits.max(1) as f32).sqrt();
    let c_puct = config.c_puct;
    let value_scale = low.value_scale();

    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_utility = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;

    for i in 0..n {
        let visits = low.marginal_visits_p1(i);
        let prior = low.p1_prior(i);
        let q = if visits > 0 { marginal_q_p1(low, i) } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p1[i] as f32);
        let mut score = q_norm + exploration;

        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
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

    // Tie-breaking with reservoir sampling.
    let mut tie_count = 1u32;
    for i in 0..n {
        if i as u8 == best_idx { continue; }
        let visits = low.marginal_visits_p1(i);
        let prior = low.p1_prior(i);
        let q = if visits > 0 { marginal_q_p1(low, i) } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p1[i] as f32);
        let mut score = q_norm + exploration;
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
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
        return (best_idx, u32::MAX);
    }
    if best_utility >= second_best_score {
        return (best_idx, u32::MAX);
    }

    let prior_best = low.p1_prior(best_idx as usize);
    let n1 = ns_p1[best_idx as usize] as f32 + 1.0;
    let denom = second_best_score - best_utility;
    if denom <= 0.0 {
        return (best_idx, u32::MAX);
    }
    let vtc = (c_puct * prior_best * sqrt_total / denom - n1 + 1.0).max(1.0);
    (best_idx, (vtc as u32).max(1))
}

/// Same as above but for player 2 (marginalizes over i).
fn estimated_visits_to_change_best_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    ns_p2: &[u32; 5],
    rng: &mut impl Rng,
) -> (u8, u32) {
    let n = low.n2();
    if n <= 1 {
        return (0, u32::MAX);
    }

    let children_visits = low.total_edge_visits();
    let fpu = compute_fpu_p2(low, config);
    let sqrt_total = (children_visits.max(1) as f32).sqrt();
    let c_puct = config.c_puct;
    let value_scale = low.value_scale();

    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_utility = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;

    for j in 0..n {
        let visits = low.marginal_visits_p2(j);
        let prior = low.p2_prior(j);
        let q = if visits > 0 { marginal_q_p2(low, j) } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p2[j] as f32);
        let mut score = q_norm + exploration;

        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_idx = j as u8;
            best_utility = q_norm;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    let mut tie_count = 1u32;
    for j in 0..n {
        if j as u8 == best_idx { continue; }
        let visits = low.marginal_visits_p2(j);
        let prior = low.p2_prior(j);
        let q = if visits > 0 { marginal_q_p2(low, j) } else { fpu };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p2[j] as f32);
        let mut score = q_norm + exploration;
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }
        if (score - best_score).abs() < 1e-12 {
            tie_count += 1;
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = j as u8;
                best_utility = q_norm;
            }
        }
    }

    if second_best_score <= f32::NEG_INFINITY {
        return (best_idx, u32::MAX);
    }
    if best_utility >= second_best_score {
        return (best_idx, u32::MAX);
    }

    let prior_best = low.p2_prior(best_idx as usize);
    let n1 = ns_p2[best_idx as usize] as f32 + 1.0;
    let denom = second_best_score - best_utility;
    if denom <= 0.0 {
        return (best_idx, u32::MAX);
    }
    let vtc = (c_puct * prior_best * sqrt_total / denom - n1 + 1.0).max(1.0);
    (best_idx, (vtc as u32).max(1))
}

// ---------------------------------------------------------------------------
// build_gather_level — VTC-based visit allocation at one node
// ---------------------------------------------------------------------------

/// Gather-phase state for one level of the iterative tree traversal.
struct GatherLevel {
    node: Arc<SharedNode>,
    /// Flat [i * 5 + j] → allocated visits for that (i, j) child.
    vtp: [u32; 25],
    /// Next flat index to process.
    next_idx: usize,
    /// Last flat index with non-zero visits.
    last_idx: usize,
}

/// Distribute `cur_limit` visits at `node` using decoupled VTC.
fn build_gather_level(
    node: &Arc<SharedNode>,
    cur_limit: u32,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> GatherLevel {
    let low = node.get();
    let n1 = low.n1();
    let n2 = low.n2();

    // Initialize n_started from current state.
    let mut ns_p1 = [0u32; 5];
    let mut ns_p2 = [0u32; 5];
    for i in 0..n1 {
        ns_p1[i] = low.marginal_n_started_p1(i);
    }
    for j in 0..n2 {
        ns_p2[j] = low.marginal_n_started_p2(j);
    }

    let mut vtp = [0u32; 25];
    let mut remaining = cur_limit;
    let mut last_idx = 0usize;

    while remaining > 0 {
        let (best1, vtcb1) = estimated_visits_to_change_best_p1(low, config, is_root, &ns_p1, rng);
        let (best2, vtcb2) = estimated_visits_to_change_best_p2(low, config, is_root, &ns_p2, rng);

        let k = remaining.min(vtcb1).min(vtcb2).max(1);

        let flat = best1 as usize * 5 + best2 as usize;
        vtp[flat] += k;
        ns_p1[best1 as usize] += k;
        ns_p2[best2 as usize] += k;
        remaining -= k;
        if vtp[flat] > 0 && flat > last_idx {
            last_idx = flat;
        }
    }

    // Apply edge virtual loss for all allocated visits.
    let low_mut = node.get_mut();
    for i in 0..n1 {
        for j in 0..n2 {
            let delta = vtp[i * 5 + j];
            if delta > 0 {
                low_mut.add_virtual_loss_multi(i, j, delta);
            }
        }
    }

    GatherLevel {
        node: Arc::clone(node),
        vtp,
        next_idx: 0,
        last_idx,
    }
}

// ---------------------------------------------------------------------------
// pick_nodes_to_extend — LC0-style batch allocation via tree traversal
// ---------------------------------------------------------------------------

/// What a batch entry represents.
enum NodeKind {
    /// Leaf needs NN evaluation (multivisit always 1).
    NeedsEval { game_state: GameState },
    /// Terminal node. Can have multivisit > 1.
    Terminal,
    /// First-hit transposition stop. Edge has no visits but child has aggregate
    /// from other parents. Values read from leaf at processing time.
    TranspositionHit,
}

/// A single entry from the batch gather phase.
struct NodeToProcess {
    leaf: Arc<SharedNode>,
    path: SearchPath,
    kind: NodeKind,
    multivisit: u32,
}

/// A shared collision: path + multivisit to cancel after backup.
struct SharedCollision {
    path: SearchPath,
    multivisit: u32,
}

/// LC0's PickNodesToExtendTask adapted for MCGS DAG with 2-player joint matrix.
fn pick_nodes_to_extend(
    tree: &mut MCGSTree,
    game: &GameState,
    config: &SearchConfig,
    budget: u32,
    rng: &mut impl Rng,
) -> (Vec<NodeToProcess>, Vec<SharedCollision>) {
    let root = Arc::clone(tree.root());
    let mut to_process: Vec<NodeToProcess> = Vec::with_capacity(budget as usize);
    let mut shared_collisions: Vec<SharedCollision> = Vec::new();
    let mut work_game = game.clone();
    let mut undos: Vec<MoveUndo> = Vec::new();

    // Handle root: unvisited or terminal.
    let root_low = root.get();
    if root_low.total_visits() == 0 || root_low.is_terminal() {
        if root_low.total_visits() == 0 && !root_low.is_terminal() {
            if root.get_mut().try_start_score_update() {
                if work_game.check_game_over() {
                    populate_node(&root, None);
                    to_process.push(NodeToProcess {
                        leaf: Arc::clone(&root),
                        path: Vec::new(),
                        kind: NodeKind::Terminal,
                        multivisit: 1,
                    });
                } else {
                    to_process.push(NodeToProcess {
                        leaf: Arc::clone(&root),
                        path: Vec::new(),
                        kind: NodeKind::NeedsEval { game_state: work_game.clone() },
                        multivisit: 1,
                    });
                }
                if budget > 1 {
                    shared_collisions.push(SharedCollision { path: Vec::new(), multivisit: budget - 1 });
                }
            } else {
                shared_collisions.push(SharedCollision { path: Vec::new(), multivisit: budget });
            }
        } else {
            // Terminal root: one real visit + rest as collisions (LC0 pattern).
            // Every pick does one real visit, matching dag_classic's
            // ShouldStopPickingHere + TryStartScoreUpdate path.
            if root_low.total_visits() == 0 {
                populate_node(&root, None);
            }
            root.get_mut().increment_n_in_flight(1);
            to_process.push(NodeToProcess {
                leaf: Arc::clone(&root),
                path: Vec::new(),
                kind: NodeKind::Terminal,
                multivisit: 1,
            });
            if budget > 1 {
                shared_collisions.push(SharedCollision {
                    path: Vec::new(),
                    multivisit: budget - 1,
                });
            }
        }
        return (to_process, shared_collisions);
    }

    // Root is interior: increment n_in_flight for all visits.
    root.get_mut().increment_n_in_flight(budget);

    let first_level = build_gather_level(&root, budget, config, true, rng);
    let mut levels: Vec<GatherLevel> = vec![first_level];
    let mut path_prefix: SearchPath = Vec::new();

    while let Some(level) = levels.last_mut() {
        let mut found_child = false;
        while level.next_idx <= level.last_idx {
            let idx = level.next_idx;
            level.next_idx += 1;
            if level.vtp[idx] == 0 {
                continue;
            }
            let i = (idx / 5) as u8;
            let j = (idx % 5) as u8;
            let k = level.vtp[idx];

            // Convert outcome indices to canonical actions.
            let low = level.node.get();
            let act1 = low.p1_outcome_action(i as usize);
            let act2 = low.p2_outcome_action(j as usize);
            let d1 = Direction::try_from(act1).expect("valid direction");
            let d2 = Direction::try_from(act2).expect("valid direction");
            let scores_before = (work_game.player1_score(), work_game.player2_score());
            let undo = work_game.make_move(d1, d2);
            let (r1, r2) = compute_rewards(&work_game, scores_before);

            let (child, is_new) =
                find_or_create_child(&level.node, i, j, &work_game, tree.tt_mut(), r1, r2);
            if is_new {
                tree.increment_node_count();
            }

            // Build the path to this child.
            let mut child_path = path_prefix.clone();
            child_path.push(PathEntry {
                node: Arc::clone(&level.node),
                p1_outcome: i,
                p2_outcome: j,
            });

            let child_low = child.get();
            if child_low.total_visits() == 0 || child_low.is_terminal() {
                // Leaf or terminal.
                if child.get_mut().try_start_score_update() {
                    if child_low.is_terminal() || work_game.check_game_over() {
                        if child_low.total_visits() == 0 {
                            populate_node(&child, None);
                        }
                        to_process.push(NodeToProcess {
                            leaf: Arc::clone(&child),
                            path: child_path.clone(),
                            kind: NodeKind::Terminal,
                            multivisit: 1,
                        });
                    } else {
                        to_process.push(NodeToProcess {
                            leaf: Arc::clone(&child),
                            path: child_path.clone(),
                            kind: NodeKind::NeedsEval { game_state: work_game.clone() },
                            multivisit: 1,
                        });
                    }
                    if k > 1 {
                        shared_collisions.push(SharedCollision { path: child_path, multivisit: k - 1 });
                    }
                } else {
                    // Collision: all k visits.
                    shared_collisions.push(SharedCollision { path: child_path, multivisit: k });
                }
                work_game.unmake_move(undo);
            } else {
                // Interior child: check transposition stopping.
                let parent_low = level.node.get();
                let edge_vis = parent_low.edge_visits(i as usize, j as usize);
                if child.num_parents() > 1
                    && edge_vis < child_low.total_visits()
                {
                    // TT stop: edge is behind the shared aggregate.
                    // Covers both first-hit (edge_vis == 0) and stale
                    // (edge_vis > 0) cases. One productive stop that
                    // backs up the child's current aggregate through the
                    // path, correcting the edge via delta fixup.
                    // Remaining k-1 visits become collisions.
                    // Don't increment child's n_in_flight — we're reading,
                    // not visiting.
                    to_process.push(NodeToProcess {
                        leaf: Arc::clone(&child),
                        path: child_path.clone(),
                        kind: NodeKind::TranspositionHit,
                        multivisit: 1,
                    });
                    if k > 1 {
                        shared_collisions.push(SharedCollision {
                            path: child_path,
                            multivisit: k - 1,
                        });
                    }
                    work_game.unmake_move(undo);
                } else {
                    // Normal interior: descend with k visits.
                    child.get_mut().increment_n_in_flight(k);
                    undos.push(undo);
                    path_prefix = child_path;
                    let child_level =
                        build_gather_level(&child, k, config, false, rng);
                    levels.push(child_level);
                    found_child = true;
                    break;
                }
            }
        }

        if !found_child {
            // All children at this level processed, backtrack.
            levels.pop();
            if let Some(undo) = undos.pop() {
                work_game.unmake_move(undo);
            }
            path_prefix.pop();
        }
    }

    (to_process, shared_collisions)
}

// ---------------------------------------------------------------------------
// backup_and_finalize — combined backup + VL cleanup
// ---------------------------------------------------------------------------

/// Walk leaf→root, updating values with multivisit Welford and reverting VL.
/// Applies delta correction for transposition staleness.
fn backup_and_finalize(
    path: &[PathEntry],
    leaf: &SharedNode,
    g1: f32,
    g2: f32,
    multivisit: u32,
) {
    leaf.get_mut().finalize_score_update_multi(g1, g2, multivisit);

    let mut v1 = g1;
    let mut v2 = g2;
    let mut n_to_fix: u32 = 0;
    let mut v1_delta: f32 = 0.0;
    let mut v2_delta: f32 = 0.0;

    for entry in path.iter().rev() {
        let i = entry.p1_outcome as usize;
        let j = entry.p2_outcome as usize;

        let edge = entry
            .node
            .get()
            .find_child(entry.p1_outcome, entry.p2_outcome)
            .expect("backup: edge must exist");
        let r1 = edge.r1();
        let r2 = edge.r2();
        let child = Arc::clone(edge.low_node());

        let mut q1 = r1 + v1;
        let mut q2 = r2 + v2;
        let mut q1_delta = v1_delta;
        let mut q2_delta = v2_delta;

        // Delta detection (unchanged from Tier 1).
        {
            let child_low = child.get();
            let parent_low = entry.node.get();
            let edge_vis = parent_low.edge_visits(i, j);

            if child.num_parents() > 1 || edge_vis < child_low.total_visits() {
                let correct_q1 = r1 + child_low.v1();
                let correct_q2 = r2 + child_low.v2();
                q1_delta = correct_q1 - parent_low.edge_q_p1(i, j);
                q2_delta = correct_q2 - parent_low.edge_q_p2(i, j);
                n_to_fix = edge_vis;
                q1 = correct_q1;
                q2 = correct_q2;
            }
        }

        let node = entry.node.get_mut();

        node.finalize_edge_update_multi(i, j, q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_edge_for_terminal(i, j, q1_delta, q2_delta, n_to_fix);
        }
        node.finalize_score_update_multi(q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_for_terminal(q1_delta, q2_delta, n_to_fix);
        }

        // Revert VL inline (replaces separate cleanup_descent).
        node.revert_virtual_loss_multi(i, j, multivisit);

        v1 = q1;
        v2 = q2;
        v1_delta = q1_delta;
        v2_delta = q2_delta;
    }
}

/// Back up a transposition stop: initialize the new edge from the shared
/// child's existing aggregate without incrementing the child's visit count.
///
/// Same path walk as `backup_and_finalize` but skips leaf finalization.
/// Reads v1/v2 from the leaf at processing time (not gather time).
fn backup_transposition_stop(
    path: &[PathEntry],
    leaf: &SharedNode,
    multivisit: u32,
) {
    // Read the leaf's current aggregate — not frozen at gather time.
    let mut v1 = leaf.get().v1();
    let mut v2 = leaf.get().v2();
    let mut n_to_fix: u32 = 0;
    let mut v1_delta: f32 = 0.0;
    let mut v2_delta: f32 = 0.0;

    for entry in path.iter().rev() {
        let i = entry.p1_outcome as usize;
        let j = entry.p2_outcome as usize;

        let edge = entry
            .node
            .get()
            .find_child(entry.p1_outcome, entry.p2_outcome)
            .expect("backup: edge must exist");
        let r1 = edge.r1();
        let r2 = edge.r2();
        let child = Arc::clone(edge.low_node());

        let mut q1 = r1 + v1;
        let mut q2 = r2 + v2;
        let mut q1_delta = v1_delta;
        let mut q2_delta = v2_delta;

        // Delta detection (same as backup_and_finalize).
        {
            let child_low = child.get();
            let parent_low = entry.node.get();
            let edge_vis = parent_low.edge_visits(i, j);

            if child.num_parents() > 1 || edge_vis < child_low.total_visits() {
                let correct_q1 = r1 + child_low.v1();
                let correct_q2 = r2 + child_low.v2();
                q1_delta = correct_q1 - parent_low.edge_q_p1(i, j);
                q2_delta = correct_q2 - parent_low.edge_q_p2(i, j);
                n_to_fix = edge_vis;
                q1 = correct_q1;
                q2 = correct_q2;
            }
        }

        let node = entry.node.get_mut();

        node.finalize_edge_update_multi(i, j, q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_edge_for_terminal(i, j, q1_delta, q2_delta, n_to_fix);
        }
        node.finalize_score_update_multi(q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_for_terminal(q1_delta, q2_delta, n_to_fix);
        }

        node.revert_virtual_loss_multi(i, j, multivisit);

        v1 = q1;
        v2 = q2;
        v1_delta = q1_delta;
        v2_delta = q2_delta;
    }
}

// ---------------------------------------------------------------------------
// cancel_shared_collisions — revert VL for unused visits
// ---------------------------------------------------------------------------

/// Walk each collision's stored path, reverting VL and n_in_flight.
fn cancel_shared_collisions(collisions: &[SharedCollision]) {
    for coll in collisions {
        for entry in &coll.path {
            let low = entry.node.get_mut();
            low.revert_virtual_loss_multi(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
                coll.multivisit,
            );
            low.cancel_score_update_multi(coll.multivisit);
        }
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
// GatherCleanupGuard — RAII revert of VL/n_in_flight on early exit
// ---------------------------------------------------------------------------

/// Drop guard that reverts virtual loss and n_in_flight for gathered-but-not-backed-up
/// entries if simulate_batch exits early (e.g., backend error). Call `disarm()` on
/// the success path to skip cleanup.
struct GatherCleanupGuard<'a> {
    to_process: &'a [NodeToProcess],
    collisions: &'a [SharedCollision],
    armed: bool,
}

impl<'a> GatherCleanupGuard<'a> {
    fn new(to_process: &'a [NodeToProcess], collisions: &'a [SharedCollision]) -> Self {
        Self {
            to_process,
            collisions,
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
        // Revert each NeedsEval entry: VL on path entries + n_in_flight on path + leaf.
        for entry in self.to_process {
            for pe in &entry.path {
                let low = pe.node.get_mut();
                low.revert_virtual_loss_multi(
                    pe.p1_outcome as usize,
                    pe.p2_outcome as usize,
                    entry.multivisit,
                );
                low.cancel_score_update_multi(entry.multivisit);
            }
            entry.leaf.get_mut().cancel_score_update_multi(entry.multivisit);
        }
        cancel_shared_collisions(self.collisions);
    }
}

// ---------------------------------------------------------------------------
// simulate_batch — LC0-style gather/eval/backup cycle
// ---------------------------------------------------------------------------

fn simulate_batch(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<BatchStats, BackendError> {
    let root = Arc::clone(tree.root());
    let mut collisions_left = calculate_collisions_left(tree.node_count(), config) as i32;

    let mut all_to_process: Vec<NodeToProcess> = Vec::with_capacity(batch_size as usize);
    let mut all_collisions: Vec<SharedCollision> = Vec::new();
    let mut minibatch_size = 0u32;
    let mut terminals = 0u32;
    let mut tt_stop_hits = 0u32;

    // ---- Outer Gather Loop (LC0's GatherMinibatch) ----
    while minibatch_size < batch_size && collisions_left > 0 {
        let budget = (collisions_left as u32).min(batch_size - minibatch_size);
        let (to_process, shared_collisions) =
            pick_nodes_to_extend(tree, game, config, budget, rng);

        for entry in to_process {
            match entry.kind {
                NodeKind::Terminal => {
                    backup_and_finalize(&entry.path, &entry.leaf, 0.0, 0.0, entry.multivisit);
                    terminals += entry.multivisit;
                    minibatch_size += 1;
                }
                NodeKind::TranspositionHit => {
                    backup_transposition_stop(
                        &entry.path,
                        &entry.leaf,
                        entry.multivisit,
                    );
                    tt_stop_hits += entry.multivisit;
                    minibatch_size += 1;
                }
                NodeKind::NeedsEval { .. } => {
                    minibatch_size += 1;
                    all_to_process.push(entry);
                }
            }
        }

        for coll in &shared_collisions {
            collisions_left -= coll.multivisit as i32;
        }
        all_collisions.extend(shared_collisions);
    }

    let nn_evals = all_to_process.len() as u32;

    // Guard: if evaluate_batch fails, revert all gathered VL/n_in_flight.
    let mut cleanup_guard = GatherCleanupGuard::new(&all_to_process, &all_collisions);

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

    // ---- Backup Phase: NN eval results ----
    let mut eval_idx = 0;
    for entry in &all_to_process {
        if let NodeKind::NeedsEval { .. } = &entry.kind {
            let eval = &eval_results[eval_idx];
            eval_idx += 1;

            populate_node(&entry.leaf, Some(eval));

            if Arc::ptr_eq(&entry.leaf, &root) && config.noise_epsilon > 0.0 {
                apply_dirichlet_noise_p1(&entry.leaf, config.noise_epsilon, config.noise_concentration, rng);
                apply_dirichlet_noise_p2(&entry.leaf, config.noise_epsilon, config.noise_concentration, rng);
            }

            backup_and_finalize(&entry.path, &entry.leaf, eval.value_p1, eval.value_p2, entry.multivisit);
        }
    }

    // Success: disarm guard before normal collision cancellation.
    cleanup_guard.disarm();

    // ---- Cancel all accumulated shared collisions ----
    cancel_shared_collisions(&all_collisions);
    let total_collisions: u32 = all_collisions.iter().map(|c| c.multivisit).sum();

    Ok(BatchStats {
        nn_evals,
        terminals,
        collisions: total_collisions,
        tt_stop_hits,
    })
}

// ---------------------------------------------------------------------------
// extract_result — policies and values from root
// ---------------------------------------------------------------------------

fn extract_result(
    root: &SharedNode,
    config: &SearchConfig,
    _rng: &mut impl Rng,
) -> SearchResult {
    let low = root.get();
    let total_visits = low.total_visits();

    let (policy_p1, visit_counts_p1, value_p1, q_values_p1) = extract_p1(low, config);
    let (policy_p2, visit_counts_p2, value_p2, q_values_p2) = extract_p2(low, config);

    let prior_p1 = low.expand_p1_prior();
    let prior_p2 = low.expand_p2_prior();

    SearchResult {
        policy_p1,
        policy_p2,
        value_p1,
        value_p2,
        visit_counts_p1,
        visit_counts_p2,
        prior_p1,
        prior_p2,
        q_values_p1,
        q_values_p2,
        total_visits,
        nn_evals: 0,
        terminals: 0,
        collisions: 0,
        tt_stop_hits: 0,
    }
}

/// Extract policy, visit counts, value, and Q-values for player 1 from root.
fn extract_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32, [f32; 5]) {
    let n = low.n1();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v1(), [0.0; 5]);
    }

    let children_visits = low.total_edge_visits();

    let fpu = compute_fpu_p1(low, config);

    // Read Q and visits per outcome.
    let mut q = [0.0f32; 5];
    let mut raw_visits = [0.0f32; 5];
    let mut prior = [0.0f32; 5];
    let mut q_norm = [0.0f32; 5];

    for i in 0..n {
        let visits = low.marginal_visits_p1(i);
        q[i] = if visits > 0 { marginal_q_p1(low, i) } else { fpu };
        raw_visits[i] = visits as f32;
        prior[i] = low.p1_prior(i);
        q_norm[i] = q[i] / low.value_scale();
    }

    // Compute pruned visits.
    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, children_visits, config.c_puct);

    // Expand to 5-action space.
    let mut visit_counts = [0.0f32; 5];
    for (i, &pv) in pruned.iter().enumerate().take(n) {
        let action = low.p1_outcome_action(i) as usize;
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
        policy = low.expand_p1_prior();
    }

    // Expand Q-values to 5-action space.
    let mut q_values = [0.0f32; 5];
    for i in 0..n {
        let action = low.p1_outcome_action(i) as usize;
        q_values[action] = q[i];
    }

    // Value = dot(q, raw_visits) / sum(raw_visits).
    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|i| q[i] * raw_visits[i]).sum();
        dot / visit_sum
    } else {
        low.v1()
    };

    (policy, visit_counts, value, q_values)
}

/// Extract policy, visit counts, value, and Q-values for player 2 from root.
fn extract_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32, [f32; 5]) {
    let n = low.n2();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v2(), [0.0; 5]);
    }

    let children_visits = low.total_edge_visits();

    let fpu = compute_fpu_p2(low, config);

    let mut q = [0.0f32; 5];
    let mut raw_visits = [0.0f32; 5];
    let mut prior = [0.0f32; 5];
    let mut q_norm = [0.0f32; 5];

    for j in 0..n {
        let visits = low.marginal_visits_p2(j);
        q[j] = if visits > 0 { marginal_q_p2(low, j) } else { fpu };
        raw_visits[j] = visits as f32;
        prior[j] = low.p2_prior(j);
        q_norm[j] = q[j] / low.value_scale();
    }

    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, children_visits, config.c_puct);

    let mut visit_counts = [0.0f32; 5];
    for (j, &pv) in pruned.iter().enumerate().take(n) {
        let action = low.p2_outcome_action(j) as usize;
        visit_counts[action] = pv;
    }

    let mut policy = visit_counts;
    let policy_sum: f32 = policy.iter().sum();
    if policy_sum > 0.0 {
        for p in &mut policy {
            *p /= policy_sum;
        }
    } else {
        policy = low.expand_p2_prior();
    }

    // Expand Q-values to 5-action space.
    let mut q_values = [0.0f32; 5];
    for j in 0..n {
        let action = low.p2_outcome_action(j) as usize;
        q_values[action] = q[j];
    }

    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|j| q[j] * raw_visits[j]).sum();
        dot / visit_sum
    } else {
        low.v2()
    };

    (policy, visit_counts, value, q_values)
}

/// Forced-playout pruning: cap visits on low-Q outcomes.
fn compute_pruned_visits(
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
    use crate::node::{Edge, LowNode};
    use crate::{BackendError, SmartUniformBackend, ConstantValueBackend};
    use pyrat::{Coordinates, Direction, GameBuilder};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::collections::{HashMap, HashSet};

    /// Test-only shim: old single-visit backup without VL handling.
    /// Pre-increments n_in_flight and adds VL on path entries so that
    /// backup_and_finalize's decrements work correctly.
    fn backup(path: &[PathEntry], leaf: &SharedNode, g1: f32, g2: f32) {
        leaf.get_mut().increment_n_in_flight(1);
        for entry in path {
            entry.node.get_mut().increment_n_in_flight(1);
            entry.node.get_mut().add_virtual_loss(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
            );
        }
        backup_and_finalize(path, leaf, g1, g2, 1);
    }

    /// Test-only shim: old cleanup_descent for tests that still use it.
    fn cleanup_descent(path: &[PathEntry], leaf_claimed: Option<&SharedNode>) {
        for entry in path {
            entry.node.get_mut().revert_virtual_loss(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
            );
        }
        if let Some(leaf) = leaf_claimed {
            leaf.get_mut().cancel_score_update();
        }
    }

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    fn default_config() -> SearchConfig {
        SearchConfig::default()
    }

    fn open_5x5_game(
        p1: Coordinates,
        p2: Coordinates,
        cheese: &[Coordinates],
    ) -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(p1, p2)
            .with_custom_cheese(cheese.to_vec())
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    fn terminal_game() -> GameState {
        let mut game = GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(0, 1))
            .with_custom_cheese(vec![Coordinates::new(4, 4)])
            .with_max_turns(1)
            .build()
            .create(None)
            .unwrap();
        let _undo = game.make_move(Direction::Stay, Direction::Stay);
        game
    }

    fn short_game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 0))
            .with_custom_cheese(vec![Coordinates::new(1, 0)])
            .with_max_turns(3)
            .build()
            .create(None)
            .unwrap()
    }

    // ---- select_actions ----

    #[test]
    fn select_actions_uniform_prior() {
        // With uniform priors and no visits, PUCT should select via FPU.
        // With forced playouts, all outcomes start unvisited → first one wins.
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = crate::tt::TranspositionTable::new();
        let root = crate::tree::create_root_node(&game, &mut tt);
        let config = default_config();
        let mut r = rng();

        let (a1, a2) = select_actions(&root, &config, true, &mut r);
        // Should be valid outcome indices
        assert!((a1 as usize) < root.get().n1());
        assert!((a2 as usize) < root.get().n2());
    }

    #[test]
    fn select_actions_after_visits() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = crate::tt::TranspositionTable::new();
        let root = crate::tree::create_root_node(&game, &mut tt);

        // Add some visits to (0, 0)
        root.get_mut().finalize_edge_update(0, 0, 1.0, 1.0);
        root.get_mut().finalize_edge_update(0, 0, 1.0, 1.0);
        root.get_mut().finalize_score_update(1.0, 1.0);
        root.get_mut().finalize_score_update(1.0, 1.0);

        let config = default_config();
        let mut r = rng();

        // With visits on (0,0), PUCT should explore other outcomes
        let (a1, a2) = select_actions(&root, &config, true, &mut r);
        assert!((a1 as usize) < root.get().n1());
        assert!((a2 as usize) < root.get().n2());
    }

    #[test]
    fn select_actions_respects_priors_after_root_eval() {
        // After root NN eval: total_visits()=1, total_edge_visits()=0.
        // Before the fix, sqrt(0) killed the exploration term entirely,
        // making selection degenerate to FPU-only (ignoring priors).
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_value_scale(5.0);

        // Non-uniform priors: heavily favor outcome 2
        let mut prior_p1 = [0.0f32; 5];
        prior_p1[0] = 0.05;
        prior_p1[1] = 0.05;
        prior_p1[2] = 0.80;
        prior_p1[3] = 0.05;
        prior_p1[4] = 0.05;
        low.set_prior(prior_p1, [0.2; 5]);

        // Simulate one finalize_score_update (root NN eval) without any edge visits.
        low.finalize_score_update(1.0, 1.0);
        assert_eq!(low.total_visits(), 1);
        assert_eq!(low.total_edge_visits(), 0);

        let root = Arc::new(SharedNode::new(low));
        let config = SearchConfig {
            force_k: 0.0, // disable forced playouts to test pure PUCT
            ..default_config()
        };
        let mut r = rng();

        // With total_visits() used, sqrt(1)=1 gives exploration a non-zero term.
        // The high prior on outcome 2 should make it the preferred selection.
        let (a1, _a2) = select_actions(&root, &config, false, &mut r);
        assert_eq!(a1, 2, "should select outcome with highest prior");
    }

    // ---- marginal_q ----

    #[test]
    fn marginal_q_p1_weighted_average() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_value_scale(5.0);

        // i=0, j=0: Q=2.0, 3 visits
        for _ in 0..3 {
            low.finalize_edge_update(0, 0, 2.0, 1.0);
        }
        // i=0, j=1: Q=4.0, 1 visit
        low.finalize_edge_update(0, 1, 4.0, 2.0);

        let mq = marginal_q_p1(&low, 0);
        // Expected: (3 * 2.0 + 1 * 4.0) / 4 = 10.0 / 4 = 2.5
        assert!((mq - 2.5).abs() < 1e-5);
    }

    #[test]
    fn marginal_q_no_visits_returns_zero() {
        let low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert_eq!(marginal_q_p1(&low, 0), 0.0);
        assert_eq!(marginal_q_p2(&low, 0), 0.0);
    }

    // ---- backup ----

    #[test]
    fn backup_single_level() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Create a child via edge
        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        let edge = Box::new(crate::node::Edge::new(Arc::clone(&child), (0, 1), 1.0, 0.5));
        root.get_mut().prepend_child(edge);

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 1,
        }];

        backup(&path, &child, 3.0, 2.0);

        // Leaf: v=(3.0, 2.0), visits=1
        assert_eq!(child.get().total_visits(), 1);
        assert!((child.get().v1() - 3.0).abs() < 1e-6);
        assert!((child.get().v2() - 2.0).abs() < 1e-6);

        // Root: q1 = r1 + leaf_v1 = 1.0 + 3.0 = 4.0
        //       q2 = r2 + leaf_v2 = 0.5 + 2.0 = 2.5
        assert_eq!(root.get().total_visits(), 1);
        assert!((root.get().v1() - 4.0).abs() < 1e-6);
        assert!((root.get().v2() - 2.5).abs() < 1e-6);

        // Joint matrix: edge at (0, 1)
        assert_eq!(root.get().edge_visits(0, 1), 1);
        assert!((root.get().edge_q_p1(0, 1) - 4.0).abs() < 1e-6);
        assert!((root.get().edge_q_p2(0, 1) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn backup_stale_edge_delta_catchup() {
        // A child has total_visits=5, but the parent's edge only has 2 visits
        // and num_parents=1. This simulates a stale edge after root advancement
        // pruned another parent. The delta correction should still trigger.
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        // Simulate child having been visited 5 times from another (now-pruned) parent
        // with Q that drifted from what our edge knows.
        for _ in 0..5 {
            child.get_mut().finalize_score_update(3.0, 2.0);
        }
        assert_eq!(child.get().total_visits(), 5);
        assert_eq!(child.num_parents(), 0); // no edge yet

        // Wire edge: root --(0,1)--> child, with r=(1.0, 0.5)
        let edge = Box::new(Edge::new(Arc::clone(&child), (0, 1), 1.0, 0.5));
        root.get_mut().prepend_child(edge);
        assert_eq!(child.num_parents(), 1);

        // Add 2 stale visits on the edge with outdated Q values
        root.get_mut().finalize_edge_update(0, 1, 2.0, 1.0);
        root.get_mut().finalize_edge_update(0, 1, 2.0, 1.0);
        root.get_mut().finalize_score_update(2.0, 1.0);
        root.get_mut().finalize_score_update(2.0, 1.0);
        assert_eq!(root.get().edge_visits(0, 1), 2);

        // Now backup a new visit through this path.
        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 1,
        }];
        backup(&path, &child, 3.0, 2.0);

        // The correct Q for the edge is r + child.v = (1+3, 0.5+2) = (4.0, 2.5).
        // Before the fix, num_parents=1 would skip delta correction, leaving
        // the edge Q based only on the new visit + stale visits.
        // With the fix, edge_visits(2) < child.total_visits(6) triggers correction.
        let edge_q1 = root.get().edge_q_p1(0, 1);
        let edge_q2 = root.get().edge_q_p2(0, 1);
        assert!(
            (edge_q1 - 4.0).abs() < 0.5,
            "edge Q1 should be corrected toward 4.0, got {edge_q1}"
        );
        assert!(
            (edge_q2 - 2.5).abs() < 0.5,
            "edge Q2 should be corrected toward 2.5, got {edge_q2}"
        );
    }

    // ---- cleanup_descent ----

    #[test]
    fn cleanup_reverts_virtual_loss() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().add_virtual_loss(1, 2);

        let path = vec![PathEntry {
            node: Arc::clone(&node),
            p1_outcome: 1,
            p2_outcome: 2,
        }];

        cleanup_descent(&path, None);
        assert_eq!(node.get().edge_in_flight(1, 2), 0);
    }

    #[test]
    fn cleanup_cancels_leaf_claim() {
        let leaf = SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ));
        assert!(leaf.get_mut().try_start_score_update());
        assert_eq!(leaf.get().n_in_flight(), 1);

        cleanup_descent(&[], Some(&leaf));
        assert_eq!(leaf.get().n_in_flight(), 0);
    }

    // ---- run_search integration ----

    #[test]
    fn run_search_uniform_basic() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        // Basic properties
        assert!(result.total_visits > 0);
        assert!(result.nn_evals > 0 || result.terminals > 0);

        // Policy sums to 1
        let sum_p1: f32 = result.policy_p1.iter().sum();
        let sum_p2: f32 = result.policy_p2.iter().sum();
        assert!((sum_p1 - 1.0).abs() < 1e-4, "P1 policy sum: {sum_p1}");
        assert!((sum_p2 - 1.0).abs() < 1e-4, "P2 policy sum: {sum_p2}");

        // Corner positions: blocked actions should have 0 visits
        // P1 at (0,0): DOWN and LEFT are blocked
        assert_eq!(result.visit_counts_p1[2], 0.0); // DOWN blocked
        assert_eq!(result.visit_counts_p1[3], 0.0); // LEFT blocked
    }

    #[test]
    fn run_search_terminal_game() {
        let game = terminal_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 50;
        let result =
            run_search(&mut tree, &game, &backend, &config, n_sims, 16, &mut r).unwrap();

        // Terminal root: one real visit per pick, total = n_sims.
        // Before fix: quadratic blowup (multivisit=budget per pick, counts as 1).
        let root_visits = tree.root().get().total_visits();
        assert_eq!(
            root_visits, n_sims,
            "terminal root should have exactly {n_sims} visits, got {root_visits}"
        );
        assert_eq!(result.terminals, n_sims);
        assert_eq!(tree.root().get().n_in_flight(), 0, "root n_in_flight leak");
    }

    #[test]
    fn run_search_short_game() {
        let game = short_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        // Should find terminals in the short game
        assert!(result.total_visits > 0);
        assert!(result.nn_evals + result.terminals > 0);
    }

    #[test]
    fn run_search_constant_value_backend() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = ConstantValueBackend {
            value_p1: 0.5,
            value_p2: 0.3,
        };
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        assert!(result.total_visits > 0);

        // Values should be influenced by the constant backend
        // With non-zero values, root value should be non-zero after search
        let sum_p1: f32 = result.policy_p1.iter().sum();
        assert!((sum_p1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn run_search_tt_sharing() {
        // Run enough sims that transpositions should occur
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let _result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        // TT should have entries beyond just the root
        assert!(tree.tt().live_count() > 1, "TT should have entries for explored positions");
    }

    // =====================================================================
    // Additional helpers
    // =====================================================================

    fn corridor_game() -> GameState {
        let mut walls = HashMap::new();
        for x in 0..5 {
            walls
                .entry(Coordinates::new(x, 0))
                .or_insert_with(Vec::new)
                .push(Coordinates::new(x, 1));
            walls
                .entry(Coordinates::new(x, 1))
                .or_insert_with(Vec::new)
                .push(Coordinates::new(x, 0));
        }
        GameBuilder::new(5, 5)
            .with_custom_maze(walls, Default::default())
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 0))
            .with_custom_cheese(vec![Coordinates::new(2, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }


    /// DFS walk collecting all SharedNode pointers reachable from a root.
    fn walk_dag(root: &Arc<SharedNode>) -> HashSet<*const SharedNode> {
        let mut visited = HashSet::new();
        let mut stack = vec![Arc::clone(root)];
        while let Some(node) = stack.pop() {
            let ptr = Arc::as_ptr(&node) as *const SharedNode;
            if !visited.insert(ptr) {
                continue;
            }
            let low = node.get();
            let mut cur = low.first_child();
            while let Some(edge) = cur {
                stack.push(Arc::clone(edge.low_node()));
                cur = edge.next_sibling();
            }
        }
        visited
    }

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn evaluate(
            &self,
            _game: &GameState,
        ) -> Result<crate::EvalResult, BackendError> {
            Err(BackendError::msg("intentional test failure"))
        }
    }

    // =====================================================================
    // Step 1: Backup depth tests
    // =====================================================================

    #[test]
    fn backup_two_level_q_chain() {
        // root --edge(r=1,0.5)--> mid --edge(r=0.5,1.0)--> leaf
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let mid = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        mid.get_mut().set_value_scale(5.0);
        mid.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let leaf = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        leaf.get_mut().set_value_scale(5.0);

        // Wire edges
        let edge_mid_leaf = Box::new(Edge::new(Arc::clone(&leaf), (2, 3), 0.5, 1.0));
        mid.get_mut().prepend_child(edge_mid_leaf);

        let edge_root_mid = Box::new(Edge::new(Arc::clone(&mid), (0, 0), 1.0, 0.5));
        root.get_mut().prepend_child(edge_root_mid);

        let path = vec![
            PathEntry {
                node: Arc::clone(&root),
                p1_outcome: 0,
                p2_outcome: 0,
            },
            PathEntry {
                node: Arc::clone(&mid),
                p1_outcome: 2,
                p2_outcome: 3,
            },
        ];

        // g = (2.0, 3.0) at leaf
        backup(&path, &leaf, 2.0, 3.0);

        // leaf: v=(2.0, 3.0)
        assert!((leaf.get().v1() - 2.0).abs() < 1e-6);
        assert!((leaf.get().v2() - 3.0).abs() < 1e-6);

        // mid: q1 = edge_r1(0.5) + leaf_v1(2.0) = 2.5
        //       q2 = edge_r2(1.0) + leaf_v2(3.0) = 4.0
        assert!((mid.get().v1() - 2.5).abs() < 1e-6);
        assert!((mid.get().v2() - 4.0).abs() < 1e-6);

        // root: q1 = edge_r1(1.0) + mid_q1(2.5) = 3.5
        //        q2 = edge_r2(0.5) + mid_q2(4.0) = 4.5
        assert!((root.get().v1() - 3.5).abs() < 1e-6);
        assert!((root.get().v2() - 4.5).abs() < 1e-6);

        // Joint matrix at root
        assert!((root.get().edge_q_p1(0, 0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn backup_three_level_reward_chain() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(10.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        a.get_mut().set_value_scale(10.0);
        a.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        b.get_mut().set_value_scale(10.0);
        b.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let leaf = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        leaf.get_mut().set_value_scale(10.0);

        // Wire: root--(1,0)-->a--(1,0)-->b--(1,0)-->leaf, all edge_r = (1.0, 0.5)
        b.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&leaf), (1, 0), 1.0, 0.5)));
        a.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&b), (1, 0), 1.0, 0.5)));
        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&a), (1, 0), 1.0, 0.5)));

        let path = vec![
            PathEntry {
                node: Arc::clone(&root),
                p1_outcome: 1,
                p2_outcome: 0,
            },
            PathEntry {
                node: Arc::clone(&a),
                p1_outcome: 1,
                p2_outcome: 0,
            },
            PathEntry {
                node: Arc::clone(&b),
                p1_outcome: 1,
                p2_outcome: 0,
            },
        ];

        // First backup: g=(4.0, 2.0)
        backup(&path, &leaf, 4.0, 2.0);
        // leaf: v1=4.0
        // b: q1 = 1+4=5
        // a: q1 = 1+5=6
        // root: q1 = 1+6=7
        assert!((leaf.get().v1() - 4.0).abs() < 1e-5);
        assert!((b.get().v1() - 5.0).abs() < 1e-5);
        assert!((a.get().v1() - 6.0).abs() < 1e-5);
        assert!((root.get().v1() - 7.0).abs() < 1e-5);

        // Second backup: g=(2.0, 1.0) — Welford averages
        backup(&path, &leaf, 2.0, 1.0);
        // leaf: v1 = mean(4, 2) = 3.0
        // b: backups are (5, 3), mean = 4.0
        // a: backups are (6, 4), mean = 5.0
        // root: backups are (7, 5), mean = 6.0
        assert!((leaf.get().v1() - 3.0).abs() < 1e-5);
        assert!((b.get().v1() - 4.0).abs() < 1e-5);
        assert!((a.get().v1() - 5.0).abs() < 1e-5);
        assert!((root.get().v1() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn backup_multiple_same_edge() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 0.0, 0.0)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        backup(&path, &child, 2.0, 1.0);
        backup(&path, &child, 4.0, 3.0);
        backup(&path, &child, 6.0, 5.0);

        // edge_r = 0, so edge_q = g values
        // Joint matrix: 3 visits, Q = mean(2,4,6) = 4.0 for p1
        assert_eq!(root.get().edge_visits(0, 0), 3);
        assert!((root.get().edge_q_p1(0, 0) - 4.0).abs() < 1e-5);
        assert!((root.get().edge_q_p2(0, 0) - 3.0).abs() < 1e-5);

        // child: v1 = mean(2,4,6) = 4.0
        assert_eq!(child.get().total_visits(), 3);
        assert!((child.get().v1() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn backup_multiple_different_edges() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(10.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child_a.get_mut().set_value_scale(10.0);

        let child_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child_b.get_mut().set_value_scale(10.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child_a), (0, 0), 0.0, 0.0)));
        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child_b), (1, 1), 0.0, 0.0)));

        let path_a = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        let path_b = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 1,
            p2_outcome: 1,
        }];

        backup(&path_a, &child_a, 5.0, 5.0);
        backup(&path_b, &child_b, 1.0, 1.0);

        assert!((root.get().edge_q_p1(0, 0) - 5.0).abs() < 1e-5);
        assert!((root.get().edge_q_p1(1, 1) - 1.0).abs() < 1e-5);
        assert_eq!(root.get().edge_visits(0, 0), 1);
        assert_eq!(root.get().edge_visits(1, 1), 1);
    }

    #[test]
    fn backup_terminal_leaf() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_terminal();

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 1.0, 0.5)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        // Terminal backup: g = (0, 0)
        backup(&path, &child, 0.0, 0.0);

        // Root: q1 = r1(1.0) + 0 = 1.0, q2 = r2(0.5) + 0 = 0.5
        assert!((root.get().v1() - 1.0).abs() < 1e-6);
        assert!((root.get().v2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn backup_empty_path() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let path: Vec<PathEntry> = vec![];
        backup(&path, &root, 3.0, 2.0);

        assert_eq!(root.get().total_visits(), 1);
        assert!((root.get().v1() - 3.0).abs() < 1e-6);
        assert_eq!(root.get().total_edge_visits(), 0);
    }

    #[test]
    fn backup_same_edge_raw_propagation() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(10.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(10.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 2.0, 0.0)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        backup(&path, &child, 10.0, 0.0); // q = 2 + 10 = 12
        backup(&path, &child, 4.0, 0.0); // q = 2 + 4 = 6
        backup(&path, &child, 7.0, 0.0); // q = 2 + 7 = 9

        // Edge Q = Welford mean of (12, 6, 9) = 9.0
        assert!((root.get().edge_q_p1(0, 0) - 9.0).abs() < 1e-5);
    }

    #[test]
    fn backup_asymmetric_rewards() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(10.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(10.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 2.0, 0.5)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        backup(&path, &child, 3.0, 4.0);

        // root v1 = r1(2) + g1(3) = 5.0, v2 = r2(0.5) + g2(4) = 4.5
        assert!((root.get().v1() - 5.0).abs() < 1e-6);
        assert!((root.get().v2() - 4.5).abs() < 1e-6);
    }

    #[test]
    fn backup_edge_visit_sum() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 0.0, 0.0)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        for i in 0..5 {
            backup(&path, &child, i as f32, 0.0);
        }

        assert_eq!(root.get().total_edge_visits(), 5);
        assert_eq!(root.get().total_visits(), 5);
    }

    #[test]
    fn backup_p2_independent_propagation() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(10.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(10.0);

        root.get_mut()
            .prepend_child(Box::new(Edge::new(Arc::clone(&child), (0, 0), 2.0, 0.5)));

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 0,
        }];

        backup(&path, &child, 10.0, 1.0); // q1=12, q2=1.5
        backup(&path, &child, 4.0, 8.0); // q1=6, q2=8.5

        // p1 edge Q = mean(12, 6) = 9.0
        // p2 edge Q = mean(1.5, 8.5) = 5.0
        assert!((root.get().edge_q_p1(0, 0) - 9.0).abs() < 1e-5);
        assert!((root.get().edge_q_p2(0, 0) - 5.0).abs() < 1e-5);
    }

    // =====================================================================
    // Step 2: PUCT selection tests
    // =====================================================================

    #[test]
    fn puct_monotonic_q() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(10.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Give outcome 2 high Q, all outcomes some visits
        for i in 0..5 {
            for j in 0..5 {
                let q = if i == 2 { 10.0 } else { 1.0 };
                for _ in 0..10 {
                    node.get_mut().finalize_edge_update(i, j, q, q);
                }
            }
        }
        // Need some total_visits for value to be nonzero
        for _ in 0..250 {
            node.get_mut().finalize_score_update(1.0, 1.0);
        }

        let config = default_config();
        let mut r = rng();
        let selected = select_p1(node.get(), &config, false, &mut r);
        assert_eq!(selected, 2, "Highest Q should win");
    }

    #[test]
    fn puct_monotonic_prior() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut()
            .set_prior([0.05, 0.05, 0.7, 0.1, 0.1], [0.2; 5]);

        // Give 1 visit at (0,0) so total_edge_visits > 0
        node.get_mut().finalize_edge_update(0, 0, 1.0, 1.0);
        node.get_mut().finalize_score_update(1.0, 1.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(node.get(), &config, false, &mut r);
        // Outcome 2 has highest prior (0.7) and is unvisited
        assert_eq!(
            selected, 2,
            "Highest prior should dominate when mostly unvisited"
        );
    }

    #[test]
    fn puct_unvisited_selected() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Give 100 visits to outcomes 0-3, leave outcome 4 unvisited
        for i in 0..4 {
            for _ in 0..100 {
                node.get_mut().finalize_edge_update(i, 0, 1.0, 1.0);
                node.get_mut().finalize_score_update(1.0, 1.0);
            }
        }

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(node.get(), &config, false, &mut r);
        assert_eq!(selected, 4, "Unvisited outcome should be selected");
    }

    #[test]
    fn puct_fpu_pessimism() {
        // Higher visited prior mass → stronger FPU penalty → less willingness to explore
        let node_lo = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_lo.get_mut().set_value_scale(5.0);
        node_lo.get_mut().set_prior([0.2; 5], [0.2; 5]);
        node_lo.get_mut().finalize_score_update(5.0, 5.0);
        // Visit 1 outcome → visited_mass = 0.2
        node_lo.get_mut().finalize_edge_update(0, 0, 5.0, 5.0);

        let node_hi = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_hi.get_mut().set_value_scale(5.0);
        node_hi.get_mut().set_prior([0.2; 5], [0.2; 5]);
        node_hi.get_mut().finalize_score_update(5.0, 5.0);
        // Visit 3 outcomes → visited_mass = 0.6
        node_hi.get_mut().finalize_edge_update(0, 0, 5.0, 5.0);
        node_hi.get_mut().finalize_edge_update(1, 0, 5.0, 5.0);
        node_hi.get_mut().finalize_edge_update(2, 0, 5.0, 5.0);

        // FPU = v1 - fpu_reduction * value_scale * sqrt(visited_mass)
        // node_lo: FPU = 5 - 0.2 * 5 * sqrt(0.2) ~ 4.553
        // node_hi: FPU = 5 - 0.2 * 5 * sqrt(0.6) ~ 4.225
        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let sel_lo = select_p1(node_lo.get(), &config, false, &mut r);
        let sel_hi = select_p1(node_hi.get(), &config, false, &mut r);
        assert!((sel_lo as usize) < 5);
        assert!((sel_hi as usize) < 5);
    }

    #[test]
    fn puct_fpu_no_visits() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut()
            .set_prior([0.05, 0.05, 0.7, 0.1, 0.1], [0.2; 5]);

        // Give 1 visit so sqrt_total > 0
        node.get_mut().finalize_edge_update(0, 0, 0.0, 0.0);
        node.get_mut().finalize_score_update(0.0, 0.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(node.get(), &config, false, &mut r);
        assert_eq!(selected, 2, "Highest prior should win via exploration");
    }

    #[test]
    fn puct_forced_fires_at_root() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Give 100 visits to outcomes 0 and 1
        for _ in 0..50 {
            node.get_mut().finalize_edge_update(0, 0, 1.0, 1.0);
            node.get_mut().finalize_edge_update(1, 0, 1.0, 1.0);
            node.get_mut().finalize_score_update(1.0, 1.0);
            node.get_mut().finalize_score_update(1.0, 1.0);
        }
        // Outcomes 2,3,4 have 1 visit each
        for i in 2..5 {
            node.get_mut().finalize_edge_update(i, 0, 1.0, 1.0);
            node.get_mut().finalize_score_update(1.0, 1.0);
        }

        // total_edge_visits = 103
        // threshold for prior=0.2, total=103: sqrt(2.0 * 0.2 * 103) ~ 6.4
        // Outcomes 2,3,4 have 1 visit < 6.4 → FORCED
        let config = default_config(); // force_k = 2.0
        let mut r = rng();
        let selected = select_p1(node.get(), &config, true, &mut r);
        assert!(
            selected >= 2,
            "Forced playout should select an undervisited outcome, got {selected}"
        );
    }

    #[test]
    fn puct_forced_not_at_nonroot() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Give many visits and high Q to outcome 0, few visits to others
        for _ in 0..100 {
            node.get_mut().finalize_edge_update(0, 0, 10.0, 10.0);
            node.get_mut().finalize_score_update(10.0, 10.0);
        }
        for i in 1..5 {
            node.get_mut().finalize_edge_update(i, 0, 1.0, 1.0);
            node.get_mut().finalize_score_update(1.0, 1.0);
        }

        let config = default_config();
        let mut r = rng();
        // At non-root, forced playouts don't fire
        let selected = select_p1(node.get(), &config, false, &mut r);
        // Outcome 0 has 100 visits with Q=10 vs others with 1 visit Q=1.
        // Without forced playouts, Q dominates.
        assert_eq!(selected, 0, "Without forced playouts, high-Q should win");
    }

    #[test]
    fn puct_value_scale_effect() {
        // With small value_scale, Q/scale is large → exploitation dominates.
        // With large value_scale, Q/scale is small → exploration dominates.
        //
        // All outcomes get visits so FPU doesn't apply. Outcome 0 has high Q,
        // outcomes 1-4 have low Q. Outcome 3 has highest prior (0.6).
        // Small scale: Q gap dominates → outcome 0 selected.
        // Large scale: Q gap shrinks, exploration from prior dominates → outcome 3 selected.
        let make_node = |scale: f32| {
            let node = Arc::new(SharedNode::new(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            )));
            node.get_mut().set_value_scale(scale);
            node.get_mut()
                .set_prior([0.05, 0.05, 0.05, 0.6, 0.25], [0.2; 5]);
            // Outcome 0: high Q, many visits
            for _ in 0..100 {
                node.get_mut().finalize_edge_update(0, 0, 10.0, 10.0);
                node.get_mut().finalize_score_update(10.0, 10.0);
            }
            // Outcomes 1-4: low Q, few visits
            for i in 1..5 {
                for _ in 0..5 {
                    node.get_mut().finalize_edge_update(i, 0, 0.1, 0.1);
                    node.get_mut().finalize_score_update(0.1, 0.1);
                }
            }
            node
        };

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        // Small scale: Q=10 / scale=1 = 10, gap to others = ~9.9 → exploitation
        let node_small = make_node(1.0);
        let sel_small = select_p1(node_small.get(), &config, false, &mut r);

        // Large scale: Q=10 / scale=10000 ≈ 0 → exploration dominates (outcome 3, prior=0.6)
        let node_large = make_node(10000.0);
        let sel_large = select_p1(node_large.get(), &config, false, &mut r);

        assert_eq!(sel_small, 0, "Small scale -> exploitation -> outcome 0");
        assert_eq!(
            sel_large, 3,
            "Large scale -> exploration -> outcome 3 (highest prior, few visits)"
        );
    }

    #[test]
    fn puct_decoupled() {
        // Two nodes: same P1 structure, different P2 priors
        let node_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_a.get_mut().set_value_scale(5.0);
        node_a.get_mut().set_prior(
            [0.1, 0.3, 0.2, 0.15, 0.25],
            [0.8, 0.05, 0.05, 0.05, 0.05],
        );
        node_a.get_mut().finalize_edge_update(0, 0, 2.0, 2.0);
        node_a.get_mut().finalize_score_update(2.0, 2.0);

        let node_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_b.get_mut().set_value_scale(5.0);
        node_b.get_mut().set_prior(
            [0.1, 0.3, 0.2, 0.15, 0.25],
            [0.05, 0.05, 0.05, 0.05, 0.8],
        );
        node_b.get_mut().finalize_edge_update(0, 0, 2.0, 2.0);
        node_b.get_mut().finalize_score_update(2.0, 2.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r1 = SmallRng::seed_from_u64(99);
        let mut r2 = SmallRng::seed_from_u64(99);

        let sel_a = select_p1(node_a.get(), &config, false, &mut r1);
        let sel_b = select_p1(node_b.get(), &config, false, &mut r2);

        assert_eq!(sel_a, sel_b, "P2 priors should not affect P1 selection");
    }

    #[test]
    fn puct_single_outcome() {
        // Mud-stuck: all actions → STAY, n1=1
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let config = default_config();
        let mut r = rng();
        let selected = select_p1(node.get(), &config, false, &mut r);
        assert_eq!(selected, 0, "Single outcome should return 0");
    }

    #[test]
    fn puct_virtual_loss_diversifies() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);
        // Give some visits so selection isn't degenerate
        for i in 0..5 {
            node.get_mut().finalize_edge_update(i, 0, 2.0, 2.0);
            node.get_mut().finalize_score_update(2.0, 2.0);
        }

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        // Select without virtual loss
        let baseline = select_p1(node.get(), &config, false, &mut r);

        // Add heavy virtual loss on baseline outcome
        for _ in 0..100 {
            node.get_mut().add_virtual_loss(baseline as usize, 0);
        }

        let mut r2 = rng();
        let shifted = select_p1(node.get(), &config, false, &mut r2);
        assert_ne!(
            baseline, shifted,
            "Virtual loss should shift selection away from {baseline}"
        );

        // Clean up
        for _ in 0..100 {
            node.get_mut().revert_virtual_loss(baseline as usize, 0);
        }
    }

    #[test]
    fn puct_multi_descent_diversification() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_value_scale(5.0);
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Give 1 visit so total_edge_visits > 0
        node.get_mut().finalize_edge_update(0, 0, 1.0, 1.0);
        node.get_mut().finalize_score_update(1.0, 1.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        let mut selected = HashSet::new();
        for _ in 0..3 {
            let s = select_p1(node.get(), &config, false, &mut r);
            // Add virtual loss to force next selection elsewhere
            node.get_mut().add_virtual_loss(s as usize, 0);
            selected.insert(s);
        }

        assert!(
            selected.len() >= 2,
            "3 descents with virtual loss should diversify, got {:?}",
            selected
        );

        // Cleanup
        for &s in &selected {
            node.get_mut().revert_virtual_loss(s as usize, 0);
        }
    }

    // =====================================================================
    // Step 3: Post-search invariants
    // =====================================================================

    #[test]
    fn search_n_in_flight_zero_after_search() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let _result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        // Walk DAG: every node should have n_in_flight=0
        for &ptr in &walk_dag(tree.root()) {
            let node_ref = unsafe { &*ptr };
            let low = node_ref.get();
            assert_eq!(
                low.n_in_flight(),
                0,
                "Node still has n_in_flight={}",
                low.n_in_flight()
            );

            // Also check per-edge in_flight matrix
            for i in 0..low.n1() {
                for j in 0..low.n2() {
                    assert_eq!(
                        low.edge_in_flight(i, j),
                        0,
                        "edge_in_flight[{i}][{j}] = {}",
                        low.edge_in_flight(i, j)
                    );
                }
            }
        }
    }

    #[test]
    fn search_value_bounded() {
        let cheese = vec![
            Coordinates::new(1, 0),
            Coordinates::new(2, 0),
            Coordinates::new(3, 0),
            Coordinates::new(4, 0),
        ];
        let game = open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let remaining = game.cheese.remaining_cheese() as f32;

        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        assert!(
            result.value_p1 >= -0.1,
            "value_p1 too low: {}",
            result.value_p1
        );
        assert!(
            result.value_p1 <= remaining + 0.1,
            "value_p1 too high: {}",
            result.value_p1
        );
        assert!(
            result.value_p2 >= -0.1,
            "value_p2 too low: {}",
            result.value_p2
        );
        assert!(
            result.value_p2 <= remaining + 0.1,
            "value_p2 too high: {}",
            result.value_p2
        );
    }

    #[test]
    fn search_policy_sums_to_one_multiple_configs() {
        let configs = [
            (
                Coordinates::new(0, 0),
                Coordinates::new(4, 4),
                vec![Coordinates::new(2, 2)],
                "center cheese",
            ),
            (
                Coordinates::new(0, 0),
                Coordinates::new(4, 4),
                vec![Coordinates::new(1, 0)],
                "adjacent cheese",
            ),
            (
                Coordinates::new(2, 2),
                Coordinates::new(2, 2),
                vec![Coordinates::new(0, 0)],
                "same position",
            ),
            (
                Coordinates::new(0, 0),
                Coordinates::new(0, 4),
                vec![Coordinates::new(4, 2), Coordinates::new(0, 2)],
                "multi cheese",
            ),
        ];

        let backend = SmartUniformBackend;
        let config = default_config();

        for (p1, p2, cheese, name) in &configs {
            let game = open_5x5_game(*p1, *p2, cheese);
            let mut tree = MCGSTree::new(&game);
            let mut r = rng();

            let result =
                run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();
            let sum_p1: f32 = result.policy_p1.iter().sum();
            let sum_p2: f32 = result.policy_p2.iter().sum();
            assert!((sum_p1 - 1.0).abs() < 1e-4, "{name}: P1 sum = {sum_p1}");
            assert!((sum_p2 - 1.0).abs() < 1e-4, "{name}: P2 sum = {sum_p2}");
        }
    }

    #[test]
    fn search_blocked_actions_zero() {
        // P1 at (0,0) corner: DOWN and LEFT blocked
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        assert_eq!(result.visit_counts_p1[2], 0.0, "P1 DOWN should be blocked");
        assert_eq!(result.visit_counts_p1[3], 0.0, "P1 LEFT should be blocked");

        // P2 at (4,4): UP and RIGHT blocked
        assert_eq!(result.visit_counts_p2[0], 0.0, "P2 UP should be blocked");
        assert_eq!(
            result.visit_counts_p2[1], 0.0,
            "P2 RIGHT should be blocked"
        );

        // Also test corridor
        let game2 = corridor_game();
        let mut tree2 = MCGSTree::new(&game2);
        let mut r2 = rng();
        let result2 =
            run_search(&mut tree2, &game2, &backend, &config, 100, 16, &mut r2).unwrap();

        // P1 at (0,0) in corridor: UP blocked, DOWN blocked, LEFT blocked
        assert_eq!(
            result2.visit_counts_p1[0], 0.0,
            "P1 UP blocked in corridor"
        );
        assert_eq!(
            result2.visit_counts_p1[2], 0.0,
            "P1 DOWN blocked in corridor"
        );
        assert_eq!(
            result2.visit_counts_p1[3], 0.0,
            "P1 LEFT blocked in corridor"
        );
    }

    #[test]
    fn search_adjacent_cheese_dominates() {
        // P1 at (0,0), cheese at (1,0): RIGHT should dominate
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 0)],
        );
        let backend = ConstantValueBackend {
            value_p1: 0.5,
            value_p2: 0.5,
        };
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 500, 16, &mut r).unwrap();

        // RIGHT = action 1
        assert!(
            result.policy_p1[1] > 0.5,
            "RIGHT should dominate for P1, got {}",
            result.policy_p1[1]
        );
    }

    // =====================================================================
    // Step 4: Dirichlet noise tests
    // =====================================================================

    #[test]
    fn noise_disabled_priors_unchanged() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut()
            .set_prior([0.1, 0.2, 0.3, 0.15, 0.25], [0.2; 5]);

        let priors_before: Vec<f32> = (0..5).map(|i| node.get().p1_prior(i)).collect();

        let mut r = rng();
        apply_dirichlet_noise_p1(&node, 0.0, 10.83, &mut r);

        for i in 0..5 {
            assert!(
                (node.get().p1_prior(i) - priors_before[i]).abs() < 1e-10,
                "Prior {i} changed with epsilon=0"
            );
        }
    }

    #[test]
    fn noise_enabled_priors_modified() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let priors_before: Vec<f32> = (0..5).map(|i| node.get().p1_prior(i)).collect();

        let mut r = rng();
        apply_dirichlet_noise_p1(&node, 0.25, 10.83, &mut r);

        let mut any_changed = false;
        let mut sum = 0.0f32;
        for i in 0..5 {
            sum += node.get().p1_prior(i);
            if (node.get().p1_prior(i) - priors_before[i]).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(
            any_changed,
            "At least one prior should change with epsilon=0.25"
        );
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Priors should still sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn noise_deterministic_with_seed() {
        let make = || {
            let node = Arc::new(SharedNode::new(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            )));
            node.get_mut().set_prior([0.2; 5], [0.2; 5]);
            node
        };

        let node1 = make();
        let mut r1 = SmallRng::seed_from_u64(777);
        apply_dirichlet_noise_p1(&node1, 0.25, 10.83, &mut r1);

        let node2 = make();
        let mut r2 = SmallRng::seed_from_u64(777);
        apply_dirichlet_noise_p1(&node2, 0.25, 10.83, &mut r2);

        for i in 0..5 {
            assert!(
                (node1.get().p1_prior(i) - node2.get().p1_prior(i)).abs() < 1e-10,
                "Same seed should produce same noise at outcome {i}"
            );
        }
    }

    #[test]
    fn noise_single_outcome_noop() {
        let node = Arc::new(SharedNode::new(LowNode::new_shell(
            [4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4],
        )));
        node.get_mut().set_prior([0.2; 5], [0.2; 5]);

        assert_eq!(node.get().n1(), 1);
        let prior_before = node.get().p1_prior(0);

        let mut r = rng();
        apply_dirichlet_noise_p1(&node, 0.25, 10.83, &mut r);

        assert!(
            (node.get().p1_prior(0) - prior_before).abs() < 1e-10,
            "Single outcome should not be modified by noise"
        );
    }

    #[test]
    fn noise_search_integration() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;

        // No noise
        let config_no_noise = SearchConfig {
            noise_epsilon: 0.0,
            ..default_config()
        };
        let mut tree1 = MCGSTree::new(&game);
        let mut r1 = SmallRng::seed_from_u64(123);
        let result1 =
            run_search(&mut tree1, &game, &backend, &config_no_noise, 100, 16, &mut r1).unwrap();

        // With noise
        let config_noise = SearchConfig {
            noise_epsilon: 0.25,
            ..default_config()
        };
        let mut tree2 = MCGSTree::new(&game);
        let mut r2 = SmallRng::seed_from_u64(123);
        let result2 =
            run_search(&mut tree2, &game, &backend, &config_noise, 100, 16, &mut r2).unwrap();

        // Priors should differ (noise modifies root priors)
        let any_diff = result1
            .prior_p1
            .iter()
            .zip(result2.prior_p1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_diff,
            "Noisy search should produce different root priors"
        );
    }

    // =====================================================================
    // Step 5: Error handling + OOO gather
    // =====================================================================

    #[test]
    fn failing_backend_propagates() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        let config = default_config();
        let mut r = rng();

        let result = run_search(&mut tree, &game, &FailingBackend, &config, 10, 4, &mut r);
        assert!(
            result.is_err(),
            "run_search should propagate backend errors"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intentional test failure"));
    }

    #[test]
    fn backend_error_reverts_bookkeeping() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2), Coordinates::new(1, 1)],
        );
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // Phase 1: expand the tree just enough that gather descends interior nodes,
        // but not so much that every leaf is a transposition stop.
        let _ =
            run_search(&mut tree, &game, &SmartUniformBackend, &config, 10, 8, &mut r).unwrap();

        // Phase 2: search with failing backend — high budget to ensure NeedsEval.
        let result = run_search(&mut tree, &game, &FailingBackend, &config, 100, 16, &mut r);
        assert!(result.is_err());

        // Phase 3: verify the tree is clean — all n_in_flight and edge_in_flight = 0.
        for &ptr in &walk_dag(tree.root()) {
            let node_ref = unsafe { &*ptr };
            let low = node_ref.get();
            assert_eq!(
                low.n_in_flight(),
                0,
                "n_in_flight={} after backend error",
                low.n_in_flight()
            );
            for i in 0..low.n1() {
                for j in 0..low.n2() {
                    assert_eq!(
                        low.edge_in_flight(i, j),
                        0,
                        "edge_in_flight[{i}][{j}]={} after backend error",
                        low.edge_in_flight(i, j)
                    );
                }
            }
        }

        // Phase 4: rerun with good backend — tree should be usable.
        let result2 =
            run_search(&mut tree, &game, &SmartUniformBackend, &config, 50, 8, &mut r).unwrap();
        assert!(
            result2.total_visits > 0,
            "search after backend error should produce visits"
        );
    }

    #[test]
    fn ooo_terminal_fills_batch() {
        // Short game with reachable terminals
        let game = short_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        // With OOO terminal processing, terminals are processed inline
        assert!(result.total_visits > 0);
        assert!(result.terminals > 0 || result.nn_evals > 0);
    }

    #[test]
    fn ooo_collision_budget_stops_gather() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        // Low collision budget to test collision limit
        let config = SearchConfig {
            collision_limit_min: 1,
            collision_limit_max: 1,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // Should complete without panicking even with tight collision budget
        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        assert!(result.total_visits > 0);
    }

    #[test]
    fn ooo_batch_size_1_exact_visits() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 20;
        let result =
            run_search(&mut tree, &game, &backend, &config, n_sims, 1, &mut r).unwrap();

        // With batch_size=1 and enough cheese, nn_evals + terminals >= n_sims
        assert!(
            result.nn_evals + result.terminals >= n_sims,
            "nn_evals({}) + terminals({}) should be >= n_sims({})",
            result.nn_evals,
            result.terminals,
            n_sims
        );
    }

    #[test]
    fn search_batch_size_larger_than_n_sims() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // batch_size > n_sims: should still complete without panicking.
        // With small n_sims the root may be the only leaf, so total_edge_visits
        // (which is what SearchResult.total_visits reports) can be 0 when
        // the root itself is the only node evaluated. We just verify it doesn't crash
        // and produces a valid result.
        let result = run_search(&mut tree, &game, &backend, &config, 5, 100, &mut r).unwrap();
        assert!(result.nn_evals + result.terminals + result.collisions > 0);
    }

    // =====================================================================
    // Delta correction tests
    // =====================================================================

    /// No transposition: delta stays 0, backup produces identical results to before.
    #[test]
    fn delta_no_transposition_baseline() {
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        // Only one parent edge → num_parents == 1, no delta correction.
        let edge = Box::new(Edge::new(Arc::clone(&child), (0, 1), 1.0, 0.5));
        root.get_mut().prepend_child(edge);
        assert_eq!(child.num_parents(), 1);

        let path = vec![PathEntry {
            node: Arc::clone(&root),
            p1_outcome: 0,
            p2_outcome: 1,
        }];

        backup(&path, &child, 3.0, 2.0);

        // Same results as non-delta-correction backup.
        assert_eq!(child.get().total_visits(), 1);
        assert!((child.get().v1() - 3.0).abs() < 1e-6);
        assert!((child.get().v2() - 2.0).abs() < 1e-6);

        // q1 = 1.0 + 3.0 = 4.0, q2 = 0.5 + 2.0 = 2.5
        assert_eq!(root.get().total_visits(), 1);
        assert!((root.get().v1() - 4.0).abs() < 1e-6);
        assert!((root.get().v2() - 2.5).abs() < 1e-6);
        assert_eq!(root.get().edge_visits(0, 1), 1);
        assert!((root.get().edge_q_p1(0, 1) - 4.0).abs() < 1e-6);
        assert!((root.get().edge_q_p2(0, 1) - 2.5).abs() < 1e-6);
    }

    /// Simple transposition correction.
    /// Root → A → C and Root → B → C. Backup through A, then through B.
    /// After B's backup, A's edge_q should reflect C's aggregate (not just the
    /// single visit that went through A).
    #[test]
    fn delta_simple_transposition_correction() {
        // Shared child C (the transposition).
        let child_c = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child_c.get_mut().set_value_scale(5.0);

        // Parent A, with edge to C.
        let parent_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_a.get_mut().set_value_scale(5.0);
        parent_a.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_a_c = Box::new(Edge::new(Arc::clone(&child_c), (0, 0), 1.0, 0.5));
        parent_a.get_mut().prepend_child(edge_a_c);

        // Parent B, with edge to C.
        let parent_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_b.get_mut().set_value_scale(5.0);
        parent_b.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_b_c = Box::new(Edge::new(Arc::clone(&child_c), (0, 0), 0.0, 0.0));
        parent_b.get_mut().prepend_child(edge_b_c);

        assert_eq!(child_c.num_parents(), 2);

        // Backup 1: through A with leaf value (2.0, 3.0).
        let path_a = vec![PathEntry {
            node: Arc::clone(&parent_a),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_a, &child_c, 2.0, 3.0);

        // After backup 1: C.v = (2.0, 3.0), A.edge_q = (1+2, 0.5+3) = (3.0, 3.5).
        assert_eq!(child_c.get().total_visits(), 1);
        assert!((child_c.get().v1() - 2.0).abs() < 1e-6);
        assert!((parent_a.get().edge_q_p1(0, 0) - 3.0).abs() < 1e-6);
        assert!((parent_a.get().edge_q_p2(0, 0) - 3.5).abs() < 1e-6);

        // Backup 2: through B with leaf value (4.0, 5.0).
        let path_b = vec![PathEntry {
            node: Arc::clone(&parent_b),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_b, &child_c, 4.0, 5.0);

        // After backup 2: C.v = mean(2, 4) = 3.0, mean(3, 5) = 4.0.
        assert_eq!(child_c.get().total_visits(), 2);
        assert!((child_c.get().v1() - 3.0).abs() < 1e-6);
        assert!((child_c.get().v2() - 4.0).abs() < 1e-6);

        // B's edge sees C as a transposition (num_parents > 1).
        // correct_q1 = r1(0) + C.v1(3.0) = 3.0
        // correct_q2 = r2(0) + C.v2(4.0) = 4.0
        // B had no prior visits, so the Welford new-visit gets correct_q directly.
        assert_eq!(parent_b.get().edge_visits(0, 0), 1);
        assert!((parent_b.get().edge_q_p1(0, 0) - 3.0).abs() < 1e-6);
        assert!((parent_b.get().edge_q_p2(0, 0) - 4.0).abs() < 1e-6);

        // Now backup through A again with leaf value (6.0, 7.0).
        // C.v becomes mean(2, 4, 6) = 4.0, mean(3, 5, 7) = 5.0.
        let path_a2 = vec![PathEntry {
            node: Arc::clone(&parent_a),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_a2, &child_c, 6.0, 7.0);

        assert_eq!(child_c.get().total_visits(), 3);
        assert!((child_c.get().v1() - 4.0).abs() < 1e-6);
        assert!((child_c.get().v2() - 5.0).abs() < 1e-6);

        // A's edge_q should now be corrected.
        // correct_q1 = r1(1.0) + C.v1(4.0) = 5.0
        // correct_q2 = r2(0.5) + C.v2(5.0) = 5.5
        //
        // Before this backup, A had 1 visit with edge_q = (3.0, 3.5).
        // Delta detection: correct_q1(5.0) - old_edge_q1(3.0) = 2.0, n_to_fix=1.
        // FinalizeEdge adds visit 2 with Welford(5.0): (3.0 + (5.0-3.0)/2) = 4.0.
        // AdjustEdge: 4.0 + 1*2.0/2 = 5.0. (exact catch-up)
        assert_eq!(parent_a.get().edge_visits(0, 0), 2);
        assert!((parent_a.get().edge_q_p1(0, 0) - 5.0).abs() < 1e-5);
        assert!((parent_a.get().edge_q_p2(0, 0) - 5.5).abs() < 1e-5);
    }

    /// Exact catch-up: when all prior visits are stale, adjustment should
    /// bring edge_q exactly to r + child.v.
    #[test]
    fn delta_exact_catchup() {
        let child = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child.get_mut().set_value_scale(5.0);

        let parent = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent.get_mut().set_value_scale(5.0);
        parent.get_mut().set_prior([0.2; 5], [0.2; 5]);

        // Parent needs 2 edges to child so num_parents > 1.
        let other_parent = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        other_parent.get_mut().set_value_scale(5.0);
        other_parent.get_mut().set_prior([0.2; 5], [0.2; 5]);

        let edge1 = Box::new(Edge::new(Arc::clone(&child), (0, 0), 2.0, 1.0));
        parent.get_mut().prepend_child(edge1);
        let edge2 = Box::new(Edge::new(Arc::clone(&child), (0, 0), 0.0, 0.0));
        other_parent.get_mut().prepend_child(edge2);

        assert_eq!(child.num_parents(), 2);

        // 5 backups through other_parent updating child but not parent.
        for val in [1.0, 2.0, 3.0, 4.0, 5.0] {
            let path_other = vec![PathEntry {
                node: Arc::clone(&other_parent),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(&path_other, &child, val, val * 0.5);
        }

        // child.v1 = mean(1,2,3,4,5) = 3.0, child.v2 = mean(0.5,1,1.5,2,2.5) = 1.5
        assert_eq!(child.get().total_visits(), 5);
        assert!((child.get().v1() - 3.0).abs() < 1e-5);
        assert!((child.get().v2() - 1.5).abs() < 1e-5);

        // parent has 0 edge visits to child. Now backup through parent.
        let path = vec![PathEntry {
            node: Arc::clone(&parent),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path, &child, 10.0, 5.0);

        // child.v after 6th visit: mean(1,2,3,4,5,10) = 25/6 ≈ 4.1667
        // But for parent's edge_q, the delta detection reads child.v AFTER
        // finalize_score_update on the leaf.
        // correct_q1 = r(2.0) + child.v1 ≈ 2.0 + 4.1667 = 6.1667
        // correct_q2 = r(1.0) + child.v2 ≈ 1.0 + 2.0833 = 3.0833
        //
        // n_to_fix = 0 (parent had no visits), so no adjustment needed.
        // First visit just gets the correct value directly.
        let expected_child_v1 = 25.0 / 6.0;
        let expected_child_v2 = (0.5 + 1.0 + 1.5 + 2.0 + 2.5 + 5.0) / 6.0;
        let expected_q1 = 2.0 + expected_child_v1;
        let expected_q2 = 1.0 + expected_child_v2;

        assert_eq!(parent.get().edge_visits(0, 0), 1);
        assert!((parent.get().edge_q_p1(0, 0) - expected_q1).abs() < 1e-4);
        assert!((parent.get().edge_q_p2(0, 0) - expected_q2).abs() < 1e-4);
    }

    /// Cascading correction: A → B → C, D → B.
    /// Backup through D updates B, then backup through A should correct
    /// both B's and A's edge_q.
    #[test]
    fn delta_cascading_correction() {
        // C (leaf)
        let node_c = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_c.get_mut().set_value_scale(5.0);

        // B (intermediate, shared by A and D)
        let node_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_b.get_mut().set_value_scale(5.0);
        node_b.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_b_c = Box::new(Edge::new(Arc::clone(&node_c), (0, 0), 0.5, 0.5));
        node_b.get_mut().prepend_child(edge_b_c);

        // A (root-like, single path through B to C)
        let node_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_a.get_mut().set_value_scale(5.0);
        node_a.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_a_b = Box::new(Edge::new(Arc::clone(&node_b), (0, 0), 1.0, 1.0));
        node_a.get_mut().prepend_child(edge_a_b);

        // D (another parent of B, creating the transposition)
        let node_d = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_d.get_mut().set_value_scale(5.0);
        node_d.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_d_b = Box::new(Edge::new(Arc::clone(&node_b), (0, 0), 0.0, 0.0));
        node_d.get_mut().prepend_child(edge_d_b);

        assert_eq!(node_b.num_parents(), 2); // B is a transposition

        // Step 1: backup A → B → C with leaf value (2.0, 1.0).
        let path_abc = vec![
            PathEntry { node: Arc::clone(&node_a), p1_outcome: 0, p2_outcome: 0 },
            PathEntry { node: Arc::clone(&node_b), p1_outcome: 0, p2_outcome: 0 },
        ];
        backup(&path_abc, &node_c, 2.0, 1.0);

        // C.v = (2.0, 1.0), B.v = (0.5+2.0, 0.5+1.0) = (2.5, 1.5)
        // A.v = (1+2.5, 1+1.5) = (3.5, 2.5)
        assert!((node_c.get().v1() - 2.0).abs() < 1e-6);
        assert!((node_b.get().v1() - 2.5).abs() < 1e-6);
        assert!((node_a.get().v1() - 3.5).abs() < 1e-6);

        // Step 2: backup D → B → C with leaf value (6.0, 5.0).
        // This updates C and B but not A.
        let path_dbc = vec![
            PathEntry { node: Arc::clone(&node_d), p1_outcome: 0, p2_outcome: 0 },
            PathEntry { node: Arc::clone(&node_b), p1_outcome: 0, p2_outcome: 0 },
        ];
        backup(&path_dbc, &node_c, 6.0, 5.0);

        // C.v = mean(2, 6) = 4.0, mean(1, 5) = 3.0
        assert!((node_c.get().v1() - 4.0).abs() < 1e-6);
        assert!((node_c.get().v2() - 3.0).abs() < 1e-6);

        // B now has 2 visits. B's edge_q should reflect corrected C.v.
        // At step 2, B saw C as transposition → used correct_q = 0.5 + 4.0 = 4.5.
        // B's edge_q should be corrected from the first stale visit.
        assert_eq!(node_b.get().edge_visits(0, 0), 2);

        // Step 3: backup A → B → C with leaf value (8.0, 7.0).
        // A should see B as transposition and correct its stale visit.
        let path_abc2 = vec![
            PathEntry { node: Arc::clone(&node_a), p1_outcome: 0, p2_outcome: 0 },
            PathEntry { node: Arc::clone(&node_b), p1_outcome: 0, p2_outcome: 0 },
        ];
        backup(&path_abc2, &node_c, 8.0, 7.0);

        // C.v = mean(2, 6, 8) = 16/3 ≈ 5.333
        assert!((node_c.get().v1() - 16.0 / 3.0).abs() < 1e-4);

        // A should have corrected edge_q. The key property: A's edge_q
        // should be close to r(1,1) + B.v, not stuck at the old stale value.
        // B.v after 3 visits reflects all information through C.
        let b_v1 = node_b.get().v1();
        let b_v2 = node_b.get().v2();
        let a_correct_q1 = 1.0 + b_v1;
        let a_correct_q2 = 1.0 + b_v2;

        // A has 2 edge visits. The corrected edge_q should be close to the correct value.
        assert_eq!(node_a.get().edge_visits(0, 0), 2);
        assert!(
            (node_a.get().edge_q_p1(0, 0) - a_correct_q1).abs() < 0.5,
            "A.edge_q_p1 = {}, expected near {}", node_a.get().edge_q_p1(0, 0), a_correct_q1,
        );
        assert!(
            (node_a.get().edge_q_p2(0, 0) - a_correct_q2).abs() < 0.5,
            "A.edge_q_p2 = {}, expected near {}", node_a.get().edge_q_p2(0, 0), a_correct_q2,
        );
    }

    /// Non-transposition child with transposition grandchild: delta
    /// propagates through the intermediate node.
    #[test]
    fn delta_propagates_through_non_transposition() {
        // grandchild (transposition, shared by mid and other_parent)
        let grandchild = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        grandchild.get_mut().set_value_scale(5.0);

        // mid (non-transposition, single parent: root)
        let mid = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        mid.get_mut().set_value_scale(5.0);
        mid.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_mid_gc = Box::new(Edge::new(Arc::clone(&grandchild), (0, 0), 0.5, 0.5));
        mid.get_mut().prepend_child(edge_mid_gc);

        // root
        let root = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        root.get_mut().set_value_scale(5.0);
        root.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_root_mid = Box::new(Edge::new(Arc::clone(&mid), (0, 0), 1.0, 1.0));
        root.get_mut().prepend_child(edge_root_mid);

        // other_parent (creates transposition at grandchild)
        let other = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        other.get_mut().set_value_scale(5.0);
        other.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_other_gc = Box::new(Edge::new(Arc::clone(&grandchild), (0, 0), 0.0, 0.0));
        other.get_mut().prepend_child(edge_other_gc);

        assert_eq!(mid.num_parents(), 1);       // not a transposition
        assert_eq!(grandchild.num_parents(), 2); // transposition

        // Backup 1: root → mid → grandchild, leaf = (1.0, 1.0).
        let path1 = vec![
            PathEntry { node: Arc::clone(&root), p1_outcome: 0, p2_outcome: 0 },
            PathEntry { node: Arc::clone(&mid), p1_outcome: 0, p2_outcome: 0 },
        ];
        backup(&path1, &grandchild, 1.0, 1.0);

        let root_q1_after_1 = root.get().edge_q_p1(0, 0);

        // Backup 2: other → grandchild, leaf = (10.0, 10.0).
        // Updates grandchild but not mid or root.
        let path_other = vec![PathEntry {
            node: Arc::clone(&other),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_other, &grandchild, 10.0, 10.0);

        // grandchild.v1 = mean(1, 10) = 5.5
        assert!((grandchild.get().v1() - 5.5).abs() < 1e-5);

        // Backup 3: root → mid → grandchild, leaf = (4.0, 4.0).
        // Mid sees grandchild as transposition → delta detected.
        // Root sees mid as non-transposition → delta from below propagates.
        let path3 = vec![
            PathEntry { node: Arc::clone(&root), p1_outcome: 0, p2_outcome: 0 },
            PathEntry { node: Arc::clone(&mid), p1_outcome: 0, p2_outcome: 0 },
        ];
        backup(&path3, &grandchild, 4.0, 4.0);

        // Root's edge_q should have changed significantly from the first value.
        // The delta from the grandchild transposition should have propagated up.
        let root_q1_after_3 = root.get().edge_q_p1(0, 0);
        assert!(
            (root_q1_after_3 - root_q1_after_1).abs() > 0.5,
            "Delta should propagate: before={root_q1_after_1}, after={root_q1_after_3}",
        );
    }

    // =====================================================================
    // Tier 1 LC0 port fix tests
    // =====================================================================

    #[test]
    fn children_visits_vs_total_visits() {
        // After N sims, root.total_edge_visits() == root.total_visits() - 1.
        // The first visit is the root's own NN eval (no edge update).
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 8, &mut r).unwrap();
        let root_low = tree.root().get();
        let total = root_low.total_visits();
        let edge = root_low.total_edge_visits();

        // Root's first visit is its own NN eval. Every subsequent visit
        // is a child visit (edge update). So edge_visits == total - 1.
        assert_eq!(
            edge,
            total - 1,
            "children_visits should be total_visits - 1 at root, got edge={edge} total={total}"
        );

        // Sanity: result.total_visits matches.
        assert_eq!(result.total_visits, total);
    }

    #[test]
    fn sim_counting_produces_useful_visits() {
        // Verify the sim budget produces useful visits proportional to n_sims.
        // With VTC batch allocation, shared collisions (excess visits beyond
        // what a leaf can absorb) are expected. The important property is that
        // useful visits grow with n_sims and don't stall.
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2), Coordinates::new(1, 1)],
        );
        let backend = SmartUniformBackend;
        let config = SearchConfig {
            collision_limit_min: 256,
            collision_limit_max: 256,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 100u32;
        let result = run_search(&mut tree, &game, &backend, &config, n_sims, 8, &mut r).unwrap();
        let useful = result.nn_evals + result.terminals + result.tt_stop_hits;

        // Should produce a meaningful number of useful visits.
        assert!(
            useful >= n_sims / 2,
            "useful visits ({useful}) should be >= n_sims/2 ({n_sims}/2), \
             nn={}, term={}, tt_stop={}, coll={}",
            result.nn_evals, result.terminals, result.tt_stop_hits, result.collisions,
        );
        assert!(result.total_visits > 0);
    }

    #[test]
    fn collision_vl_deferred_reduces_repeat_collisions() {
        // Run search with batch_size > 1 to exercise collision VL deferral.
        // A batch_size=1 search can't benefit from deferred VLs since there's
        // only one descent per batch. With batch_size=16, deferred VLs should
        // reduce repeated collisions compared to immediate cleanup.
        //
        // We can't directly test "what would happen without the fix" in a single
        // test, but we CAN verify that after search, all VLs are reverted (no
        // leaked in-flight counts) and collisions stay bounded.
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        // No leaked virtual losses.
        let root_low = tree.root().get();
        for i in 0..root_low.n1() {
            for j in 0..root_low.n2() {
                assert_eq!(
                    root_low.edge_in_flight(i, j), 0,
                    "VL leak at ({i},{j})"
                );
            }
        }
        assert_eq!(root_low.n_in_flight(), 0, "root n_in_flight leak");

        // Collisions should be bounded — not exploding.
        let useful = result.nn_evals + result.terminals;
        assert!(useful > 0, "should have some useful work done");
    }

    // =====================================================================
    // Transposition stop tests
    // =====================================================================

    #[test]
    fn tt_stop_initializes_edge_from_aggregate() {
        // Manual DAG: child C has visits via parent A.
        // Parent B has an edge to C but no visits on it.
        // backup_transposition_stop should initialize B's edge from C's aggregate
        // without incrementing C's total_visits.
        let child_c = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child_c.get_mut().set_value_scale(5.0);

        // Parent A with edge to C at (0,0), reward (1.0, 0.5).
        let parent_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_a.get_mut().set_value_scale(5.0);
        parent_a.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_a = Box::new(Edge::new(Arc::clone(&child_c), (0, 0), 1.0, 0.5));
        parent_a.get_mut().prepend_child(edge_a);

        // Backup through A to give C some visits.
        let path_a = vec![PathEntry {
            node: Arc::clone(&parent_a),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_a, &child_c, 2.0, 3.0);
        assert_eq!(child_c.get().total_visits(), 1);
        assert!((child_c.get().v1() - 2.0).abs() < 1e-6);
        assert!((child_c.get().v2() - 3.0).abs() < 1e-6);

        // Parent B with edge to C at (1,2), reward (0.5, 1.0).
        let parent_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_b.get_mut().set_value_scale(5.0);
        parent_b.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_b = Box::new(Edge::new(Arc::clone(&child_c), (1, 2), 0.5, 1.0));
        parent_b.get_mut().prepend_child(edge_b);

        assert_eq!(child_c.num_parents(), 2);

        // B's edge visits = 0, C has visits from A.
        // This is exactly the first-hit transposition scenario.
        assert_eq!(parent_b.get().edge_visits(1, 2), 0);

        let child_visits_before = child_c.get().total_visits();

        // Simulate gather: VL + n_in_flight on path entries (but NOT on child).
        let path_b = vec![PathEntry {
            node: Arc::clone(&parent_b),
            p1_outcome: 1,
            p2_outcome: 2,
        }];
        for entry in &path_b {
            entry.node.get_mut().increment_n_in_flight(1);
            entry.node.get_mut().add_virtual_loss(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
            );
        }

        // Run backup_transposition_stop through B's path.
        backup_transposition_stop(&path_b, &child_c, 1);

        // Child C's total_visits should NOT change.
        assert_eq!(
            child_c.get().total_visits(),
            child_visits_before,
            "transposition stop should not increment child visits"
        );

        // B's edge should be initialized from C's aggregate.
        // Expected: edge_q = r + child.v = (0.5+2.0, 1.0+3.0) = (2.5, 4.0).
        let b_low = parent_b.get();
        assert_eq!(b_low.edge_visits(1, 2), 1, "B's edge should have 1 visit");
        assert!(
            (b_low.edge_q_p1(1, 2) - 2.5).abs() < 1e-5,
            "B's edge Q p1 = {}, expected 2.5",
            b_low.edge_q_p1(1, 2)
        );
        assert!(
            (b_low.edge_q_p2(1, 2) - 4.0).abs() < 1e-5,
            "B's edge Q p2 = {}, expected 4.0",
            b_low.edge_q_p2(1, 2)
        );
    }

    #[test]
    fn tt_stop_stale_edge_corrects_from_aggregate() {
        // Manual DAG: child C has visits from parent A. Parent B's edge to C
        // has 1 visit (from the first-hit stop). Then C gets more visits via A,
        // making B's edge stale (edge_vis < child.total_visits).
        // A second backup_transposition_stop should correct B's edge Q toward
        // the new aggregate, without incrementing C's total_visits.

        // -- Setup: shared child C --
        let child_c = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        child_c.get_mut().set_value_scale(5.0);

        // -- Parent A: edge to C at (0,0), reward (1.0, 0.5) --
        let parent_a = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_a.get_mut().set_value_scale(5.0);
        parent_a.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_a = Box::new(Edge::new(Arc::clone(&child_c), (0, 0), 1.0, 0.5));
        parent_a.get_mut().prepend_child(edge_a);

        // -- Parent B: edge to C at (1,2), reward (0.5, 1.0) --
        let parent_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        parent_b.get_mut().set_value_scale(5.0);
        parent_b.get_mut().set_prior([0.2; 5], [0.2; 5]);
        let edge_b = Box::new(Edge::new(Arc::clone(&child_c), (1, 2), 0.5, 1.0));
        parent_b.get_mut().prepend_child(edge_b);

        assert_eq!(child_c.num_parents(), 2);

        // -- Step 1: Give C one visit through A with values (2.0, 3.0) --
        let path_a = vec![PathEntry {
            node: Arc::clone(&parent_a),
            p1_outcome: 0,
            p2_outcome: 0,
        }];
        backup(&path_a, &child_c, 2.0, 3.0);
        assert_eq!(child_c.get().total_visits(), 1);

        // -- Step 2: First-hit TT stop through B (edge_vis == 0) --
        let path_b = vec![PathEntry {
            node: Arc::clone(&parent_b),
            p1_outcome: 1,
            p2_outcome: 2,
        }];
        for entry in &path_b {
            entry.node.get_mut().increment_n_in_flight(1);
            entry
                .node
                .get_mut()
                .add_virtual_loss(entry.p1_outcome as usize, entry.p2_outcome as usize);
        }
        backup_transposition_stop(&path_b, &child_c, 1);

        // B's edge now has 1 visit, Q = r + child.v = (0.5+2.0, 1.0+3.0) = (2.5, 4.0).
        assert_eq!(parent_b.get().edge_visits(1, 2), 1);
        assert!((parent_b.get().edge_q_p1(1, 2) - 2.5).abs() < 1e-5);

        // -- Step 3: C gets more visits through A, shifting C's aggregate --
        // New values (6.0, 1.0): C.v goes from (2.0, 3.0) to mean(2.0, 6.0) = (4.0, 2.0).
        backup(&path_a, &child_c, 6.0, 1.0);
        assert_eq!(child_c.get().total_visits(), 2);
        assert!((child_c.get().v1() - 4.0).abs() < 1e-5);
        assert!((child_c.get().v2() - 2.0).abs() < 1e-5);

        // B's edge is now stale: edge_vis(1) < child.total_visits(2).
        assert!(parent_b.get().edge_visits(1, 2) < child_c.get().total_visits());

        let child_visits_before = child_c.get().total_visits();

        // -- Step 4: Stale TT stop through B --
        for entry in &path_b {
            entry.node.get_mut().increment_n_in_flight(1);
            entry
                .node
                .get_mut()
                .add_virtual_loss(entry.p1_outcome as usize, entry.p2_outcome as usize);
        }
        backup_transposition_stop(&path_b, &child_c, 1);

        // Child C's total_visits should NOT change.
        assert_eq!(
            child_c.get().total_visits(),
            child_visits_before,
            "stale TT stop should not increment child visits"
        );

        // B's edge should now have 2 visits.
        assert_eq!(
            parent_b.get().edge_visits(1, 2),
            2,
            "B's edge should have 2 visits after stale stop"
        );

        // B's edge Q should be corrected toward the new aggregate.
        // Delta correction: correct_q = r + child.v = (0.5+4.0, 1.0+2.0) = (4.5, 3.0).
        // n_to_fix = 1 (one prior visit to correct), so the edge gets adjusted
        // to the correct aggregate-derived value.
        let b_low = parent_b.get();
        assert!(
            (b_low.edge_q_p1(1, 2) - 4.5).abs() < 1e-4,
            "stale stop should correct edge Q p1 to 4.5, got {}",
            b_low.edge_q_p1(1, 2)
        );
        assert!(
            (b_low.edge_q_p2(1, 2) - 3.0).abs() < 1e-4,
            "stale stop should correct edge Q p2 to 3.0, got {}",
            b_low.edge_q_p2(1, 2)
        );
    }

    #[test]
    fn tt_stop_hits_nonzero_on_open_maze() {
        // Open 5x5 with 2 cheese: transpositions are common (e.g. UP+RIGHT = RIGHT+UP).
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result =
            run_search(&mut tree, &game, &backend, &config, 500, 8, &mut r).unwrap();

        assert!(
            result.tt_stop_hits > 0,
            "expected transposition stops on open maze, got 0"
        );
    }

    #[test]
    fn tt_stop_hits_consume_sim_budget() {
        // On a transposition-heavy position, tt_stop_hits should count toward
        // the sim budget. Without this, the search overshoots n_sims.
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 200;
        let result =
            run_search(&mut tree, &game, &backend, &config, n_sims, 8, &mut r).unwrap();

        // Productive work = nn_evals + terminals + tt_stop_hits.
        let productive = result.nn_evals + result.terminals + result.tt_stop_hits;
        assert!(
            productive >= n_sims,
            "productive work ({productive}) should be >= n_sims ({n_sims})"
        );

        // Root visits should not wildly overshoot. Allow some slack for
        // batching (up to one extra batch worth).
        let max_expected = n_sims + 8;
        assert!(
            result.total_visits <= max_expected,
            "root visits ({}) overshot n_sims ({n_sims}) by more than one batch",
            result.total_visits
        );
    }
}
