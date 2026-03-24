use std::sync::Arc;

use rand::Rng;
use rand_distr::Gamma;

use crate::node::SharedNode;
use crate::tree::{compute_rewards, find_or_create_child, populate_node, MCGSTree};
use crate::{Backend, BackendError};
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
    pub noise_concentration: f32,
    /// Maximum collision retries per batch before stopping the gather phase.
    /// 0 = use batch_size as the limit (default).
    pub max_collisions: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            fpu_reduction: 0.2,
            force_k: 2.0,
            noise_epsilon: 0.0,
            noise_concentration: 10.83,
            max_collisions: 0,
        }
    }
}

/// A single step on the search path.
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

/// What happened at the leaf of a single PUCT descent.
enum DescentOutcome {
    /// Leaf needs NN evaluation.
    NeedsEval {
        path: SearchPath,
        leaf: Arc<SharedNode>,
        game_state: GameState,
    },
    /// Leaf is terminal (game over).
    Terminal {
        path: SearchPath,
        leaf: Arc<SharedNode>,
        /// Whether try_start_score_update was called on the leaf.
        leaf_claimed: bool,
    },
    /// Collision: another descent already claimed this unvisited leaf.
    Collision { path: SearchPath },
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
    while remaining > 0 {
        let actual = remaining.min(batch_size);
        let batch = simulate_batch(tree, game, backend, config, actual, rng)?;
        total_nn_evals += batch.nn_evals;
        total_terminals += batch.terminals;
        total_collisions += batch.collisions;
        remaining -= actual;
    }

    let root = Arc::clone(tree.root());
    let mut result = extract_result(&root, config, rng);
    result.nn_evals = total_nn_evals;
    result.terminals = total_terminals;
    result.collisions = total_collisions;
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

    let total_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    // FPU: compute visited prior mass.
    let mut visited_prior_mass = 0.0f32;
    for i in 0..n {
        if low.marginal_visits_p1(i) > 0 {
            visited_prior_mass += low.p1_prior(i);
        }
    }
    let fpu = low.v1() - config.fpu_reduction * value_scale * visited_prior_mass.sqrt();

    let sqrt_total = (total_visits as f32).sqrt();

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
            let threshold = (config.force_k * prior * total_visits as f32).sqrt();
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

    let total_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    // FPU: compute visited prior mass.
    let mut visited_prior_mass = 0.0f32;
    for j in 0..n {
        if low.marginal_visits_p2(j) > 0 {
            visited_prior_mass += low.p2_prior(j);
        }
    }
    let fpu = low.v2() - config.fpu_reduction * value_scale * visited_prior_mass.sqrt();

    let sqrt_total = (total_visits as f32).sqrt();

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
            let threshold = (config.force_k * prior * total_visits as f32).sqrt();
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
// simulate_batch — gather/eval/backup cycle
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
    let max_collisions = if config.max_collisions > 0 {
        config.max_collisions
    } else {
        batch_size
    };

    // ---- OOO Gather Phase ----
    let max_ooo = 2 * batch_size;
    let mut nn_outcomes: Vec<DescentOutcome> = Vec::with_capacity(batch_size as usize);
    let mut collisions = 0u32;
    let mut terminals = 0u32;
    let mut n_ooo = 0u32;

    while (nn_outcomes.len() as u32) < batch_size
        && collisions < max_collisions
        && n_ooo < max_ooo
    {
        let outcome = descend(tree, &root, game, config, rng);

        match outcome {
            DescentOutcome::NeedsEval { .. } => {
                nn_outcomes.push(outcome);
            }
            DescentOutcome::Terminal {
                ref path,
                ref leaf,
                leaf_claimed,
            } => {
                if leaf.get().total_visits() == 0 {
                    populate_node(leaf, None);
                }
                backup(path, leaf, 0.0, 0.0);
                cleanup_descent(path, if leaf_claimed { Some(leaf) } else { None });
                terminals += 1;
                n_ooo += 1;
            }
            DescentOutcome::Collision { ref path } => {
                cleanup_descent(path, None);
                collisions += 1;
            }
        }
    }

    let nn_evals = nn_outcomes.len() as u32;

    // ---- Eval Phase ----
    let needs_eval_refs: Vec<&GameState> = nn_outcomes
        .iter()
        .filter_map(|o| match o {
            DescentOutcome::NeedsEval { game_state, .. } => Some(game_state),
            _ => None,
        })
        .collect();

    let eval_results = if needs_eval_refs.is_empty() {
        Vec::new()
    } else {
        backend.evaluate_batch(&needs_eval_refs)?
    };

    // ---- Backup Phase ----
    let mut eval_idx = 0;
    for outcome in nn_outcomes {
        if let DescentOutcome::NeedsEval { path, leaf, .. } = outcome {
            let eval = &eval_results[eval_idx];
            eval_idx += 1;

            populate_node(&leaf, Some(eval));

            if Arc::ptr_eq(&leaf, &root) && config.noise_epsilon > 0.0 {
                apply_dirichlet_noise_p1(&leaf, config.noise_epsilon, config.noise_concentration, rng);
                apply_dirichlet_noise_p2(&leaf, config.noise_epsilon, config.noise_concentration, rng);
            }

            backup(&path, &leaf, eval.value_p1, eval.value_p2);
            cleanup_descent(&path, Some(&leaf));
        }
    }

    Ok(BatchStats {
        nn_evals,
        terminals,
        collisions,
    })
}

// ---------------------------------------------------------------------------
// descend — single PUCT descent with virtual loss
// ---------------------------------------------------------------------------

fn descend(
    tree: &mut MCGSTree,
    root: &Arc<SharedNode>,
    game: &GameState,
    config: &SearchConfig,
    rng: &mut impl Rng,
) -> DescentOutcome {
    let mut current = Arc::clone(root);
    let mut path: SearchPath = Vec::new();
    let mut game = game.clone();

    loop {
        let low = current.get();

        // Unvisited leaf — try to claim it.
        if low.total_visits() == 0 && !low.is_terminal() {
            if !current.get_mut().try_start_score_update() {
                return DescentOutcome::Collision { path };
            }
            if game.check_game_over() {
                return DescentOutcome::Terminal {
                    path,
                    leaf: current,
                    leaf_claimed: true,
                };
            }
            return DescentOutcome::NeedsEval {
                path,
                leaf: current,
                game_state: game,
            };
        }

        // Revisited terminal — no claim needed.
        if low.is_terminal() {
            return DescentOutcome::Terminal {
                path,
                leaf: current,
                leaf_claimed: false,
            };
        }

        // Interior node: select actions via PUCT.
        let is_root = Arc::ptr_eq(&current, root);
        let (idx1, idx2) = select_actions(&current, config, is_root, rng);

        // Add virtual loss on selected (i, j).
        current.get_mut().add_virtual_loss(idx1 as usize, idx2 as usize);

        // Convert outcome indices to actions.
        let a1 = low.p1_outcome_action(idx1 as usize);
        let a2 = low.p2_outcome_action(idx2 as usize);

        // Record path step.
        path.push(PathEntry {
            node: Arc::clone(&current),
            p1_outcome: idx1,
            p2_outcome: idx2,
        });

        // Advance game state.
        let scores_before = (game.player1_score(), game.player2_score());
        let d1 = Direction::try_from(a1).expect("valid direction");
        let d2 = Direction::try_from(a2).expect("valid direction");
        let _undo = game.make_move(d1, d2);
        let (r1, r2) = compute_rewards(&game, scores_before);

        // Find or create child — TT interaction happens here.
        let (child, _is_new) =
            find_or_create_child(&current, idx1, idx2, &game, tree.tt_mut(), r1, r2);

        current = child;
    }
}

// ---------------------------------------------------------------------------
// backup — walk leaf→root, updating values
// ---------------------------------------------------------------------------

/// Walk leaf→root, updating LowNode values and Edge Q along the path.
///
/// `g1, g2` are the leaf evaluation (NN value or terminal reward).
fn backup(path: &[PathEntry], leaf: &SharedNode, g1: f32, g2: f32) {
    // Visit 1 on the leaf: NN eval or terminal value.
    leaf.get_mut().update_value(g1, g2);

    let mut v1 = g1;
    let mut v2 = g2;

    for entry in path.iter().rev() {
        let i = entry.p1_outcome as usize;
        let j = entry.p2_outcome as usize;

        // Find the edge to get transition rewards.
        let edge = entry.node.get().find_child(entry.p1_outcome, entry.p2_outcome)
            .expect("backup: edge must exist");
        let q1 = edge.r1() + v1;
        let q2 = edge.r2() + v2;

        // Update per-edge aggregate.
        let edge_mut = entry.node.get_mut().find_child_mut(entry.p1_outcome, entry.p2_outcome)
            .expect("backup: edge must exist");
        edge_mut.update_value(q1, q2);

        // Update LowNode aggregate.
        let node = entry.node.get_mut();
        node.update_value(q1, q2);

        // Update joint matrix.
        node.update_edge(i, j, q1, q2);

        v1 = q1;
        v2 = q2;
    }
}

// ---------------------------------------------------------------------------
// cleanup_descent — revert virtual losses
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// extract_result — policies and values from root
// ---------------------------------------------------------------------------

fn extract_result(
    root: &SharedNode,
    config: &SearchConfig,
    _rng: &mut impl Rng,
) -> SearchResult {
    let low = root.get();
    let total_visits = low.total_edge_visits();

    let (policy_p1, visit_counts_p1, value_p1) = extract_p1(low, config);
    let (policy_p2, visit_counts_p2, value_p2) = extract_p2(low, config);

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
        total_visits,
        nn_evals: 0,
        terminals: 0,
        collisions: 0,
    }
}

/// Extract policy, visit counts, and value for player 1 from root.
fn extract_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32) {
    let n = low.n1();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v1());
    }

    let total_visits = low.total_edge_visits();

    // Compute FPU for unvisited outcomes.
    let mut visited_prior_mass = 0.0f32;
    for i in 0..n {
        if low.marginal_visits_p1(i) > 0 {
            visited_prior_mass += low.p1_prior(i);
        }
    }
    let fpu = low.v1() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt();

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
    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, total_visits, config.c_puct);

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

    // Value = dot(q, raw_visits) / sum(raw_visits).
    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|i| q[i] * raw_visits[i]).sum();
        dot / visit_sum
    } else {
        low.v1()
    };

    (policy, visit_counts, value)
}

/// Extract policy, visit counts, and value for player 2 from root.
fn extract_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32) {
    let n = low.n2();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v2());
    }

    let total_visits = low.total_edge_visits();

    let mut visited_prior_mass = 0.0f32;
    for j in 0..n {
        if low.marginal_visits_p2(j) > 0 {
            visited_prior_mass += low.p2_prior(j);
        }
    }
    let fpu = low.v2() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt();

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

    let pruned = compute_pruned_visits(&q_norm, &prior, &raw_visits, n, total_visits, config.c_puct);

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

    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|j| q[j] * raw_visits[j]).sum();
        dot / visit_sum
    } else {
        low.v2()
    };

    (policy, visit_counts, value)
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
        root.get_mut().update_edge(0, 0, 1.0, 1.0);
        root.get_mut().update_edge(0, 0, 1.0, 1.0);
        root.get_mut().update_value(1.0, 1.0);
        root.get_mut().update_value(1.0, 1.0);

        let config = default_config();
        let mut r = rng();

        // With visits on (0,0), PUCT should explore other outcomes
        let (a1, a2) = select_actions(&root, &config, true, &mut r);
        assert!((a1 as usize) < root.get().n1());
        assert!((a2 as usize) < root.get().n2());
    }

    // ---- marginal_q ----

    #[test]
    fn marginal_q_p1_weighted_average() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_value_scale(5.0);

        // i=0, j=0: Q=2.0, 3 visits
        for _ in 0..3 {
            low.update_edge(0, 0, 2.0, 1.0);
        }
        // i=0, j=1: Q=4.0, 1 visit
        low.update_edge(0, 1, 4.0, 2.0);

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

        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();

        // Terminal root: should be detected immediately
        assert!(result.terminals > 0 || result.total_visits == 0);
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
                    node.get_mut().update_edge(i, j, q, q);
                }
            }
        }
        // Need some total_visits for value to be nonzero
        for _ in 0..250 {
            node.get_mut().update_value(1.0, 1.0);
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
        node.get_mut().update_edge(0, 0, 1.0, 1.0);
        node.get_mut().update_value(1.0, 1.0);

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
                node.get_mut().update_edge(i, 0, 1.0, 1.0);
                node.get_mut().update_value(1.0, 1.0);
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
        node_lo.get_mut().update_value(5.0, 5.0);
        // Visit 1 outcome → visited_mass = 0.2
        node_lo.get_mut().update_edge(0, 0, 5.0, 5.0);

        let node_hi = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_hi.get_mut().set_value_scale(5.0);
        node_hi.get_mut().set_prior([0.2; 5], [0.2; 5]);
        node_hi.get_mut().update_value(5.0, 5.0);
        // Visit 3 outcomes → visited_mass = 0.6
        node_hi.get_mut().update_edge(0, 0, 5.0, 5.0);
        node_hi.get_mut().update_edge(1, 0, 5.0, 5.0);
        node_hi.get_mut().update_edge(2, 0, 5.0, 5.0);

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
        node.get_mut().update_edge(0, 0, 0.0, 0.0);
        node.get_mut().update_value(0.0, 0.0);

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
            node.get_mut().update_edge(0, 0, 1.0, 1.0);
            node.get_mut().update_edge(1, 0, 1.0, 1.0);
            node.get_mut().update_value(1.0, 1.0);
            node.get_mut().update_value(1.0, 1.0);
        }
        // Outcomes 2,3,4 have 1 visit each
        for i in 2..5 {
            node.get_mut().update_edge(i, 0, 1.0, 1.0);
            node.get_mut().update_value(1.0, 1.0);
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
            node.get_mut().update_edge(0, 0, 10.0, 10.0);
            node.get_mut().update_value(10.0, 10.0);
        }
        for i in 1..5 {
            node.get_mut().update_edge(i, 0, 1.0, 1.0);
            node.get_mut().update_value(1.0, 1.0);
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
                node.get_mut().update_edge(0, 0, 10.0, 10.0);
                node.get_mut().update_value(10.0, 10.0);
            }
            // Outcomes 1-4: low Q, few visits
            for i in 1..5 {
                for _ in 0..5 {
                    node.get_mut().update_edge(i, 0, 0.1, 0.1);
                    node.get_mut().update_value(0.1, 0.1);
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
        node_a.get_mut().update_edge(0, 0, 2.0, 2.0);
        node_a.get_mut().update_value(2.0, 2.0);

        let node_b = Arc::new(SharedNode::new(LowNode::new_shell(
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        )));
        node_b.get_mut().set_value_scale(5.0);
        node_b.get_mut().set_prior(
            [0.1, 0.3, 0.2, 0.15, 0.25],
            [0.05, 0.05, 0.05, 0.05, 0.8],
        );
        node_b.get_mut().update_edge(0, 0, 2.0, 2.0);
        node_b.get_mut().update_value(2.0, 2.0);

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
            node.get_mut().update_edge(i, 0, 2.0, 2.0);
            node.get_mut().update_value(2.0, 2.0);
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
        node.get_mut().update_edge(0, 0, 1.0, 1.0);
        node.get_mut().update_value(1.0, 1.0);

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
        // Low max_collisions to test collision budget
        let config = SearchConfig {
            max_collisions: 1,
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
}
