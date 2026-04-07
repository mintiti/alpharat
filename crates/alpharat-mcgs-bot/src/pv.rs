use std::sync::Arc;

use alpharat_mcgs::{Edge, LowNode, MCGSTree, SharedNode};
use pyrat_sdk::{Coordinates, Direction, GameSim};

/// A single principal variation line rooted at one of the top root moves.
pub struct PvLine {
    /// Sequence of moves for the subject player along this line.
    pub moves: Vec<Direction>,
    /// First cheese the subject player collects along this PV, as (x, y).
    pub target: Option<(u8, u8)>,
    /// Marginal Q-value of the root action (expected remaining cheese).
    pub score: f32,
}

/// Extract multi-PV lines for one player from the current tree.
///
/// Returns up to `max_lines` PV lines, walking until the tree runs out.
/// Lines are sorted by marginal visit count (most-visited first),
/// with marginal Q then prior as tiebreakers (lc0-style).
pub fn extract_pvs(
    tree: &MCGSTree,
    sim: &GameSim,
    is_player1: bool,
    max_lines: usize,
) -> Vec<PvLine> {
    let root_shared = tree.root();
    let root_low = root_shared.get();

    let n = if is_player1 { root_low.n1() } else { root_low.n2() };
    if n == 0 {
        return Vec::new();
    }

    // Sort outcome indices: visits desc -> Q desc -> prior desc (lc0-style).
    let mut indices: Vec<usize> = (0..n).collect();
    if is_player1 {
        indices.sort_unstable_by(|&a, &b| {
            rank_key_p1(root_low, b)
                .partial_cmp(&rank_key_p1(root_low, a))
                .unwrap()
        });
        indices.retain(|&i| root_low.marginal_visits_p1(i) > 0);
    } else {
        indices.sort_unstable_by(|&a, &b| {
            rank_key_p2(root_low, b)
                .partial_cmp(&rank_key_p2(root_low, a))
                .unwrap()
        });
        indices.retain(|&j| root_low.marginal_visits_p2(j) > 0);
    }
    indices.truncate(max_lines);

    // Subject player's starting position.
    let start_pos = if is_player1 {
        sim.player1_position()
    } else {
        sim.player2_position()
    };

    indices
        .into_iter()
        .map(|root_outcome_idx| {
            let score = if is_player1 {
                marginal_q_p1(root_low, root_outcome_idx)
            } else {
                marginal_q_p2(root_low, root_outcome_idx)
            };
            let root_action = if is_player1 {
                root_low.p1_outcome_action(root_outcome_idx)
            } else {
                root_low.p2_outcome_action(root_outcome_idx)
            };
            let root_dir =
                Direction::try_from(root_action).expect("invalid action from outcome_action");

            let mut moves = vec![root_dir];
            let mut target = None;
            let mut pos = root_dir.apply_to(start_pos);

            // Opponent's best root action (most-visited, with tiebreaking).
            let opp_best_idx = if is_player1 {
                best_p2_outcome_idx(root_low)
            } else {
                best_p1_outcome_idx(root_low)
            };

            // Map (subject_idx, opponent_idx) -> (p1_idx, p2_idx).
            let (p1_idx, p2_idx) = if is_player1 {
                (root_outcome_idx as u8, opp_best_idx)
            } else {
                (opp_best_idx, root_outcome_idx as u8)
            };

            // Find the root child edge and start walking.
            let mut current: Option<Arc<SharedNode>> =
                root_low.find_child(p1_idx, p2_idx).map(|edge| {
                    check_edge_reward(edge, is_player1, pos, &mut target);
                    Arc::clone(edge.low_node())
                });

            // Walk deeper into the tree.
            loop {
                let next = {
                    let Some(node) = current.as_ref() else {
                        break;
                    };
                    let low = node.get();
                    if low.is_terminal() || low.total_edge_visits() == 0 {
                        break;
                    }
                    if low.n1() == 0 || low.n2() == 0 {
                        break;
                    }

                    let subj_idx = if is_player1 {
                        best_p1_outcome_idx(low)
                    } else {
                        best_p2_outcome_idx(low)
                    };
                    let opp_idx = if is_player1 {
                        best_p2_outcome_idx(low)
                    } else {
                        best_p1_outcome_idx(low)
                    };

                    let subj_action = if is_player1 {
                        low.p1_outcome_action(subj_idx as usize)
                    } else {
                        low.p2_outcome_action(subj_idx as usize)
                    };
                    let subj_dir =
                        Direction::try_from(subj_action).unwrap_or(Direction::Stay);
                    moves.push(subj_dir);
                    pos = subj_dir.apply_to(pos);

                    // Map back to (p1_idx, p2_idx) for child lookup.
                    let (ci, cj) = if is_player1 {
                        (subj_idx, opp_idx)
                    } else {
                        (opp_idx, subj_idx)
                    };

                    low.find_child(ci, cj).map(|edge| {
                        check_edge_reward(edge, is_player1, pos, &mut target);
                        Arc::clone(edge.low_node())
                    })
                };
                current = next;
            }

            PvLine {
                moves,
                target,
                score,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Marginal Q for p1 outcome i: visit-weighted average over j.
fn marginal_q_p1(low: &LowNode, i: usize) -> f32 {
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
fn marginal_q_p2(low: &LowNode, j: usize) -> f32 {
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

/// Ranking key for p1 outcomes: (marginal visits, marginal Q, prior). All descending.
fn rank_key_p1(low: &LowNode, i: usize) -> (u32, f32, f32) {
    (
        low.marginal_visits_p1(i),
        marginal_q_p1(low, i),
        low.p1_prior(i),
    )
}

/// Ranking key for p2 outcomes: (marginal visits, marginal Q, prior). All descending.
fn rank_key_p2(low: &LowNode, j: usize) -> (u32, f32, f32) {
    (
        low.marginal_visits_p2(j),
        marginal_q_p2(low, j),
        low.p2_prior(j),
    )
}

/// Return the p1 outcome index with the best ranking (visits -> Q -> prior).
pub(crate) fn best_p1_outcome_idx(low: &LowNode) -> u8 {
    let n = low.n1();
    let mut best = 0u8;
    let mut best_key = rank_key_p1(low, 0);
    for i in 1..n {
        let key = rank_key_p1(low, i);
        if key > best_key {
            best_key = key;
            best = i as u8;
        }
    }
    best
}

/// Return the p2 outcome index with the best ranking (visits -> Q -> prior).
pub(crate) fn best_p2_outcome_idx(low: &LowNode) -> u8 {
    let n = low.n2();
    let mut best = 0u8;
    let mut best_key = rank_key_p2(low, 0);
    for j in 1..n {
        let key = rank_key_p2(low, j);
        if key > best_key {
            best_key = key;
            best = j as u8;
        }
    }
    best
}

/// If the edge reward is positive, record the position as the target cheese (first only).
fn check_edge_reward(
    edge: &Edge,
    is_player1: bool,
    pos: Coordinates,
    target: &mut Option<(u8, u8)>,
) {
    if target.is_some() {
        return;
    }
    let reward = if is_player1 { edge.r1() } else { edge.r2() };
    if reward > 0.0 {
        *target = Some((pos.x, pos.y));
    }
}
