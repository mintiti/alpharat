use alpharat_mcts::{find_child, HalfNode, MCTSTree};
use pyrat_sdk::{Coordinates, Direction, GameSim};

/// A single principal variation line rooted at one of the top root moves.
pub struct PvLine {
    /// Sequence of moves for the subject player along this line.
    pub moves: Vec<Direction>,
    /// First cheese the subject player collects along this PV, as (x, y).
    pub target: Option<(u8, u8)>,
    /// Edge Q-value of the root action (expected remaining cheese).
    pub score: f32,
}

/// Extract multi-PV lines for one player from the current tree.
///
/// Returns up to `max_lines` PV lines, walking until the tree runs out.
/// Lines are sorted by root edge visit count (most-visited first),
/// with Q then prior as tiebreakers (lc0-style).
pub fn extract_pvs(
    tree: &MCTSTree,
    sim: &GameSim,
    is_player1: bool,
    max_lines: usize,
) -> Vec<PvLine> {
    let arena = tree.arena();
    let root = tree.root();
    let root_node = &arena[root];

    let (half, opponent_half) = if is_player1 {
        (&root_node.p1, &root_node.p2)
    } else {
        (&root_node.p2, &root_node.p1)
    };
    let n = half.n_outcomes();
    if n == 0 {
        return Vec::new();
    }

    // Sort outcome indices: visits desc → Q desc → prior desc (lc0-style).
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| rank_key(half, b).partial_cmp(&rank_key(half, a)).unwrap());
    indices.truncate(max_lines);
    indices.retain(|&i| half.edge(i).visits > 0);

    // Subject player's starting position.
    let start_pos = if is_player1 {
        sim.player1_position()
    } else {
        sim.player2_position()
    };

    indices
        .into_iter()
        .map(|root_outcome_idx| {
            let score = half.edge(root_outcome_idx).q;
            let root_action = half.outcome_action(root_outcome_idx);
            let root_dir =
                Direction::try_from(root_action).expect("invalid action from outcome_action");

            let mut moves = vec![root_dir];
            let mut target = None;
            let mut pos = root_dir.apply_to(start_pos);

            // Opponent's best root action (most-visited, with tiebreaking).
            let opp_best_idx = best_outcome_idx(opponent_half);

            // Map (subject_idx, opponent_idx) → (p1_idx, p2_idx).
            let (p1_idx, p2_idx) = if is_player1 {
                (root_outcome_idx as u8, opp_best_idx)
            } else {
                (opp_best_idx, root_outcome_idx as u8)
            };

            // Check edge reward on root's child for cheese collection.
            let mut current = find_child(arena, root, p1_idx, p2_idx);
            if let Some(child_idx) = current {
                check_edge_reward(&arena[child_idx], is_player1, pos, &mut target);
            }

            // Walk deeper into the tree.
            while let Some(node_idx) = current {
                let node = &arena[node_idx];
                if node.is_terminal() || node.total_visits() == 0 {
                    break;
                }

                let (subj_half, opp_half) = if is_player1 {
                    (&node.p1, &node.p2)
                } else {
                    (&node.p2, &node.p1)
                };

                if subj_half.n_outcomes() == 0 || opp_half.n_outcomes() == 0 {
                    break;
                }

                let subj_idx = best_outcome_idx(subj_half);
                let opp_idx = best_outcome_idx(opp_half);

                let subj_dir =
                    Direction::try_from(subj_half.outcome_action(subj_idx as usize))
                        .unwrap_or(Direction::Stay);
                moves.push(subj_dir);
                pos = subj_dir.apply_to(pos);

                // Map back to (p1_idx, p2_idx) for child lookup.
                let (ci, cj) = if is_player1 {
                    (subj_idx, opp_idx)
                } else {
                    (opp_idx, subj_idx)
                };

                current = find_child(arena, node_idx, ci, cj);
                if let Some(child_idx) = current {
                    check_edge_reward(&arena[child_idx], is_player1, pos, &mut target);
                }
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

/// Ranking key for sorting outcomes: (visits, Q, prior). All descending.
pub(crate) fn rank_key(half: &HalfNode, idx: usize) -> (u32, f32, f32) {
    let e = half.edge(idx);
    (e.visits, e.q, half.prior(idx))
}

/// Return the outcome index with the best ranking (visits → Q → prior).
pub(crate) fn best_outcome_idx(half: &HalfNode) -> u8 {
    let n = half.n_outcomes();
    let mut best = 0u8;
    let mut best_key = rank_key(half, 0);
    for i in 1..n {
        let key = rank_key(half, i);
        if key > best_key {
            best_key = key;
            best = i as u8;
        }
    }
    best
}

/// If the edge reward is positive, record the position as the target cheese (first only).
fn check_edge_reward(
    node: &alpharat_mcts::Node,
    is_player1: bool,
    pos: Coordinates,
    target: &mut Option<(u8, u8)>,
) {
    if target.is_some() {
        return;
    }
    let reward = if is_player1 {
        node.edge_r1()
    } else {
        node.edge_r2()
    };
    if reward > 0.0 {
        *target = Some((pos.x, pos.y));
    }
}
