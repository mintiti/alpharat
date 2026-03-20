use alpharat_mcts::{find_child, HalfNode, MCTSTree};
use pyrat_sdk::{Direction, GameSim};

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
/// Returns up to `max_lines` PV lines, each up to `max_depth` moves deep.
/// Lines are sorted by root edge visit count (most-visited first),
/// with Q then prior as tiebreakers (lc0-style).
pub fn extract_pvs(
    tree: &MCTSTree,
    sim: &GameSim,
    is_player1: bool,
    max_lines: usize,
    max_depth: usize,
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

    indices
        .into_iter()
        .map(|root_outcome_idx| {
            let score = half.edge(root_outcome_idx).q;
            let root_action = half.outcome_action(root_outcome_idx);
            let root_dir =
                Direction::try_from(root_action).expect("invalid action from outcome_action");

            let mut moves = vec![root_dir];
            let mut target = None;

            // Opponent's best root action (most-visited, with tiebreaking).
            let opp_best_idx = best_outcome_idx(opponent_half);

            // Map (subject_idx, opponent_idx) → (p1_idx, p2_idx).
            let (p1_idx, p2_idx) = if is_player1 {
                (root_outcome_idx as u8, opp_best_idx)
            } else {
                (opp_best_idx, root_outcome_idx as u8)
            };

            // Clone sim to track positions and cheese collection.
            let mut sim_copy = sim.clone();

            // Make the root move.
            let p1_dir = dir_from_half(&root_node.p1, p1_idx);
            let p2_dir = dir_from_half(&root_node.p2, p2_idx);
            let prev = subject_score(&sim_copy, is_player1);
            sim_copy.make_move(p1_dir, p2_dir);
            check_cheese(&sim_copy, is_player1, prev, &mut target);

            // Walk deeper into the tree.
            let mut current = find_child(arena, root, p1_idx, p2_idx);

            while let Some(node_idx) = current {
                if moves.len() >= max_depth {
                    break;
                }
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

                moves.push(
                    Direction::try_from(subj_half.outcome_action(subj_idx as usize))
                        .unwrap_or(Direction::Stay),
                );

                // Map back to (p1_idx, p2_idx) for child lookup and sim advance.
                let (ci, cj) = if is_player1 {
                    (subj_idx, opp_idx)
                } else {
                    (opp_idx, subj_idx)
                };

                let p1_d = dir_from_half(&node.p1, ci);
                let p2_d = dir_from_half(&node.p2, cj);

                let prev = subject_score(&sim_copy, is_player1);
                sim_copy.make_move(p1_d, p2_d);
                check_cheese(&sim_copy, is_player1, prev, &mut target);

                current = find_child(arena, node_idx, ci, cj);
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
fn rank_key(half: &HalfNode, idx: usize) -> (u32, f32, f32) {
    let e = half.edge(idx);
    (e.visits, e.q, half.prior(idx))
}

/// Return the outcome index with the best ranking (visits → Q → prior).
fn best_outcome_idx(half: &HalfNode) -> u8 {
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

fn dir_from_half(half: &HalfNode, outcome_idx: u8) -> Direction {
    Direction::try_from(half.outcome_action(outcome_idx as usize)).unwrap_or(Direction::Stay)
}

fn subject_score(sim: &GameSim, is_player1: bool) -> f32 {
    if is_player1 {
        sim.player1_score()
    } else {
        sim.player2_score()
    }
}

/// If subject's score increased, record the current position as the target cheese.
fn check_cheese(sim: &GameSim, is_player1: bool, prev_score: f32, target: &mut Option<(u8, u8)>) {
    if target.is_some() {
        return;
    }
    let cur = subject_score(sim, is_player1);
    if cur > prev_score {
        let pos = if is_player1 {
            sim.player1_position()
        } else {
            sim.player2_position()
        };
        *target = Some((pos.x, pos.y));
    }
}
