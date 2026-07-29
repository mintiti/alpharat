use alpharat_mcgs::{EdgeTransition, MCGSTree, NodeView, OutcomeStats, SearchPlayer, TreeView};
use pyrat_sdk::{Coordinates, Direction, GameSim};

/// A single principal variation line rooted at one of the top root moves.
#[derive(Debug, PartialEq)]
pub struct PvLine {
    /// Sequence of moves for the subject player along this line.
    pub moves: Vec<Direction>,
    /// First cheese the subject player collects along this PV.
    pub target: Option<Coordinates>,
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
    tree.observe(|view| extract_pvs_from(&view, sim, is_player1, max_lines))
}

fn extract_pvs_from(
    tree: &TreeView<'_, '_>,
    sim: &GameSim,
    is_player1: bool,
    max_lines: usize,
) -> Vec<PvLine> {
    let player = if is_player1 {
        SearchPlayer::Player1
    } else {
        SearchPlayer::Player2
    };
    let opponent = if is_player1 {
        SearchPlayer::Player2
    } else {
        SearchPlayer::Player1
    };
    let start_pos = if is_player1 {
        sim.player1_position()
    } else {
        sim.player2_position()
    };

    let root_lines = tree.with_root(|root| {
        let outcomes = ranked_visited_outcomes(root.outcomes(player), max_lines);

        let opponent_best = best_outcome(root, opponent);
        outcomes
            .into_iter()
            .map(|subject| {
                let direction = Direction::try_from(subject.action)
                    .expect("invalid action from outcome_action");
                let (p1_outcome, p2_outcome) = if is_player1 {
                    (subject.index, opponent_best.index)
                } else {
                    (opponent_best.index, subject.index)
                };
                let child = root.edge(p1_outcome, p2_outcome).map(|edge| {
                    let transition = edge.transition();
                    (transition, edge.child())
                });
                (subject, direction, child)
            })
            .collect::<Vec<_>>()
    });

    root_lines
        .into_iter()
        .map(|(root_outcome, root_direction, root_child)| {
            let mut moves = vec![root_direction];
            let mut target = None;
            let mut pos = root_direction.apply_to(start_pos);
            let mut current = root_child.map(|(transition, child)| {
                check_edge_reward(transition, is_player1, pos, &mut target);
                child
            });

            loop {
                let Some(node) = current.as_ref() else {
                    break;
                };
                let next = tree.with_node(node, |node| {
                    let stats = node.stats();
                    if stats.is_terminal || stats.total_edge_visits == 0 {
                        return None;
                    }
                    if node.outcomes(SearchPlayer::Player1).next().is_none()
                        || node.outcomes(SearchPlayer::Player2).next().is_none()
                    {
                        return None;
                    }

                    let subject = best_outcome(node, player);
                    let opponent = best_outcome(node, opponent);
                    let direction = Direction::try_from(subject.action).unwrap_or(Direction::Stay);
                    moves.push(direction);
                    pos = direction.apply_to(pos);

                    let (p1_outcome, p2_outcome) = if is_player1 {
                        (subject.index, opponent.index)
                    } else {
                        (opponent.index, subject.index)
                    };
                    node.edge(p1_outcome, p2_outcome).map(|edge| {
                        let transition = edge.transition();
                        check_edge_reward(transition, is_player1, pos, &mut target);
                        edge.child()
                    })
                });
                current = next;
            }

            PvLine {
                moves,
                target,
                score: root_outcome.q,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rank_key(outcome: OutcomeStats) -> (u32, f32, f32) {
    (outcome.visits, outcome.q, outcome.prior)
}

fn ranked_visited_outcomes(
    outcomes: impl IntoIterator<Item = OutcomeStats>,
    max_lines: usize,
) -> Vec<OutcomeStats> {
    let mut outcomes: Vec<_> = outcomes.into_iter().collect();
    outcomes.sort_unstable_by(|a, b| rank_key(*b).partial_cmp(&rank_key(*a)).unwrap());
    outcomes.retain(|outcome| outcome.visits > 0);
    outcomes.truncate(max_lines);
    outcomes
}

fn best_ranked_outcome(outcomes: impl IntoIterator<Item = OutcomeStats>) -> OutcomeStats {
    let mut outcomes = outcomes.into_iter();
    let mut best = outcomes.next().expect("node has no outcomes");
    let mut best_key = rank_key(best);
    for outcome in outcomes {
        let key = rank_key(outcome);
        if key > best_key {
            best = outcome;
            best_key = key;
        }
    }
    best
}

fn best_outcome(node: NodeView<'_, '_>, player: SearchPlayer) -> OutcomeStats {
    best_ranked_outcome(node.outcomes(player))
}

/// Return the p1 outcome with the best visits -> Q -> prior ranking.
pub(crate) fn best_p1_outcome(node: NodeView<'_, '_>) -> OutcomeStats {
    best_outcome(node, SearchPlayer::Player1)
}

/// Return the p2 outcome with the best visits -> Q -> prior ranking.
pub(crate) fn best_p2_outcome(node: NodeView<'_, '_>) -> OutcomeStats {
    best_outcome(node, SearchPlayer::Player2)
}

/// If the edge reward is positive, record the position as the target cheese (first only).
fn check_edge_reward(
    edge: EdgeTransition,
    is_player1: bool,
    pos: Coordinates,
    target: &mut Option<Coordinates>,
) {
    if target.is_some() {
        return;
    }
    let reward = if is_player1 {
        edge.reward_p1
    } else {
        edge.reward_p2
    };
    if reward > 0.0 {
        *target = Some(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alpharat_mcgs::{run_search, SearchConfig, SmartUniformBackend};
    use pyrat::{GameBuilder, GameState};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn outcome(index: u8, visits: u32, q: f32, prior: f32) -> OutcomeStats {
        OutcomeStats {
            index,
            action: index,
            visits,
            q,
            prior,
        }
    }

    #[test]
    fn ranking_is_visits_then_q_then_prior_with_lowest_index_on_exact_tie() {
        let visits_win =
            best_ranked_outcome([outcome(0, 11, -10.0, 0.0), outcome(1, 10, 10.0, 1.0)]);
        assert_eq!(visits_win.index, 0);

        let q_wins = best_ranked_outcome([outcome(0, 10, 2.0, 1.0), outcome(1, 10, 3.0, 0.0)]);
        assert_eq!(q_wins.index, 1);

        let prior_wins = best_ranked_outcome([outcome(0, 10, 3.0, 0.1), outcome(1, 10, 3.0, 0.9)]);
        assert_eq!(prior_wins.index, 1);

        let exact_tie = best_ranked_outcome([outcome(0, 10, 3.0, 0.9), outcome(1, 10, 3.0, 0.9)]);
        assert_eq!(exact_tie.index, 0);
    }

    #[test]
    fn multi_pv_excludes_zero_visit_outcomes_and_truncates_in_rank_order() {
        let ranked = ranked_visited_outcomes(
            [
                outcome(0, 0, 100.0, 1.0),
                outcome(1, 4, 1.0, 0.5),
                outcome(2, 7, 0.0, 0.1),
                outcome(3, 4, 2.0, 0.1),
            ],
            2,
        );
        assert_eq!(
            ranked
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            [2, 3]
        );
    }

    #[test]
    fn target_uses_the_subject_reward_and_keeps_the_first_positive_edge() {
        let edge = EdgeTransition {
            p1_outcome: 0,
            p2_outcome: 1,
            reward_p1: 1.0,
            reward_p2: 0.0,
        };
        let later = EdgeTransition {
            reward_p1: 2.0,
            reward_p2: 3.0,
            ..edge
        };
        let first_pos = Coordinates::new(1, 2);
        let later_pos = Coordinates::new(2, 2);

        let mut p1_target = None;
        check_edge_reward(edge, true, first_pos, &mut p1_target);
        check_edge_reward(later, true, later_pos, &mut p1_target);
        assert_eq!(p1_target, Some(Coordinates::new(1, 2)));

        let mut p2_target = None;
        check_edge_reward(edge, false, first_pos, &mut p2_target);
        assert_eq!(p2_target, None);
        check_edge_reward(later, false, later_pos, &mut p2_target);
        assert_eq!(p2_target, Some(Coordinates::new(2, 2)));
    }

    fn fixed_pv_game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![
                Coordinates::new(1, 0),
                Coordinates::new(0, 1),
                Coordinates::new(2, 2),
                Coordinates::new(3, 4),
                Coordinates::new(4, 3),
            ])
            .with_max_turns(30)
            .build()
            .create(None)
            .unwrap()
    }

    #[test]
    fn fixed_search_preserves_multi_pv_lines() {
        let game = fixed_pv_game();
        let mut tree = MCGSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(42);
        run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &SearchConfig::default(),
            256,
            8,
            &mut rng,
        )
        .unwrap();

        let p1 = extract_pvs(&tree, &game, true, 3);
        let p2 = extract_pvs(&tree, &game, false, 3);

        assert_eq!(p1.len(), 3);
        assert_eq!(
            p1[0].moves,
            [
                Direction::Right,
                Direction::Left,
                Direction::Up,
                Direction::Right
            ]
        );
        assert_eq!(p1[0].target, Some(Coordinates::new(1, 0)));
        assert!((p1[0].score - 1.207_547_2).abs() < 1e-6);
        assert_eq!(
            p1[1].moves,
            [
                Direction::Up,
                Direction::Down,
                Direction::Right,
                Direction::Stay
            ]
        );
        assert_eq!(p1[1].target, Some(Coordinates::new(0, 1)));
        assert!((p1[1].score - 1.170_068).abs() < 1e-6);
        assert_eq!(
            p1[2].moves,
            [Direction::Stay, Direction::Up, Direction::Down]
        );
        assert_eq!(p1[2].target, Some(Coordinates::new(0, 1)));
        assert!((p1[2].score - 0.803_921_6).abs() < 1e-6);

        assert_eq!(p2.len(), 3);
        assert_eq!(
            p2[0].moves,
            [
                Direction::Left,
                Direction::Right,
                Direction::Down,
                Direction::Up
            ]
        );
        assert_eq!(p2[0].target, Some(Coordinates::new(3, 4)));
        assert!((p2[0].score - 1.174_311_9).abs() < 1e-6);
        assert_eq!(
            p2[1].moves,
            [Direction::Down, Direction::Up, Direction::Down]
        );
        assert_eq!(p2[1].target, Some(Coordinates::new(4, 3)));
        assert!((p2[1].score - 1.121_276_6).abs() < 1e-6);
        assert_eq!(
            p2[2].moves,
            [Direction::Stay, Direction::Down, Direction::Up]
        );
        assert_eq!(p2[2].target, Some(Coordinates::new(4, 3)));
        assert!((p2[2].score - 0.788_461_57).abs() < 1e-6);
    }

    #[test]
    fn pv_keeps_the_selected_move_when_the_joint_child_is_missing() {
        let game = fixed_pv_game();
        let mut tree = MCGSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(0);
        run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &SearchConfig::default(),
            4,
            4,
            &mut rng,
        )
        .unwrap();

        let selected_edge = tree.observe(|view| {
            view.with_root(|root| {
                let subject = best_p1_outcome(root);
                let opponent = best_p2_outcome(root);
                (
                    subject.action,
                    root.edge(subject.index, opponent.index).is_some(),
                )
            })
        });
        assert_eq!(selected_edge, (Direction::Up as u8, false));

        let pvs = extract_pvs(&tree, &game, true, 5);
        assert_eq!(pvs[0].moves, [Direction::Up]);
        assert_eq!(pvs[0].target, None);
        assert_eq!(pvs[0].score, 1.0);
    }
}
