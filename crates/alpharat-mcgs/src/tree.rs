use std::sync::Arc;

use pyrat::GameState;

#[cfg(test)]
use pyrat::Coordinates;

use crate::access::ExclusiveAccess;
use crate::node::{LowNode, OwnerToken, SharedNode};
use crate::observer::{
    TranspositionEviction, TranspositionStats, TreeStats, TreeView,
};
use crate::tt::TranspositionTable;
use crate::smart_uniform_prior;

#[cfg(test)]
use crate::EvalResult;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Score diffs after advancing game state.
///
/// Call AFTER make_move — compares current scores against `scores_before`.
pub(crate) fn compute_rewards(game: &GameState, scores_before: (f32, f32)) -> (f32, f32) {
    (
        game.player1_score() - scores_before.0,
        game.player2_score() - scores_before.1,
    )
}

// ---------------------------------------------------------------------------
// Three-phase lifecycle: create_root_node + populate_node + find_or_create_child
// ---------------------------------------------------------------------------

/// Create a root node from the current game state.
///
/// Checks TT first — if the position already exists (e.g. from a prior search),
/// returns the existing node. Otherwise creates a fresh node with smart uniform
/// priors and `value_scale = max(remaining_cheese, 1)`, inserts into TT.
fn create_root_node_with_owner(
    game: &GameState,
    tt: &mut TranspositionTable,
    owner: &Arc<OwnerToken>,
) -> Arc<SharedNode> {
    let hash = game.state_hash();
    if let Some(existing) = tt.lookup(hash) {
        assert!(
            Arc::ptr_eq(owner, existing.owner()),
            "transposition table contains a node from a different MCGS tree"
        );
        return existing;
    }

    let eff_p1 = game.effective_actions_p1();
    let eff_p2 = game.effective_actions_p2();

    let prior_p1 = smart_uniform_prior(&eff_p1);
    let prior_p2 = smart_uniform_prior(&eff_p2);

    let mut node = LowNode::new_shell(eff_p1, eff_p2);
    node.set_prior(prior_p1, prior_p2);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

    let shared = Arc::new(SharedNode::with_owner(node, Arc::clone(owner)));
    tt.insert(hash, &shared);
    shared
}

// ---------------------------------------------------------------------------
// MCGSTree
// ---------------------------------------------------------------------------

/// Thin lifecycle manager for the MCGS DAG and root node.
///
/// Bundles root + TT. Search logic operates through the accessors.
pub struct MCGSTree {
    owner: Arc<OwnerToken>,
    root: Arc<SharedNode>,
    tt: TranspositionTable,
    node_count: u32,
}

impl MCGSTree {
    /// Create a new tree with a root node derived from `game`.
    pub fn new(game: &GameState) -> Self {
        let owner = Arc::new(OwnerToken);
        let mut tt = TranspositionTable::new();
        let root = create_root_node_with_owner(game, &mut tt, &owner);
        Self {
            owner,
            root,
            tt,
            node_count: 1,
        }
    }

    pub(crate) fn root(&self) -> &Arc<SharedNode> {
        &self.root
    }

    pub(crate) fn owner(&self) -> &Arc<OwnerToken> {
        &self.owner
    }

    #[allow(dead_code)]
    pub(crate) fn tt(&self) -> &TranspositionTable {
        &self.tt
    }

    pub(crate) fn tt_mut(&mut self) -> &mut TranspositionTable {
        &mut self.tt
    }

    pub(crate) fn node_count(&self) -> u32 {
        self.node_count
    }

    pub(crate) fn increment_node_count(&mut self) {
        self.node_count += 1;
    }

    pub(crate) fn create_root_for_owner(&mut self, game: &GameState) -> Arc<SharedNode> {
        create_root_node_with_owner(game, &mut self.tt, &self.owner)
    }

    pub(crate) fn install_root(&mut self, root: Arc<SharedNode>) {
        assert!(
            Arc::ptr_eq(&self.owner, root.owner()),
            "cannot install a root from a different MCGS tree"
        );
        self.root = root;
    }

    pub(crate) fn recount_nodes(&mut self) {
        self.node_count = self.tt.live_count() as u32;
    }

    /// Enter a fresh branded, lock-free exclusive access session.
    pub(crate) fn with_exclusive<'tree, R>(
        &'tree mut self,
        use_access: impl for<'session> FnOnce(ExclusiveAccess<'tree, 'session>) -> R,
    ) -> R {
        use_access(ExclusiveAccess::new(self))
    }

    /// Observe the tree through a fresh, non-escaping read-only session.
    ///
    /// Node handles may be retained between `TreeView::with_node` calls inside
    /// `inspect`, but neither handles nor borrowed views can leave this callback.
    /// The returned `R` must contain only owned observations.
    ///
    /// ```compile_fail
    /// # use alpharat_mcgs::MCGSTree;
    /// # use pyrat::{Coordinates, GameBuilder};
    /// # let game = GameBuilder::new(3, 3)
    /// #     .with_open_maze()
    /// #     .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 2))
    /// #     .with_custom_cheese(vec![Coordinates::new(1, 1)])
    /// #     .with_max_turns(10)
    /// #     .build().create(None).unwrap();
    /// let tree = MCGSTree::new(&game);
    /// let escaped = tree.observe(|view| view.root());
    /// ```
    ///
    /// Handles from independently branded trees cannot be mixed either:
    ///
    /// ```compile_fail
    /// # use alpharat_mcgs::MCGSTree;
    /// # use pyrat::{Coordinates, GameBuilder};
    /// # let game = GameBuilder::new(3, 3)
    /// #     .with_open_maze()
    /// #     .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 2))
    /// #     .with_custom_cheese(vec![Coordinates::new(1, 1)])
    /// #     .with_max_turns(10)
    /// #     .build().create(None).unwrap();
    /// let first = MCGSTree::new(&game);
    /// let second = MCGSTree::new(&game);
    /// first.observe(|first_view| {
    ///     let first_root = first_view.root();
    ///     second.observe(|second_view| {
    ///         second_view.with_node(&first_root, |_| ());
    ///     });
    /// });
    /// ```
    pub fn observe<'tree, R>(
        &'tree self,
        inspect: impl for<'session> FnOnce(TreeView<'tree, 'session>) -> R,
    ) -> R {
        inspect(TreeView::new(self))
    }

    /// Return owned tree and transposition-table statistics.
    pub fn stats(&self) -> TreeStats {
        TreeStats {
            node_count: self.node_count,
            transpositions: self.transposition_stats(),
        }
    }

    /// Best-effort removal of expired weak transposition-table entries.
    ///
    /// This operation is memory-safe before the background GC has drained, but
    /// it may remove nothing while queued edges still own nodes. Call
    /// `gc::stop()` and `gc::wait()` first when deterministic reclamation is
    /// required.
    pub fn evict_expired(&mut self) -> TranspositionEviction {
        let before = self.transposition_stats();
        self.tt.evict_expired();
        let after = self.transposition_stats();
        TranspositionEviction {
            before,
            after,
            removed_entries: before.entries.saturating_sub(after.entries),
        }
    }

    fn transposition_stats(&self) -> TranspositionStats {
        TranspositionStats::from_counts(self.tt.len(), self.tt.live_count())
    }

    /// Advance the root to the child reached by `(p1_action, p2_action)`.
    ///
    /// Reuses the existing subtree when possible. Pruned siblings are sent to
    /// the background GC. If no matching child edge exists (unexplored move),
    /// falls back to `create_root_node` which checks TT before creating fresh.
    ///
    /// TT entries expire naturally via Weak references as the GC drops
    /// unreachable nodes. Call `evict_expired()` explicitly to reclaim space.
    ///
    /// `game` must already reflect the state after the move.
    pub fn advance_root(&mut self, game: &GameState, p1_action: u8, p2_action: u8) {
        self.with_exclusive(|mut access| {
            access.advance_root(game, p1_action, p2_action);
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pyrat::GameBuilder;

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

    // ---- compute_rewards ----

    #[test]
    fn compute_rewards_no_cheese() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let scores_before = (game.player1_score(), game.player2_score());
        // No cheese collected on this move
        let (r1, r2) = compute_rewards(&game, scores_before);
        assert_eq!(r1, 0.0);
        assert_eq!(r2, 0.0);
    }

    // ---- state_hash ----

    #[test]
    fn hash_same_state_deterministic() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        assert_eq!(game.state_hash(), game.state_hash());
    }

    #[test]
    fn hash_different_positions() {
        let g1 = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let g2 = open_5x5_game(
            Coordinates::new(1, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        assert_ne!(g1.state_hash(), g2.state_hash());
    }

    #[test]
    fn hash_different_cheese() {
        let g1 = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let g2 = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(3, 3)],
        );
        assert_ne!(g1.state_hash(), g2.state_hash());
    }

    // ---- create_root_node ----

    #[test]
    fn create_root_inserts_in_tt() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);
        let root = tree.observe(|view| view.with_root(|root| root.stats()));

        assert!(root.is_evaluated);
        assert_eq!(root.total_visits, 0);
        assert_eq!(
            tree.stats().transpositions,
            TranspositionStats {
                entries: 1,
                live_entries: 1,
                expired_entries: 0,
            }
        );
        tree.with_exclusive(|access| {
            let root = access.root();
            let stored = access
                .test_tt_lookup(game.state_hash())
                .expect("root must be stored under its state hash");
            assert!(access.same_node(&root, &stored));
        });
    }

    #[test]
    fn create_root_value_scale() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 0), Coordinates::new(2, 0), Coordinates::new(3, 0)],
        );
        let tree = MCGSTree::new(&game);
        let value_scale = tree.observe(|view| view.with_root(|root| root.stats().value_scale));
        assert_eq!(value_scale, 3.0);
    }

    // ---- populate_node ----

    #[test]
    fn populate_node_with_eval() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 2.0,
        };
        tree.with_exclusive(|mut access| {
            let node = access.test_node(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ));
            access.populate_node(&node, Some(&eval));
            access.test_install_root(&node);
        });

        let node = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(node.is_evaluated);
        assert!(!node.is_terminal);
    }

    #[test]
    fn populate_node_terminal() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        tree.with_exclusive(|mut access| {
            let node = access.test_node(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ));
            access.populate_node(&node, None);
            access.test_install_root(&node);
        });

        let node = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(node.is_terminal);
    }

    // ---- find_or_create_child ----

    #[test]
    fn find_or_create_child_tt_miss() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        // Advance game
        use pyrat::Direction;
        let mut child_game = game.clone();
        let scores_before = (child_game.player1_score(), child_game.player2_score());
        let _undo = child_game.make_move(Direction::Up, Direction::Down);
        let (r1, r2) = compute_rewards(&child_game, scores_before);

        let (i, j, is_new, parents) = tree.with_exclusive(|mut access| {
            let root = access.root();
            let (i, j) = {
                let root = access.node(&root);
                (
                    root.p1_action_to_outcome_idx(0), // UP
                    root.p2_action_to_outcome_idx(2), // DOWN
                )
            };
            let before = access.node_count();
            let child = access.find_or_create_child(&root, i, j, &child_game, r1, r2);
            (
                i,
                j,
                access.node_count() == before + 1,
                access.num_parents(&child),
            )
        });

        assert!(is_new);
        assert_eq!(parents, 1);
        assert_eq!(tree.stats().transpositions.live_entries, 2);
        let child = tree.observe(|view| {
            view.with_root(|root| {
                let child = root.edge(i, j).expect("new child edge").child();
                view.with_node(&child, |child| child.stats())
            })
        });
        assert!(!child.is_evaluated);
    }

    #[test]
    fn find_or_create_child_edge_reuse() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        tree.with_exclusive(|mut access| {
            let root = access.root();
            let (i, j) = {
                let root = access.node(&root);
                (
                    root.p1_action_to_outcome_idx(0),
                    root.p2_action_to_outcome_idx(2),
                )
            };

            let before = access.node_count();
            let child1 =
                access.find_or_create_child(&root, i, j, &child_game, 0.0, 0.0);
            assert_eq!(access.node_count(), before + 1);

            // Second call with same (i, j) — reuses existing edge.
            let child2 =
                access.find_or_create_child(&root, i, j, &child_game, 0.0, 0.0);
            assert_eq!(access.node_count(), before + 1);
            assert!(access.same_node(&child1, &child2));
        });
    }

    #[test]
    fn find_or_create_child_tt_hit() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        tree.with_exclusive(|mut access| {
            let root = access.root();
            let (i, j) = {
                let root = access.node(&root);
                (
                    root.p1_action_to_outcome_idx(0),
                    root.p2_action_to_outcome_idx(2),
                )
            };

            // Publish the canonical child through the first parent.
            let existing =
                access.find_or_create_child(&root, i, j, &child_game, 0.0, 0.0);
            let count_after_insert = access.node_count();

            // A detached, owner-consistent second parent has no edge yet, so
            // this call must take the TT-hit path and reuse `existing`.
            let second_parent = access.test_node(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ));
            let child = access.find_or_create_child(
                &second_parent,
                i,
                j,
                &child_game,
                0.0,
                0.0,
            );

            assert_eq!(access.node_count(), count_after_insert);
            assert!(access.same_node(&child, &existing));
            assert_eq!(access.num_parents(&existing), 2);
        });
    }

    // ---- MCGSTree ----

    #[test]
    fn mcgs_tree_new() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let tree = MCGSTree::new(&game);
        let root = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(root.is_evaluated);
        assert_eq!(tree.stats().transpositions.live_entries, 1);
    }

    // ---- Additional helpers ----

    fn one_cheese_adjacent_game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(1, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    fn contested_cheese_game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 0))
            .with_custom_cheese(vec![Coordinates::new(1, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    // ---- compute_rewards ----

    #[test]
    fn compute_rewards_cheese_collected() {
        let mut game = one_cheese_adjacent_game();
        let scores_before = (game.player1_score(), game.player2_score());
        use pyrat::Direction;
        let _undo = game.make_move(Direction::Right, Direction::Stay);
        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6);
        assert!((r2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn compute_rewards_contested_cheese() {
        let mut game = contested_cheese_game();
        let scores_before = (game.player1_score(), game.player2_score());
        use pyrat::Direction;
        let _undo = game.make_move(Direction::Right, Direction::Left);
        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.5).abs() < 1e-6, "r1={r1}");
        assert!((r2 - 0.5).abs() < 1e-6, "r2={r2}");
    }

    #[test]
    fn compute_rewards_both_collect_different() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 0), Coordinates::new(3, 4)],
        );
        let mut game = game;
        let scores_before = (game.player1_score(), game.player2_score());
        use pyrat::Direction;
        let _undo = game.make_move(Direction::Right, Direction::Left);
        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6, "r1={r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "r2={r2}");
    }

    #[test]
    fn compute_rewards_p2_collects() {
        let mut game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(3, 4)],
        );
        let scores_before = (game.player1_score(), game.player2_score());
        use pyrat::Direction;
        let _undo = game.make_move(Direction::Stay, Direction::Left);
        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.0).abs() < 1e-6);
        assert!((r2 - 1.0).abs() < 1e-6, "r2={r2}");
    }

    // ---- find_or_create_child rewards ----

    #[test]
    fn find_or_create_child_reward_immutability() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        let (r1, r2) = tree.with_exclusive(|mut access| {
            let root = access.root();
            let (i, j) = {
                let root = access.node(&root);
                (
                    root.p1_action_to_outcome_idx(0),
                    root.p2_action_to_outcome_idx(2),
                )
            };

            // First call creates the edge with r1=1.0, r2=0.5.
            let before = access.node_count();
            let child1 =
                access.find_or_create_child(&root, i, j, &child_game, 1.0, 0.5);
            assert_eq!(access.node_count(), before + 1);

            // The existing edge wins; replacement rewards are ignored.
            let child2 =
                access.find_or_create_child(&root, i, j, &child_game, 9.0, 9.0);
            assert_eq!(access.node_count(), before + 1);
            assert!(access.same_node(&child1, &child2));

            let edge = access.child(&root, i, j).expect("fixture child edge");
            let (_, r1, r2) = edge.into_parts();
            (r1, r2)
        });

        assert!((r1 - 1.0).abs() < 1e-6);
        assert!((r2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn find_or_create_child_rewards_stored() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let scores_before = (child_game.player1_score(), child_game.player2_score());
        let _undo = child_game.make_move(Direction::Up, Direction::Down);
        let (r1, r2) = compute_rewards(&child_game, scores_before);

        let stored = tree.with_exclusive(|mut access| {
            let root = access.root();
            let (i, j) = {
                let root = access.node(&root);
                (
                    root.p1_action_to_outcome_idx(0),
                    root.p2_action_to_outcome_idx(2),
                )
            };
            access.find_or_create_child(&root, i, j, &child_game, r1, r2);
            let edge = access.child(&root, i, j).expect("fixture child edge");
            let (_, r1, r2) = edge.into_parts();
            (r1, r2)
        });

        assert!((stored.0 - r1).abs() < 1e-6);
        assert!((stored.1 - r2).abs() < 1e-6);
    }

    // ---- populate_node ----

    #[test]
    #[should_panic(expected = "populate_node: node already has")]
    fn populate_node_on_visited_panics() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        let eval = crate::EvalResult {
            policy_p1: [0.2; 5],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 1.0,
        };
        tree.with_exclusive(|mut access| {
            let node = access.test_node(LowNode::new_shell(
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ));
            access
                .node_mut(&node)
                .finalize_score_update(1.0, 1.0);
            access.populate_node(&node, Some(&eval));
        });
    }

    // ---- advance_root ----

    /// Search a tree for a few sims, then advance root to a child.
    fn search_and_advance(
        tree: &mut MCGSTree,
        game: &mut GameState,
        p1_action: u8,
        p2_action: u8,
        n_sims: u32,
    ) {
        use crate::{SearchConfig, SmartUniformBackend, run_search};
        use pyrat::Direction;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let config = SearchConfig {
            c_puct: 1.5,
            ..Default::default()
        };
        let backend = SmartUniformBackend;
        let mut rng = SmallRng::seed_from_u64(42);

        run_search(tree, game, &backend, &config, n_sims, 8, &mut rng).unwrap();

        let _undo = game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        tree.advance_root(game, p1_action, p2_action);
    }

    #[test]
    fn advance_root_reuses_explored_child() {
        crate::gc::init();

        let mut game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);

        // Search to expand children
        use crate::{SearchConfig, SmartUniformBackend, run_search};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let config = SearchConfig::default();
        let backend = SmartUniformBackend;
        let mut rng = SmallRng::seed_from_u64(42);
        run_search(&mut tree, &game, &backend, &config, 50, 8, &mut rng).unwrap();

        let (p1_action, p2_action, expected_child) = tree.observe(|view| {
            view.with_root(|root| {
                let p1_action = root
                    .outcome(crate::SearchPlayer::Player1, 0)
                    .unwrap()
                    .action;
                let p2_action = root
                    .outcome(crate::SearchPlayer::Player2, 0)
                    .unwrap()
                    .action;
                let child = root.edge(0, 0).expect("searched root child").child();
                let child_stats = view.with_node(&child, |child| child.stats());
                (p1_action, p2_action, child_stats)
            })
        });

        use pyrat::Direction;
        let _undo = game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        tree.with_exclusive(|mut access| {
            let old_root = access.root();
            let expected_child = access
                .child(&old_root, 0, 0)
                .expect("searched root child")
                .into_child();
            access.advance_root(&game, p1_action, p2_action);
            let promoted = access.root();
            assert!(access.same_node(&promoted, &expected_child));
        });

        let promoted = tree.observe(|view| view.with_root(|root| root.stats()));
        assert_eq!(promoted, expected_child);
    }

    #[test]
    fn advance_root_unexplored_creates_fresh() {
        crate::gc::init();

        let mut game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tree = MCGSTree::new(&game);

        use pyrat::Direction;
        let _undo = game.make_move(Direction::Up, Direction::Down);
        tree.with_exclusive(|mut access| {
            let old_root = access.root();
            let old_root_id = access.test_node_id(&old_root);
            drop(old_root);
            access.advance_root(&game, 0, 2);
            let new_root = access.root();
            assert_ne!(old_root_id, access.test_node_id(&new_root));
            let stored = access
                .test_tt_lookup(game.state_hash())
                .expect("fresh root must be stored under the advanced state hash");
            assert!(access.same_node(&new_root, &stored));
        });

        let root = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(root.is_evaluated); // create_root_node sets smart-uniform priors
        assert!(!root.is_terminal);
        assert_eq!(root.total_visits, 0);
        assert_eq!(root.total_edge_visits, 0);
        let stats = tree.stats();
        assert_eq!(stats.node_count, 1);
        assert_eq!(stats.transpositions.live_entries, 1);
    }

    #[test]
    fn advance_root_preserves_search_stats() {
        crate::gc::init();

        let mut game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);

        use crate::{SearchConfig, SmartUniformBackend, run_search};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let config = SearchConfig::default();
        let backend = SmartUniformBackend;
        let mut rng = SmallRng::seed_from_u64(42);
        run_search(&mut tree, &game, &backend, &config, 50, 8, &mut rng).unwrap();

        let (p1_action, p2_action, before) = tree.observe(|view| {
            view.with_root(|root| {
                let p1_action = root
                    .outcome(crate::SearchPlayer::Player1, 0)
                    .unwrap()
                    .action;
                let p2_action = root
                    .outcome(crate::SearchPlayer::Player2, 0)
                    .unwrap()
                    .action;
                let child = root.edge(0, 0).expect("searched root child").child();
                let stats = view.with_node(&child, |child| child.stats());
                (p1_action, p2_action, stats)
            })
        });

        use pyrat::Direction;
        let _undo = game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        tree.advance_root(&game, p1_action, p2_action);

        let after = tree.observe(|view| view.with_root(|root| root.stats()));
        assert_eq!(after.total_visits, before.total_visits);
        assert!((after.value_p1 - before.value_p1).abs() < 1e-6);
        assert!((after.value_p2 - before.value_p2).abs() < 1e-6);
    }

    #[test]
    fn advance_root_tt_eviction() {
        crate::gc::init();

        let mut game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);

        use crate::{SearchConfig, SmartUniformBackend, run_search};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let config = SearchConfig::default();
        let backend = SmartUniformBackend;
        let mut rng = SmallRng::seed_from_u64(42);
        run_search(&mut tree, &game, &backend, &config, 100, 8, &mut rng).unwrap();

        let tt_before = tree.stats().transpositions.live_entries;
        assert!(tt_before > 1, "should have explored multiple nodes");

        // Advance
        let (p1_action, p2_action) = tree.observe(|view| {
            view.with_root(|root| {
                (
                    root.outcome(crate::SearchPlayer::Player1, 0)
                        .unwrap()
                        .action,
                    root.outcome(crate::SearchPlayer::Player2, 0)
                        .unwrap()
                        .action,
                )
            })
        });

        use pyrat::Direction;
        let _undo = game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        tree.advance_root(&game, p1_action, p2_action);

        // Deterministic sync: let GC process all pruned edges
        crate::gc::start();
        crate::gc::stop();
        crate::gc::wait();

        // Evict stale TT entries now that GC has dropped pruned subtrees.
        let eviction = tree.evict_expired();

        let tt_after = tree.stats().transpositions.live_entries;
        assert!(
            tt_after < tt_before,
            "TT should shrink after evicting pruned subtrees: before={tt_before}, after={tt_after}"
        );
        assert!(eviction.removed_entries > 0);
    }

    #[test]
    fn advance_root_multiple_consecutive() {
        crate::gc::init();

        let mut game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2), Coordinates::new(3, 3)],
        );
        let mut tree = MCGSTree::new(&game);

        // Three consecutive advance-root cycles
        for _ in 0..3 {
            search_and_advance(&mut tree, &mut game, 1, 3, 30); // RIGHT, LEFT
            if game.check_game_over() {
                break;
            }
        }

        // No crash, tree is still functional
        let root = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(root.is_evaluated);
        assert!(tree.stats().transpositions.live_entries >= 1);
    }

    #[test]
    fn advance_root_transposition_survives() {
        crate::gc::init();

        // Set up a position where transpositions are likely:
        // both players near center, cheese at corners
        let mut game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let mut tree = MCGSTree::new(&game);

        use crate::{SearchConfig, SmartUniformBackend, run_search};
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let config = SearchConfig::default();
        let backend = SmartUniformBackend;
        let mut rng = SmallRng::seed_from_u64(42);
        // Use enough sims to produce transpositions in this symmetric setup
        run_search(&mut tree, &game, &backend, &config, 1000, 8, &mut rng).unwrap();

        // Keep the exact identity check inside one branded access session.
        // Return only the surviving edge coordinates for the post-GC observer
        // assertion.
        let surviving_edge = tree.with_exclusive(|mut access| {
            let root = access.root();
            let (best_i, best_j, p1_action, p2_action) = {
                let root = access.node(&root);
                let mut best_i = 0u8;
                let mut best_visits = 0u32;
                for i in 0..root.n1() as u8 {
                    let visits = root.marginal_visits_p1(i as usize);
                    if visits > best_visits {
                        best_visits = visits;
                        best_i = i;
                    }
                }

                let mut best_j = 0u8;
                best_visits = 0;
                for j in 0..root.n2() as u8 {
                    let visits = root.marginal_visits_p2(j as usize);
                    if visits > best_visits {
                        best_visits = visits;
                        best_j = j;
                    }
                }

                (
                    best_i,
                    best_j,
                    root.p1_outcome_action(best_i as usize),
                    root.p2_outcome_action(best_j as usize),
                )
            };

            let target_child = access
                .child(&root, best_i, best_j)
                .expect("best edge must exist")
                .into_child();
            let (n1, n2) = {
                let child = access.node(&target_child);
                (child.n1() as u8, child.n2() as u8)
            };

            let mut transposition = None;
            'children: for i in 0..n1 {
                for j in 0..n2 {
                    if let Some(edge) = access.child(&target_child, i, j) {
                        let child = edge.into_child();
                        if access.num_parents(&child) > 1 {
                            transposition = Some((i, j, child));
                            break 'children;
                        }
                    }
                }
            }
            let (transposition_i, transposition_j, transposition) = transposition.expect(
                "search should produce at least one transposition with 1000 sims on symmetric setup",
            );

            use pyrat::Direction;
            let _undo = game.make_move(
                Direction::try_from(p1_action).unwrap(),
                Direction::try_from(p2_action).unwrap(),
            );
            access.advance_root(&game, p1_action, p2_action);

            let promoted = access.root();
            assert!(access.same_node(&promoted, &target_child));
            let surviving = access
                .child(&promoted, transposition_i, transposition_j)
                .expect("transposition edge must survive root promotion")
                .into_child();
            assert!(access.same_node(&surviving, &transposition));

            (transposition_i, transposition_j)
        });

        // Let GC process pruned edges
        crate::gc::start();
        crate::gc::stop();
        crate::gc::wait();

        let transposition_survives = tree.observe(|view| {
            view.with_root(|root| {
                root.edge(surviving_edge.0, surviving_edge.1)
                    .is_some()
            })
        });
        assert!(transposition_survives);
    }

    #[test]
    fn observer_preserves_fixed_node_edge_and_tt_statistics() {
        let open = [0, 1, 2, 3, 4];
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let initial_root_hash = game.state_hash();
        let fixture_root_hash = initial_root_hash.wrapping_add(1);
        let fixture_child_hash = initial_root_hash.wrapping_add(2);
        let mut tree = MCGSTree::new(&game);

        let mut child_low = LowNode::new_shell(open, open);
        child_low.set_value_scale(3.0);
        child_low.set_terminal();

        let mut root_low = LowNode::new_shell(open, open);
        root_low.set_prior(
            [0.1, 0.2, 0.3, 0.15, 0.25],
            [0.25, 0.15, 0.3, 0.2, 0.1],
        );
        root_low.set_value_scale(5.0);
        root_low.finalize_edge_update(1, 2, 3.0, 4.0);
        root_low.finalize_edge_update(1, 3, 1.0, 2.0);
        root_low.finalize_score_update(2.0, 4.0);
        root_low.finalize_score_update(4.0, 6.0);

        tree.with_exclusive(|mut access| {
            let child = access.test_node(child_low);
            let root = access.test_node(root_low);
            access.test_connect(&root, &child, (1, 2), 1.0, -0.5);
            assert!(access.test_insert_tt(fixture_root_hash, &root));
            assert!(access.test_insert_tt(fixture_child_hash, &child));
            access.test_install_root(&root);
        });
        // `test_node` deliberately leaves production accounting unchanged.
        // The old constructor root's TT slot is now the one expired fixture
        // entry, while the installed root and its child are the two live nodes.
        tree.increment_node_count();

        let (root_stats, p1_outcomes, action_visits, transition, child_stats) =
            tree.observe(|view| {
                let (root_stats, p1_outcomes, action_visits, transition, child) =
                    view.with_root(|root| {
                        let edge = root.edge(1, 2).expect("fixture child edge");
                        (
                            root.stats(),
                            root.outcomes(crate::SearchPlayer::Player1)
                                .collect::<Vec<_>>(),
                            root.action_visits(crate::SearchPlayer::Player1),
                            edge.transition(),
                            edge.child(),
                        )
                    });
                let child_stats = view.with_node(&child, |child| child.stats());
                (
                    root_stats,
                    p1_outcomes,
                    action_visits,
                    transition,
                    child_stats,
                )
            });

        assert_eq!(root_stats.total_visits, 2);
        assert_eq!(root_stats.total_edge_visits, 2);
        assert_eq!(root_stats.value_p1, 3.0);
        assert_eq!(root_stats.value_p2, 5.0);
        assert_eq!(root_stats.value_scale, 5.0);
        assert!(root_stats.is_evaluated);
        assert_eq!(p1_outcomes[1].visits, 2);
        assert_eq!(p1_outcomes[1].q, 2.0);
        assert_eq!(p1_outcomes[1].prior, 0.2);
        assert_eq!(action_visits, [0.0, 2.0, 0.0, 0.0, 0.0]);
        assert_eq!(transition.p1_outcome, 1);
        assert_eq!(transition.p2_outcome, 2);
        assert_eq!(transition.reward_p1, 1.0);
        assert_eq!(transition.reward_p2, -0.5);
        assert!(child_stats.is_terminal);
        assert_eq!(child_stats.value_scale, 3.0);

        assert_eq!(
            tree.stats().transpositions,
            TranspositionStats {
                entries: 3,
                live_entries: 2,
                expired_entries: 1,
            }
        );
        let eviction = tree.evict_expired();
        assert_eq!(eviction.removed_entries, 1);
        assert_eq!(eviction.after.entries, 2);
        assert_eq!(eviction.after.live_entries, 2);
        assert_eq!(eviction.after.expired_entries, 0);
    }
}
