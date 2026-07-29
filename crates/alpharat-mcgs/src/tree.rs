use std::sync::Arc;

use pyrat::GameState;

#[cfg(test)]
use pyrat::Coordinates;

use crate::node::{Edge, LowNode, SharedNode};
use crate::observer::{
    TranspositionEviction, TranspositionStats, TreeStats, TreeView,
};
use crate::tt::TranspositionTable;
use crate::{smart_uniform_prior, EvalResult};

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
pub(crate) fn create_root_node(
    game: &GameState,
    tt: &mut TranspositionTable,
) -> Arc<SharedNode> {
    let hash = game.state_hash();
    if let Some(existing) = tt.lookup(hash) {
        return existing;
    }

    let eff_p1 = game.effective_actions_p1();
    let eff_p2 = game.effective_actions_p2();

    let prior_p1 = smart_uniform_prior(&eff_p1);
    let prior_p2 = smart_uniform_prior(&eff_p2);

    let mut node = LowNode::new_shell(eff_p1, eff_p2);
    node.set_prior(prior_p1, prior_p2);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

    let shared = Arc::new(SharedNode::new(node));
    tt.insert(hash, &shared);
    shared
}

/// Set priors on a shell node after batch NN evaluation.
///
/// - `Some(result)`: reduces NN policies into outcome-indexed priors.
/// - `None`: marks the node as terminal (no priors needed).
pub(crate) fn populate_node(node: &SharedNode, eval_result: Option<&EvalResult>) {
    let low = node.get_mut();
    debug_assert!(
        low.total_visits() == 0,
        "populate_node: node already has {} visits",
        low.total_visits()
    );

    match eval_result {
        Some(result) => {
            low.set_prior(result.policy_p1, result.policy_p2);
        }
        None => {
            low.set_terminal();
        }
    }
}

/// Find existing child or create one, using TT for transposition detection.
///
/// Three cases:
/// 1. Edge at (i,j) already exists → return its LowNode, `false`
/// 2. No edge, TT hit → create Edge → existing LowNode, `false`
/// 3. No edge, TT miss → create shell LowNode, insert in TT, create Edge, `true`
///
/// `game` must already be advanced to the child position.
/// Always prepends Edge to parent's child list when creating new edge.
///
/// Returns (child_node, is_new_lownode).
pub(crate) fn find_or_create_child(
    parent: &SharedNode,
    i: u8,
    j: u8,
    game: &GameState,
    tt: &mut TranspositionTable,
    r1: f32,
    r2: f32,
) -> (Arc<SharedNode>, bool) {
    // Case 1: edge already exists
    if let Some(edge) = parent.get().find_child(i, j) {
        return (Arc::clone(edge.low_node()), false);
    }

    let hash = game.state_hash();

    // Case 2: TT hit — reuse existing LowNode
    if let Some(existing) = tt.lookup(hash) {
        let edge = Box::new(Edge::new(Arc::clone(&existing), (i, j), r1, r2));
        parent.get_mut().prepend_child(edge);
        return (existing, false);
    }

    // Case 3: TT miss — create new shell LowNode
    let eff_p1 = game.effective_actions_p1();
    let eff_p2 = game.effective_actions_p2();
    let mut child_node = LowNode::new_shell(eff_p1, eff_p2);
    child_node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

    let child = Arc::new(SharedNode::new(child_node));
    tt.insert(hash, &child);

    let edge = Box::new(Edge::new(Arc::clone(&child), (i, j), r1, r2));
    parent.get_mut().prepend_child(edge);

    (child, true)
}

// ---------------------------------------------------------------------------
// MCGSTree
// ---------------------------------------------------------------------------

/// Thin lifecycle manager for the MCGS DAG and root node.
///
/// Bundles root + TT. Search logic operates through the accessors.
pub struct MCGSTree {
    root: Arc<SharedNode>,
    tt: TranspositionTable,
    node_count: u32,
}

impl MCGSTree {
    /// Create a new tree with a root node derived from `game`.
    pub fn new(game: &GameState) -> Self {
        let mut tt = TranspositionTable::new();
        let root = create_root_node(game, &mut tt);
        Self { root, tt, node_count: 1 }
    }

    pub(crate) fn root(&self) -> &Arc<SharedNode> {
        &self.root
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
        let old_root = &self.root;
        let old_low = old_root.get();

        // Map raw actions to outcome indices
        let i = old_low.p1_action_to_outcome_idx(p1_action);
        let j = old_low.p2_action_to_outcome_idx(p2_action);

        // Detach the child list from the old root
        let mut cursor = old_root.get_mut().take_first_child();
        let mut new_root: Option<Arc<SharedNode>> = None;

        // Walk the linked list: find the matching edge, queue the rest
        while let Some(mut edge) = cursor {
            // Detach next sibling before we consume this edge
            cursor = edge.take_next_sibling();

            let (ei, ej) = edge.parent_outcome();
            if ei == i && ej == j && new_root.is_none() {
                new_root = Some(Arc::clone(edge.low_node()));
                // Drop the edge (decrements num_parents), don't queue it
                drop(edge);
            } else {
                crate::gc::queue(edge);
            }
        }

        // Flush thread-local GC batch so edges are visible to the GC thread
        crate::gc::flush();

        // Set the new root
        self.root = match new_root {
            Some(node) => node,
            None => create_root_node(game, &mut self.tt),
        };

        // Recount after pruning (TT live_count is O(n) but advance_root is infrequent).
        self.node_count = self.tt.live_count() as u32;
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
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        assert!(root.get().is_evaluated());
        assert_eq!(root.get().total_visits(), 0);
        assert_eq!(tt.live_count(), 1);

        // TT lookup returns the same node
        let hash = game.state_hash();
        let found = tt.lookup(hash).unwrap();
        assert!(Arc::ptr_eq(&found, &root));
    }

    #[test]
    fn create_root_value_scale() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 0), Coordinates::new(2, 0), Coordinates::new(3, 0)],
        );
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);
        assert_eq!(root.get().value_scale(), 3.0);
    }

    // ---- populate_node ----

    #[test]
    fn populate_node_with_eval() {
        let node = SharedNode::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 2.0,
        };
        populate_node(&node, Some(&eval));
        assert!(node.get().is_evaluated());
        assert!(!node.get().is_terminal());
    }

    #[test]
    fn populate_node_terminal() {
        let node = SharedNode::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        populate_node(&node, None);
        assert!(node.get().is_terminal());
    }

    // ---- find_or_create_child ----

    #[test]
    fn find_or_create_child_tt_miss() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        // Advance game
        use pyrat::Direction;
        let mut child_game = game.clone();
        let scores_before = (child_game.player1_score(), child_game.player2_score());
        let _undo = child_game.make_move(Direction::Up, Direction::Down);
        let (r1, r2) = compute_rewards(&child_game, scores_before);

        let i = root.get().p1_action_to_outcome_idx(0); // UP
        let j = root.get().p2_action_to_outcome_idx(2); // DOWN

        let (child, is_new) = find_or_create_child(&root, i, j, &child_game, &mut tt, r1, r2);
        assert!(is_new);
        assert!(!child.get().is_evaluated());
        assert_eq!(child.num_parents(), 1);
        assert_eq!(tt.live_count(), 2); // root + child

        // Edge exists on root
        assert!(root.get().find_child(i, j).is_some());
    }

    #[test]
    fn find_or_create_child_edge_reuse() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        let i = root.get().p1_action_to_outcome_idx(0);
        let j = root.get().p2_action_to_outcome_idx(2);

        let (child1, is_new1) =
            find_or_create_child(&root, i, j, &child_game, &mut tt, 0.0, 0.0);
        assert!(is_new1);

        // Second call with same (i, j) — reuses existing edge
        let (child2, is_new2) =
            find_or_create_child(&root, i, j, &child_game, &mut tt, 0.0, 0.0);
        assert!(!is_new2);
        assert!(Arc::ptr_eq(&child1, &child2));
    }

    #[test]
    fn find_or_create_child_tt_hit() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        // Pre-insert the child in TT at its hash
        let child_hash = child_game.state_hash();
        let eff_p1 = child_game.effective_actions_p1();
        let eff_p2 = child_game.effective_actions_p2();
        let existing = Arc::new(SharedNode::new(LowNode::new_shell(eff_p1, eff_p2)));
        tt.insert(child_hash, &existing);

        let i = root.get().p1_action_to_outcome_idx(0);
        let j = root.get().p2_action_to_outcome_idx(2);

        let (child, is_new) = find_or_create_child(&root, i, j, &child_game, &mut tt, 0.0, 0.0);
        assert!(!is_new);
        assert!(Arc::ptr_eq(&child, &existing));
        assert_eq!(existing.num_parents(), 1); // edge incremented it
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
        assert!(tree.root().get().is_evaluated());
        assert_eq!(tree.tt().live_count(), 1);
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
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let _undo = child_game.make_move(Direction::Up, Direction::Down);

        let i = root.get().p1_action_to_outcome_idx(0);
        let j = root.get().p2_action_to_outcome_idx(2);

        // First call: creates edge with r1=1.0, r2=0.5
        let (_child1, is_new1) =
            find_or_create_child(&root, i, j, &child_game, &mut tt, 1.0, 0.5);
        assert!(is_new1);

        // Second call: same (i,j) — edge already exists, r values ignored
        let (_child2, is_new2) =
            find_or_create_child(&root, i, j, &child_game, &mut tt, 9.0, 9.0);
        assert!(!is_new2);

        // Verify original rewards preserved
        let edge = root.get().find_child(i, j).unwrap();
        assert!((edge.r1() - 1.0).abs() < 1e-6);
        assert!((edge.r2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn find_or_create_child_rewards_stored() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let mut tt = TranspositionTable::new();
        let root = create_root_node(&game, &mut tt);

        use pyrat::Direction;
        let mut child_game = game.clone();
        let scores_before = (child_game.player1_score(), child_game.player2_score());
        let _undo = child_game.make_move(Direction::Up, Direction::Down);
        let (r1, r2) = compute_rewards(&child_game, scores_before);

        let i = root.get().p1_action_to_outcome_idx(0);
        let j = root.get().p2_action_to_outcome_idx(2);

        let (_child, _is_new) =
            find_or_create_child(&root, i, j, &child_game, &mut tt, r1, r2);

        let edge = root.get().find_child(i, j).unwrap();
        assert!((edge.r1() - r1).abs() < 1e-6);
        assert!((edge.r2() - r2).abs() < 1e-6);
    }

    // ---- populate_node ----

    #[test]
    #[should_panic(expected = "populate_node: node already has")]
    fn populate_node_on_visited_panics() {
        let node = SharedNode::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        node.get_mut().finalize_score_update(1.0, 1.0);
        let eval = crate::EvalResult {
            policy_p1: [0.2; 5],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 1.0,
        };
        populate_node(&node, Some(&eval));
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

        // Keep the internal identity assertion exact while separately proving
        // the public observer preserves the child's complete state.
        let expected_child_identity = {
            let root = tree.root().get();
            Arc::clone(root.find_child(0, 0).unwrap().low_node())
        };
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
        tree.advance_root(&game, p1_action, p2_action);

        assert!(Arc::ptr_eq(tree.root(), &expected_child_identity));
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
        let old_root = Arc::clone(tree.root());

        use pyrat::Direction;
        let _undo = game.make_move(Direction::Up, Direction::Down);
        tree.advance_root(&game, 0, 2);

        let root = tree.observe(|view| view.with_root(|root| root.stats()));
        assert!(root.is_evaluated); // create_root_node sets smart-uniform priors
        assert!(!root.is_terminal);
        assert_eq!(root.total_visits, 0);
        assert_eq!(root.total_edge_visits, 0);
        assert!(!Arc::ptr_eq(tree.root(), &old_root));
        assert!(tree.tt().lookup(game.state_hash()).is_some());
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
        let root_low = tree.root().get();
        let p1_action = root_low.p1_outcome_action(0);
        let p2_action = root_low.p2_outcome_action(0);

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
        assert!(tree.root().get().is_evaluated());
        assert!(tree.tt().live_count() >= 1);
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

        // Advance to the most-visited child so we keep the largest subtree.
        let root_low = tree.root().get();
        let mut best_i = 0u8;
        let mut best_visits = 0u32;
        for i in 0..root_low.n1() as u8 {
            let v = root_low.marginal_visits_p1(i as usize);
            if v > best_visits {
                best_visits = v;
                best_i = i;
            }
        }
        let mut best_j = 0u8;
        best_visits = 0;
        for j in 0..root_low.n2() as u8 {
            let v = root_low.marginal_visits_p2(j as usize);
            if v > best_visits {
                best_visits = v;
                best_j = j;
            }
        }
        let p1_action = root_low.p1_outcome_action(best_i as usize);
        let p2_action = root_low.p2_outcome_action(best_j as usize);

        // Find a transposition reachable from the chosen child's subtree.
        let target_edge = root_low.find_child(best_i, best_j)
            .expect("best edge must exist");
        let target_child = target_edge.low_node();
        let mut transposition_weak: Option<std::sync::Weak<SharedNode>> = None;
        let mut cursor = target_child.get().first_child();
        while let Some(edge) = cursor {
            let grandchild = edge.low_node();
            if grandchild.num_parents() > 1 {
                transposition_weak = Some(Arc::downgrade(grandchild));
                break;
            }
            cursor = edge.next_sibling();
        }

        let weak = transposition_weak.expect(
            "search should produce at least one transposition with 1000 sims on symmetric setup",
        );

        use pyrat::Direction;
        let _undo = game.make_move(
            Direction::try_from(p1_action).unwrap(),
            Direction::try_from(p2_action).unwrap(),
        );
        tree.advance_root(&game, p1_action, p2_action);

        // Let GC process pruned edges
        crate::gc::start();
        crate::gc::stop();
        crate::gc::wait();

        // The transposed node should still be alive (kept by multiple parent edges
        // in the surviving subtree, or by the TT)
        assert!(
            weak.upgrade().is_some(),
            "transposition node should survive advance_root (still reachable from new subtree)"
        );
    }

    #[test]
    fn observer_preserves_fixed_node_edge_and_tt_statistics() {
        let open = [0, 1, 2, 3, 4];

        let mut child_low = LowNode::new_shell(open, open);
        child_low.set_value_scale(3.0);
        child_low.set_terminal();
        let child = Arc::new(SharedNode::new(child_low));

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
        root_low.prepend_child(Box::new(Edge::new(
            Arc::clone(&child),
            (1, 2),
            1.0,
            -0.5,
        )));
        let root = Arc::new(SharedNode::new(root_low));

        let mut tt = TranspositionTable::new();
        tt.insert(1, &root);
        tt.insert(2, &child);
        let expired = Arc::new(SharedNode::new(LowNode::new_shell(open, open)));
        tt.insert(3, &expired);
        drop(expired);

        let mut tree = MCGSTree {
            root,
            tt,
            node_count: 2,
        };

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
