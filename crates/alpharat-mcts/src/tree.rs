use crate::{EvalResult, HalfNode, Node, NodeArena, NodeIndex};
use pyrat::GameState;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Walk the child linked list of `parent`, return the child whose
/// `parent_outcome` matches `(i, j)`.
pub fn find_child(arena: &NodeArena, parent: NodeIndex, i: u8, j: u8) -> Option<NodeIndex> {
    let mut cur = arena[parent].first_child();
    while let Some(idx) = cur {
        if arena[idx].parent_outcome() == (i, j) {
            return Some(idx);
        }
        cur = arena[idx].next_sibling();
    }
    None
}

/// Uniform prior over unique effective actions only.
///
/// Each unique outcome action gets `1/n_unique`; all others get 0.
/// Same semantics as Python's `_smart_uniform_prior`.
pub fn smart_uniform_prior(effective: &[u8; 5]) -> [f32; 5] {
    let mut seen = [false; 5];
    let mut count = 0u8;
    for &e in effective {
        if !seen[e as usize] {
            seen[e as usize] = true;
            count += 1;
        }
    }
    let p = 1.0 / count as f32;
    let mut prior = [0.0f32; 5];
    for &e in effective {
        prior[e as usize] = p;
    }
    prior
}

/// Score diffs after advancing game state.
///
/// Call AFTER make_move — compares current scores against `scores_before`.
pub fn compute_rewards(game: &GameState, scores_before: (f32, f32)) -> (f32, f32) {
    (
        game.player1_score() - scores_before.0,
        game.player2_score() - scores_before.1,
    )
}

// ---------------------------------------------------------------------------
// Three-phase lifecycle: extend_node + populate_node
// ---------------------------------------------------------------------------

/// Create a shell child node under `parent` at the given outcome indices.
///
/// The child gets outcome mappings from `game` (which must already be advanced
/// to the child position), but priors are zeroed — they arrive later via
/// `populate_node`. Does NOT set priors, values, or terminal status.
///
/// Wires the child into the parent's linked list (prepend).
pub fn extend_node(
    arena: &mut NodeArena,
    parent: NodeIndex,
    outcome_p1: u8,
    outcome_p2: u8,
    game: &GameState,
) -> NodeIndex {
    debug_assert!(
        (outcome_p1 as usize) < arena[parent].p1.n_outcomes(),
        "extend_node: outcome_p1={outcome_p1} out of bounds (n_outcomes={})",
        arena[parent].p1.n_outcomes()
    );
    debug_assert!(
        (outcome_p2 as usize) < arena[parent].p2.n_outcomes(),
        "extend_node: outcome_p2={outcome_p2} out of bounds (n_outcomes={})",
        arena[parent].p2.n_outcomes()
    );
    debug_assert!(
        find_child(arena, parent, outcome_p1, outcome_p2).is_none(),
        "extend_node: child already exists at outcome ({outcome_p1}, {outcome_p2})"
    );

    let p1_shell = HalfNode::new_shell(game.effective_actions_p1());
    let p2_shell = HalfNode::new_shell(game.effective_actions_p2());

    let mut node = Node::new(p1_shell, p2_shell);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);
    node.set_parent(Some(parent), (outcome_p1, outcome_p2));

    let child_idx = arena.alloc(node);

    // Prepend to parent's linked list.
    let old_first = arena[parent].first_child();
    arena[child_idx].set_next_sibling(old_first);
    arena[parent].set_first_child(Some(child_idx));

    child_idx
}

/// Set priors on a shell node after batch NN evaluation.
///
/// - `Some(result)`: reduces NN policies into outcome-indexed priors.
/// - `None`: marks the node as terminal (no priors needed).
///
/// Does NOT call `update_value` — value flows through backup (chunk 4).
pub fn populate_node(arena: &mut NodeArena, node_idx: NodeIndex, eval_result: Option<&EvalResult>) {
    debug_assert!(
        arena[node_idx].total_visits() == 0,
        "populate_node: node already has {} visits",
        arena[node_idx].total_visits()
    );

    let node = &mut arena[node_idx];
    match eval_result {
        Some(result) => {
            node.p1.set_prior(result.policy_p1);
            node.p2.set_prior(result.policy_p2);
        }
        None => {
            node.set_terminal();
        }
    }
}

/// Find existing child or create a shell at the given outcome pair.
///
/// For new children: creates shell (extend_node), sets edge rewards.
/// For existing children, `r1`/`r2` are ignored — edge rewards are set
/// once at creation.
///
/// New nodes are shells — caller must handle terminal detection and
/// populate_node before backup. (Terminal detection is search-layer
/// logic, matching lc0's pattern where SearchWorker owns ExtendNode.)
///
/// Returns (child_index, is_new).
pub fn find_or_extend_child(
    arena: &mut NodeArena,
    parent: NodeIndex,
    outcome_p1: u8,
    outcome_p2: u8,
    game: &GameState,
    r1: f32,
    r2: f32,
) -> (NodeIndex, bool) {
    if let Some(child) = find_child(arena, parent, outcome_p1, outcome_p2) {
        return (child, false);
    }

    let child = extend_node(arena, parent, outcome_p1, outcome_p2, game);
    arena[child].set_edge_rewards(r1, r2);
    (child, true)
}

// ---------------------------------------------------------------------------
// MCTSTree
// ---------------------------------------------------------------------------

/// Thin lifecycle manager for the MCTS arena and root node.
///
/// Bundles arena + root index. Does not own the game state — takes `&GameState`
/// for init/reinit. Search logic (chunk 3+) operates on the arena through the
/// accessors.
pub struct MCTSTree {
    arena: NodeArena,
    root: NodeIndex,
}

impl MCTSTree {
    /// Create a new tree with a root node derived from `game`.
    ///
    /// Root gets smart uniform priors and `value_scale = max(remaining_cheese, 1)`.
    /// Starts unevaluated (v1=v2=0, total_visits=0) — LC0 style.
    pub fn new(game: &GameState) -> Self {
        let mut arena = NodeArena::new();
        let root = alloc_root(&mut arena, game);
        Self { arena, root }
    }

    pub fn root(&self) -> NodeIndex {
        self.root
    }

    pub fn arena(&self) -> &NodeArena {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut NodeArena {
        &mut self.arena
    }

    /// Move root to the child matching the given action pair.
    ///
    /// Returns `true` if the child was found (root advanced), `false` if
    /// the child doesn't exist (caller should `reinit`).
    ///
    /// Abandoned nodes (old root, siblings) stay in the arena — no pruning.
    /// If arena growth becomes a problem, we can add a compact/reroot step.
    pub fn advance_root(&mut self, p1_action: u8, p2_action: u8) -> bool {
        let root_node = &self.arena[self.root];
        let i = root_node.p1.action_to_outcome_idx(p1_action);
        let j = root_node.p2.action_to_outcome_idx(p2_action);

        if let Some(child) = find_child(&self.arena, self.root, i, j) {
            self.root = child;
            true
        } else {
            false
        }
    }

    /// Clear the arena and create a fresh root from `game`.
    pub fn reinit(&mut self, game: &GameState) {
        self.arena.clear();
        self.root = alloc_root(&mut self.arena, game);
    }
}

/// Allocate a root node from the current game state.
fn alloc_root(arena: &mut NodeArena, game: &GameState) -> NodeIndex {
    let eff_p1 = game.effective_actions_p1();
    let eff_p2 = game.effective_actions_p2();

    let prior_p1 = smart_uniform_prior(&eff_p1);
    let prior_p2 = smart_uniform_prior(&eff_p2);

    let p1_half = HalfNode::new(prior_p1, eff_p1);
    let p2_half = HalfNode::new(prior_p2, eff_p2);

    let mut node = Node::new(p1_half, p2_half);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

    arena.alloc(node)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
    use pyrat::{Coordinates, GameBuilder};

    // ---- find_child ----

    #[test]
    fn find_child_found() {
        let mut arena = NodeArena::new();
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let parent = arena.alloc(Node::new(h, h));
        let c0 = arena.alloc(Node::new(h, h));
        let c1 = arena.alloc(Node::new(h, h));
        let c2 = arena.alloc(Node::new(h, h));

        // Wire parent → c0 → c1 → c2
        arena[parent].set_first_child(Some(c0));
        arena[c0].set_parent(Some(parent), (0, 1));
        arena[c0].set_next_sibling(Some(c1));
        arena[c1].set_parent(Some(parent), (2, 3));
        arena[c1].set_next_sibling(Some(c2));
        arena[c2].set_parent(Some(parent), (4, 0));

        assert_eq!(find_child(&arena, parent, 0, 1), Some(c0));
        assert_eq!(find_child(&arena, parent, 2, 3), Some(c1));
        assert_eq!(find_child(&arena, parent, 4, 0), Some(c2));
    }

    #[test]
    fn find_child_not_found() {
        let mut arena = NodeArena::new();
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let parent = arena.alloc(Node::new(h, h));
        let child = arena.alloc(Node::new(h, h));

        arena[parent].set_first_child(Some(child));
        arena[child].set_parent(Some(parent), (0, 0));

        assert_eq!(find_child(&arena, parent, 1, 1), None);
    }

    #[test]
    fn find_child_empty() {
        let mut arena = NodeArena::new();
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let parent = arena.alloc(Node::new(h, h));
        assert_eq!(find_child(&arena, parent, 0, 0), None);
    }

    // ---- smart_uniform_prior ----

    #[test]
    fn smart_uniform_open() {
        let effective = [0, 1, 2, 3, 4];
        let prior = smart_uniform_prior(&effective);
        for p in &prior {
            assert!((*p - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn smart_uniform_one_wall() {
        // UP blocked → maps to STAY
        let effective = [4, 1, 2, 3, 4];
        let prior = smart_uniform_prior(&effective);
        assert_eq!(prior[0], 0.0); // action 0 not a unique outcome
        assert!((prior[1] - 0.25).abs() < 1e-6);
        assert!((prior[2] - 0.25).abs() < 1e-6);
        assert!((prior[3] - 0.25).abs() < 1e-6);
        assert!((prior[4] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn smart_uniform_mud() {
        let effective = [4, 4, 4, 4, 4];
        let prior = smart_uniform_prior(&effective);
        for a in 0..4 {
            assert_eq!(prior[a], 0.0);
        }
        assert!((prior[4] - 1.0).abs() < 1e-6);
    }

    // ---- MCTSTree: root init ----

    #[test]
    fn root_init_open_maze() {
        let cheese = [
            Coordinates::new(0, 0),
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            Coordinates::new(4, 4),
            Coordinates::new(2, 0),
        ];
        let game = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        let root = &tree.arena()[tree.root()];

        // Center of open 5×5: 5 unique outcomes per player
        assert_eq!(root.p1.n_outcomes(), 5);
        // P2 at (4,4) — top-right corner, 2 edges blocked → 3 unique outcomes
        // Actually on a 5×5 grid, (4,4) is the top-right corner.
        // UP and RIGHT are board edges → blocked → map to STAY
        // So effective = [4, 4, 2, 3, 4] → outcomes = [2, 3, 4] → 3 unique
        assert_eq!(root.p2.n_outcomes(), 3);

        // P1 priors: uniform over 5
        let p1_prior = root.p1.expand_prior();
        for p in &p1_prior {
            assert!((*p - 0.2).abs() < 1e-6);
        }

        // Unevaluated
        assert_eq!(root.total_visits(), 0);
        assert_eq!(root.v1(), 0.0);
        assert_eq!(root.v2(), 0.0);
    }

    #[test]
    fn root_init_corner() {
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        let root = &tree.arena()[tree.root()];

        // P1 at (0,0): DOWN and LEFT are board edges → blocked
        // effective = [0, 1, 4, 4, 4] → outcomes = [0, 1, 4] → 3 unique
        assert_eq!(root.p1.n_outcomes(), 3);

        let p1_prior = root.p1.expand_prior();
        let third = 1.0 / 3.0;
        assert!((p1_prior[0] - third).abs() < 1e-6); // UP
        assert!((p1_prior[1] - third).abs() < 1e-6); // RIGHT
        assert_eq!(p1_prior[2], 0.0); // DOWN blocked
        assert_eq!(p1_prior[3], 0.0); // LEFT blocked
        assert!((p1_prior[4] - third).abs() < 1e-6); // STAY
    }

    #[test]
    fn root_init_value_scale() {
        // 5 cheese → value_scale = 5
        let cheese: Vec<_> = (0..5).map(|i| Coordinates::new(i, 0)).collect();
        let game = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        assert!((tree.arena()[tree.root()].value_scale() - 5.0).abs() < 1e-6);

        // 0 cheese → value_scale = 1 (clamped)
        let game_no_cheese = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &[]);
        let tree_no_cheese = MCTSTree::new(&game_no_cheese);
        assert!((tree_no_cheese.arena()[tree_no_cheese.root()].value_scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn root_init_mud() {
        let game = test_util::mud_game_p1_stuck();
        let tree = MCTSTree::new(&game);
        let root = &tree.arena()[tree.root()];

        // All actions → STAY when in mud
        assert_eq!(root.p1.n_outcomes(), 1);
        let p1_prior = root.p1.expand_prior();
        assert!((p1_prior[4] - 1.0).abs() < 1e-6);
        for a in 0..4 {
            assert_eq!(p1_prior[a], 0.0);
        }
    }

    // ---- arena_clear + reinit ----

    #[test]
    fn arena_clear_reinit() {
        let cheese = [Coordinates::new(1, 1), Coordinates::new(3, 3)];
        let game1 = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game1);

        // Verify initial state
        assert_eq!(tree.arena().len(), 1);
        assert!((tree.arena()[tree.root()].value_scale() - 2.0).abs() < 1e-6);

        // Add a dummy child so arena has >1 node
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        tree.arena_mut().alloc(Node::new(h, h));
        assert_eq!(tree.arena().len(), 2);

        // Reinit with different game state (1 cheese → value_scale = 1)
        let cheese2 = [Coordinates::new(0, 0)];
        let game2 = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese2);
        tree.reinit(&game2);

        // Old nodes gone, single fresh root
        assert_eq!(tree.arena().len(), 1);
        assert!((tree.arena()[tree.root()].value_scale() - 1.0).abs() < 1e-6);
    }

    // ---- advance_root ----

    #[test]
    fn advance_root_found() {
        let cheese = [Coordinates::new(1, 1)];
        let game = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // Manually create a child at outcome (0, 0) — UP for both players
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let child = tree.arena_mut().alloc(Node::new(h, h));
        let i = tree.arena()[root].p1.action_to_outcome_idx(0); // UP
        let j = tree.arena()[root].p2.action_to_outcome_idx(0); // UP
        tree.arena_mut()[child].set_parent(Some(root), (i, j));
        tree.arena_mut()[root].set_first_child(Some(child));

        // Advance to that child
        assert!(tree.advance_root(0, 0));
        assert_eq!(tree.root(), child);
    }

    #[test]
    fn advance_root_blocked_action_resolves_via_equivalence() {
        // P1 at (0,0): DOWN(2) and LEFT(3) are board edges → map to STAY(4).
        // Wire a child at the STAY outcome, then advance with DOWN.
        // Should resolve to the same child through action_to_outcome_idx.
        let cheese = [Coordinates::new(2, 2)];
        let game = test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // P2 at (4,4): use DOWN(2) — valid move for P2
        let stay_idx = tree.arena()[root].p1.action_to_outcome_idx(4); // STAY
        let j = tree.arena()[root].p2.action_to_outcome_idx(2); // DOWN

        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let child = tree.arena_mut().alloc(Node::new(h, h));
        tree.arena_mut()[child].set_parent(Some(root), (stay_idx, j));
        tree.arena_mut()[root].set_first_child(Some(child));

        // DOWN(2) is blocked at (0,0) → same outcome as STAY(4)
        assert!(tree.advance_root(2, 2));
        assert_eq!(tree.root(), child);
    }

    #[test]
    fn advance_root_not_found() {
        let cheese = [Coordinates::new(1, 1)];
        let game = test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // No children → advance should fail
        assert!(!tree.advance_root(0, 0));
        assert_eq!(tree.root(), root); // Root unchanged
    }

    // ---- extend_node ----

    #[test]
    fn extend_node_shell_properties() {
        // P1 center (2,2): 5 outcomes. P2 corner (4,4): 3 outcomes.
        // Extend with P1=UP, P2=DOWN → child at new positions.
        let cheese = [Coordinates::new(0, 0), Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(0); // UP
        let j = arena[parent].p2.action_to_outcome_idx(2); // DOWN

        // Advance game to child position
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let child_idx = extend_node(&mut arena, parent, i, j, &game);
        let child = &arena[child_idx];

        // Shell properties
        assert_eq!(child.total_visits(), 0);
        assert_eq!(child.v1(), 0.0);
        assert_eq!(child.v2(), 0.0);
        assert!(!child.is_terminal());
        assert_eq!(child.edge_r1(), 0.0);
        assert_eq!(child.edge_r2(), 0.0);

        // Effective actions from game at child position
        // P1 now at (2,3): open interior → 5 outcomes
        assert_eq!(child.p1.n_outcomes(), 5);

        // Priors are zero (shell)
        for idx in 0..child.p1.n_outcomes() {
            assert_eq!(child.p1.prior(idx), 0.0);
        }
        for idx in 0..child.p2.n_outcomes() {
            assert_eq!(child.p2.prior(idx), 0.0);
        }

        // Value scale = remaining cheese
        assert!((child.value_scale() - 2.0).abs() < 1e-6);

        // Parent link
        assert_eq!(child.parent(), Some(parent));
        assert_eq!(child.parent_outcome(), (i, j));

        // Parent's first_child updated
        assert_eq!(arena[parent].first_child(), Some(child_idx));
    }

    #[test]
    fn extend_node_linked_list_two_children() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let parent = arena.alloc(Node::new(root_h, root_h));

        // Child 1: P1=UP, P2=RIGHT
        let i1 = arena[parent].p1.action_to_outcome_idx(0);
        let j1 = arena[parent].p2.action_to_outcome_idx(1);
        let undo1 = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);
        let c1 = extend_node(&mut arena, parent, i1, j1, &game);
        game.unmake_move(undo1);

        // Child 2: P1=DOWN, P2=LEFT
        let i2 = arena[parent].p1.action_to_outcome_idx(2);
        let j2 = arena[parent].p2.action_to_outcome_idx(3);
        let undo2 = game.make_move(pyrat::Direction::Down, pyrat::Direction::Left);
        let c2 = extend_node(&mut arena, parent, i2, j2, &game);
        game.unmake_move(undo2);

        // Prepend order: c2 is first_child (added last)
        assert_eq!(arena[parent].first_child(), Some(c2));
        assert_eq!(arena[c2].next_sibling(), Some(c1));
        assert!(arena[c1].next_sibling().is_none());

        // find_child locates both
        assert_eq!(find_child(&arena, parent, i1, j1), Some(c1));
        assert_eq!(find_child(&arena, parent, i2, j2), Some(c2));

        // Parent outcomes correct
        assert_eq!(arena[c1].parent_outcome(), (i1, j1));
        assert_eq!(arena[c2].parent_outcome(), (i2, j2));
    }

    #[test]
    fn extend_node_multiple_children() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let parent = arena.alloc(Node::new(root_h, root_h));

        // Create 5 children: P1=each direction, P2=STAY
        let j = arena[parent].p2.action_to_outcome_idx(4); // STAY
        let dirs = [
            pyrat::Direction::Up,
            pyrat::Direction::Right,
            pyrat::Direction::Down,
            pyrat::Direction::Left,
            pyrat::Direction::Stay,
        ];
        let mut child_indices = Vec::new();
        for &dir in &dirs {
            let action = dir as u8;
            let i = arena[parent].p1.action_to_outcome_idx(action);
            let undo = game.make_move(dir, pyrat::Direction::Stay);
            let c = extend_node(&mut arena, parent, i, j, &game);
            child_indices.push((c, i));
            game.unmake_move(undo);
        }

        // All findable
        for &(c, i) in &child_indices {
            assert_eq!(find_child(&arena, parent, i, j), Some(c));
        }

        // Linked list walk counts 5 children
        let mut count = 0;
        let mut cur = arena[parent].first_child();
        while let Some(idx) = cur {
            count += 1;
            cur = arena[idx].next_sibling();
        }
        assert_eq!(count, 5);
    }

    // ---- populate_node ----

    #[test]
    fn populate_node_sets_priors() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_idx = arena.alloc(Node::new(shell_p1, shell_p2));

        let eval = EvalResult {
            policy_p1: [0.1, 0.2, 0.3, 0.25, 0.15],
            policy_p2: [0.2; 5],
            value_p1: 0.5,
            value_p2: 0.3,
        };

        populate_node(&mut arena, node_idx, Some(&eval));
        let node = &arena[node_idx];

        // Open position: 5 outcomes, no merging — reduced priors = input priors
        let expected_p1 = [0.1, 0.2, 0.3, 0.25, 0.15];
        for i in 0..node.p1.n_outcomes() {
            assert!(
                (node.p1.prior(i) - expected_p1[i]).abs() < 1e-6,
                "P1 outcome {i}: expected {} got {}",
                expected_p1[i],
                node.p1.prior(i)
            );
        }

        // Priors sum to 1
        let sum_p1: f32 = (0..node.p1.n_outcomes()).map(|i| node.p1.prior(i)).sum();
        let sum_p2: f32 = (0..node.p2.n_outcomes()).map(|i| node.p2.prior(i)).sum();
        assert!((sum_p1 - 1.0).abs() < 1e-6);
        assert!((sum_p2 - 1.0).abs() < 1e-6);

        // Still unvisited — value not set by populate
        assert_eq!(node.total_visits(), 0);
        assert!(!node.is_terminal());
    }

    #[test]
    fn populate_node_terminal() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_idx = arena.alloc(Node::new(shell_p1, shell_p2));

        populate_node(&mut arena, node_idx, None);
        let node = &arena[node_idx];

        assert!(node.is_terminal());
        assert_eq!(node.total_visits(), 0);

        // Priors stay zero
        for i in 0..node.p1.n_outcomes() {
            assert_eq!(node.p1.prior(i), 0.0);
        }
    }

    #[test]
    fn populate_node_wall_topology() {
        // P1 at corner (0,0): DOWN and LEFT blocked → 3 outcomes
        let cheese = [Coordinates::new(2, 2)];
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_idx = arena.alloc(Node::new(shell_p1, shell_p2));

        assert_eq!(arena[node_idx].p1.n_outcomes(), 3);

        // Non-uniform prior where blocked actions have mass
        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 2.0,
        };

        populate_node(&mut arena, node_idx, Some(&eval));
        let node = &arena[node_idx];

        // P1: actions 2,3 blocked → merge into STAY
        // STAY outcome gets: 0.2 (DOWN) + 0.15 (LEFT) + 0.25 (STAY) = 0.6
        let stay_idx = node.p1.action_to_outcome_idx(4) as usize;
        assert!((node.p1.prior(stay_idx) - 0.6).abs() < 1e-6);

        // Expand and verify blocked actions get 0
        let expanded = node.p1.expand_prior();
        assert_eq!(expanded[2], 0.0); // DOWN blocked
        assert_eq!(expanded[3], 0.0); // LEFT blocked

        let sum: f32 = (0..node.p1.n_outcomes()).map(|i| node.p1.prior(i)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ---- integration: extend → populate roundtrip ----

    #[test]
    fn extend_then_populate_roundtrip() {
        use crate::backend::{Backend, SmartUniformBackend};

        let cheese = [Coordinates::new(0, 0), Coordinates::new(4, 4)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        // Root with priors
        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        // Extend: P1=UP, P2=RIGHT
        let i = arena[parent].p1.action_to_outcome_idx(0);
        let j = arena[parent].p2.action_to_outcome_idx(1);
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        let child_idx = extend_node(&mut arena, parent, i, j, &game);

        // Shell: priors zero, visits 0
        assert_eq!(arena[child_idx].total_visits(), 0);
        for idx in 0..arena[child_idx].p1.n_outcomes() {
            assert_eq!(arena[child_idx].p1.prior(idx), 0.0);
        }

        // Populate with SmartUniformBackend
        let backend = SmartUniformBackend;
        let eval = backend.evaluate(&game).unwrap();
        populate_node(&mut arena, child_idx, Some(&eval));

        // Verify: expand_prior matches backend output
        let child = &arena[child_idx];
        let expanded_p1 = child.p1.expand_prior();
        let expanded_p2 = child.p2.expand_prior();

        // Smart uniform: each unique outcome gets equal weight
        let expected_p1 = smart_uniform_prior(&game.effective_actions_p1());
        let expected_p2 = smart_uniform_prior(&game.effective_actions_p2());

        for a in 0..5 {
            assert!(
                (expanded_p1[a] - expected_p1[a]).abs() < 1e-6,
                "P1 action {a}: expanded={} expected={}",
                expanded_p1[a],
                expected_p1[a]
            );
            assert!(
                (expanded_p2[a] - expected_p2[a]).abs() < 1e-6,
                "P2 action {a}: expanded={} expected={}",
                expanded_p2[a],
                expected_p2[a]
            );
        }

        // Still unvisited
        assert_eq!(child.total_visits(), 0);
    }

    #[test]
    fn extend_then_populate_wall_topology() {
        // Parent P1 at (1,0), P2 at (2,2). Extend with P1=LEFT → child P1 at (0,0).
        // Corner (0,0): DOWN and LEFT blocked → 3 outcomes (UP, RIGHT, STAY).
        // Populate with non-uniform priors to exercise merging.
        let cheese = [Coordinates::new(3, 3), Coordinates::new(4, 4)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(1, 0), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        // P1=LEFT(3), P2=STAY(4)
        let i = arena[parent].p1.action_to_outcome_idx(3);
        let j = arena[parent].p2.action_to_outcome_idx(4);
        let _undo = game.make_move(pyrat::Direction::Left, pyrat::Direction::Stay);

        let child_idx = extend_node(&mut arena, parent, i, j, &game);

        // Shell: P1 at corner (0,0) → 3 outcomes
        assert_eq!(arena[child_idx].p1.n_outcomes(), 3);

        // Populate with non-uniform priors — blocked actions carry mass
        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 0.5,
            value_p2: 0.3,
        };
        populate_node(&mut arena, child_idx, Some(&eval));
        let child = &arena[child_idx];

        // DOWN(2) and LEFT(3) blocked → merge into STAY
        // STAY outcome gets: 0.2 (DOWN) + 0.15 (LEFT) + 0.25 (STAY) = 0.6
        let stay_idx = child.p1.action_to_outcome_idx(4) as usize;
        assert!((child.p1.prior(stay_idx) - 0.6).abs() < 1e-6);

        // expand_prior: blocked actions get 0
        let expanded = child.p1.expand_prior();
        assert_eq!(expanded[2], 0.0); // DOWN blocked
        assert_eq!(expanded[3], 0.0); // LEFT blocked

        // STAY in expanded space carries the merged mass
        assert!((expanded[4] - 0.6).abs() < 1e-6);

        // Still unvisited
        assert_eq!(child.total_visits(), 0);
    }

    // ---- find_or_extend_child: basic ----

    #[test]
    fn find_or_extend_new_child() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(0); // UP
        let j = arena[parent].p2.action_to_outcome_idx(2); // DOWN

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (child_idx, is_new) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
        assert!(is_new);

        // Shell properties: priors zero, visits 0
        let child = &arena[child_idx];
        assert_eq!(child.total_visits(), 0);
        for idx in 0..child.p1.n_outcomes() {
            assert_eq!(child.p1.prior(idx), 0.0);
        }

        // Effective actions derived from child position
        // P1 at (2,3): open interior → 5 outcomes
        assert_eq!(child.p1.n_outcomes(), 5);
    }

    #[test]
    fn find_or_extend_existing_child() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(0);
        let j = arena[parent].p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (first_idx, is_new1) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
        assert!(is_new1);
        let arena_len_after_first = arena.len();

        // Second call: same (i, j) → existing child
        let (second_idx, is_new2) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
        assert!(!is_new2);
        assert_eq!(first_idx, second_idx);
        assert_eq!(arena.len(), arena_len_after_first); // No new allocation
    }

    #[test]
    fn find_or_extend_multiple_children() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let parent = arena.alloc(Node::new(root_h, root_h));

        let j = arena[parent].p2.action_to_outcome_idx(4); // P2=STAY
        let moves = [
            (0, pyrat::Direction::Up),
            (1, pyrat::Direction::Right),
            (2, pyrat::Direction::Down),
        ];

        let mut children = Vec::new();
        for &(action, dir) in &moves {
            let i = arena[parent].p1.action_to_outcome_idx(action);
            let undo = game.make_move(dir, pyrat::Direction::Stay);
            let (child, is_new) =
                find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
            assert!(is_new);
            children.push((child, i));
            game.unmake_move(undo);
        }

        // All distinct
        assert_ne!(children[0].0, children[1].0);
        assert_ne!(children[1].0, children[2].0);

        // All findable
        for &(child, i) in &children {
            assert_eq!(find_child(&arena, parent, i, j), Some(child));
        }
    }

    // ---- compute_rewards ----

    #[test]
    fn compute_rewards_cheese_collected() {
        let mut game = test_util::one_cheese_adjacent_game();
        let scores_before = (game.player1_score(), game.player2_score());

        // P1 at (0,0) moves RIGHT to (1,0) where cheese is
        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Stay);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6, "P1 should collect 1 cheese, got {r1}");
        assert!((r2 - 0.0).abs() < 1e-6, "P2 should collect nothing, got {r2}");
    }

    #[test]
    fn compute_rewards_no_cheese() {
        let mut game = test_util::one_cheese_adjacent_game();
        let scores_before = (game.player1_score(), game.player2_score());

        // P1 moves UP (no cheese at (0,1))
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Stay);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.0).abs() < 1e-6);
        assert!((r2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn compute_rewards_contested_cheese() {
        // Both players reach the same cheese simultaneously → 0.5 each
        let mut game = test_util::contested_cheese_game();
        let scores_before = (game.player1_score(), game.player2_score());

        // P1 RIGHT (0,0)→(1,0), P2 LEFT (2,0)→(1,0) — both land on cheese
        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.5).abs() < 1e-6, "P1 should get 0.5, got {r1}");
        assert!((r2 - 0.5).abs() < 1e-6, "P2 should get 0.5, got {r2}");
    }

    #[test]
    fn compute_rewards_both_collect_different() {
        // Each player collects a different cheese → (1.0, 1.0)
        let mut game = GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(
                Coordinates::new(0, 0), // P1 adjacent to cheese at (1,0)
                Coordinates::new(4, 0), // P2 adjacent to cheese at (3,0)
            )
            .with_custom_cheese(vec![Coordinates::new(1, 0), Coordinates::new(3, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        let scores_before = (game.player1_score(), game.player2_score());

        // P1 RIGHT → (1,0), P2 LEFT → (3,0)
        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6, "P1 should collect 1, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "P2 should collect 1, got {r2}");
    }

    #[test]
    fn compute_rewards_p2_collects() {
        // Only P2 collects — symmetry check
        let mut game = test_util::contested_cheese_game();
        let scores_before = (game.player1_score(), game.player2_score());

        // P1 stays, P2 LEFT (2,0)→(1,0) collects cheese
        let _undo = game.make_move(pyrat::Direction::Stay, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.0).abs() < 1e-6, "P1 should get 0, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "P2 should get 1, got {r2}");
    }

    #[test]
    fn find_or_extend_rewards_on_child() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(0);
        let j = arena[parent].p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (child_idx, _) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 1.5, 0.5);

        assert!((arena[child_idx].edge_r1() - 1.5).abs() < 1e-6);
        assert!((arena[child_idx].edge_r2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn find_or_extend_existing_child_ignores_rewards() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(0);
        let j = arena[parent].p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        // First call: creates child with rewards (1.5, 0.5)
        let (child_idx, is_new) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 1.5, 0.5);
        assert!(is_new);

        // Second call: same outcome pair, different rewards — should be ignored
        let (same_idx, is_new2) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 99.0, 99.0);
        assert!(!is_new2);
        assert_eq!(child_idx, same_idx);

        // Edge rewards unchanged from first creation
        assert!((arena[child_idx].edge_r1() - 1.5).abs() < 1e-6);
        assert!((arena[child_idx].edge_r2() - 0.5).abs() < 1e-6);
    }

    // ---- debug_assert guards ----

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: child already exists")]
    fn extend_node_duplicate_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let parent = arena.alloc(Node::new(root_h, root_h));

        let i = arena[parent].p1.action_to_outcome_idx(0);
        let j = arena[parent].p2.action_to_outcome_idx(1);
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        // First extend: fine
        let _ = extend_node(&mut arena, parent, i, j, &game);
        // Second extend at same (i, j): should panic
        let _ = extend_node(&mut arena, parent, i, j, &game);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "populate_node: node already has")]
    fn populate_node_visited_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_idx = arena.alloc(Node::new(shell_p1, shell_p2));

        // Simulate a visit before populate
        arena[node_idx].update_value(1.0, 1.0);
        assert_eq!(arena[node_idx].total_visits(), 1);

        let eval = EvalResult {
            policy_p1: [0.2; 5],
            policy_p2: [0.2; 5],
            value_p1: 0.0,
            value_p2: 0.0,
        };
        // Should panic: node already has visits
        populate_node(&mut arena, node_idx, Some(&eval));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: outcome_p1")]
    fn extend_node_outcome_p1_out_of_bounds_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        // P1 at (0,0): 3 outcomes. Outcome index 3 is out of bounds.
        let _ = extend_node(&mut arena, parent, 3, 0, &game);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: outcome_p2")]
    fn extend_node_outcome_p2_out_of_bounds_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        // P2 at (4,4): 3 outcomes. Outcome index 3 is out of bounds.
        let _ = extend_node(&mut arena, parent, 0, 3, &game);
    }

    // ---- integration / topology ----

    #[test]
    fn find_or_extend_corner_topology() {
        // P1 at (1,0) moves LEFT → child P1 at (0,0), corner with 3 outcomes
        let cheese = [Coordinates::new(3, 3)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(1, 0), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        let i = arena[parent].p1.action_to_outcome_idx(3); // LEFT
        let j = arena[parent].p2.action_to_outcome_idx(4); // STAY

        let _undo = game.make_move(pyrat::Direction::Left, pyrat::Direction::Stay);

        let (child_idx, is_new) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
        assert!(is_new);

        // (0,0) corner: DOWN and LEFT blocked → 3 outcomes
        assert_eq!(arena[child_idx].p1.n_outcomes(), 3);
    }

    #[test]
    fn find_or_extend_then_populate_corner_merges_priors() {
        // Extend to a corner, then populate with non-uniform priors that
        // exercise prior merging — distinct from extend_then_populate_roundtrip
        // which uses an open interior with SmartUniformBackend.
        let cheese = [Coordinates::new(3, 3), Coordinates::new(4, 4)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(1, 0), Coordinates::new(2, 2), &cheese);
        let mut arena = NodeArena::new();

        let root_h1 = HalfNode::new([0.2; 5], game.effective_actions_p1());
        let root_h2 = HalfNode::new([0.2; 5], game.effective_actions_p2());
        let parent = arena.alloc(Node::new(root_h1, root_h2));

        // P1=LEFT(3) → child P1 at (0,0), corner: DOWN and LEFT blocked → 3 outcomes
        let i = arena[parent].p1.action_to_outcome_idx(3);
        let j = arena[parent].p2.action_to_outcome_idx(4); // STAY

        let _undo = game.make_move(pyrat::Direction::Left, pyrat::Direction::Stay);

        let (child_idx, is_new) =
            find_or_extend_child(&mut arena, parent, i, j, &game, 0.0, 0.0);
        assert!(is_new);
        assert_eq!(arena[child_idx].p1.n_outcomes(), 3);

        // Populate with non-uniform priors — blocked actions carry mass
        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 0.5,
            value_p2: 0.3,
        };
        populate_node(&mut arena, child_idx, Some(&eval));
        let child = &arena[child_idx];

        // STAY outcome merges: DOWN(0.2) + LEFT(0.15) + STAY(0.25) = 0.6
        let stay_idx = child.p1.action_to_outcome_idx(4) as usize;
        assert!((child.p1.prior(stay_idx) - 0.6).abs() < 1e-6);

        // Blocked actions expand to 0
        let expanded = child.p1.expand_prior();
        assert_eq!(expanded[2], 0.0); // DOWN blocked
        assert_eq!(expanded[3], 0.0); // LEFT blocked
        assert!((expanded[4] - 0.6).abs() < 1e-6);

        assert_eq!(child.total_visits(), 0);
    }

    // ---- find_or_extend + advance_root interaction ----

    #[test]
    fn find_or_extend_then_advance_root() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(4, 4)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);

        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // Create child via find_or_extend_child on the arena
        let i = tree.arena()[root].p1.action_to_outcome_idx(0); // UP
        let j = tree.arena()[root].p2.action_to_outcome_idx(1); // RIGHT

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        let (child_idx, is_new) =
            find_or_extend_child(tree.arena_mut(), root, i, j, &game, 0.0, 0.0);
        assert!(is_new);

        // advance_root with the same actions should land on that child
        assert!(tree.advance_root(0, 1));
        assert_eq!(tree.root(), child_idx);
    }
}
