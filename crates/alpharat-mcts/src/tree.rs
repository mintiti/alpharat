use std::mem::ManuallyDrop;
use std::sync::{mpsc, Mutex, OnceLock};
use std::thread;

use crate::{EvalResult, HalfNode, Node, NodePtr};
use pyrat::GameState;

// ---------------------------------------------------------------------------
// Global NodeGc — process-wide async subtree deallocation
// ---------------------------------------------------------------------------

static NODE_GC: OnceLock<NodeGc> = OnceLock::new();

struct NodeGc {
    sender: Mutex<mpsc::Sender<Box<Node>>>,
}

impl NodeGc {
    fn init() -> Self {
        let (sender, receiver) = mpsc::channel();
        thread::Builder::new()
            .name("alpharat-node-gc".into())
            .spawn(move || {
                while let Ok(subtree) = receiver.recv() {
                    drop(subtree);
                }
            })
            .expect("failed to spawn node-gc thread");
        Self {
            sender: Mutex::new(sender),
        }
    }

    fn send(&self, node: Box<Node>) {
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(node);
        }
        // Mutex poisoned or send failed: node drops locally (safe fallback)
    }
}

fn node_gc() -> &'static NodeGc {
    NODE_GC.get_or_init(NodeGc::init)
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Walk the child linked list of `parent`, return the child whose
/// `parent_outcome` matches `(i, j)`.
pub fn find_child(parent: NodePtr, i: u8, j: u8) -> Option<NodePtr> {
    let parent_node = unsafe { parent.as_ref() };
    let mut cur = parent_node.first_child();
    while let Some(ptr) = cur {
        let node = unsafe { ptr.as_ref() };
        if node.parent_outcome() == (i, j) {
            return Some(ptr);
        }
        cur = node.next_sibling();
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
    parent: NodePtr,
    outcome_p1: u8,
    outcome_p2: u8,
    game: &GameState,
) -> NodePtr {
    #[cfg(debug_assertions)]
    unsafe {
        let parent_ref = parent.as_ref();
        debug_assert!(
            (outcome_p1 as usize) < parent_ref.p1.n_outcomes(),
            "extend_node: outcome_p1={outcome_p1} out of bounds (n_outcomes={})",
            parent_ref.p1.n_outcomes()
        );
        debug_assert!(
            (outcome_p2 as usize) < parent_ref.p2.n_outcomes(),
            "extend_node: outcome_p2={outcome_p2} out of bounds (n_outcomes={})",
            parent_ref.p2.n_outcomes()
        );
        debug_assert!(
            find_child(parent, outcome_p1, outcome_p2).is_none(),
            "extend_node: child already exists at outcome ({outcome_p1}, {outcome_p2})"
        );
    }

    let p1_shell = HalfNode::new_shell(game.effective_actions_p1());
    let p2_shell = HalfNode::new_shell(game.effective_actions_p2());

    let mut node = Node::new(p1_shell, p2_shell);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);
    node.set_parent(Some(parent), (outcome_p1, outcome_p2));

    // Prepend to parent's linked list.
    let parent_mut = unsafe { parent.as_mut() };
    node.next_sibling = parent_mut.first_child.take();

    let child_box = Box::new(node);
    let child_ptr = NodePtr::from_ref(&child_box);
    parent_mut.first_child = Some(child_box);

    child_ptr
}

/// Set priors on a shell node after batch NN evaluation.
///
/// - `Some(result)`: reduces NN policies into outcome-indexed priors.
/// - `None`: marks the node as terminal (no priors needed).
///
/// Does NOT call `update_value` — value flows through backup (chunk 4).
pub fn populate_node(node: NodePtr, eval_result: Option<&EvalResult>) {
    let node_mut = unsafe { node.as_mut() };
    debug_assert!(
        node_mut.total_visits() == 0,
        "populate_node: node already has {} visits",
        node_mut.total_visits()
    );

    match eval_result {
        Some(result) => {
            node_mut.p1.set_prior(result.policy_p1);
            node_mut.p2.set_prior(result.policy_p2);
        }
        None => {
            node_mut.set_terminal();
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
/// Returns (child_ptr, is_new).
pub fn find_or_extend_child(
    parent: NodePtr,
    outcome_p1: u8,
    outcome_p2: u8,
    game: &GameState,
    r1: f32,
    r2: f32,
) -> (NodePtr, bool) {
    if let Some(child) = find_child(parent, outcome_p1, outcome_p2) {
        return (child, false);
    }

    let child = extend_node(parent, outcome_p1, outcome_p2, game);
    unsafe { child.as_mut() }.set_edge_rewards(r1, r2);
    (child, true)
}

// ---------------------------------------------------------------------------
// MCTSTree
// ---------------------------------------------------------------------------

/// Lifecycle manager for the MCTS tree and root node.
///
/// Owns the root via Box<Node>. Child nodes are owned via the linked-list
/// structure (Box<Node> in first_child / next_sibling). Orphaned subtrees
/// from advance_root / reinit / drop are sent to the global GC thread for
/// async deallocation.
pub struct MCTSTree {
    root_box: ManuallyDrop<Box<Node>>,
    node_count: u32,
}

impl MCTSTree {
    /// Create a new tree with a root node derived from `game`.
    ///
    /// Root gets smart uniform priors and `value_scale = max(remaining_cheese, 1)`.
    /// Starts unevaluated (v1=v2=0, total_visits=0) — LC0 style.
    pub fn new(game: &GameState) -> Self {
        // Ensure the global GC thread is running.
        node_gc();
        Self {
            root_box: ManuallyDrop::new(alloc_root(game)),
            node_count: 1,
        }
    }

    pub fn root(&self) -> NodePtr {
        NodePtr::from_ref(&self.root_box)
    }

    /// Immutable reference to the root node.
    pub fn root_node(&self) -> &Node {
        &self.root_box
    }

    /// Number of nodes in the tree. Used for collision budget scaling.
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Increment node count (called when a new node is created).
    pub fn increment_node_count(&mut self) {
        self.node_count += 1;
    }

    /// Move root to the child matching the given action pair.
    ///
    /// Returns `true` if the child was found (root advanced), `false` if
    /// the child doesn't exist (caller should `reinit`).
    ///
    /// Orphaned subtree (old root + remaining siblings) is sent to the
    /// GC thread for async deallocation.
    pub fn advance_root(&mut self, p1_action: u8, p2_action: u8) -> bool {
        let i = self.root_box.p1.action_to_outcome_idx(p1_action);
        let j = self.root_box.p2.action_to_outcome_idx(p2_action);

        if let Some(child_box) = self.detach_child(i, j) {
            let old_root = std::mem::replace(&mut *self.root_box, child_box);
            node_gc().send(old_root);
            true
        } else {
            false
        }
    }

    /// Clear the tree and create a fresh root from `game`.
    pub fn reinit(&mut self, game: &GameState) {
        let old_root = std::mem::replace(&mut *self.root_box, alloc_root(game));
        node_gc().send(old_root);
        self.node_count = 1;
    }

    /// Detach the child matching outcome `(i, j)` from root's linked list.
    fn detach_child(&mut self, i: u8, j: u8) -> Option<Box<Node>> {
        // Check first child.
        if self
            .root_box
            .first_child
            .as_ref()
            .is_some_and(|c| c.parent_outcome() == (i, j))
        {
            let mut child = self.root_box.first_child.take().unwrap();
            self.root_box.first_child = child.next_sibling.take();
            child.parent = None;
            return Some(child);
        }

        // Walk sibling chain via NodePtr to find prev → target.
        let mut node_ptr = self.root_box.first_child();
        while let Some(ptr) = node_ptr {
            // SAFETY: ptr points to a child node (distinct heap allocation from
            // root_box), so mutating it doesn't alias root_box or other siblings.
            let node = unsafe { ptr.as_mut() };
            if node
                .next_sibling
                .as_ref()
                .is_some_and(|s| s.parent_outcome() == (i, j))
            {
                let mut found = node.next_sibling.take().unwrap();
                node.next_sibling = found.next_sibling.take();
                found.parent = None;
                return Some(found);
            }
            node_ptr = node.next_sibling();
        }

        None
    }
}

impl Drop for MCTSTree {
    fn drop(&mut self) {
        // SAFETY: drop runs exactly once; root_box is not accessed after.
        let root = unsafe { ManuallyDrop::take(&mut self.root_box) };
        node_gc().send(root);
    }
}

/// Allocate a root node from the current game state.
fn alloc_root(game: &GameState) -> Box<Node> {
    let eff_p1 = game.effective_actions_p1();
    let eff_p2 = game.effective_actions_p2();

    let prior_p1 = smart_uniform_prior(&eff_p1);
    let prior_p2 = smart_uniform_prior(&eff_p2);

    let p1_half = HalfNode::new(prior_p1, eff_p1);
    let p2_half = HalfNode::new(prior_p2, eff_p2);

    let mut node = Node::new(p1_half, p2_half);
    node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

    Box::new(node)
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
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let mut parent = Box::new(Node::new(h, h));
        let parent_ptr = NodePtr::from_ref(&parent);

        let mut c2 = Box::new(Node::new(h, h));
        let c2_ptr = NodePtr::from_ref(&c2);
        c2.set_parent(Some(parent_ptr), (4, 0));

        let mut c1 = Box::new(Node::new(h, h));
        let c1_ptr = NodePtr::from_ref(&c1);
        c1.set_parent(Some(parent_ptr), (2, 3));
        c1.next_sibling = Some(c2);

        let mut c0 = Box::new(Node::new(h, h));
        let c0_ptr = NodePtr::from_ref(&c0);
        c0.set_parent(Some(parent_ptr), (0, 1));
        c0.next_sibling = Some(c1);

        parent.first_child = Some(c0);

        assert_eq!(find_child(parent_ptr, 0, 1), Some(c0_ptr));
        assert_eq!(find_child(parent_ptr, 2, 3), Some(c1_ptr));
        assert_eq!(find_child(parent_ptr, 4, 0), Some(c2_ptr));
    }

    #[test]
    fn find_child_not_found() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let mut parent = Box::new(Node::new(h, h));
        let parent_ptr = NodePtr::from_ref(&parent);

        let mut child = Box::new(Node::new(h, h));
        child.set_parent(Some(parent_ptr), (0, 0));
        parent.first_child = Some(child);

        assert_eq!(find_child(parent_ptr, 1, 1), None);
    }

    #[test]
    fn find_child_empty() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let parent = Box::new(Node::new(h, h));
        let parent_ptr = NodePtr::from_ref(&parent);

        assert_eq!(find_child(parent_ptr, 0, 0), None);
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
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        let root = tree.root_node();

        // Center of open 5×5: 5 unique outcomes per player
        assert_eq!(root.p1.n_outcomes(), 5);
        // P2 at (4,4) — top-right corner, UP and RIGHT are board edges
        // → blocked → map to STAY → 3 unique outcomes
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
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        let root = tree.root_node();

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
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);
        assert!((tree.root_node().value_scale() - 5.0).abs() < 1e-6);

        // 0 cheese → value_scale = 1 (clamped)
        let game_no_cheese =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &[]);
        let tree_no_cheese = MCTSTree::new(&game_no_cheese);
        assert!((tree_no_cheese.root_node().value_scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn root_init_mud() {
        let game = test_util::mud_game_p1_stuck();
        let tree = MCTSTree::new(&game);
        let root = tree.root_node();

        // All actions → STAY when in mud
        assert_eq!(root.p1.n_outcomes(), 1);
        let p1_prior = root.p1.expand_prior();
        assert!((p1_prior[4] - 1.0).abs() < 1e-6);
        for a in 0..4 {
            assert_eq!(p1_prior[a], 0.0);
        }
    }

    // ---- reinit ----

    #[test]
    fn reinit_creates_fresh_root() {
        let cheese = [Coordinates::new(1, 1), Coordinates::new(3, 3)];
        let game1 =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game1);

        // Verify initial state
        assert!((tree.root_node().value_scale() - 2.0).abs() < 1e-6);

        // Reinit with different game state (1 cheese → value_scale = 1)
        let cheese2 = [Coordinates::new(0, 0)];
        let game2 =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese2);
        tree.reinit(&game2);

        assert!((tree.root_node().value_scale() - 1.0).abs() < 1e-6);
        assert!(tree.root_node().first_child().is_none());
    }

    // ---- advance_root ----

    #[test]
    fn advance_root_found() {
        let cheese = [Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0); // UP
        let j = tree.root_node().p2.action_to_outcome_idx(0); // UP

        // Create a child via extend_node
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Up);
        let child_ptr = extend_node(tree.root(), i, j, &game);
        game.unmake_move(_undo);

        assert!(tree.advance_root(0, 0));
        assert_eq!(tree.root(), child_ptr);
    }

    #[test]
    fn advance_root_blocked_action_resolves_via_equivalence() {
        // P1 at (0,0): DOWN(2) and LEFT(3) are board edges → map to STAY(4).
        // Wire a child at the STAY outcome, then advance with DOWN.
        // Should resolve to the same child through action_to_outcome_idx.
        let cheese = [Coordinates::new(2, 2)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);

        let stay_idx = tree.root_node().p1.action_to_outcome_idx(4); // STAY
        let j = tree.root_node().p2.action_to_outcome_idx(2); // DOWN

        let _undo = game.make_move(pyrat::Direction::Stay, pyrat::Direction::Down);
        let _child_ptr = extend_node(tree.root(), stay_idx, j, &game);
        game.unmake_move(_undo);

        // DOWN(2) is blocked at (0,0) → same outcome as STAY(4)
        assert!(tree.advance_root(2, 2));
    }

    #[test]
    fn advance_root_not_found() {
        let cheese = [Coordinates::new(1, 1)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // No children → advance should fail
        assert!(!tree.advance_root(0, 0));
        assert_eq!(tree.root(), root);
    }

    #[test]
    fn advance_root_non_first_child() {
        let cheese = [Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let mut tree = MCTSTree::new(&game);

        let i1 = tree.root_node().p1.action_to_outcome_idx(0);
        let j1 = tree.root_node().p2.action_to_outcome_idx(1);
        let i2 = tree.root_node().p1.action_to_outcome_idx(2);
        let j2 = tree.root_node().p2.action_to_outcome_idx(3);

        // Create c1 at (UP, RIGHT).
        let undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);
        let c1_ptr = extend_node(tree.root(), i1, j1, &game);
        game.unmake_move(undo);

        // Create c2 at (DOWN, LEFT) — prepend makes c2 first_child.
        let undo = game.make_move(pyrat::Direction::Down, pyrat::Direction::Left);
        let _c2_ptr = extend_node(tree.root(), i2, j2, &game);
        game.unmake_move(undo);

        // advance_root with c1's actions — must walk past c2 to find c1.
        assert!(tree.advance_root(0, 1));
        assert_eq!(tree.root(), c1_ptr);
        assert!(tree.root_node().parent().is_none());
    }

    // ---- extend_node ----

    #[test]
    fn extend_node_shell_properties() {
        // P1 center (2,2): 5 outcomes. P2 corner (4,4): 3 outcomes.
        // Extend with P1=UP, P2=DOWN → child at new positions.
        let cheese = [Coordinates::new(0, 0), Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0); // UP
        let j = tree.root_node().p2.action_to_outcome_idx(2); // DOWN

        // Advance game to child position
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);
        let child_ptr = extend_node(tree.root(), i, j, &game);
        let child = unsafe { child_ptr.as_ref() };

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

        assert!((child.value_scale() - 2.0).abs() < 1e-6);
        assert_eq!(child.parent(), Some(tree.root()));
        assert_eq!(child.parent_outcome(), (i, j));
        assert_eq!(tree.root_node().first_child(), Some(child_ptr));
    }

    #[test]
    fn extend_node_linked_list_two_children() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let tree = MCTSTree::new(&game);
        let root = tree.root();

        let root_node = tree.root_node();
        let i1 = root_node.p1.action_to_outcome_idx(0);
        let j1 = root_node.p2.action_to_outcome_idx(1);
        let undo1 = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);
        let c1 = extend_node(root, i1, j1, &game);
        game.unmake_move(undo1);

        let i2 = root_node.p1.action_to_outcome_idx(2);
        let j2 = root_node.p2.action_to_outcome_idx(3);
        let undo2 = game.make_move(pyrat::Direction::Down, pyrat::Direction::Left);
        let c2 = extend_node(root, i2, j2, &game);
        game.unmake_move(undo2);

        // Prepend order: c2 is first_child (added last)
        assert_eq!(tree.root_node().first_child(), Some(c2));
        assert_eq!(unsafe { c2.as_ref() }.next_sibling(), Some(c1));
        assert!(unsafe { c1.as_ref() }.next_sibling().is_none());

        assert_eq!(find_child(root, i1, j1), Some(c1));
        assert_eq!(find_child(root, i2, j2), Some(c2));

        assert_eq!(unsafe { c1.as_ref() }.parent_outcome(), (i1, j1));
        assert_eq!(unsafe { c2.as_ref() }.parent_outcome(), (i2, j2));
    }

    #[test]
    fn extend_node_multiple_children() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let tree = MCTSTree::new(&game);
        let root = tree.root();

        let j = tree.root_node().p2.action_to_outcome_idx(4); // STAY
        let dirs = [
            pyrat::Direction::Up,
            pyrat::Direction::Right,
            pyrat::Direction::Down,
            pyrat::Direction::Left,
            pyrat::Direction::Stay,
        ];
        let mut child_ptrs = Vec::new();
        for &dir in &dirs {
            let action = dir as u8;
            let i = tree.root_node().p1.action_to_outcome_idx(action);
            let undo = game.make_move(dir, pyrat::Direction::Stay);
            let c = extend_node(root, i, j, &game);
            child_ptrs.push((c, i));
            game.unmake_move(undo);
        }

        for &(c, i) in &child_ptrs {
            assert_eq!(find_child(root, i, j), Some(c));
        }

        // Linked list walk counts 5 children
        let mut count = 0;
        let mut cur = tree.root_node().first_child();
        while let Some(ptr) = cur {
            count += 1;
            cur = unsafe { ptr.as_ref() }.next_sibling();
        }
        assert_eq!(count, 5);
    }

    // ---- populate_node ----

    #[test]
    fn populate_node_sets_priors() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_box = Box::new(Node::new(shell_p1, shell_p2));
        let node_ptr = NodePtr::from_ref(&node_box);

        let eval = EvalResult {
            policy_p1: [0.1, 0.2, 0.3, 0.25, 0.15],
            policy_p2: [0.2; 5],
            value_p1: 0.5,
            value_p2: 0.3,
        };

        populate_node(node_ptr, Some(&eval));
        let node = unsafe { node_ptr.as_ref() };

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

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_box = Box::new(Node::new(shell_p1, shell_p2));
        let node_ptr = NodePtr::from_ref(&node_box);

        populate_node(node_ptr, None);
        let node = unsafe { node_ptr.as_ref() };

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

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_box = Box::new(Node::new(shell_p1, shell_p2));
        let node_ptr = NodePtr::from_ref(&node_box);

        assert_eq!(unsafe { node_ptr.as_ref() }.p1.n_outcomes(), 3);

        let eval = EvalResult {
            policy_p1: [0.1, 0.3, 0.2, 0.15, 0.25],
            policy_p2: [0.2; 5],
            value_p1: 1.0,
            value_p2: 2.0,
        };

        populate_node(node_ptr, Some(&eval));
        let node = unsafe { node_ptr.as_ref() };

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
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(1);
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        let child_ptr = extend_node(tree.root(), i, j, &game);

        // Shell: priors zero, visits 0
        assert_eq!(unsafe { child_ptr.as_ref() }.total_visits(), 0);
        for idx in 0..unsafe { child_ptr.as_ref() }.p1.n_outcomes() {
            assert_eq!(unsafe { child_ptr.as_ref() }.p1.prior(idx), 0.0);
        }

        // Populate with SmartUniformBackend
        let backend = SmartUniformBackend;
        let eval = backend.evaluate(&game).unwrap();
        populate_node(child_ptr, Some(&eval));

        // Verify: expand_prior matches backend output
        let child = unsafe { child_ptr.as_ref() };
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

    // ---- compute_rewards ----

    #[test]
    fn compute_rewards_cheese_collected() {
        let mut game = test_util::one_cheese_adjacent_game();
        let scores_before = (game.player1_score(), game.player2_score());

        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Stay);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6, "P1 should collect 1 cheese, got {r1}");
        assert!((r2 - 0.0).abs() < 1e-6, "P2 should collect nothing, got {r2}");
    }

    #[test]
    fn compute_rewards_no_cheese() {
        let mut game = test_util::one_cheese_adjacent_game();
        let scores_before = (game.player1_score(), game.player2_score());

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Stay);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.0).abs() < 1e-6);
        assert!((r2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn compute_rewards_contested_cheese() {
        let mut game = test_util::contested_cheese_game();
        let scores_before = (game.player1_score(), game.player2_score());

        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.5).abs() < 1e-6, "P1 should get 0.5, got {r1}");
        assert!((r2 - 0.5).abs() < 1e-6, "P2 should get 0.5, got {r2}");
    }

    #[test]
    fn compute_rewards_both_collect_different() {
        let mut game = GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 0))
            .with_custom_cheese(vec![Coordinates::new(1, 0), Coordinates::new(3, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        let scores_before = (game.player1_score(), game.player2_score());

        let _undo = game.make_move(pyrat::Direction::Right, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 1.0).abs() < 1e-6, "P1 should collect 1, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "P2 should collect 1, got {r2}");
    }

    #[test]
    fn compute_rewards_p2_collects() {
        let mut game = test_util::contested_cheese_game();
        let scores_before = (game.player1_score(), game.player2_score());

        let _undo = game.make_move(pyrat::Direction::Stay, pyrat::Direction::Left);

        let (r1, r2) = compute_rewards(&game, scores_before);
        assert!((r1 - 0.0).abs() < 1e-6, "P1 should get 0, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "P2 should get 1, got {r2}");
    }

    #[test]
    fn find_or_extend_new_child() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(1, 1)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (child_ptr, is_new) = find_or_extend_child(tree.root(), i, j, &game, 0.0, 0.0);
        assert!(is_new);

        let child = unsafe { child_ptr.as_ref() };
        assert_eq!(child.total_visits(), 0);
        for idx in 0..child.p1.n_outcomes() {
            assert_eq!(child.p1.prior(idx), 0.0);
        }
        assert_eq!(child.p1.n_outcomes(), 5);
    }

    #[test]
    fn find_or_extend_existing_child() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (first_ptr, is_new1) = find_or_extend_child(tree.root(), i, j, &game, 0.0, 0.0);
        assert!(is_new1);

        let (second_ptr, is_new2) = find_or_extend_child(tree.root(), i, j, &game, 0.0, 0.0);
        assert!(!is_new2);
        assert_eq!(first_ptr, second_ptr);
    }

    #[test]
    fn find_or_extend_rewards_on_child() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (child_ptr, _) = find_or_extend_child(tree.root(), i, j, &game, 1.5, 0.5);

        let child = unsafe { child_ptr.as_ref() };
        assert!((child.edge_r1() - 1.5).abs() < 1e-6);
        assert!((child.edge_r2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn find_or_extend_existing_child_ignores_rewards() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(2);

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Down);

        let (child_ptr, is_new) = find_or_extend_child(tree.root(), i, j, &game, 1.5, 0.5);
        assert!(is_new);

        let (same_ptr, is_new2) = find_or_extend_child(tree.root(), i, j, &game, 99.0, 99.0);
        assert!(!is_new2);
        assert_eq!(child_ptr, same_ptr);

        let child = unsafe { child_ptr.as_ref() };
        assert!((child.edge_r1() - 1.5).abs() < 1e-6);
        assert!((child.edge_r2() - 0.5).abs() < 1e-6);
    }

    // ---- debug_assert guards ----

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: child already exists")]
    fn extend_node_duplicate_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);
        let tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0);
        let j = tree.root_node().p2.action_to_outcome_idx(1);
        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        let _ = extend_node(tree.root(), i, j, &game);
        let _ = extend_node(tree.root(), i, j, &game);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "populate_node: node already has")]
    fn populate_node_visited_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);

        let shell_p1 = HalfNode::new_shell(game.effective_actions_p1());
        let shell_p2 = HalfNode::new_shell(game.effective_actions_p2());
        let node_box = Box::new(Node::new(shell_p1, shell_p2));
        let node_ptr = NodePtr::from_ref(&node_box);

        unsafe { node_ptr.as_mut() }.update_value(1.0, 1.0);

        let eval = EvalResult {
            policy_p1: [0.2; 5],
            policy_p2: [0.2; 5],
            value_p1: 0.0,
            value_p2: 0.0,
        };
        populate_node(node_ptr, Some(&eval));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: outcome_p1")]
    fn extend_node_outcome_p1_out_of_bounds_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let _ = extend_node(tree.root(), 3, 0, &game);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "extend_node: outcome_p2")]
    fn extend_node_outcome_p2_out_of_bounds_panics() {
        let cheese = [Coordinates::new(0, 0)];
        let game =
            test_util::open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let tree = MCTSTree::new(&game);

        let _ = extend_node(tree.root(), 0, 3, &game);
    }

    // ---- integration: find_or_extend + advance_root ----

    #[test]
    fn find_or_extend_then_advance_root() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(4, 4)];
        let mut game =
            test_util::open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &cheese);

        let mut tree = MCTSTree::new(&game);

        let i = tree.root_node().p1.action_to_outcome_idx(0); // UP
        let j = tree.root_node().p2.action_to_outcome_idx(1); // RIGHT

        let _undo = game.make_move(pyrat::Direction::Up, pyrat::Direction::Right);

        let (child_ptr, is_new) = find_or_extend_child(tree.root(), i, j, &game, 0.0, 0.0);
        assert!(is_new);

        assert!(tree.advance_root(0, 1));
        assert_eq!(tree.root(), child_ptr);
    }
}
