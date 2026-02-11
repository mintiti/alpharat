use crate::{HalfNode, Node, NodeArena, NodeIndex};
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
    use pyrat::{Coordinates, GameState};
    use std::collections::HashMap;

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

    // ---- Helper: create simple games for tree tests ----

    fn open_5x5_game(cheese: &[Coordinates]) -> GameState {
        GameState::new_with_config(
            5,
            5,
            HashMap::new(), // no walls
            Default::default(),
            cheese,
            Coordinates::new(2, 2), // center — open in all directions
            Coordinates::new(4, 4),
            100,
        )
    }

    fn corner_game(cheese: &[Coordinates]) -> GameState {
        // P1 at (0,0) — blocked UP on 1-tall and LEFT on 1-wide? No — (0,0) is bottom-left.
        // On a 5×5 board, (0,0) has walls on DOWN and LEFT (board edges).
        GameState::new_with_config(
            5,
            5,
            HashMap::new(),
            Default::default(),
            cheese,
            Coordinates::new(0, 0), // bottom-left corner
            Coordinates::new(4, 4),
            100,
        )
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
        let game = open_5x5_game(&cheese);
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
        let game = corner_game(&cheese);
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
        let game = open_5x5_game(&cheese);
        let tree = MCTSTree::new(&game);
        assert!((tree.arena()[tree.root()].value_scale() - 5.0).abs() < 1e-6);

        // 0 cheese → value_scale = 1 (clamped)
        let game_no_cheese = open_5x5_game(&[]);
        let tree_no_cheese = MCTSTree::new(&game_no_cheese);
        assert!((tree_no_cheese.arena()[tree_no_cheese.root()].value_scale() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn root_init_mud() {
        // Create game, then put P1 in mud manually via make_move into mud.
        // Simpler: use new_with_config with mud and position P1 so they're in mud.
        //
        // Actually, mud_timer is set when a player *moves into* a mud passage.
        // We can't just place P1 on mud at init — they need to walk into it.
        // Instead, test effective_actions_p1 directly with a game where P1 is stuck.
        use pyrat::game::types::MudMap;

        let mut mud = MudMap::new();
        // Mud between (2,2) and (2,3) with value 3
        mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);

        let game = GameState::new_with_config(
            5,
            5,
            HashMap::new(),
            mud,
            &[Coordinates::new(0, 0)],
            Coordinates::new(2, 2), // P1 at mud start
            Coordinates::new(4, 4),
            100,
        );

        // P1 isn't *in* mud yet (mud_timer=0), just at a position with mud.
        // To actually get stuck, P1 must move UP into the mud passage.
        // Let's do that.
        let mut game = game;
        let _undo = game.make_move(
            pyrat::Direction::Up, // P1 moves into mud (2,2)→(2,3)
            pyrat::Direction::Stay,
        );

        // Now P1 should be stuck in mud.
        assert!(game.player1.mud_timer > 0);

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
        let game1 = open_5x5_game(&cheese);
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
        let game2 = open_5x5_game(&cheese2);
        tree.reinit(&game2);

        // Old nodes gone, single fresh root
        assert_eq!(tree.arena().len(), 1);
        assert!((tree.arena()[tree.root()].value_scale() - 1.0).abs() < 1e-6);
    }

    // ---- advance_root ----

    #[test]
    fn advance_root_found() {
        let cheese = [Coordinates::new(1, 1)];
        let game = open_5x5_game(&cheese);
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
        let game = corner_game(&cheese);
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
        let game = open_5x5_game(&cheese);
        let mut tree = MCTSTree::new(&game);
        let root = tree.root();

        // No children → advance should fail
        assert!(!tree.advance_root(0, 0));
        assert_eq!(tree.root(), root); // Root unchanged
    }
}
