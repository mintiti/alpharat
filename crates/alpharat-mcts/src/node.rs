use std::ops::{Index, IndexMut};

// ---------------------------------------------------------------------------
// NodeIndex — typed arena index
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeIndex(u32);

impl NodeIndex {
    fn as_usize(self) -> usize {
        self.0 as usize
    }
}

// ---------------------------------------------------------------------------
// HalfEdge — per-outcome stats (8 bytes)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    pub q: f32,
    pub visits: u32,
}

impl Default for HalfEdge {
    fn default() -> Self {
        Self { q: 0.0, visits: 0 }
    }
}

impl HalfEdge {
    /// Welford running-average update: q ← q + (value - q) / visits
    pub fn update(&mut self, value: f32) {
        self.visits += 1;
        self.q += (value - self.q) / self.visits as f32;
    }
}

// ---------------------------------------------------------------------------
// HalfNode — per-player data
// ---------------------------------------------------------------------------

/// Per-player MCTS statistics. Stores priors and edges in outcome-indexed
/// space (deduplicated actions). Padded arrays of length 5; only the first
/// `n_outcomes` entries are valid.
#[derive(Clone, Copy, Debug)]
pub struct HalfNode {
    prior: [f32; 5],
    edges: [HalfEdge; 5],
    outcomes: [u8; 5],
    action_to_idx: [u8; 5],
    n_outcomes: u8,
}

impl HalfNode {
    /// Build a HalfNode from a 5-action prior and the effective-action map.
    ///
    /// `prior_5[a]` is the raw prior for action `a` (0..5).
    /// `effective[a]` is the outcome action for action `a` — e.g. if UP is
    /// blocked, `effective[0] = 4` (maps to STAY).
    ///
    /// The prior is reduced by summing probabilities of actions that share
    /// the same outcome.
    pub fn new(prior_5: [f32; 5], effective: [u8; 5]) -> Self {
        let (outcomes, n_outcomes, action_to_idx) = compute_outcomes(effective);

        // Reduce prior: sum probabilities of equivalent actions.
        let mut prior = [0.0f32; 5];
        for action in 0..5u8 {
            let idx = action_to_idx[action as usize];
            prior[idx as usize] += prior_5[action as usize];
        }

        Self {
            prior,
            edges: [HalfEdge::default(); 5],
            outcomes,
            action_to_idx,
            n_outcomes,
        }
    }

    pub fn n_outcomes(&self) -> usize {
        self.n_outcomes as usize
    }

    pub fn prior(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.n_outcomes());
        self.prior[idx]
    }

    pub fn edge(&self, idx: usize) -> &HalfEdge {
        debug_assert!(idx < self.n_outcomes());
        &self.edges[idx]
    }

    pub fn edge_mut(&mut self, idx: usize) -> &mut HalfEdge {
        debug_assert!(idx < self.n_outcomes());
        &mut self.edges[idx]
    }

    /// Outcome index → the canonical action for that outcome.
    pub fn outcome_action(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.n_outcomes());
        self.outcomes[idx]
    }

    /// Action (0..5) → outcome index in this HalfNode.
    pub fn action_to_outcome_idx(&self, action: u8) -> u8 {
        debug_assert!((action as usize) < 5);
        self.action_to_idx[action as usize]
    }

    /// Expand edge visits back to 5-action space.
    ///
    /// Each canonical action gets its outcome's visit count as f32.
    /// Non-canonical actions (blocked duplicates) get 0.
    pub fn expand_visits(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n_outcomes() {
            let action = self.outcomes[idx] as usize;
            out[action] = self.edges[idx].visits as f32;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// compute_outcomes — effective actions → deduplicated outcome mapping
// ---------------------------------------------------------------------------

/// Given the effective-action array (action → outcome action), compute:
/// - `outcomes`: sorted unique outcome actions (padded with 0)
/// - `n_outcomes`: how many unique outcomes
/// - `action_to_idx`: for each action 0..5, the index into `outcomes`
fn compute_outcomes(effective: [u8; 5]) -> ([u8; 5], u8, [u8; 5]) {
    // Collect unique values via a small sorted buffer.
    let mut unique = [0u8; 5];
    let mut n = 0u8;

    for &val in &effective {
        // Insert into sorted position if not already present.
        let pos = unique[..n as usize].partition_point(|&v| v < val);
        if pos < n as usize && unique[pos] == val {
            continue; // already present
        }
        // Shift right to make room.
        let mut i = n as usize;
        while i > pos {
            unique[i] = unique[i - 1];
            i -= 1;
        }
        unique[pos] = val;
        n += 1;
    }

    // Build reverse map.
    let mut action_to_idx = [0u8; 5];
    for action in 0..5 {
        let outcome = effective[action];
        // Binary search in the sorted unique array.
        let idx = unique[..n as usize]
            .partition_point(|&v| v < outcome);
        debug_assert!(idx < n as usize && unique[idx] == outcome);
        action_to_idx[action] = idx as u8;
    }

    (unique, n, action_to_idx)
}

// ---------------------------------------------------------------------------
// Node — full MCTS node
// ---------------------------------------------------------------------------

pub struct Node {
    pub p1: HalfNode,
    pub p2: HalfNode,

    // Value estimates (Welford running averages)
    v1: f32,
    v2: f32,
    total_visits: u32,

    // PUCT normalization
    value_scale: f32,

    // Edge from parent (reward on transition)
    edge_r1: f32,
    edge_r2: f32,

    // Linked-list children
    first_child: Option<NodeIndex>,
    next_sibling: Option<NodeIndex>,

    // Parent link
    parent: Option<NodeIndex>,
    parent_outcome: (u8, u8),

    is_terminal: bool,
}

impl Node {
    pub fn new(p1: HalfNode, p2: HalfNode, v1: f32, v2: f32) -> Self {
        Self {
            p1,
            p2,
            v1,
            v2,
            total_visits: 0,
            value_scale: 0.0,
            edge_r1: 0.0,
            edge_r2: 0.0,
            first_child: None,
            next_sibling: None,
            parent: None,
            parent_outcome: (0, 0),
            is_terminal: false,
        }
    }

    // --- Getters ---

    pub fn v1(&self) -> f32 {
        self.v1
    }
    pub fn v2(&self) -> f32 {
        self.v2
    }
    pub fn total_visits(&self) -> u32 {
        self.total_visits
    }
    pub fn value_scale(&self) -> f32 {
        self.value_scale
    }
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }
    pub fn edge_r1(&self) -> f32 {
        self.edge_r1
    }
    pub fn edge_r2(&self) -> f32 {
        self.edge_r2
    }
    pub fn first_child(&self) -> Option<NodeIndex> {
        self.first_child
    }
    pub fn next_sibling(&self) -> Option<NodeIndex> {
        self.next_sibling
    }
    pub fn parent(&self) -> Option<NodeIndex> {
        self.parent
    }
    pub fn parent_outcome(&self) -> (u8, u8) {
        self.parent_outcome
    }

    // --- Setters ---

    pub fn set_value_scale(&mut self, scale: f32) {
        self.value_scale = scale;
    }

    pub fn set_terminal(&mut self) {
        self.is_terminal = true;
    }

    pub fn set_edge_rewards(&mut self, r1: f32, r2: f32) {
        self.edge_r1 = r1;
        self.edge_r2 = r2;
    }

    pub fn set_first_child(&mut self, idx: Option<NodeIndex>) {
        self.first_child = idx;
    }

    pub fn set_next_sibling(&mut self, idx: Option<NodeIndex>) {
        self.next_sibling = idx;
    }

    pub fn set_parent(&mut self, idx: Option<NodeIndex>, outcome: (u8, u8)) {
        self.parent = idx;
        self.parent_outcome = outcome;
    }

    // --- Value update ---

    /// Welford running-average update on v1/v2. Increments total_visits.
    pub fn update_value(&mut self, q1: f32, q2: f32) {
        self.total_visits += 1;
        let n = self.total_visits as f32;
        self.v1 += (q1 - self.v1) / n;
        self.v2 += (q2 - self.v2) / n;
    }
}

// ---------------------------------------------------------------------------
// NodeArena — arena allocator
// ---------------------------------------------------------------------------

pub struct NodeArena {
    nodes: Vec<Node>,
}

impl NodeArena {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
        }
    }

    pub fn alloc(&mut self, node: Node) -> NodeIndex {
        let idx = NodeIndex(self.nodes.len() as u32);
        self.nodes.push(node);
        idx
    }

    pub fn get(&self, idx: NodeIndex) -> &Node {
        &self.nodes[idx.as_usize()]
    }

    pub fn get_mut(&mut self, idx: NodeIndex) -> &mut Node {
        &mut self.nodes[idx.as_usize()]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for NodeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<NodeIndex> for NodeArena {
    type Output = Node;
    fn index(&self, idx: NodeIndex) -> &Self::Output {
        self.get(idx)
    }
}

impl IndexMut<NodeIndex> for NodeArena {
    fn index_mut(&mut self, idx: NodeIndex) -> &mut Self::Output {
        self.get_mut(idx)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- compute_outcomes ----

    #[test]
    fn outcomes_open_position() {
        // No walls: every action maps to itself.
        let effective = [0, 1, 2, 3, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 5);
        assert_eq!(&outcomes[..5], &[0, 1, 2, 3, 4]);
        // Identity mapping.
        for a in 0..5 {
            assert_eq!(a2i[a], a as u8);
        }
    }

    #[test]
    fn outcomes_one_wall() {
        // UP blocked → maps to STAY.
        let effective = [4, 1, 2, 3, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 4);
        assert_eq!(&outcomes[..4], &[1, 2, 3, 4]);
        // action 0 and action 4 share the same outcome index.
        assert_eq!(a2i[0], a2i[4]);
        // That index points to outcome 4 (STAY).
        assert_eq!(outcomes[a2i[0] as usize], 4);
    }

    #[test]
    fn outcomes_corner() {
        // UP and LEFT blocked → both map to STAY.
        let effective = [4, 1, 2, 4, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 3);
        assert_eq!(&outcomes[..3], &[1, 2, 4]);
        assert_eq!(a2i[0], a2i[3]);
        assert_eq!(a2i[0], a2i[4]);
    }

    #[test]
    fn outcomes_mud() {
        // All actions → STAY.
        let effective = [4, 4, 4, 4, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 1);
        assert_eq!(outcomes[0], 4);
        for a in 0..5 {
            assert_eq!(a2i[a], 0);
        }
    }

    // ---- HalfNode prior reduction ----

    #[test]
    fn prior_reduction_uniform_one_wall() {
        let prior_5 = [0.2; 5];
        let effective = [4, 1, 2, 3, 4]; // UP blocked
        let half = HalfNode::new(prior_5, effective);

        assert_eq!(half.n_outcomes(), 4);
        // STAY outcome should have merged prior: 0.2 (action 0) + 0.2 (action 4) = 0.4
        let stay_idx = half.action_to_outcome_idx(4);
        assert!((half.prior(stay_idx as usize) - 0.4).abs() < 1e-6);

        // Total prior should sum to 1.0.
        let total: f32 = (0..half.n_outcomes()).map(|i| half.prior(i)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn prior_reduction_nonuniform() {
        let prior_5 = [0.1, 0.3, 0.2, 0.15, 0.25];
        let effective = [4, 1, 2, 3, 4]; // UP blocked
        let half = HalfNode::new(prior_5, effective);

        assert_eq!(half.n_outcomes(), 4);
        let stay_idx = half.action_to_outcome_idx(4) as usize;
        // action 0 (0.1) + action 4 (0.25) = 0.35
        assert!((half.prior(stay_idx) - 0.35).abs() < 1e-6);

        let total: f32 = (0..half.n_outcomes()).map(|i| half.prior(i)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn prior_reduction_mud() {
        let prior_5 = [0.1, 0.2, 0.3, 0.15, 0.25];
        let effective = [4, 4, 4, 4, 4];
        let half = HalfNode::new(prior_5, effective);

        assert_eq!(half.n_outcomes(), 1);
        assert!((half.prior(0) - 1.0).abs() < 1e-6);
    }

    // ---- expand_visits ----

    #[test]
    fn expand_visits_basic() {
        let prior_5 = [0.2; 5];
        let effective = [4, 1, 2, 3, 4]; // UP blocked → 4 outcomes: [1,2,3,4]
        let mut half = HalfNode::new(prior_5, effective);

        // Give some visits to outcome index 0 (action 1) and index 3 (action 4/STAY).
        half.edges[0].visits = 10;
        half.edges[3].visits = 7;

        let expanded = half.expand_visits();
        // outcome 0 → action 1
        assert_eq!(expanded[1], 10.0);
        // outcome 3 → action 4 (STAY)
        assert_eq!(expanded[4], 7.0);
        // Blocked action 0 gets nothing (it's not a canonical outcome).
        assert_eq!(expanded[0], 0.0);
    }

    #[test]
    fn expand_visits_open() {
        let prior_5 = [0.2; 5];
        let effective = [0, 1, 2, 3, 4];
        let mut half = HalfNode::new(prior_5, effective);

        for i in 0..5 {
            half.edges[i].visits = (i as u32 + 1) * 5;
        }

        let expanded = half.expand_visits();
        for a in 0..5 {
            assert_eq!(expanded[a], ((a as u32 + 1) * 5) as f32);
        }
    }

    // ---- Node Welford update ----

    #[test]
    fn node_welford_single() {
        let h1 = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let h2 = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h1, h2, 0.0, 0.0);

        node.update_value(3.0, 5.0);
        assert_eq!(node.total_visits(), 1);
        assert!((node.v1() - 3.0).abs() < 1e-6);
        assert!((node.v2() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn node_welford_sequence() {
        let h1 = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let h2 = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h1, h2, 0.0, 0.0);

        let values = [(2.0, 1.0), (4.0, 3.0), (6.0, 5.0), (8.0, 7.0)];
        for (q1, q2) in &values {
            node.update_value(*q1, *q2);
        }

        assert_eq!(node.total_visits(), 4);
        // Mean of [2,4,6,8] = 5.0
        assert!((node.v1() - 5.0).abs() < 1e-5);
        // Mean of [1,3,5,7] = 4.0
        assert!((node.v2() - 4.0).abs() < 1e-5);
    }

    // ---- HalfEdge Welford ----

    #[test]
    fn edge_welford_sequence() {
        let mut edge = HalfEdge::default();
        let values = [10.0, 20.0, 30.0];
        for v in &values {
            edge.update(*v);
        }
        assert_eq!(edge.visits, 3);
        assert!((edge.q - 20.0).abs() < 1e-5);
    }

    // ---- NodeArena ----

    #[test]
    fn arena_alloc_and_get() {
        let mut arena = NodeArena::new();
        assert!(arena.is_empty());

        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let idx0 = arena.alloc(Node::new(h, h, 1.0, 2.0));
        let idx1 = arena.alloc(Node::new(h, h, 3.0, 4.0));

        assert_eq!(arena.len(), 2);
        assert!((arena[idx0].v1() - 1.0).abs() < 1e-6);
        assert!((arena[idx1].v1() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn arena_index_mut() {
        let mut arena = NodeArena::new();
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let idx = arena.alloc(Node::new(h, h, 0.0, 0.0));

        arena[idx].update_value(5.0, 10.0);
        assert!((arena[idx].v1() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn arena_with_capacity() {
        let arena = NodeArena::with_capacity(1000);
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
    }

    // ---- Linked list wiring ----

    #[test]
    fn linked_list_wiring() {
        let mut arena = NodeArena::new();
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        let parent_idx = arena.alloc(Node::new(h, h, 0.0, 0.0));
        let child0_idx = arena.alloc(Node::new(h, h, 1.0, 0.0));
        let child1_idx = arena.alloc(Node::new(h, h, 2.0, 0.0));

        // Wire: parent → child0 → child1
        arena[parent_idx].set_first_child(Some(child0_idx));
        arena[child0_idx].set_next_sibling(Some(child1_idx));
        arena[child0_idx].set_parent(Some(parent_idx), (0, 1));
        arena[child1_idx].set_parent(Some(parent_idx), (2, 3));

        // Walk the list.
        let first = arena[parent_idx].first_child().unwrap();
        assert_eq!(first, child0_idx);
        assert!((arena[first].v1() - 1.0).abs() < 1e-6);
        assert_eq!(arena[first].parent_outcome(), (0, 1));

        let second = arena[first].next_sibling().unwrap();
        assert_eq!(second, child1_idx);
        assert!((arena[second].v1() - 2.0).abs() < 1e-6);
        assert_eq!(arena[second].parent_outcome(), (2, 3));

        assert!(arena[second].next_sibling().is_none());
    }
}
