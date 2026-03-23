use std::ptr::NonNull;

// ---------------------------------------------------------------------------
// NodePtr — typed heap pointer (replaces NodeIndex)
// ---------------------------------------------------------------------------

/// Non-owning pointer to a heap-allocated Node.
///
/// Ownership lives in `Box<Node>` (via the tree's linked-list structure).
/// NodePtr is a copyable handle for traversal and mutation during search.
///
/// # Safety
///
/// All dereferences are unsafe. Soundness relies on:
/// - Single-threaded search (no data races)
/// - Tree-structured ownership (no aliasing — each node is a distinct allocation)
/// - Pointers only used while the owning Box is alive
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodePtr(NonNull<Node>);

// SAFETY: NodePtr is only dereferenced within single-threaded search.
// Box<Node> is sent to the GC thread, but NodePtr itself is never
// dereferenced across threads.
unsafe impl Send for NodePtr {}

impl NodePtr {
    /// Create a NodePtr from a reference to a Node.
    ///
    /// Typically used on a `&*box_node` or `box_node.as_ref()`.
    /// The pointer is valid as long as the owning Box is alive.
    pub fn from_ref(node: &Node) -> Self {
        Self(NonNull::from(node))
    }

    /// # Safety
    ///
    /// The pointed-to Node must be alive and no mutable reference to it
    /// may exist simultaneously.
    pub unsafe fn as_ref<'a>(self) -> &'a Node {
        self.0.as_ref()
    }

    /// # Safety
    ///
    /// The pointed-to Node must be alive and no other reference to it
    /// (mutable or immutable) may exist simultaneously.
    pub unsafe fn as_mut<'a>(mut self) -> &'a mut Node {
        self.0.as_mut()
    }
}

// ---------------------------------------------------------------------------
// HalfEdge — per-outcome stats (12 bytes)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    pub q: f32,
    pub visits: u32,
    n_in_flight: u32,
}

impl Default for HalfEdge {
    fn default() -> Self {
        Self {
            q: 0.0,
            visits: 0,
            n_in_flight: 0,
        }
    }
}

impl HalfEdge {
    /// Welford running-average update: q ← q + (value - q) / visits
    pub fn update(&mut self, value: f32) {
        self.visits += 1;
        self.q += (value - self.q) / self.visits as f32;
    }

    pub fn n_in_flight(&self) -> u32 {
        self.n_in_flight
    }

    pub fn add_virtual_loss(&mut self) {
        self.n_in_flight += 1;
    }

    pub fn revert_virtual_loss(&mut self) {
        debug_assert!(
            self.n_in_flight > 0,
            "revert_virtual_loss: n_in_flight is already 0"
        );
        self.n_in_flight -= 1;
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
        let mut half = Self::new_shell(effective);
        half.set_prior(prior_5);
        half
    }

    /// Shell constructor: outcomes computed, priors zeroed.
    ///
    /// Used in the three-phase lifecycle: shell exists so other descents
    /// detect the child, priors arrive later from batch NN eval.
    pub fn new_shell(effective: [u8; 5]) -> Self {
        let (outcomes, n_outcomes, action_to_idx) = compute_outcomes(effective);
        Self {
            prior: [0.0f32; 5],
            edges: [HalfEdge::default(); 5],
            outcomes,
            action_to_idx,
            n_outcomes,
        }
    }

    /// Set priors from a 5-action policy, reducing into outcome-indexed space.
    ///
    /// Clears existing priors, then scatter-adds: for each action,
    /// `prior[action_to_idx[a]] += prior_5[a]`.
    pub fn set_prior(&mut self, prior_5: [f32; 5]) {
        self.prior = [0.0f32; 5];
        for action in 0..5u8 {
            let idx = self.action_to_idx[action as usize];
            self.prior[idx as usize] += prior_5[action as usize];
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

    /// Set a single outcome-indexed prior directly.
    pub fn set_prior_at(&mut self, idx: usize, value: f32) {
        debug_assert!(idx < self.n_outcomes());
        self.prior[idx] = value;
    }

    /// Expand outcome-indexed priors back to 5-action space.
    ///
    /// Canonical actions get their outcome's prior; blocked actions get 0.
    pub fn expand_prior(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n_outcomes() {
            out[self.outcomes[idx] as usize] = self.prior[idx];
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
        let idx = unique[..n as usize].partition_point(|&v| v < outcome);
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
    n_in_flight: u32,

    // PUCT normalization
    value_scale: f32,

    // Edge from parent (reward on transition)
    edge_r1: f32,
    edge_r2: f32,

    // Linked-list children (owned)
    pub(crate) first_child: Option<Box<Node>>,
    pub(crate) next_sibling: Option<Box<Node>>,

    // Parent link (non-owning)
    pub(crate) parent: Option<NodePtr>,
    parent_outcome: (u8, u8),

    is_terminal: bool,
}

impl Node {
    pub fn new(p1: HalfNode, p2: HalfNode) -> Self {
        Self {
            p1,
            p2,
            v1: 0.0,
            v2: 0.0,
            total_visits: 0,
            n_in_flight: 0,
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
    pub fn first_child(&self) -> Option<NodePtr> {
        self.first_child.as_ref().map(|b| NodePtr::from_ref(b))
    }
    pub fn next_sibling(&self) -> Option<NodePtr> {
        self.next_sibling.as_ref().map(|b| NodePtr::from_ref(b))
    }
    pub fn parent(&self) -> Option<NodePtr> {
        self.parent
    }
    pub fn parent_outcome(&self) -> (u8, u8) {
        self.parent_outcome
    }
    pub fn n_in_flight(&self) -> u32 {
        self.n_in_flight
    }

    /// lc0 pattern. For unvisited nodes (total_visits == 0): fails if already
    /// claimed (n_in_flight > 0). For visited nodes: always succeeds.
    pub fn try_start_score_update(&mut self) -> bool {
        if self.total_visits == 0 && self.n_in_flight > 0 {
            return false;
        }
        self.n_in_flight += 1;
        true
    }

    /// Revert a score update claim (collision or post-backup).
    pub fn cancel_score_update(&mut self) {
        debug_assert!(
            self.n_in_flight > 0,
            "cancel_score_update: n_in_flight is already 0"
        );
        self.n_in_flight -= 1;
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

    pub fn set_parent(&mut self, ptr: Option<NodePtr>, outcome: (u8, u8)) {
        self.parent = ptr;
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
// Drop — iterative to avoid stack overflow on deep trees
// ---------------------------------------------------------------------------

impl Drop for Node {
    fn drop(&mut self) {
        let mut stack = Vec::new();
        if let Some(c) = self.first_child.take() {
            stack.push(c);
        }
        if let Some(s) = self.next_sibling.take() {
            stack.push(s);
        }
        while let Some(mut node) = stack.pop() {
            if let Some(c) = node.first_child.take() {
                stack.push(c);
            }
            if let Some(s) = node.next_sibling.take() {
                stack.push(s);
            }
            // node drops here with first_child=None, next_sibling=None
            // so its Drop doesn't recurse
        }
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
        let effective = [0, 1, 2, 3, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 5);
        assert_eq!(&outcomes[..5], &[0, 1, 2, 3, 4]);
        for a in 0..5 {
            assert_eq!(a2i[a], a as u8);
        }
    }

    #[test]
    fn outcomes_one_wall() {
        let effective = [4, 1, 2, 3, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 4);
        assert_eq!(&outcomes[..4], &[1, 2, 3, 4]);
        assert_eq!(a2i[0], a2i[4]);
        assert_eq!(outcomes[a2i[0] as usize], 4);
    }

    #[test]
    fn outcomes_corner() {
        let effective = [4, 1, 2, 4, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 3);
        assert_eq!(&outcomes[..3], &[1, 2, 4]);
        assert_eq!(a2i[0], a2i[3]);
        assert_eq!(a2i[0], a2i[4]);
    }

    #[test]
    fn outcomes_mud() {
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
        let effective = [4, 1, 2, 3, 4];
        let half = HalfNode::new(prior_5, effective);

        assert_eq!(half.n_outcomes(), 4);
        let stay_idx = half.action_to_outcome_idx(4);
        assert!((half.prior(stay_idx as usize) - 0.4).abs() < 1e-6);

        let total: f32 = (0..half.n_outcomes()).map(|i| half.prior(i)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn prior_reduction_nonuniform() {
        let prior_5 = [0.1, 0.3, 0.2, 0.15, 0.25];
        let effective = [4, 1, 2, 3, 4];
        let half = HalfNode::new(prior_5, effective);

        assert_eq!(half.n_outcomes(), 4);
        let stay_idx = half.action_to_outcome_idx(4) as usize;
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
        let effective = [4, 1, 2, 3, 4];
        let mut half = HalfNode::new(prior_5, effective);

        half.edges[0].visits = 10;
        half.edges[3].visits = 7;

        let expanded = half.expand_visits();
        assert_eq!(expanded[1], 10.0);
        assert_eq!(expanded[4], 7.0);
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

    // ---- Node value: LC0-style (NN eval = visit 1) ----

    #[test]
    fn node_starts_unvisited() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let node = Node::new(h, h);

        assert_eq!(node.total_visits(), 0);
        assert_eq!(node.v1(), 0.0);
        assert_eq!(node.v2(), 0.0);
    }

    #[test]
    fn nn_eval_counts_as_first_visit() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        node.update_value(3.0, 5.0);
        assert_eq!(node.total_visits(), 1);
        assert!((node.v1() - 3.0).abs() < 1e-6);
        assert!((node.v2() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn backup_averages_with_nn_value() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        node.update_value(5.0, 3.0);
        node.update_value(3.0, 7.0);

        assert_eq!(node.total_visits(), 2);
        assert!((node.v1() - 4.0).abs() < 1e-6);
        assert!((node.v2() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn node_welford_sequence() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        let values = [(2.0, 1.0), (4.0, 3.0), (6.0, 5.0), (8.0, 7.0)];
        for (q1, q2) in &values {
            node.update_value(*q1, *q2);
        }

        assert_eq!(node.total_visits(), 4);
        assert!((node.v1() - 5.0).abs() < 1e-5);
        assert!((node.v2() - 4.0).abs() < 1e-5);
    }

    // ---- HalfEdge: marginal Q accumulator ----

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

    #[test]
    fn edge_marginal_q_across_opponent_actions() {
        let mut edge = HalfEdge::default();
        let gamma = 1.0;

        edge.update(1.0 + gamma * 4.0);
        edge.update(0.5 + gamma * 4.0);
        edge.update(1.0 + gamma * 4.0);

        assert_eq!(edge.visits, 3);
        let expected = (5.0 + 4.5 + 5.0) / 3.0;
        assert!((edge.q - expected).abs() < 1e-5);
    }

    // ---- expand_prior ----

    #[test]
    fn expand_prior_one_wall() {
        let prior_5 = [0.1, 0.3, 0.2, 0.15, 0.25];
        let effective = [4, 1, 2, 3, 4];
        let half = HalfNode::new(prior_5, effective);

        let expanded = half.expand_prior();
        assert_eq!(expanded[0], 0.0);
        assert!((expanded[1] - 0.3).abs() < 1e-6);
        assert!((expanded[2] - 0.2).abs() < 1e-6);
        assert!((expanded[3] - 0.15).abs() < 1e-6);
        assert!((expanded[4] - 0.35).abs() < 1e-6);

        let total: f32 = expanded.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn expand_prior_open_identity() {
        let prior_5 = [0.2; 5];
        let effective = [0, 1, 2, 3, 4];
        let half = HalfNode::new(prior_5, effective);

        let expanded = half.expand_prior();
        for a in 0..5 {
            assert!((expanded[a] - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn expand_prior_mud() {
        let prior_5 = [0.1, 0.2, 0.3, 0.15, 0.25];
        let effective = [4, 4, 4, 4, 4];
        let half = HalfNode::new(prior_5, effective);

        let expanded = half.expand_prior();
        for a in 0..4 {
            assert_eq!(expanded[a], 0.0);
        }
        assert!((expanded[4] - 1.0).abs() < 1e-6);
    }

    // ---- expand_visits: mud ----

    #[test]
    fn expand_visits_mud() {
        let prior_5 = [0.2; 5];
        let effective = [4, 4, 4, 4, 4];
        let mut half = HalfNode::new(prior_5, effective);

        half.edges[0].visits = 42;

        let expanded = half.expand_visits();
        assert_eq!(expanded[4], 42.0);
        for a in 0..4 {
            assert_eq!(expanded[a], 0.0);
        }
    }

    // ---- outcomes: corridor ----

    #[test]
    fn outcomes_corridor() {
        let effective = [4, 1, 4, 3, 4];
        let (outcomes, n, a2i) = compute_outcomes(effective);
        assert_eq!(n, 3);
        assert_eq!(&outcomes[..3], &[1, 3, 4]);
        assert_eq!(a2i[0], a2i[2]);
        assert_eq!(a2i[0], a2i[4]);
    }

    // ---- HalfNode shell lifecycle ----

    #[test]
    fn shell_then_set_prior_open() {
        let effective = [0, 1, 2, 3, 4];
        let shell = HalfNode::new_shell(effective);

        assert_eq!(shell.n_outcomes(), 5);
        for i in 0..5 {
            assert_eq!(shell.prior(i), 0.0);
        }

        let mut half = shell;
        half.set_prior([0.2; 5]);
        for i in 0..5 {
            assert!((half.prior(i) - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn shell_then_set_prior_one_wall() {
        let effective = [4, 1, 2, 3, 4];
        let mut shell = HalfNode::new_shell(effective);
        assert_eq!(shell.n_outcomes(), 4);

        shell.set_prior([0.2; 5]);
        let stay_idx = shell.action_to_outcome_idx(4) as usize;
        assert!((shell.prior(stay_idx) - 0.4).abs() < 1e-6);

        let total: f32 = (0..shell.n_outcomes()).map(|i| shell.prior(i)).sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn shell_set_prior_matches_new() {
        let prior_5 = [0.1, 0.3, 0.2, 0.15, 0.25];
        let effective = [4, 1, 2, 3, 4];

        let from_new = HalfNode::new(prior_5, effective);
        let mut from_shell = HalfNode::new_shell(effective);
        from_shell.set_prior(prior_5);

        assert_eq!(from_new.n_outcomes(), from_shell.n_outcomes());
        for i in 0..from_new.n_outcomes() {
            assert!(
                (from_new.prior(i) - from_shell.prior(i)).abs() < 1e-6,
                "outcome {i}: new={} shell={}",
                from_new.prior(i),
                from_shell.prior(i)
            );
            assert_eq!(from_new.outcome_action(i), from_shell.outcome_action(i));
        }
        for a in 0..5u8 {
            assert_eq!(
                from_new.action_to_outcome_idx(a),
                from_shell.action_to_outcome_idx(a)
            );
        }
    }

    // ---- Linked list wiring ----

    #[test]
    fn linked_list_wiring() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);

        // Create parent first so we can capture its pointer.
        let mut parent = Box::new(Node::new(h, h));
        let parent_ptr = NodePtr::from_ref(&parent);

        // Build children bottom-up: child1 first (it becomes the tail).
        let mut child1 = Box::new(Node::new(h, h));
        let child1_ptr = NodePtr::from_ref(&child1);
        child1.update_value(2.0, 0.0);
        child1.set_parent(Some(parent_ptr), (2, 3));

        let mut child0 = Box::new(Node::new(h, h));
        let child0_ptr = NodePtr::from_ref(&child0);
        child0.update_value(1.0, 0.0);
        child0.set_parent(Some(parent_ptr), (0, 1));
        child0.next_sibling = Some(child1);

        parent.first_child = Some(child0);

        // Walk the list.
        let first = parent.first_child().unwrap();
        assert_eq!(first, child0_ptr);
        assert!((unsafe { first.as_ref() }.v1() - 1.0).abs() < 1e-6);
        assert_eq!(unsafe { first.as_ref() }.parent_outcome(), (0, 1));

        let second = unsafe { first.as_ref() }.next_sibling().unwrap();
        assert_eq!(second, child1_ptr);
        assert!((unsafe { second.as_ref() }.v1() - 2.0).abs() < 1e-6);
        assert_eq!(unsafe { second.as_ref() }.parent_outcome(), (2, 3));

        assert!(unsafe { second.as_ref() }.next_sibling().is_none());
    }

    // ---- Virtual loss: Node ----

    #[test]
    fn try_start_score_update_fresh_node() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        assert!(node.try_start_score_update());
        assert_eq!(node.n_in_flight(), 1);
    }

    #[test]
    fn try_start_score_update_collision_unvisited() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        assert!(node.try_start_score_update());
        assert!(!node.try_start_score_update());
        assert_eq!(node.n_in_flight(), 1);
    }

    #[test]
    fn try_start_score_update_visited_always_succeeds() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        node.update_value(1.0, 1.0);
        assert!(node.try_start_score_update());
        assert!(node.try_start_score_update());
        assert!(node.try_start_score_update());
        assert_eq!(node.n_in_flight(), 3);
    }

    #[test]
    fn try_start_score_update_visited_with_existing_in_flight() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        assert!(node.try_start_score_update());
        node.update_value(1.0, 1.0);

        assert!(node.try_start_score_update());
        assert_eq!(node.n_in_flight(), 2);
    }

    #[test]
    fn cancel_score_update_decrements() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);

        node.update_value(1.0, 2.0);
        assert!(node.try_start_score_update());
        assert_eq!(node.n_in_flight(), 1);

        node.cancel_score_update();
        assert_eq!(node.n_in_flight(), 0);
        assert!((node.v1() - 1.0).abs() < 1e-6);
        assert!((node.v2() - 2.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "cancel_score_update: n_in_flight is already 0")]
    fn cancel_score_update_at_zero_panics() {
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut node = Node::new(h, h);
        node.cancel_score_update();
    }

    // ---- Virtual loss: HalfEdge ----

    #[test]
    fn half_edge_virtual_loss_round_trip() {
        let mut edge = HalfEdge::default();
        edge.update(5.0);
        assert_eq!(edge.visits, 1);

        edge.add_virtual_loss();
        edge.add_virtual_loss();
        assert_eq!(edge.n_in_flight(), 2);
        assert!((edge.q - 5.0).abs() < 1e-6);
        assert_eq!(edge.visits, 1);

        edge.revert_virtual_loss();
        edge.revert_virtual_loss();
        assert_eq!(edge.n_in_flight(), 0);
        assert!((edge.q - 5.0).abs() < 1e-6);
        assert_eq!(edge.visits, 1);
    }

    #[test]
    #[should_panic(expected = "revert_virtual_loss: n_in_flight is already 0")]
    fn half_edge_revert_at_zero_panics() {
        let mut edge = HalfEdge::default();
        edge.revert_virtual_loss();
    }

    // ---- Custom Drop: iterative ----

    #[test]
    fn drop_deep_tree_no_stack_overflow() {
        // Build a chain of 10_000 nodes as first_child links.
        // Without iterative drop, this would overflow the call stack.
        let h = HalfNode::new([0.2; 5], [0, 1, 2, 3, 4]);
        let mut chain: Option<Box<Node>> = None;
        for _ in 0..10_000 {
            let mut node = Box::new(Node::new(h, h));
            node.first_child = chain.take();
            chain = Some(node);
        }
        // Dropping `chain` here exercises the iterative Drop.
        drop(chain);
    }
}
