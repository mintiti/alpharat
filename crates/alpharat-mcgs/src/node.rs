use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// HalfEdge — per-outcome stats (8 bytes)
// ---------------------------------------------------------------------------
//
// Copied from alpharat-mcts. Accumulates marginal Q for one player's outcome
// via Welford running average. In MCGS, these live on Edge (per-parent),
// not on LowNode (shared).

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
    /// Welford running-average update: q <- q + (value - q) / visits
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
// compute_outcomes — effective actions -> deduplicated outcome mapping
// ---------------------------------------------------------------------------
//
// Copied from alpharat-mcts. Given effective[a] = outcome action for action a,
// produces sorted unique outcomes and the reverse mapping.

fn compute_outcomes(effective: [u8; 5]) -> ([u8; 5], u8, [u8; 5]) {
    let mut unique = [0u8; 5];
    let mut n = 0u8;

    for &val in &effective {
        let pos = unique[..n as usize].partition_point(|&v| v < val);
        if pos < n as usize && unique[pos] == val {
            continue;
        }
        let mut i = n as usize;
        while i > pos {
            unique[i] = unique[i - 1];
            i -= 1;
        }
        unique[pos] = val;
        n += 1;
    }

    let mut action_to_idx = [0u8; 5];
    for action in 0..5 {
        let outcome = effective[action];
        let idx = unique[..n as usize].partition_point(|&v| v < outcome);
        debug_assert!(idx < n as usize && unique[idx] == outcome);
        action_to_idx[action] = idx as u8;
    }

    (unique, n, action_to_idx)
}

/// Uniform prior over unique effective actions only.
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
// LowNode — shared per-position data (lc0 pattern)
// ---------------------------------------------------------------------------
//
// One LowNode per unique game state in the DAG. Multiple Edges can point to
// the same LowNode (transpositions). Priors and outcome mappings are frozen
// after NN evaluation; aggregate values are updated during backup.

pub struct LowNode {
    // Outcome mappings (frozen after creation)
    p1_prior: [f32; 5],
    p2_prior: [f32; 5],
    p1_outcomes: [u8; 5],
    p2_outcomes: [u8; 5],
    p1_action_to_idx: [u8; 5],
    p2_action_to_idx: [u8; 5],
    n1: u8,
    n2: u8,

    // Aggregate values across all parent edges (Welford running average)
    v1: f32,
    v2: f32,
    total_visits: u32,

    // Children: head of Edge linked list
    first_child: Option<Box<Edge>>,

    // Position metadata
    value_scale: f32,
    is_terminal: bool,
    is_evaluated: bool,

    // Transposition tracking
    num_parents: AtomicU16,
}

impl std::fmt::Debug for LowNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowNode")
            .field("n1", &self.n1)
            .field("n2", &self.n2)
            .field("v1", &self.v1)
            .field("v2", &self.v2)
            .field("total_visits", &self.total_visits)
            .field("is_terminal", &self.is_terminal)
            .field("is_evaluated", &self.is_evaluated)
            .field("num_parents", &self.num_parents.load(Ordering::Relaxed))
            .finish()
    }
}

impl LowNode {
    /// Shell constructor: outcome mappings computed, priors zeroed, unevaluated.
    ///
    /// `effective_p1[a]` and `effective_p2[a]` are the outcome actions for each
    /// raw action a (0..5). Blocked actions map to STAY (4).
    pub fn new_shell(effective_p1: [u8; 5], effective_p2: [u8; 5]) -> Self {
        let (p1_outcomes, n1, p1_action_to_idx) = compute_outcomes(effective_p1);
        let (p2_outcomes, n2, p2_action_to_idx) = compute_outcomes(effective_p2);

        Self {
            p1_prior: [0.0; 5],
            p2_prior: [0.0; 5],
            p1_outcomes,
            p2_outcomes,
            p1_action_to_idx,
            p2_action_to_idx,
            n1,
            n2,
            v1: 0.0,
            v2: 0.0,
            total_visits: 0,
            first_child: None,
            value_scale: 0.0,
            is_terminal: false,
            is_evaluated: false,
            num_parents: AtomicU16::new(0),
        }
    }

    /// Set priors from 5-action NN policies, reducing into outcome-indexed space.
    ///
    /// Scatter-adds: for each action, `prior[action_to_idx[a]] += policy[a]`.
    /// Marks the node as evaluated.
    pub fn set_prior(&mut self, policy_p1: [f32; 5], policy_p2: [f32; 5]) {
        self.p1_prior = [0.0; 5];
        self.p2_prior = [0.0; 5];
        for a in 0..5usize {
            self.p1_prior[self.p1_action_to_idx[a] as usize] += policy_p1[a];
            self.p2_prior[self.p2_action_to_idx[a] as usize] += policy_p2[a];
        }
        self.is_evaluated = true;
    }

    pub fn set_terminal(&mut self) {
        self.is_terminal = true;
    }

    pub fn set_value_scale(&mut self, scale: f32) {
        self.value_scale = scale;
    }

    /// Welford running-average update on aggregate values.
    pub fn update_value(&mut self, q1: f32, q2: f32) {
        self.total_visits += 1;
        let n = self.total_visits as f32;
        self.v1 += (q1 - self.v1) / n;
        self.v2 += (q2 - self.v2) / n;
    }

    // --- Getters ---

    pub fn n1(&self) -> usize {
        self.n1 as usize
    }

    pub fn n2(&self) -> usize {
        self.n2 as usize
    }

    pub fn p1_prior(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.n1());
        self.p1_prior[idx]
    }

    pub fn p2_prior(&self, idx: usize) -> f32 {
        debug_assert!(idx < self.n2());
        self.p2_prior[idx]
    }

    pub fn p1_outcome_action(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.n1());
        self.p1_outcomes[idx]
    }

    pub fn p2_outcome_action(&self, idx: usize) -> u8 {
        debug_assert!(idx < self.n2());
        self.p2_outcomes[idx]
    }

    pub fn p1_action_to_outcome_idx(&self, action: u8) -> u8 {
        debug_assert!((action as usize) < 5);
        self.p1_action_to_idx[action as usize]
    }

    pub fn p2_action_to_outcome_idx(&self, action: u8) -> u8 {
        debug_assert!((action as usize) < 5);
        self.p2_action_to_idx[action as usize]
    }

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

    pub fn is_evaluated(&self) -> bool {
        self.is_evaluated
    }

    // --- Child list ---

    pub fn first_child(&self) -> Option<&Edge> {
        self.first_child.as_deref()
    }

    pub fn first_child_mut(&mut self) -> Option<&mut Edge> {
        self.first_child.as_deref_mut()
    }

    /// Prepend an edge to this node's child list.
    pub fn prepend_child(&mut self, mut edge: Box<Edge>) {
        edge.next_sibling = self.first_child.take();
        self.first_child = Some(edge);
    }

    /// Walk the child list, return the edge at outcome pair (i, j).
    pub fn find_child(&self, i: u8, j: u8) -> Option<&Edge> {
        let mut cur = self.first_child();
        while let Some(edge) = cur {
            if edge.parent_outcome() == (i, j) {
                return Some(edge);
            }
            cur = edge.next_sibling();
        }
        None
    }

    /// Walk the child list, return mutable ref to edge at outcome pair (i, j).
    pub fn find_child_mut(&mut self, i: u8, j: u8) -> Option<&mut Edge> {
        let mut cur = self.first_child_mut();
        while let Some(edge) = cur {
            if edge.parent_outcome() == (i, j) {
                return Some(edge);
            }
            cur = edge.next_sibling_mut();
        }
        None
    }

    // --- Transposition tracking ---

    pub fn add_parent(&self) {
        self.num_parents.fetch_add(1, Ordering::Relaxed);
    }

    pub fn remove_parent(&self) {
        let prev = self.num_parents.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prev > 0, "remove_parent: num_parents is already 0");
    }

    pub fn num_parents(&self) -> u16 {
        self.num_parents.load(Ordering::Relaxed)
    }

    pub fn is_transposition(&self) -> bool {
        self.num_parents() > 1
    }

    // --- Prior expansion (for policy extraction) ---

    /// Expand p1 outcome-indexed priors back to 5-action space.
    pub fn expand_p1_prior(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n1() {
            out[self.p1_outcomes[idx] as usize] = self.p1_prior[idx];
        }
        out
    }

    /// Expand p2 outcome-indexed priors back to 5-action space.
    pub fn expand_p2_prior(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n2() {
            out[self.p2_outcomes[idx] as usize] = self.p2_prior[idx];
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Edge — per-parent link (lc0 Node equivalent)
// ---------------------------------------------------------------------------
//
// One Edge per (parent LowNode, outcome pair) visited. Carries per-parent
// marginal Q stats (HalfEdge arrays) and the transition reward. Points to
// the shared LowNode for the child position via Arc.

pub struct Edge {
    // Per-parent marginal Q stats (outcome-indexed, only n1/n2 valid)
    p1_edges: [HalfEdge; 5],
    p2_edges: [HalfEdge; 5],

    // Per-edge aggregate values (for delta detection during backup)
    v1: f32,
    v2: f32,
    total_visits: u32,
    n_in_flight: u32,

    // Transition reward from parent to child (per-parent, not shared)
    edge_r1: f32,
    edge_r2: f32,

    // Which outcome pair at the parent LowNode this edge represents
    parent_outcome: (u8, u8),

    // Linked list under parent LowNode
    next_sibling: Option<Box<Edge>>,

    // Shared child position
    low_node: Arc<LowNode>,
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edge")
            .field("parent_outcome", &self.parent_outcome)
            .field("v1", &self.v1)
            .field("v2", &self.v2)
            .field("total_visits", &self.total_visits)
            .field("n_in_flight", &self.n_in_flight)
            .field("edge_r1", &self.edge_r1)
            .field("edge_r2", &self.edge_r2)
            .finish()
    }
}

impl Edge {
    /// Create an edge linking a parent to a child position.
    ///
    /// `parent_outcome`: (i, j) outcome indices at the parent.
    /// `r1, r2`: immediate rewards on this transition.
    /// `low_node`: shared position data for the child.
    ///
    /// Automatically increments `low_node.num_parents`.
    pub fn new(
        low_node: Arc<LowNode>,
        parent_outcome: (u8, u8),
        r1: f32,
        r2: f32,
    ) -> Self {
        low_node.add_parent();
        Self {
            p1_edges: [HalfEdge::default(); 5],
            p2_edges: [HalfEdge::default(); 5],
            v1: 0.0,
            v2: 0.0,
            total_visits: 0,
            n_in_flight: 0,
            edge_r1: r1,
            edge_r2: r2,
            parent_outcome,
            next_sibling: None,
            low_node,
        }
    }

    // --- HalfEdge access ---

    pub fn p1_edge(&self, idx: usize) -> &HalfEdge {
        debug_assert!(idx < self.low_node.n1());
        &self.p1_edges[idx]
    }

    pub fn p1_edge_mut(&mut self, idx: usize) -> &mut HalfEdge {
        debug_assert!(idx < self.low_node.n1());
        &mut self.p1_edges[idx]
    }

    pub fn p2_edge(&self, idx: usize) -> &HalfEdge {
        debug_assert!(idx < self.low_node.n2());
        &self.p2_edges[idx]
    }

    pub fn p2_edge_mut(&mut self, idx: usize) -> &mut HalfEdge {
        debug_assert!(idx < self.low_node.n2());
        &mut self.p2_edges[idx]
    }

    // --- Aggregate value ---

    /// Welford running-average update on edge's aggregate values.
    pub fn update_value(&mut self, q1: f32, q2: f32) {
        self.total_visits += 1;
        let n = self.total_visits as f32;
        self.v1 += (q1 - self.v1) / n;
        self.v2 += (q2 - self.v2) / n;
    }

    // --- Virtual loss ---

    /// lc0 pattern. For unvisited nodes: fails if already claimed.
    /// For visited nodes: always succeeds.
    pub fn try_start_score_update(&mut self) -> bool {
        if self.total_visits == 0 && self.n_in_flight > 0 {
            return false;
        }
        self.n_in_flight += 1;
        true
    }

    pub fn cancel_score_update(&mut self) {
        debug_assert!(
            self.n_in_flight > 0,
            "cancel_score_update: n_in_flight is already 0"
        );
        self.n_in_flight -= 1;
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

    pub fn n_in_flight(&self) -> u32 {
        self.n_in_flight
    }

    pub fn r1(&self) -> f32 {
        self.edge_r1
    }

    pub fn r2(&self) -> f32 {
        self.edge_r2
    }

    pub fn parent_outcome(&self) -> (u8, u8) {
        self.parent_outcome
    }

    pub fn low_node(&self) -> &Arc<LowNode> {
        &self.low_node
    }

    // --- Sibling navigation ---

    pub fn next_sibling(&self) -> Option<&Edge> {
        self.next_sibling.as_deref()
    }

    pub fn next_sibling_mut(&mut self) -> Option<&mut Edge> {
        self.next_sibling.as_deref_mut()
    }

    /// Expand p1 edge visits back to 5-action space.
    pub fn expand_p1_visits(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.low_node.n1() {
            let action = self.low_node.p1_outcome_action(idx) as usize;
            out[action] = self.p1_edges[idx].visits as f32;
        }
        out
    }

    /// Expand p2 edge visits back to 5-action space.
    pub fn expand_p2_visits(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.low_node.n2() {
            let action = self.low_node.p2_outcome_action(idx) as usize;
            out[action] = self.p2_edges[idx].visits as f32;
        }
        out
    }
}

impl Drop for Edge {
    fn drop(&mut self) {
        self.low_node.remove_parent();
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

    // ---- LowNode: creation and outcome mapping ----

    #[test]
    fn low_node_shell_open() {
        let low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert_eq!(low.n1(), 5);
        assert_eq!(low.n2(), 5);
        assert!(!low.is_evaluated());
        assert!(!low.is_terminal());
        assert_eq!(low.total_visits(), 0);

        // Priors zero before evaluation
        for i in 0..5 {
            assert_eq!(low.p1_prior(i), 0.0);
            assert_eq!(low.p2_prior(i), 0.0);
        }
    }

    #[test]
    fn low_node_shell_asymmetric() {
        // P1 open, P2 in corner (UP + RIGHT blocked)
        let low = LowNode::new_shell([0, 1, 2, 3, 4], [4, 4, 2, 3, 4]);
        assert_eq!(low.n1(), 5);
        assert_eq!(low.n2(), 3); // outcomes: [2, 3, 4]
    }

    #[test]
    fn low_node_set_prior() {
        let mut low = LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        // P1: UP blocked -> 4 outcomes [1,2,3,4]
        // P2: open -> 5 outcomes
        let p1_policy = [0.1, 0.3, 0.2, 0.15, 0.25];
        let p2_policy = [0.2; 5];
        low.set_prior(p1_policy, p2_policy);

        assert!(low.is_evaluated());

        // P1: action 0 (0.1) + action 4 (0.25) -> STAY outcome = 0.35
        let stay_idx = low.p1_action_to_outcome_idx(4) as usize;
        assert!((low.p1_prior(stay_idx) - 0.35).abs() < 1e-6);

        // P1 prior sums to 1
        let p1_total: f32 = (0..low.n1()).map(|i| low.p1_prior(i)).sum();
        assert!((p1_total - 1.0).abs() < 1e-6);

        // P2 uniform, all 0.2
        for i in 0..low.n2() {
            assert!((low.p2_prior(i) - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn low_node_set_prior_mud() {
        let mut low = LowNode::new_shell([4, 4, 4, 4, 4], [0, 1, 2, 3, 4]);
        low.set_prior([0.1, 0.2, 0.3, 0.15, 0.25], [0.2; 5]);

        assert_eq!(low.n1(), 1);
        assert!((low.p1_prior(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn low_node_outcome_round_trip() {
        // action -> outcome_idx -> canonical action preserves canonical
        let low = LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        // Canonical actions for P1: 1, 2, 3, 4
        for canonical in [1u8, 2, 3, 4] {
            let idx = low.p1_action_to_outcome_idx(canonical);
            let back = low.p1_outcome_action(idx as usize);
            assert_eq!(back, canonical);
        }

        // Blocked action 0 maps to same outcome as 4 (STAY)
        let idx0 = low.p1_action_to_outcome_idx(0);
        let idx4 = low.p1_action_to_outcome_idx(4);
        assert_eq!(idx0, idx4);
    }

    #[test]
    fn low_node_expand_prior() {
        let mut low = LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_prior([0.1, 0.3, 0.2, 0.15, 0.25], [0.2; 5]);

        let expanded = low.expand_p1_prior();
        assert_eq!(expanded[0], 0.0); // blocked
        assert!((expanded[1] - 0.3).abs() < 1e-6);
        assert!((expanded[2] - 0.2).abs() < 1e-6);
        assert!((expanded[3] - 0.15).abs() < 1e-6);
        assert!((expanded[4] - 0.35).abs() < 1e-6); // merged

        let total: f32 = expanded.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    // ---- LowNode: Welford value update ----

    #[test]
    fn low_node_welford() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        low.update_value(2.0, 1.0);
        low.update_value(4.0, 3.0);
        low.update_value(6.0, 5.0);

        assert_eq!(low.total_visits(), 3);
        assert!((low.v1() - 4.0).abs() < 1e-5);
        assert!((low.v2() - 3.0).abs() < 1e-5);
    }

    // ---- LowNode: terminal ----

    #[test]
    fn low_node_terminal() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert!(!low.is_terminal());
        low.set_terminal();
        assert!(low.is_terminal());
    }

    // ---- LowNode: transposition tracking ----

    #[test]
    fn low_node_num_parents() {
        let low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert_eq!(low.num_parents(), 0);
        assert!(!low.is_transposition());

        low.add_parent();
        assert_eq!(low.num_parents(), 1);
        assert!(!low.is_transposition());

        low.add_parent();
        assert_eq!(low.num_parents(), 2);
        assert!(low.is_transposition());

        low.remove_parent();
        assert_eq!(low.num_parents(), 1);
        assert!(!low.is_transposition());
    }

    // ---- Edge: creation and Arc sharing ----

    #[test]
    fn edge_creation_increments_parents() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        assert_eq!(low.num_parents(), 0);

        let edge = Edge::new(Arc::clone(&low), (0, 1), 1.0, 0.5);
        assert_eq!(low.num_parents(), 1);
        assert_eq!(edge.parent_outcome(), (0, 1));
        assert!((edge.r1() - 1.0).abs() < 1e-6);
        assert!((edge.r2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn edge_drop_decrements_parents() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));

        {
            let _edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);
            assert_eq!(low.num_parents(), 1);
        }
        // Edge dropped
        assert_eq!(low.num_parents(), 0);
    }

    #[test]
    fn multiple_edges_share_low_node() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));

        let edge1 = Edge::new(Arc::clone(&low), (0, 0), 1.0, 0.0);
        let edge2 = Edge::new(Arc::clone(&low), (1, 0), 0.0, 1.0);
        assert_eq!(low.num_parents(), 2);
        assert!(low.is_transposition());

        // Both edges see the same LowNode
        assert!(Arc::ptr_eq(edge1.low_node(), edge2.low_node()));

        drop(edge1);
        assert_eq!(low.num_parents(), 1);
        assert!(!low.is_transposition());

        drop(edge2);
        assert_eq!(low.num_parents(), 0);
    }

    #[test]
    fn arc_keeps_low_node_alive() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let weak = Arc::downgrade(&low);

        let edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);
        drop(low); // Original Arc dropped, but edge still holds one

        assert!(weak.upgrade().is_some());
        assert_eq!(edge.low_node().num_parents(), 1);

        drop(edge);
        assert!(weak.upgrade().is_none()); // LowNode freed
    }

    // ---- Edge: Welford update ----

    #[test]
    fn edge_welford() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        edge.update_value(2.0, 1.0);
        edge.update_value(4.0, 3.0);
        edge.update_value(6.0, 5.0);

        assert_eq!(edge.total_visits(), 3);
        assert!((edge.v1() - 4.0).abs() < 1e-5);
        assert!((edge.v2() - 3.0).abs() < 1e-5);
    }

    // ---- Edge: virtual loss ----

    #[test]
    fn edge_try_start_fresh() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        assert!(edge.try_start_score_update());
        assert_eq!(edge.n_in_flight(), 1);
    }

    #[test]
    fn edge_collision_unvisited() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        assert!(edge.try_start_score_update());
        assert!(!edge.try_start_score_update()); // collision
        assert_eq!(edge.n_in_flight(), 1);
    }

    #[test]
    fn edge_visited_always_succeeds() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        edge.update_value(1.0, 1.0);
        assert!(edge.try_start_score_update());
        assert!(edge.try_start_score_update());
        assert!(edge.try_start_score_update());
        assert_eq!(edge.n_in_flight(), 3);
    }

    #[test]
    fn edge_cancel_score_update() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        edge.update_value(1.0, 2.0);
        assert!(edge.try_start_score_update());
        edge.cancel_score_update();
        assert_eq!(edge.n_in_flight(), 0);
    }

    #[test]
    #[should_panic(expected = "cancel_score_update: n_in_flight is already 0")]
    fn edge_cancel_at_zero_panics() {
        let low = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);
        edge.cancel_score_update();
    }

    // ---- Edge: HalfEdge access ----

    #[test]
    fn edge_half_edge_update() {
        let low = Arc::new(LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        // P1 has 4 outcomes. Update outcome 0.
        edge.p1_edge_mut(0).update(5.0);
        edge.p1_edge_mut(0).update(3.0);
        assert_eq!(edge.p1_edge(0).visits, 2);
        assert!((edge.p1_edge(0).q - 4.0).abs() < 1e-5);

        // P2 has 5 outcomes.
        edge.p2_edge_mut(3).update(10.0);
        assert_eq!(edge.p2_edge(3).visits, 1);
        assert!((edge.p2_edge(3).q - 10.0).abs() < 1e-5);
    }

    #[test]
    fn edge_expand_visits() {
        let low = Arc::new(LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);

        // P1 outcomes: [1, 2, 3, 4] (4 outcomes)
        // Give visits to outcome 0 (action 1) and outcome 3 (action 4)
        for _ in 0..10 {
            edge.p1_edge_mut(0).update(1.0);
        }
        for _ in 0..7 {
            edge.p1_edge_mut(3).update(1.0);
        }

        let expanded = edge.expand_p1_visits();
        assert_eq!(expanded[0], 0.0); // blocked
        assert_eq!(expanded[1], 10.0);
        assert_eq!(expanded[2], 0.0);
        assert_eq!(expanded[3], 0.0);
        assert_eq!(expanded[4], 7.0);
    }

    // ---- Child linked list ----

    #[test]
    fn child_list_prepend_and_find() {
        let parent = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let child1 = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let child2 = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));

        // Need mutable access to parent for prepend_child.
        // In real usage, parent wouldn't be behind Arc yet (or we'd use interior mutability).
        // For testing, we build the list manually.
        let mut parent_owned = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        let edge1 = Box::new(Edge::new(Arc::clone(&child1), (0, 1), 1.0, 0.0));
        let edge2 = Box::new(Edge::new(Arc::clone(&child2), (2, 3), 0.0, 1.0));

        parent_owned.prepend_child(edge1);
        parent_owned.prepend_child(edge2);

        // edge2 was prepended last, so it's first
        let first = parent_owned.first_child().unwrap();
        assert_eq!(first.parent_outcome(), (2, 3));

        let second = first.next_sibling().unwrap();
        assert_eq!(second.parent_outcome(), (0, 1));

        assert!(second.next_sibling().is_none());

        // find_child
        assert!(parent_owned.find_child(0, 1).is_some());
        assert_eq!(parent_owned.find_child(0, 1).unwrap().parent_outcome(), (0, 1));

        assert!(parent_owned.find_child(2, 3).is_some());
        assert_eq!(parent_owned.find_child(2, 3).unwrap().parent_outcome(), (2, 3));

        assert!(parent_owned.find_child(4, 4).is_none());

        // Verify Arc sharing: both edges point to distinct child LowNodes
        assert!(!Arc::ptr_eq(
            parent_owned.find_child(0, 1).unwrap().low_node(),
            parent_owned.find_child(2, 3).unwrap().low_node()
        ));

        // Verify parent counts
        assert_eq!(child1.num_parents(), 1);
        assert_eq!(child2.num_parents(), 1);

        // Dropping parent_owned should drop all edges, decrementing parent counts
        drop(parent_owned);
        assert_eq!(child1.num_parents(), 0);
        assert_eq!(child2.num_parents(), 0);

        // Clean up unused Arcs
        drop(parent);
    }

    #[test]
    fn child_list_find_mut() {
        let child = Arc::new(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
        let mut parent = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        let edge = Box::new(Edge::new(Arc::clone(&child), (1, 2), 0.5, 0.5));
        parent.prepend_child(edge);

        // Mutate through find_child_mut
        let found = parent.find_child_mut(1, 2).unwrap();
        found.update_value(3.0, 7.0);
        assert_eq!(found.total_visits(), 1);
        assert!((found.v1() - 3.0).abs() < 1e-6);
    }

    // ---- Edge: HalfEdge virtual loss ----

    #[test]
    fn half_edge_virtual_loss_round_trip() {
        let mut edge = HalfEdge::default();
        edge.update(5.0);

        edge.add_virtual_loss();
        edge.add_virtual_loss();
        assert_eq!(edge.n_in_flight(), 2);
        assert!((edge.q - 5.0).abs() < 1e-6);
        assert_eq!(edge.visits, 1);

        edge.revert_virtual_loss();
        edge.revert_virtual_loss();
        assert_eq!(edge.n_in_flight(), 0);
    }

    #[test]
    #[should_panic(expected = "revert_virtual_loss: n_in_flight is already 0")]
    fn half_edge_revert_at_zero_panics() {
        let mut edge = HalfEdge::default();
        edge.revert_virtual_loss();
    }
}
