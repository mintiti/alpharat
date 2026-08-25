use alpharat_eval_core::compute_outcomes;
use std::cell::UnsafeCell;
#[cfg(not(loom))]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::Arc;

#[cfg(loom)]
use loom::sync::atomic::AtomicU32;

// ---------------------------------------------------------------------------
// HalfEdge — per-outcome stats (12 bytes)
// ---------------------------------------------------------------------------
//
// Copied from alpharat-mcts. Welford running-average Q for one player's
// outcome. Used for virtual loss tracking and future search loop needs.

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
// ReservationState — the only concurrently mutable LowNode state
// ---------------------------------------------------------------------------

/// Node claims and joint virtual loss.
///
/// The state stays inside `LowNode` so exclusive access can reach each atomic
/// through `get_mut()` without an atomic read-modify-write. A future shared
/// gather epoch may use only the relaxed operations below; it does not publish
/// or synchronize any of the plain node payload.
#[repr(C)]
struct ReservationState {
    joint: [[AtomicU32; 5]; 5],
    node: AtomicU32,
}

impl ReservationState {
    fn new() -> Self {
        Self {
            joint: std::array::from_fn(|_| {
                std::array::from_fn(|_| AtomicU32::new(0))
            }),
            node: AtomicU32::new(0),
        }
    }

    #[inline]
    fn with_exclusive<R>(counter: &mut AtomicU32, update: impl FnOnce(&mut u32) -> R) -> R {
        #[cfg(not(loom))]
        {
            update(counter.get_mut())
        }
        #[cfg(loom)]
        {
            counter.with_mut(update)
        }
    }

    #[inline]
    fn node_exclusive(&mut self) -> u32 {
        Self::with_exclusive(&mut self.node, |value| *value)
    }

    #[inline]
    fn joint_exclusive(&mut self, i: usize, j: usize) -> u32 {
        Self::with_exclusive(&mut self.joint[i][j], |value| *value)
    }

    #[inline]
    fn reserve_node_exclusive(&mut self, count: u32, operation: &str) {
        Self::with_exclusive(&mut self.node, |value| {
            *value = value
                .checked_add(count)
                .unwrap_or_else(|| panic!("{operation}: n_in_flight overflow"));
        });
    }

    #[inline]
    fn release_node_exclusive(&mut self, count: u32, operation: &str) {
        Self::with_exclusive(&mut self.node, |value| {
            *value = value.checked_sub(count).unwrap_or_else(|| {
                if *value == 0 && count == 1 {
                    panic!("{operation}: n_in_flight is already 0");
                }
                panic!(
                    "{operation}: n_in_flight {} < count {count}",
                    *value
                )
            });
        });
    }

    #[inline]
    fn reserve_joint_exclusive(
        &mut self,
        i: usize,
        j: usize,
        count: u32,
        operation: &str,
    ) {
        Self::with_exclusive(&mut self.joint[i][j], |value| {
            *value = value.checked_add(count).unwrap_or_else(|| {
                panic!("{operation}: edge_in_flight[{i}][{j}] overflow")
            });
        });
    }

    #[inline]
    fn release_joint_exclusive(
        &mut self,
        i: usize,
        j: usize,
        count: u32,
        operation: &str,
    ) {
        Self::with_exclusive(&mut self.joint[i][j], |value| {
            *value = value.checked_sub(count).unwrap_or_else(|| {
                if *value == 0 && count == 1 {
                    panic!(
                        "{operation}: edge_in_flight[{i}][{j}] is already 0"
                    );
                }
                panic!(
                    "{operation}: edge_in_flight[{i}][{j}] {} < count {count}",
                    *value
                )
            });
        });
    }

    #[inline]
    fn node_shared(&self) -> u32 {
        self.node.load(Ordering::Relaxed)
    }

    #[inline]
    fn joint_shared(&self, i: usize, j: usize) -> u32 {
        self.joint[i][j].load(Ordering::Relaxed)
    }

    #[inline]
    fn try_claim_fresh_shared(&self) -> bool {
        self.node
            .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }

    #[inline]
    fn reserve_node_shared(&self, count: u32, operation: &str) {
        self.node
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(count)
            })
            .unwrap_or_else(|_| panic!("{operation}: n_in_flight overflow"));
    }

    #[inline]
    fn release_node_shared(&self, count: u32, operation: &str) {
        self.node
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(count)
            })
            .unwrap_or_else(|current| {
                panic!(
                    "{operation}: n_in_flight {current} < count {count}"
                )
            });
    }

    #[inline]
    fn reserve_joint_shared(
        &self,
        i: usize,
        j: usize,
        count: u32,
        operation: &str,
    ) {
        self.joint[i][j]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(count)
            })
            .unwrap_or_else(|_| {
                panic!("{operation}: edge_in_flight[{i}][{j}] overflow")
            });
    }

    #[inline]
    fn release_joint_shared(
        &self,
        i: usize,
        j: usize,
        count: u32,
        operation: &str,
    ) {
        self.joint[i][j]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(count)
            })
            .unwrap_or_else(|current| {
                panic!(
                    "{operation}: edge_in_flight[{i}][{j}] {current} < count {count}"
                )
            });
    }
}

// ---------------------------------------------------------------------------
// LowNode — shared per-position data (lc0 pattern)
// ---------------------------------------------------------------------------
//
// One LowNode per unique game state in the DAG. Multiple Edges can point to
// the same LowNode (transpositions). Priors and outcome mappings are frozen
// after NN evaluation; aggregate values are updated during backup.

// `repr(C)` pins the pre-atomic hot-field layout: joint reservations stay next
// to joint visits, while the child pointer and selection data retain their
// established offsets. This is layout control for performance, not an FFI API.
#[repr(C)]
pub struct LowNode {
    // Children: head of Edge linked list
    first_child: Option<Box<Edge>>,

    // Priors (frozen after evaluation)
    p1_prior: [f32; 5],
    p2_prior: [f32; 5],

    // Aggregate values across all parent edges (Welford running average)
    v1: f32,
    v2: f32,
    total_visits: u32,

    // Per-(i,j) Welford Q for computing marginal Q during selection.
    // Only [0..n1][0..n2] entries are valid.
    edge_q_p1: [[f32; 5]; 5],
    edge_q_p2: [[f32; 5]; 5],
    edge_visits: [[u32; 5]; 5],

    // The only concurrently mutable payload: node claims and joint virtual loss.
    reservations: ReservationState,

    // Position metadata
    value_scale: f32,

    // Outcome mappings (frozen after creation)
    p1_outcomes: [u8; 5],
    p2_outcomes: [u8; 5],
    p1_action_to_idx: [u8; 5],
    p2_action_to_idx: [u8; 5],
    n1: u8,
    n2: u8,
    is_terminal: bool,
    is_evaluated: bool,
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
            edge_q_p1: [[0.0; 5]; 5],
            edge_q_p2: [[0.0; 5]; 5],
            edge_visits: [[0; 5]; 5],
            reservations: ReservationState::new(),
            first_child: None,
            value_scale: 0.0,
            is_terminal: false,
            is_evaluated: false,
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

    /// LC0's FinalizeScoreUpdate: Welford running-average on node aggregates.
    /// Increments total_visits, then adjusts v1/v2 toward the new value.
    pub fn finalize_score_update(&mut self, q1: f32, q2: f32) {
        self.total_visits += 1;
        let n = self.total_visits as f32;
        self.v1 += (q1 - self.v1) / n;
        self.v2 += (q2 - self.v2) / n;
    }

    /// LC0's AdjustForTerminal on node-level aggregates.
    /// Retroactively adjust n_to_fix old visits by delta.
    pub fn adjust_for_terminal(&mut self, v1_delta: f32, v2_delta: f32, n_to_fix: u32) {
        let n = self.total_visits as f32;
        self.v1 += n_to_fix as f32 * v1_delta / n;
        self.v2 += n_to_fix as f32 * v2_delta / n;
    }

    /// Multivisit Welford on node aggregate. Equivalent to calling
    /// `finalize_score_update` `count` times with the same value, but O(1).
    /// Also decrements n_in_flight (LC0's combined FinalizeScoreUpdate).
    pub fn finalize_score_update_multi(&mut self, q1: f32, q2: f32, count: u32) {
        self.total_visits += count;
        let n = self.total_visits as f32;
        let w = count as f32;
        self.v1 += (q1 - self.v1) * w / n;
        self.v2 += (q2 - self.v2) * w / n;
        self.reservations
            .release_node_exclusive(count, "finalize_score_update_multi");
    }

    /// Increment n_in_flight by count. For path propagation during gather.
    pub fn increment_n_in_flight(&mut self, count: u32) {
        self.reservations
            .reserve_node_exclusive(count, "increment_n_in_flight");
    }

    /// Cancel score update by count. Decrements n_in_flight without touching visits.
    pub fn cancel_score_update_multi(&mut self, count: u32) {
        self.reservations
            .release_node_exclusive(count, "cancel_score_update_multi");
    }

    // --- Per-(i,j) edge matrix ---

    /// LC0's FinalizeScoreUpdate on a joint matrix cell.
    /// Increments edge_visits[i][j], then adjusts edge_q toward the new value.
    pub fn finalize_edge_update(&mut self, i: usize, j: usize, q1: f32, q2: f32) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.edge_visits[i][j] += 1;
        let n = self.edge_visits[i][j] as f32;
        self.edge_q_p1[i][j] += (q1 - self.edge_q_p1[i][j]) / n;
        self.edge_q_p2[i][j] += (q2 - self.edge_q_p2[i][j]) / n;
    }

    /// Multivisit Welford on a joint matrix cell.
    pub fn finalize_edge_update_multi(&mut self, i: usize, j: usize, q1: f32, q2: f32, count: u32) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.edge_visits[i][j] += count;
        let n = self.edge_visits[i][j] as f32;
        let w = count as f32;
        self.edge_q_p1[i][j] += (q1 - self.edge_q_p1[i][j]) * w / n;
        self.edge_q_p2[i][j] += (q2 - self.edge_q_p2[i][j]) * w / n;
    }

    /// LC0's AdjustForTerminal on a joint matrix cell.
    /// Retroactively adjust n_to_fix old visits by delta.
    pub fn adjust_edge_for_terminal(&mut self, i: usize, j: usize, q1_delta: f32, q2_delta: f32, n_to_fix: u32) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        let n = self.edge_visits[i][j] as f32;
        self.edge_q_p1[i][j] += n_to_fix as f32 * q1_delta / n;
        self.edge_q_p2[i][j] += n_to_fix as f32 * q2_delta / n;
    }

    // --- Virtual loss on (i,j) matrix ---

    pub fn add_virtual_loss(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations
            .reserve_joint_exclusive(i, j, 1, "add_virtual_loss");
    }

    pub fn add_virtual_loss_multi(&mut self, i: usize, j: usize, count: u32) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations
            .reserve_joint_exclusive(i, j, count, "add_virtual_loss_multi");
    }

    pub fn revert_virtual_loss(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations
            .release_joint_exclusive(i, j, 1, "revert_virtual_loss");
    }

    pub fn revert_virtual_loss_multi(&mut self, i: usize, j: usize, count: u32) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations
            .release_joint_exclusive(i, j, count, "revert_virtual_loss_multi");
    }

    pub fn edge_in_flight(&self, i: usize, j: usize) -> u32 {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations.joint_shared(i, j)
    }

    /// Sum edge_in_flight[i][j] over all j (marginal in-flight for p1 outcome i).
    pub fn marginal_in_flight_p1(&self, i: usize) -> u32 {
        debug_assert!(i < self.n1());
        let mut sum = 0u32;
        for j in 0..self.n2() {
            sum += self.reservations.joint_shared(i, j);
        }
        sum
    }

    /// Sum edge_in_flight[i][j] over all i (marginal in-flight for p2 outcome j).
    pub fn marginal_in_flight_p2(&self, j: usize) -> u32 {
        debug_assert!(j < self.n2());
        let mut sum = 0u32;
        for i in 0..self.n1() {
            sum += self.reservations.joint_shared(i, j);
        }
        sum
    }

    /// Marginal n_started for p1 outcome i: visits + in_flight, summed over j.
    /// LC0's GetNStarted() equivalent, marginalized for decoupled PUCT.
    pub fn marginal_n_started_p1(&self, i: usize) -> u32 {
        self.marginal_visits_p1(i) + self.marginal_in_flight_p1(i)
    }

    /// Marginal n_started for p2 outcome j: visits + in_flight, summed over i.
    pub fn marginal_n_started_p2(&self, j: usize) -> u32 {
        self.marginal_visits_p2(j) + self.marginal_in_flight_p2(j)
    }

    /// Snapshot marginal started counts through the exclusive fast path.
    ///
    /// The sequential gather already owns `&mut LowNode`, so every reservation
    /// read uses `AtomicU32::get_mut()` rather than an atomic load.
    pub fn marginal_n_started_exclusive(&mut self) -> ([u32; 5], [u32; 5]) {
        let mut p1 = [0u32; 5];
        let mut p2 = [0u32; 5];
        for (i, started) in p1.iter_mut().enumerate().take(self.n1()) {
            let mut in_flight = 0;
            for j in 0..self.n2() {
                in_flight += self.reservations.joint_exclusive(i, j);
            }
            *started = self.marginal_visits_p1(i) + in_flight;
        }
        for (j, started) in p2.iter_mut().enumerate().take(self.n2()) {
            let mut in_flight = 0;
            for i in 0..self.n1() {
                in_flight += self.reservations.joint_exclusive(i, j);
            }
            *started = self.marginal_visits_p2(j) + in_flight;
        }
        (p1, p2)
    }

    // --- Shared gather reservation operations ---
    //
    // These methods are deliberately payload-level primitives, not an
    // unguarded tree capability. Chunk 4's ReadAccess will establish the read
    // epoch that makes obtaining `&LowNode` safe while several gatherers use
    // only this atomic state.

    /// Claim one score update through a shared gather epoch.
    ///
    /// Exactly one gatherer may claim an unvisited node. Once the node has
    /// visits, every gatherer may reserve one additional update.
    pub(crate) fn try_start_score_update_shared(&self) -> bool {
        if self.total_visits == 0 {
            self.reservations.try_claim_fresh_shared()
        } else {
            self.reservations
                .reserve_node_shared(1, "try_start_score_update_shared");
            true
        }
    }

    pub(crate) fn increment_n_in_flight_shared(&self, count: u32) {
        self.reservations
            .reserve_node_shared(count, "increment_n_in_flight_shared");
    }

    pub(crate) fn cancel_score_update_multi_shared(&self, count: u32) {
        self.reservations
            .release_node_shared(count, "cancel_score_update_multi_shared");
    }

    pub(crate) fn add_virtual_loss_multi_shared(
        &self,
        i: usize,
        j: usize,
        count: u32,
    ) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations
            .reserve_joint_shared(i, j, count, "add_virtual_loss_multi_shared");
    }

    pub(crate) fn revert_virtual_loss_multi_shared(
        &self,
        i: usize,
        j: usize,
        count: u32,
    ) {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.reservations.release_joint_shared(
            i,
            j,
            count,
            "revert_virtual_loss_multi_shared",
        );
    }

    // --- Collision detection ---

    /// lc0 pattern: for unvisited nodes, fails if already claimed.
    /// For visited nodes, always succeeds.
    pub fn try_start_score_update(&mut self) -> bool {
        if self.total_visits == 0 && self.reservations.node_exclusive() > 0 {
            return false;
        }
        self.reservations
            .reserve_node_exclusive(1, "try_start_score_update");
        true
    }

    pub fn cancel_score_update(&mut self) {
        self.reservations
            .release_node_exclusive(1, "cancel_score_update");
    }

    pub fn n_in_flight(&self) -> u32 {
        self.reservations.node_shared()
    }

    pub fn edge_q_p1(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.edge_q_p1[i][j]
    }

    pub fn edge_q_p2(&self, i: usize, j: usize) -> f32 {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.edge_q_p2[i][j]
    }

    pub fn edge_visits(&self, i: usize, j: usize) -> u32 {
        debug_assert!(i < self.n1());
        debug_assert!(j < self.n2());
        self.edge_visits[i][j]
    }

    /// Sum edge_visits[i][j] over all j (marginal visits for p1 outcome i).
    pub fn marginal_visits_p1(&self, i: usize) -> u32 {
        debug_assert!(i < self.n1());
        let mut sum = 0u32;
        for j in 0..self.n2() {
            sum += self.edge_visits[i][j];
        }
        sum
    }

    /// Sum edge_visits[i][j] over all i (marginal visits for p2 outcome j).
    pub fn marginal_visits_p2(&self, j: usize) -> u32 {
        debug_assert!(j < self.n2());
        let mut sum = 0u32;
        for i in 0..self.n1() {
            sum += self.edge_visits[i][j];
        }
        sum
    }

    /// Sum over all valid (i, j) entries.
    pub fn total_edge_visits(&self) -> u32 {
        let mut sum = 0u32;
        for i in 0..self.n1() {
            for j in 0..self.n2() {
                sum += self.edge_visits[i][j];
            }
        }
        sum
    }

    /// Marginal p1 visits mapped to 5-action space. Blocked actions get 0.
    pub fn expand_p1_visits(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n1() {
            let action = self.p1_outcomes[idx] as usize;
            out[action] = self.marginal_visits_p1(idx) as f32;
        }
        out
    }

    /// Marginal p2 visits mapped to 5-action space. Blocked actions get 0.
    pub fn expand_p2_visits(&self) -> [f32; 5] {
        let mut out = [0.0f32; 5];
        for idx in 0..self.n2() {
            let action = self.p2_outcomes[idx] as usize;
            out[action] = self.marginal_visits_p2(idx) as f32;
        }
        out
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

    // --- Child list mutations ---

    /// Detach and return the entire child list. Leaves `first_child` as `None`.
    pub fn take_first_child(&mut self) -> Option<Box<Edge>> {
        self.first_child.take()
    }

    // --- Prior setters (for Dirichlet noise) ---

    pub fn set_p1_prior_at(&mut self, idx: usize, value: f32) {
        debug_assert!(idx < self.n1());
        self.p1_prior[idx] = value;
    }

    pub fn set_p2_prior_at(&mut self, idx: usize, value: f32) {
        debug_assert!(idx < self.n2());
        self.p2_prior[idx] = value;
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
// One Edge per (parent LowNode, outcome pair) visited. Carries the transition
// reward and per-edge aggregate values for delta detection. Points to the
// shared LowNode for the child position via Arc.
//
// Per-(i,j) Q accumulators live on LowNode (joint matrix), not here.

pub struct Edge {
    // Transition reward from parent to child (per-parent, not shared)
    edge_r1: f32,
    edge_r2: f32,

    // Which outcome pair at the parent LowNode this edge represents
    parent_outcome: (u8, u8),

    // Linked list under parent LowNode
    next_sibling: Option<Box<Edge>>,

    // Shared child position
    low_node: Arc<SharedNode>,
}

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edge")
            .field("parent_outcome", &self.parent_outcome)
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
        low_node: Arc<SharedNode>,
        parent_outcome: (u8, u8),
        r1: f32,
        r2: f32,
    ) -> Self {
        low_node.add_parent();
        Self {
            edge_r1: r1,
            edge_r2: r2,
            parent_outcome,
            next_sibling: None,
            low_node,
        }
    }

    // --- Getters ---

    pub fn r1(&self) -> f32 {
        self.edge_r1
    }

    pub fn r2(&self) -> f32 {
        self.edge_r2
    }

    pub fn parent_outcome(&self) -> (u8, u8) {
        self.parent_outcome
    }

    pub fn low_node(&self) -> &Arc<SharedNode> {
        &self.low_node
    }

    // --- Sibling navigation ---

    pub fn next_sibling(&self) -> Option<&Edge> {
        self.next_sibling.as_deref()
    }

    pub fn next_sibling_mut(&mut self) -> Option<&mut Edge> {
        self.next_sibling.as_deref_mut()
    }

    /// Detach and return the next sibling. Leaves `next_sibling` as `None`.
    pub fn take_next_sibling(&mut self) -> Option<Box<Edge>> {
        self.next_sibling.take()
    }
}

impl Drop for Edge {
    fn drop(&mut self) {
        self.low_node.remove_parent();
        if let Some(sibling) = self.next_sibling.take() {
            crate::gc::queue(sibling);
        }
    }
}

// ---------------------------------------------------------------------------
// SharedNode — owner-tagged UnsafeCell storage
// ---------------------------------------------------------------------------
//
// LowNode lives behind Arc (for TT Weak refs and Edge sharing). During search,
// we need &mut LowNode for backup, populate, virtual loss. SharedNode wraps
// LowNode in UnsafeCell for zero-cost interior mutability.
//
// Persistent storage is unbranded. Branded access sessions validate the
// immutable owner token before dereferencing `inner`.

/// Opaque identity shared by one tree and all nodes allocated for it.
pub(crate) struct OwnerToken;

pub struct SharedNode {
    owner: Arc<OwnerToken>,
    inner: UnsafeCell<LowNode>,
    // Outside UnsafeCell — safe for concurrent atomic access (e.g. from GC thread).
    num_parents: AtomicU16,
}

impl SharedNode {
    pub(crate) fn with_owner(node: LowNode, owner: Arc<OwnerToken>) -> Self {
        Self {
            owner,
            inner: UnsafeCell::new(node),
            num_parents: AtomicU16::new(0),
        }
    }

    pub(crate) fn owner(&self) -> &Arc<OwnerToken> {
        &self.owner
    }

    /// Raw payload address for the sealed capability boundary.
    ///
    /// Producing the pointer is safe; dereferencing it is not. The access
    /// capability must validate ownership and establish the appropriate
    /// read/write epoch before forming a reference.
    #[inline]
    pub(crate) fn low_node_ptr(&self) -> *mut LowNode {
        self.inner.get()
    }

    // --- Transposition tracking (atomic, safe from any thread) ---

    pub fn add_parent(&self) {
        self.num_parents
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .unwrap_or_else(|current| {
                panic!("add_parent: num_parents overflow at {current}")
            });
    }

    pub fn remove_parent(&self) {
        self.num_parents
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(1)
            })
            .unwrap_or_else(|current| {
                panic!("remove_parent: num_parents underflow at {current}")
            });
    }

    pub fn num_parents(&self) -> u16 {
        self.num_parents.load(Ordering::Acquire)
    }

    pub fn is_transposition(&self) -> bool {
        self.num_parents() > 1
    }
}

impl Drop for SharedNode {
    fn drop(&mut self) {
        // Exclusive access guaranteed: last Arc just dropped (strong_count == 0).
        let low = self.inner.get_mut();
        if let Some(child) = low.first_child.take() {
            crate::gc::queue(child);
        }
    }
}

// SAFETY: shared payload dereferences are sealed behind owner-checked access
// capabilities. Shared epochs may mutate only `ReservationState` atomics; all
// other payload fields remain immutable until an exclusive/write epoch.
// Mutable payload access requires the exclusive whole-tree capability. GC
// crosses threads only with owned, detached edges, and `Drop for SharedNode`
// has true `&mut self`. `Send` remains auto-derived from the fields; only
// shared access needs this explicit invariant.
unsafe impl Sync for SharedNode {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MCGSTree;
    use pyrat::{Coordinates, GameBuilder};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Barrier;
    use std::thread;

    /// All actions open — simplest effective-action mapping.
    const OPEN: [u8; 5] = [0, 1, 2, 3, 4];

    fn fixture_tree() -> MCGSTree {
        let game = GameBuilder::new(3, 3)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 2))
            .with_custom_cheese(vec![Coordinates::new(1, 1)])
            .with_max_turns(10)
            .build()
            .create(None)
            .unwrap();
        MCGSTree::new(&game)
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

        low.finalize_score_update(2.0, 1.0);
        low.finalize_score_update(4.0, 3.0);
        low.finalize_score_update(6.0, 5.0);

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

    // ---- SharedNode: transposition tracking ----

    #[test]
    fn shared_node_num_parents() {
        let shared = make_shared(OPEN, OPEN);
        assert_eq!(shared.num_parents(), 0);
        assert!(!shared.is_transposition());

        shared.add_parent();
        assert_eq!(shared.num_parents(), 1);
        assert!(!shared.is_transposition());

        shared.add_parent();
        assert_eq!(shared.num_parents(), 2);
        assert!(shared.is_transposition());

        shared.remove_parent();
        assert_eq!(shared.num_parents(), 1);
        assert!(!shared.is_transposition());
    }

    #[test]
    fn shared_node_parent_underflow_panics_before_wrap() {
        let shared = make_shared(OPEN, OPEN);

        let result = catch_unwind(AssertUnwindSafe(|| shared.remove_parent()));

        assert!(result.is_err());
        assert_eq!(shared.num_parents(), 0);
    }

    #[test]
    fn shared_node_parent_overflow_panics_before_wrap() {
        let shared = make_shared(OPEN, OPEN);
        for _ in 0..u16::MAX {
            shared.add_parent();
        }

        let result = catch_unwind(AssertUnwindSafe(|| shared.add_parent()));

        assert!(result.is_err());
        assert_eq!(shared.num_parents(), u16::MAX);
    }

    // ---- Edge: creation and Arc sharing ----

    /// Helper: wrap LowNode in SharedNode + Arc for Edge tests.
    fn make_shared(eff_p1: [u8; 5], eff_p2: [u8; 5]) -> Arc<SharedNode> {
        let mut tree = fixture_tree();
        tree.with_exclusive(|mut access| {
            let node = access.test_node(LowNode::new_shell(eff_p1, eff_p2));
            Arc::clone(node.arc())
        })
    }

    fn make_shared_open() -> Arc<SharedNode> {
        make_shared(OPEN, OPEN)
    }

    #[test]
    fn edge_creation_increments_parents() {
        let low = make_shared_open();
        assert_eq!(low.num_parents(), 0);

        let edge = Edge::new(Arc::clone(&low), (0, 1), 1.0, 0.5);
        assert_eq!(low.num_parents(), 1);
        assert_eq!(edge.parent_outcome(), (0, 1));
        assert!((edge.r1() - 1.0).abs() < 1e-6);
        assert!((edge.r2() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn edge_drop_decrements_parents() {
        let low = make_shared_open();

        {
            let _edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);
            assert_eq!(low.num_parents(), 1);
        }
        // Edge dropped
        assert_eq!(low.num_parents(), 0);
    }

    #[test]
    fn multiple_edges_share_low_node() {
        let low = make_shared_open();

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
        let low = make_shared_open();
        let weak = Arc::downgrade(&low);

        let edge = Edge::new(Arc::clone(&low), (0, 0), 0.0, 0.0);
        drop(low); // Original Arc dropped, but edge still holds one

        assert!(weak.upgrade().is_some());
        assert_eq!(edge.low_node().num_parents(), 1);

        drop(edge);
        assert!(weak.upgrade().is_none()); // LowNode freed
    }

    // ---- LowNode: per-(i,j) edge matrix ----

    #[test]
    fn low_node_edge_welford() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        low.finalize_edge_update(0, 0, 5.0, 2.0);
        low.finalize_edge_update(0, 0, 3.0, 4.0);
        assert_eq!(low.edge_visits(0, 0), 2);
        assert!((low.edge_q_p1(0, 0) - 4.0).abs() < 1e-5);
        assert!((low.edge_q_p2(0, 0) - 3.0).abs() < 1e-5);

        // Other entries untouched
        assert_eq!(low.edge_visits(1, 0), 0);
        assert_eq!(low.edge_q_p1(1, 0), 0.0);
    }

    #[test]
    fn low_node_edge_update_multiple_j() {
        // Different j values for the same i accumulate independently
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        low.finalize_edge_update(2, 0, 10.0, 1.0);
        low.finalize_edge_update(2, 1, 20.0, 2.0);
        low.finalize_edge_update(2, 0, 12.0, 3.0);

        assert_eq!(low.edge_visits(2, 0), 2);
        assert!((low.edge_q_p1(2, 0) - 11.0).abs() < 1e-5);

        assert_eq!(low.edge_visits(2, 1), 1);
        assert!((low.edge_q_p1(2, 1) - 20.0).abs() < 1e-5);
    }

    #[test]
    fn low_node_marginal_visits() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        low.finalize_edge_update(1, 0, 1.0, 1.0);
        low.finalize_edge_update(1, 0, 1.0, 1.0);
        low.finalize_edge_update(1, 2, 1.0, 1.0);
        low.finalize_edge_update(3, 2, 1.0, 1.0);

        // P1 marginal: i=1 visited with j=0 (2x) and j=2 (1x)
        assert_eq!(low.marginal_visits_p1(1), 3);
        assert_eq!(low.marginal_visits_p1(3), 1);
        assert_eq!(low.marginal_visits_p1(0), 0);

        // P2 marginal: j=0 visited from i=1 (2x), j=2 from i=1 (1x) + i=3 (1x)
        assert_eq!(low.marginal_visits_p2(0), 2);
        assert_eq!(low.marginal_visits_p2(2), 2);
        assert_eq!(low.marginal_visits_p2(1), 0);
    }

    #[test]
    fn low_node_total_edge_visits() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert_eq!(low.total_edge_visits(), 0);

        low.finalize_edge_update(0, 0, 1.0, 1.0);
        low.finalize_edge_update(0, 0, 1.0, 1.0);
        low.finalize_edge_update(2, 3, 1.0, 1.0);
        assert_eq!(low.total_edge_visits(), 3);
    }

    #[test]
    fn low_node_expand_visits() {
        // P1: UP blocked -> 4 outcomes [1, 2, 3, 4]
        // P2: open -> 5 outcomes
        let mut low = LowNode::new_shell([4, 1, 2, 3, 4], [0, 1, 2, 3, 4]);

        // P1 outcome 0 maps to action 1, outcome 3 maps to action 4
        // Give visits to outcome 0 with various j
        for j in 0..5 {
            low.finalize_edge_update(0, j, 1.0, 1.0);
            low.finalize_edge_update(0, j, 1.0, 1.0); // 2 visits per j = 10 total for i=0
        }
        // Give visits to outcome 3 (action 4)
        for _ in 0..7 {
            low.finalize_edge_update(3, 0, 1.0, 1.0);
        }

        let expanded = low.expand_p1_visits();
        assert_eq!(expanded[0], 0.0); // blocked
        assert_eq!(expanded[1], 10.0); // outcome 0 -> action 1
        assert_eq!(expanded[2], 0.0);
        assert_eq!(expanded[3], 0.0);
        assert_eq!(expanded[4], 7.0); // outcome 3 -> action 4
    }

    // ---- Child linked list ----

    #[test]
    fn child_list_prepend_and_find() {
        let mut tree = fixture_tree();
        tree.with_exclusive(|mut access| {
            let child1 = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let child2 = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let parent = access.test_node(LowNode::new_shell(OPEN, OPEN));

            access.test_connect(&parent, &child1, (0, 1), 1.0, 0.0);
            access.test_connect(&parent, &child2, (2, 3), 0.0, 1.0);

            {
                let parent = access.node(&parent);

                // child2 was prepended last, so it's first
                let first = parent.first_child().unwrap();
                assert_eq!(first.parent_outcome(), (2, 3));

                let second = first.next_sibling().unwrap();
                assert_eq!(second.parent_outcome(), (0, 1));
                assert!(second.next_sibling().is_none());

                assert!(parent.find_child(0, 1).is_some());
                assert_eq!(parent.find_child(0, 1).unwrap().parent_outcome(), (0, 1));
                assert!(parent.find_child(2, 3).is_some());
                assert_eq!(parent.find_child(2, 3).unwrap().parent_outcome(), (2, 3));
                assert!(parent.find_child(4, 4).is_none());

                // Both edges retain distinct child identities.
                assert!(!Arc::ptr_eq(
                    parent.find_child(0, 1).unwrap().low_node(),
                    parent.find_child(2, 3).unwrap().low_node()
                ));
            }

            assert_eq!(access.num_parents(&child1), 1);
            assert_eq!(access.num_parents(&child2), 1);
        });
    }

    #[test]
    fn child_list_find_mut() {
        let child = make_shared_open();
        let mut parent = LowNode::new_shell(OPEN, OPEN);

        let edge = Box::new(Edge::new(Arc::clone(&child), (1, 2), 0.5, 0.5));
        parent.prepend_child(edge);

        // Verify we can find the edge and read its fields
        let found = parent.find_child_mut(1, 2).unwrap();
        assert_eq!(found.parent_outcome(), (1, 2));
        assert!((found.r1() - 0.5).abs() < 1e-6);
        assert!((found.r2() - 0.5).abs() < 1e-6);
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

    // ---- LowNode: virtual loss on (i,j) matrix ----

    #[test]
    fn low_node_virtual_loss_round_trip() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        low.add_virtual_loss(1, 2);
        low.add_virtual_loss(1, 2);
        low.add_virtual_loss(1, 3);

        assert_eq!(low.edge_in_flight(1, 2), 2);
        assert_eq!(low.edge_in_flight(1, 3), 1);
        assert_eq!(low.edge_in_flight(0, 0), 0);

        low.revert_virtual_loss(1, 2);
        assert_eq!(low.edge_in_flight(1, 2), 1);

        low.revert_virtual_loss(1, 2);
        assert_eq!(low.edge_in_flight(1, 2), 0);
    }

    #[test]
    fn low_node_marginal_in_flight() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        low.add_virtual_loss(1, 0);
        low.add_virtual_loss(1, 2);
        low.add_virtual_loss(3, 2);

        // P1 marginal: i=1 has in-flight at j=0 (1) and j=2 (1)
        assert_eq!(low.marginal_in_flight_p1(1), 2);
        assert_eq!(low.marginal_in_flight_p1(3), 1);
        assert_eq!(low.marginal_in_flight_p1(0), 0);

        // P2 marginal: j=0 from i=1 (1), j=2 from i=1 (1) + i=3 (1)
        assert_eq!(low.marginal_in_flight_p2(0), 1);
        assert_eq!(low.marginal_in_flight_p2(2), 2);
        assert_eq!(low.marginal_in_flight_p2(1), 0);
    }

    #[test]
    #[should_panic(expected = "revert_virtual_loss: edge_in_flight[0][0] is already 0")]
    fn low_node_revert_virtual_loss_at_zero_panics() {
        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.revert_virtual_loss(0, 0);
    }

    // ---- LowNode: collision detection ----

    #[test]
    fn low_node_try_start_fresh() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        assert!(low.try_start_score_update());
        assert_eq!(low.n_in_flight(), 1);
    }

    #[test]
    fn low_node_collision_unvisited() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        assert!(low.try_start_score_update());
        assert!(!low.try_start_score_update()); // collision
        assert_eq!(low.n_in_flight(), 1);
    }

    #[test]
    fn low_node_visited_always_succeeds() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        low.finalize_score_update(1.0, 1.0);
        assert!(low.try_start_score_update());
        assert!(low.try_start_score_update());
        assert!(low.try_start_score_update());
        assert_eq!(low.n_in_flight(), 3);
    }

    #[test]
    fn low_node_cancel_score_update() {
        let mut low = LowNode::new_shell(OPEN, OPEN);

        low.finalize_score_update(1.0, 1.0);
        assert!(low.try_start_score_update());
        low.cancel_score_update();
        assert_eq!(low.n_in_flight(), 0);
    }

    #[test]
    #[should_panic(expected = "cancel_score_update: n_in_flight is already 0")]
    fn low_node_cancel_at_zero_panics() {
        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.cancel_score_update();
    }

    // ---- LowNode: shared reservation state ----

    #[test]
    fn reservation_state_fresh_claim_has_exactly_one_winner_and_is_reusable() {
        const WORKERS: usize = 8;

        let low = Arc::new(LowNode::new_shell(OPEN, OPEN));
        let start = Arc::new(Barrier::new(WORKERS + 1));
        let claimed = Arc::new(Barrier::new(WORKERS + 1));
        let release = Arc::new(Barrier::new(WORKERS + 1));

        let (winners, peak) = thread::scope(|scope| {
            let handles: Vec<_> = (0..WORKERS)
                .map(|_| {
                    let low = Arc::clone(&low);
                    let start = Arc::clone(&start);
                    let claimed = Arc::clone(&claimed);
                    let release = Arc::clone(&release);
                    scope.spawn(move || {
                        start.wait();
                        let won = low.try_start_score_update_shared();
                        claimed.wait();
                        release.wait();
                        if won {
                            low.cancel_score_update_multi_shared(1);
                        }
                        won
                    })
                })
                .collect();

            start.wait();
            claimed.wait();
            let peak = low.n_in_flight();
            release.wait();

            let winners = handles
                .into_iter()
                .map(|handle| handle.join().unwrap() as usize)
                .sum::<usize>();
            (winners, peak)
        });

        assert_eq!(peak, 1);
        assert_eq!(winners, 1);
        assert_eq!(low.n_in_flight(), 0);
        assert!(low.try_start_score_update_shared());
        low.cancel_score_update_multi_shared(1);
        assert_eq!(low.n_in_flight(), 0);
    }

    #[test]
    fn reservation_state_visited_claims_all_reserve_and_cancel() {
        const WORKERS: usize = 8;

        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.finalize_score_update(1.0, 1.0);
        let low = Arc::new(low);
        let start = Arc::new(Barrier::new(WORKERS + 1));
        let claimed = Arc::new(Barrier::new(WORKERS + 1));
        let release = Arc::new(Barrier::new(WORKERS + 1));

        let (successes, peak) = thread::scope(|scope| {
            let handles: Vec<_> = (0..WORKERS)
                .map(|_| {
                    let low = Arc::clone(&low);
                    let start = Arc::clone(&start);
                    let claimed = Arc::clone(&claimed);
                    let release = Arc::clone(&release);
                    scope.spawn(move || {
                        start.wait();
                        let reserved = low.try_start_score_update_shared();
                        claimed.wait();
                        release.wait();
                        if reserved {
                            low.cancel_score_update_multi_shared(1);
                        }
                        reserved
                    })
                })
                .collect();

            start.wait();
            claimed.wait();
            let peak = low.n_in_flight();
            release.wait();

            let successes = handles
                .into_iter()
                .map(|handle| handle.join().unwrap() as usize)
                .sum::<usize>();
            (successes, peak)
        });

        assert_eq!(peak, WORKERS as u32);
        assert_eq!(successes, WORKERS);
        assert_eq!(low.total_visits(), 1);
        assert_eq!(low.n_in_flight(), 0);
    }

    #[test]
    fn reservation_state_all_joint_cells_balance_and_marginalize() {
        let low = LowNode::new_shell(OPEN, OPEN);

        for i in 0..5 {
            for j in 0..5 {
                let count = (i * 5 + j + 1) as u32;
                low.increment_n_in_flight_shared(count);
                low.add_virtual_loss_multi_shared(i, j, count);
            }
        }

        assert_eq!(low.n_in_flight(), 325);
        for i in 0..5 {
            assert_eq!(low.marginal_in_flight_p1(i), (25 * i + 15) as u32);
            assert_eq!(low.marginal_n_started_p1(i), (25 * i + 15) as u32);
        }
        for j in 0..5 {
            assert_eq!(low.marginal_in_flight_p2(j), (55 + 5 * j) as u32);
            assert_eq!(low.marginal_n_started_p2(j), (55 + 5 * j) as u32);
        }

        for i in 0..5 {
            for j in 0..5 {
                let count = (i * 5 + j + 1) as u32;
                assert_eq!(low.edge_in_flight(i, j), count);
                low.revert_virtual_loss_multi_shared(i, j, count);
                low.cancel_score_update_multi_shared(count);
            }
        }

        assert_eq!(low.n_in_flight(), 0);
        for i in 0..5 {
            assert_eq!(low.marginal_in_flight_p1(i), 0);
            for j in 0..5 {
                assert_eq!(low.edge_in_flight(i, j), 0);
            }
        }
        for j in 0..5 {
            assert_eq!(low.marginal_in_flight_p2(j), 0);
        }
    }

    #[test]
    fn reservation_state_shared_node_and_edge_repeatedly_quiesce() {
        const WORKERS: usize = 8;
        const ROUNDS: usize = 256;

        let low = Arc::new(LowNode::new_shell(OPEN, OPEN));
        let reserved = Arc::new(Barrier::new(WORKERS + 1));
        let release = Arc::new(Barrier::new(WORKERS + 1));
        let quiescent = Arc::new(Barrier::new(WORKERS + 1));
        let verified = Arc::new(Barrier::new(WORKERS + 1));

        let observations = thread::scope(|scope| {
            for worker in 0..WORKERS {
                let low = Arc::clone(&low);
                let reserved = Arc::clone(&reserved);
                let release = Arc::clone(&release);
                let quiescent = Arc::clone(&quiescent);
                let verified = Arc::clone(&verified);
                scope.spawn(move || {
                    for round in 0..ROUNDS {
                        let flat = if round % 2 == 0 {
                            2 * 5 + 3
                        } else {
                            (round + worker * 7) % 25
                        };
                        let i = flat / 5;
                        let j = flat % 5;
                        let count = ((round + worker) % 3 + 1) as u32;

                        low.increment_n_in_flight_shared(count);
                        low.add_virtual_loss_multi_shared(i, j, count);
                        reserved.wait();
                        release.wait();
                        low.revert_virtual_loss_multi_shared(i, j, count);
                        low.cancel_score_update_multi_shared(count);
                        quiescent.wait();
                        verified.wait();
                    }
                });
            }

            let mut observations = Vec::with_capacity(ROUNDS);
            for round in 0..ROUNDS {
                reserved.wait();
                let expected: u32 = (0..WORKERS)
                    .map(|worker| ((round + worker) % 3 + 1) as u32)
                    .sum();
                let joint_total: u32 = (0..5)
                    .flat_map(|i| (0..5).map(move |j| (i, j)))
                    .map(|(i, j)| low.edge_in_flight(i, j))
                    .sum();
                let node_peak = low.n_in_flight();

                release.wait();
                quiescent.wait();
                let node_quiescent = low.n_in_flight();
                let p1_quiescent =
                    std::array::from_fn(|i| low.marginal_in_flight_p1(i));
                let p2_quiescent =
                    std::array::from_fn(|j| low.marginal_in_flight_p2(j));
                verified.wait();
                observations.push((
                    expected,
                    node_peak,
                    joint_total,
                    node_quiescent,
                    p1_quiescent,
                    p2_quiescent,
                ));
            }
            observations
        });

        for (
            expected,
            node_peak,
            joint_total,
            node_quiescent,
            p1_quiescent,
            p2_quiescent,
        ) in observations
        {
            assert_eq!(node_peak, expected);
            assert_eq!(joint_total, expected);
            assert_eq!(node_quiescent, 0);
            assert_eq!(p1_quiescent, [0; 5]);
            assert_eq!(p2_quiescent, [0; 5]);
        }
    }

    #[test]
    fn reservation_state_failed_shared_release_does_not_wrap() {
        let low = LowNode::new_shell(OPEN, OPEN);

        let node_release = catch_unwind(AssertUnwindSafe(|| {
            low.cancel_score_update_multi_shared(1);
        }));
        assert!(node_release.is_err());
        assert_eq!(low.n_in_flight(), 0);

        let edge_release = catch_unwind(AssertUnwindSafe(|| {
            low.revert_virtual_loss_multi_shared(2, 3, 1);
        }));
        assert!(edge_release.is_err());
        assert_eq!(low.edge_in_flight(2, 3), 0);
    }

    // ---- SharedNode ----

    #[test]
    fn shared_node_payload_uses_exclusive_access() {
        let mut tree = fixture_tree();
        tree.with_exclusive(|mut access| {
            let shared = access.test_node(LowNode::new_shell(OPEN, OPEN));
            assert_eq!(access.node(&shared).n1(), 5);
            assert_eq!(access.node(&shared).total_visits(), 0);

            access.node_mut(&shared).finalize_score_update(3.0, 2.0);
            assert_eq!(access.node(&shared).total_visits(), 1);
            assert!((access.node(&shared).v1() - 3.0).abs() < 1e-6);
        });
    }

    #[test]
    fn cloned_handle_observes_same_shared_payload() {
        let mut tree = fixture_tree();
        tree.with_exclusive(|mut access| {
            let shared = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let clone = shared.clone();

            access.node_mut(&shared).finalize_score_update(5.0, 5.0);
            assert_eq!(access.node(&clone).total_visits(), 1);
            assert!((access.node(&clone).v1() - 5.0).abs() < 1e-6);
        });
    }

    // ---- Multivisit methods ----

    #[test]
    fn finalize_score_update_multi_matches_repeated_single() {
        // Single-visit finalize_score_update doesn't touch n_in_flight.
        // Compare visit counts and values only.
        let mut single = LowNode::new_shell(OPEN, OPEN);
        single.set_value_scale(5.0);
        single.finalize_score_update(2.0, 1.0);
        single.finalize_score_update(2.0, 1.0);
        single.finalize_score_update(2.0, 1.0);

        let mut multi = LowNode::new_shell(OPEN, OPEN);
        multi.set_value_scale(5.0);
        multi.increment_n_in_flight(3); // multi version decrements n_in_flight
        multi.finalize_score_update_multi(2.0, 1.0, 3);

        assert_eq!(single.total_visits(), multi.total_visits());
        assert!((single.v1() - multi.v1()).abs() < 1e-6);
        assert!((single.v2() - multi.v2()).abs() < 1e-6);
        assert_eq!(multi.n_in_flight(), 0);
    }

    #[test]
    fn finalize_score_update_multi_mixed_values() {
        // Multi with count=3 of value 4.0, then single of 2.0
        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.increment_n_in_flight(4);
        low.finalize_score_update_multi(4.0, 4.0, 3);
        assert_eq!(low.total_visits(), 3);
        assert!((low.v1() - 4.0).abs() < 1e-6);

        low.finalize_score_update_multi(2.0, 2.0, 1);
        // mean(4,4,4,2) = 3.5
        assert_eq!(low.total_visits(), 4);
        assert!((low.v1() - 3.5).abs() < 1e-5);
        assert_eq!(low.n_in_flight(), 0);
    }

    #[test]
    fn finalize_edge_update_multi_matches_repeated_single() {
        let mut single = LowNode::new_shell(OPEN, OPEN);
        single.finalize_edge_update(1, 2, 3.0, 1.0);
        single.finalize_edge_update(1, 2, 3.0, 1.0);
        single.finalize_edge_update(1, 2, 3.0, 1.0);

        let mut multi = LowNode::new_shell(OPEN, OPEN);
        multi.finalize_edge_update_multi(1, 2, 3.0, 1.0, 3);

        assert_eq!(single.edge_visits(1, 2), multi.edge_visits(1, 2));
        assert!((single.edge_q_p1(1, 2) - multi.edge_q_p1(1, 2)).abs() < 1e-6);
        assert!((single.edge_q_p2(1, 2) - multi.edge_q_p2(1, 2)).abs() < 1e-6);
    }

    #[test]
    fn virtual_loss_multi_round_trip() {
        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.add_virtual_loss_multi(2, 3, 5);
        assert_eq!(low.edge_in_flight(2, 3), 5);
        assert_eq!(low.marginal_in_flight_p1(2), 5);

        low.revert_virtual_loss_multi(2, 3, 3);
        assert_eq!(low.edge_in_flight(2, 3), 2);

        low.revert_virtual_loss_multi(2, 3, 2);
        assert_eq!(low.edge_in_flight(2, 3), 0);
    }

    #[test]
    fn increment_and_cancel_n_in_flight() {
        let mut low = LowNode::new_shell(OPEN, OPEN);
        low.increment_n_in_flight(10);
        assert_eq!(low.n_in_flight(), 10);

        low.cancel_score_update_multi(4);
        assert_eq!(low.n_in_flight(), 6);

        low.cancel_score_update_multi(6);
        assert_eq!(low.n_in_flight(), 0);
    }

    #[test]
    fn marginal_n_started() {
        let mut low = LowNode::new_shell(OPEN, OPEN);
        // 3 visits at (1,0), 2 in-flight at (1,2)
        low.finalize_edge_update(1, 0, 1.0, 1.0);
        low.finalize_edge_update(1, 0, 1.0, 1.0);
        low.finalize_edge_update(1, 0, 1.0, 1.0);
        low.add_virtual_loss_multi(1, 2, 2);

        // p1 outcome 1: 3 visits + 2 in_flight = 5
        assert_eq!(low.marginal_n_started_p1(1), 5);

        // p2 outcome 0: 3 visits + 0 in_flight = 3
        assert_eq!(low.marginal_n_started_p2(0), 3);
        // p2 outcome 2: 0 visits + 2 in_flight = 2
        assert_eq!(low.marginal_n_started_p2(2), 2);
    }
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::ReservationState;
    use loom::sync::Arc;
    use loom::thread;

    #[test]
    fn loom_fresh_claim_has_exactly_one_winner() {
        loom::model(|| {
            let reservations = Arc::new(ReservationState::new());
            let first = {
                let reservations = Arc::clone(&reservations);
                thread::spawn(move || reservations.try_claim_fresh_shared())
            };
            let second = {
                let reservations = Arc::clone(&reservations);
                thread::spawn(move || reservations.try_claim_fresh_shared())
            };

            let winners = first.join().unwrap() as u8 + second.join().unwrap() as u8;
            assert_eq!(winners, 1);
            assert_eq!(reservations.node_shared(), 1);

            reservations.release_node_shared(1, "loom fresh claim");
            assert_eq!(reservations.node_shared(), 0);
            assert!(reservations.try_claim_fresh_shared());
            reservations.release_node_shared(1, "loom reused fresh claim");
            assert_eq!(reservations.node_shared(), 0);
        });
    }

    #[test]
    fn loom_node_and_joint_reservations_balance() {
        loom::model(|| {
            let reservations = Arc::new(ReservationState::new());
            let first = {
                let reservations = Arc::clone(&reservations);
                thread::spawn(move || {
                    reservations.reserve_node_shared(1, "loom reserve node");
                    reservations.reserve_joint_shared(2, 3, 1, "loom reserve joint");
                    reservations.release_joint_shared(2, 3, 1, "loom release joint");
                    reservations.release_node_shared(1, "loom release node");
                })
            };
            let second = {
                let reservations = Arc::clone(&reservations);
                thread::spawn(move || {
                    reservations.reserve_node_shared(2, "loom reserve node");
                    reservations.reserve_joint_shared(2, 3, 2, "loom reserve joint");
                    reservations.release_joint_shared(2, 3, 2, "loom release joint");
                    reservations.release_node_shared(2, "loom release node");
                })
            };

            first.join().unwrap();
            second.join().unwrap();
            assert_eq!(reservations.node_shared(), 0);
            assert_eq!(reservations.joint_shared(2, 3), 0);
        });
    }
}
