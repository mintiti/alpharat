//! Scoped, read-only observation of an MCGS tree.
//!
//! Public callers can inspect search results without receiving the raw DAG
//! nodes or the `UnsafeCell`-backed implementation types. A fresh invariant
//! session lifetime is minted by [`MCGSTree::observe`](crate::MCGSTree::observe):
//! owned node handles may cross short view epochs inside that callback, while
//! borrowed node and edge views cannot escape an epoch or the callback.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::node::{Edge, LowNode, SharedNode};
use crate::tree::MCGSTree;

pub(crate) type Invariant<'session> =
    PhantomData<fn(&'session mut ()) -> &'session mut ()>;

/// Which player's outcome-reduced statistics to inspect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchPlayer {
    Player1,
    Player2,
}

/// Owned aggregate statistics for one shared position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeStats {
    /// Aggregate visits to this shared position across all parent edges.
    pub total_visits: u32,
    /// Sum of visits in this node's outgoing joint-outcome matrix.
    pub total_edge_visits: u32,
    pub value_p1: f32,
    pub value_p2: f32,
    pub value_scale: f32,
    pub is_terminal: bool,
    pub is_evaluated: bool,
}

/// Owned marginal statistics for one outcome-reduced action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutcomeStats {
    /// Index in this node's outcome-reduced action space.
    pub index: u8,
    /// Representative action in the engine's five-action space.
    pub action: u8,
    pub visits: u32,
    /// Visit-weighted marginal Q, or zero when the outcome is unvisited.
    pub q: f32,
    pub prior: f32,
}

/// Owned transition data for one child edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeTransition {
    pub p1_outcome: u8,
    pub p2_outcome: u8,
    pub reward_p1: f32,
    pub reward_p2: f32,
}

/// Snapshot of the transposition table's weak-entry population.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranspositionStats {
    /// Physical weak-map slots, including expired entries.
    pub entries: usize,
    /// Entries whose weak node reference currently upgrades.
    pub live_entries: usize,
    pub expired_entries: usize,
}

impl TranspositionStats {
    pub(crate) fn from_counts(entries: usize, live_entries: usize) -> Self {
        Self {
            entries,
            live_entries,
            expired_entries: entries.saturating_sub(live_entries),
        }
    }
}

/// Owned tree-level statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TreeStats {
    /// Cached search node count, recomputed from live TT entries after root advancement.
    pub node_count: u32,
    pub transpositions: TranspositionStats,
}

/// Result of a best-effort expired-TT-entry eviction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TranspositionEviction {
    pub before: TranspositionStats,
    pub after: TranspositionStats,
    pub removed_entries: usize,
}

/// One generative, read-only observation session for a tree.
///
/// Its fields are private so callers cannot manufacture handles or reach the
/// raw tree. Do not add `Deref`/`AsRef` implementations to implementation
/// types: the scoped methods below are the safety boundary.
pub struct TreeView<'tree, 'session> {
    tree: &'tree MCGSTree,
    brand: Invariant<'session>,
}

impl<'tree, 'session> TreeView<'tree, 'session> {
    pub(crate) fn new(tree: &'tree MCGSTree) -> Self {
        Self {
            tree,
            brand: PhantomData,
        }
    }

    /// Return an owned, session-bound handle to the current root.
    pub fn root(&self) -> NodeHandle<'session> {
        NodeHandle::new(Arc::clone(self.tree.root()))
    }

    /// Inspect the current root for one non-escaping view epoch.
    pub fn with_root<R>(
        &self,
        inspect: impl for<'view> FnOnce(NodeView<'view, 'session>) -> R,
    ) -> R {
        let root = self.root();
        self.with_node(&root, inspect)
    }

    /// Inspect a session handle for one non-escaping view epoch.
    ///
    /// `R` may contain copied statistics or another `NodeHandle` from an edge,
    /// but it cannot contain the borrowed `NodeView` or `EdgeView`.
    pub fn with_node<R>(
        &self,
        handle: &NodeHandle<'session>,
        inspect: impl for<'view> FnOnce(NodeView<'view, 'session>) -> R,
    ) -> R {
        self.tree.with_observed_node(handle, |node| {
            inspect(NodeView {
                node,
                brand: PhantomData,
            })
        })
    }
}

/// Owned node identity confined to one observation session.
#[derive(Clone)]
pub struct NodeHandle<'session> {
    pub(crate) node: Arc<SharedNode>,
    pub(crate) brand: Invariant<'session>,
}

impl<'session> NodeHandle<'session> {
    pub(crate) fn new(node: Arc<SharedNode>) -> Self {
        Self {
            node,
            brand: PhantomData,
        }
    }

    pub(crate) fn inner(&self) -> &SharedNode {
        &self.node
    }

    pub(crate) fn arc(&self) -> &Arc<SharedNode> {
        &self.node
    }
}

/// Borrowed node data confined to one view epoch.
#[derive(Clone, Copy)]
pub struct NodeView<'view, 'session> {
    node: &'view LowNode,
    brand: Invariant<'session>,
}

impl<'view, 'session> NodeView<'view, 'session> {
    pub fn stats(self) -> NodeStats {
        NodeStats {
            total_visits: self.node.total_visits(),
            total_edge_visits: self.node.total_edge_visits(),
            value_p1: self.node.v1(),
            value_p2: self.node.v2(),
            value_scale: self.node.value_scale(),
            is_terminal: self.node.is_terminal(),
            is_evaluated: self.node.is_evaluated(),
        }
    }

    pub fn outcomes(self, player: SearchPlayer) -> Outcomes<'view, 'session> {
        Outcomes {
            node: self,
            player,
            next: 0,
        }
    }

    pub fn outcome(self, player: SearchPlayer, index: u8) -> Option<OutcomeStats> {
        let index_usize = index as usize;
        match player {
            SearchPlayer::Player1 if index_usize < self.node.n1() => Some(OutcomeStats {
                index,
                action: self.node.p1_outcome_action(index_usize),
                visits: self.node.marginal_visits_p1(index_usize),
                q: marginal_q_p1(self.node, index_usize),
                prior: self.node.p1_prior(index_usize),
            }),
            SearchPlayer::Player2 if index_usize < self.node.n2() => Some(OutcomeStats {
                index,
                action: self.node.p2_outcome_action(index_usize),
                visits: self.node.marginal_visits_p2(index_usize),
                q: marginal_q_p2(self.node, index_usize),
                prior: self.node.p2_prior(index_usize),
            }),
            _ => None,
        }
    }

    /// Expand marginal outcome visits back into the engine's five actions.
    /// Blocked actions remain zero.
    pub fn action_visits(self, player: SearchPlayer) -> [f32; 5] {
        match player {
            SearchPlayer::Player1 => self.node.expand_p1_visits(),
            SearchPlayer::Player2 => self.node.expand_p2_visits(),
        }
    }

    /// Find a child by outcome-reduced indices, not raw actions.
    pub fn edge(self, p1_outcome: u8, p2_outcome: u8) -> Option<EdgeView<'view, 'session>> {
        self.node
            .find_child(p1_outcome, p2_outcome)
            .map(|edge| EdgeView {
                edge,
                brand: PhantomData,
            })
    }

    pub fn children(self) -> ChildEdges<'view, 'session> {
        ChildEdges {
            next: self.node.first_child(),
            brand: PhantomData,
        }
    }
}

/// Iterator over copied marginal outcome statistics.
pub struct Outcomes<'view, 'session> {
    node: NodeView<'view, 'session>,
    player: SearchPlayer,
    next: u8,
}

impl Iterator for Outcomes<'_, '_> {
    type Item = OutcomeStats;

    fn next(&mut self) -> Option<Self::Item> {
        let outcome = self.node.outcome(self.player, self.next)?;
        self.next += 1;
        Some(outcome)
    }
}

/// Borrowed child-edge data confined to one view epoch.
#[derive(Clone, Copy)]
pub struct EdgeView<'view, 'session> {
    edge: &'view Edge,
    brand: Invariant<'session>,
}

impl<'session> EdgeView<'_, 'session> {
    pub fn transition(self) -> EdgeTransition {
        let (p1_outcome, p2_outcome) = self.edge.parent_outcome();
        EdgeTransition {
            p1_outcome,
            p2_outcome,
            reward_p1: self.edge.r1(),
            reward_p2: self.edge.r2(),
        }
    }

    /// Clone the child identity into a handle that can outlive this view epoch.
    pub fn child(self) -> NodeHandle<'session> {
        NodeHandle::new(Arc::clone(self.edge.low_node()))
    }
}

/// Iterator over guard-bound child edge views.
pub struct ChildEdges<'view, 'session> {
    next: Option<&'view Edge>,
    brand: Invariant<'session>,
}

impl<'view, 'session> Iterator for ChildEdges<'view, 'session> {
    type Item = EdgeView<'view, 'session>;

    fn next(&mut self) -> Option<Self::Item> {
        let edge = self.next?;
        self.next = edge.next_sibling();
        Some(EdgeView {
            edge,
            brand: PhantomData,
        })
    }
}

fn marginal_q_p1(node: &LowNode, i: usize) -> f32 {
    let mut visits = 0u32;
    let mut weighted_q = 0.0f32;
    for j in 0..node.n2() {
        let edge_visits = node.edge_visits(i, j);
        visits += edge_visits;
        weighted_q += edge_visits as f32 * node.edge_q_p1(i, j);
    }
    if visits == 0 {
        0.0
    } else {
        weighted_q / visits as f32
    }
}

fn marginal_q_p2(node: &LowNode, j: usize) -> f32 {
    let mut visits = 0u32;
    let mut weighted_q = 0.0f32;
    for i in 0..node.n1() {
        let edge_visits = node.edge_visits(i, j);
        visits += edge_visits;
        weighted_q += edge_visits as f32 * node.edge_q_p2(i, j);
    }
    if visits == 0 {
        0.0
    } else {
        weighted_q / visits as f32
    }
}
