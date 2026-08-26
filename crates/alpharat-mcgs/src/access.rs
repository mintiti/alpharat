//! Branded, lock-free access to one exclusively borrowed MCGS tree.
//!
//! Persistent nodes stay unbranded so the tree can survive across searches.
//! Each [`MCGSTree::with_exclusive`] callback mints a fresh invariant session
//! brand. Owned [`NodeHandle`] values may cross short read/write epochs inside
//! that callback, while references to node payloads remain tied to the access
//! borrow that produced them.

use std::marker::PhantomData;
use std::sync::{Arc, RwLock, RwLockWriteGuard};

use pyrat::GameState;

use crate::node::{Edge, LowNode, OwnerToken, SharedNode};
use crate::observer::{Invariant, NodeHandle};
use crate::tree::MCGSTree;
use crate::EvalResult;

/// An owned child-edge snapshot.
///
/// The edge itself remains borrowed only while [`ExclusiveAccess::child`] is
/// running. Rewards are copied and the child is cloned into a branded handle,
/// so callers never retain topology references across a mutation epoch.
pub(crate) struct OwnedChild<'session> {
    child: NodeHandle<'session>,
    #[cfg(test)]
    r1: f32,
    #[cfg(test)]
    r2: f32,
}

impl<'session> OwnedChild<'session> {
    pub(crate) fn into_child(self) -> NodeHandle<'session> {
        self.child
    }

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (NodeHandle<'session>, f32, f32) {
        (self.child, self.r1, self.r2)
    }
}

/// Copied facts needed for one leaf-to-root backup step.
///
/// Keeping this snapshot inside the capability lets the hot path validate and
/// read each participating node once, without retaining either payload
/// reference across the later parent mutation epoch.
pub(crate) struct BackupSnapshot {
    pub(crate) r1: f32,
    pub(crate) r2: f32,
    pub(crate) child_visits: u32,
    pub(crate) child_v1: f32,
    pub(crate) child_v2: f32,
    pub(crate) child_num_parents: u16,
    pub(crate) edge_visits: u32,
    pub(crate) edge_q1: f32,
    pub(crate) edge_q2: f32,
}

/// Zero-lock capability for the normal single-worker search path.
///
/// Construction is sealed behind [`MCGSTree::with_exclusive`]. The unique
/// borrow of the whole tree makes ordinary `LowNode` fields safe to access
/// without a runtime lock. Every `UnsafeCell` dereference also validates the
/// node's immutable per-tree owner token.
pub(crate) struct ExclusiveAccess<'tree, 'session> {
    tree: &'tree mut MCGSTree,
    owner: Arc<OwnerToken>,
    brand: Invariant<'session>,
}

/// Coarse graph gate for one search call.
///
/// Chunk 4B deliberately exposes only exclusive write epochs. Owned branded
/// handles may cross those epochs, but neither the guard nor any borrowed node
/// payload can. A future multi-worker search can add a measured shared gather
/// capability without changing the session brand or the owned work protocol.
#[allow(dead_code)]
pub(crate) struct SearchSession<'tree, 'session> {
    gate: RwLock<&'tree mut MCGSTree>,
    brand: Invariant<'session>,
}

#[allow(dead_code)]
impl<'tree, 'session> SearchSession<'tree, 'session> {
    pub(crate) fn new(tree: &'tree mut MCGSTree) -> Self {
        Self {
            gate: RwLock::new(tree),
            brand: PhantomData,
        }
    }

    /// Acquire a fresh exclusive graph epoch.
    ///
    /// Poisoning is treated as an internal invariant failure. Backend errors
    /// and panics happen without a graph guard, so their cleanup path does not
    /// poison this lock.
    pub(crate) fn write(&self) -> WriteEpoch<'_, 'tree, 'session> {
        WriteEpoch {
            guard: self.gate.write().expect("MCGS search graph gate poisoned"),
            brand: PhantomData,
        }
    }

    #[cfg(test)]
    pub(crate) fn try_write_for_test(&self) -> bool {
        self.gate.try_write().is_ok()
    }
}

/// One short exclusive mutation epoch inside a [`SearchSession`].
#[allow(dead_code)]
pub(crate) struct WriteEpoch<'guard, 'tree, 'session> {
    guard: RwLockWriteGuard<'guard, &'tree mut MCGSTree>,
    brand: Invariant<'session>,
}

#[allow(dead_code)]
impl<'tree, 'session> WriteEpoch<'_, 'tree, 'session> {
    /// Derive the existing zero-lock capability for exactly this guard borrow.
    pub(crate) fn access(&mut self) -> ExclusiveAccess<'_, 'session> {
        ExclusiveAccess::new(&mut self.guard)
    }
}

impl<'tree, 'session> ExclusiveAccess<'tree, 'session> {
    pub(crate) fn new(tree: &'tree mut MCGSTree) -> Self {
        let owner = Arc::clone(tree.owner());
        Self {
            tree,
            owner,
            brand: PhantomData,
        }
    }

    /// Return an owned, session-bound handle to the current root.
    pub(crate) fn root(&self) -> NodeHandle<'session> {
        NodeHandle::new(Arc::clone(self.tree.root()))
    }

    /// Borrow one node payload for an immutable access epoch.
    #[inline]
    pub(crate) fn node<'access>(
        &'access self,
        handle: &'access NodeHandle<'session>,
    ) -> &'access LowNode {
        assert!(
            Arc::ptr_eq(&self.owner, handle.inner().owner()),
            "node handle belongs to a different MCGS tree"
        );
        // SAFETY: the owner check ties the node to this tree, and the returned
        // reference borrows both the handle (liveness) and this access epoch.
        // `ExclusiveAccess` owns the unique borrow of the whole tree.
        unsafe { &*handle.inner().low_node_ptr() }
    }

    /// Borrow one node payload for a mutable access epoch.
    #[inline]
    pub(crate) fn node_mut<'access>(
        &'access mut self,
        handle: &'access NodeHandle<'session>,
    ) -> &'access mut LowNode {
        assert!(
            Arc::ptr_eq(&self.owner, handle.inner().owner()),
            "node handle belongs to a different MCGS tree"
        );
        // SAFETY: the owner check ties the node to this tree. `&mut self`
        // serializes payload access through the unique whole-tree capability,
        // and the handle borrow keeps the allocation alive for the result.
        unsafe { &mut *handle.inner().low_node_ptr() }
    }

    /// Return whether two checked handles identify the same shared node.
    pub(crate) fn same_node(
        &self,
        first: &NodeHandle<'session>,
        second: &NodeHandle<'session>,
    ) -> bool {
        self.assert_owned(first);
        self.assert_owned(second);
        Arc::ptr_eq(first.arc(), second.arc())
    }

    /// Read the GC-facing atomic parent count after validating tree ownership.
    pub(crate) fn num_parents(&self, handle: &NodeHandle<'session>) -> u16 {
        self.assert_owned(handle);
        handle.inner().num_parents()
    }

    pub(crate) fn node_count(&self) -> u32 {
        self.tree.node_count()
    }

    /// Copy one child edge into an owned, session-bound snapshot.
    pub(crate) fn child(
        &self,
        parent: &NodeHandle<'session>,
        p1_outcome: u8,
        p2_outcome: u8,
    ) -> Option<OwnedChild<'session>> {
        let edge = self.node(parent).find_child(p1_outcome, p2_outcome)?;
        let child = NodeHandle::new(Arc::clone(edge.low_node()));
        self.assert_owned(&child);
        Some(OwnedChild {
            child,
            #[cfg(test)]
            r1: edge.r1(),
            #[cfg(test)]
            r2: edge.r2(),
        })
    }

    /// Copy all immutable inputs for one backup step in a single read epoch.
    #[inline]
    pub(crate) fn backup_snapshot(
        &self,
        parent: &NodeHandle<'session>,
        p1_outcome: u8,
        p2_outcome: u8,
    ) -> BackupSnapshot {
        let parent_low = self.node(parent);
        let edge = parent_low
            .find_child(p1_outcome, p2_outcome)
            .expect("backup: edge must exist");
        let child = edge.low_node();
        self.assert_shared_node_owned(child);
        // SAFETY: the immediately preceding owner check ties the child to this
        // tree. The parent edge keeps it alive throughout this read epoch, and
        // `&self` excludes mutation through the exclusive capability.
        let child_low = unsafe { &*child.low_node_ptr() };
        let i = p1_outcome as usize;
        let j = p2_outcome as usize;

        BackupSnapshot {
            r1: edge.r1(),
            r2: edge.r2(),
            child_visits: child_low.total_visits(),
            child_v1: child_low.v1(),
            child_v2: child_low.v2(),
            child_num_parents: child.num_parents(),
            edge_visits: parent_low.edge_visits(i, j),
            edge_q1: parent_low.edge_q_p1(i, j),
            edge_q2: parent_low.edge_q_p2(i, j),
        }
    }

    /// Populate a shell node after evaluation, or mark it terminal.
    pub(crate) fn populate_node(
        &mut self,
        handle: &NodeHandle<'session>,
        eval_result: Option<&EvalResult>,
    ) {
        let low = self.node_mut(handle);
        debug_assert!(
            low.total_visits() == 0,
            "populate_node: node already has {} visits",
            low.total_visits()
        );

        match eval_result {
            Some(result) => low.set_prior(result.policy_p1, result.policy_p2),
            None => low.set_terminal(),
        }
    }

    /// Find an existing child or publish one canonical edge through the tree's
    /// transposition table.
    ///
    /// The game must already represent the child position. Node allocation,
    /// TT publication, parent topology mutation, and node-count maintenance are
    /// deliberately sealed into this capability operation.
    pub(crate) fn find_or_create_child(
        &mut self,
        parent: &NodeHandle<'session>,
        i: u8,
        j: u8,
        game: &GameState,
        r1: f32,
        r2: f32,
    ) -> NodeHandle<'session> {
        if let Some(existing) = self.child(parent, i, j) {
            return existing.into_child();
        }

        let hash = game.state_hash();
        if let Some(existing) = self.tree.tt().lookup(hash) {
            self.assert_shared_node_owned(&existing);
            let child = NodeHandle::new(existing);
            let edge = Box::new(Edge::new(Arc::clone(child.arc()), (i, j), r1, r2));
            self.node_mut(parent).prepend_child(edge);
            return child;
        }

        let eff_p1 = game.effective_actions_p1();
        let eff_p2 = game.effective_actions_p2();
        let mut child_node = LowNode::new_shell(eff_p1, eff_p2);
        child_node.set_value_scale(game.cheese.remaining_cheese().max(1) as f32);

        let child = Arc::new(SharedNode::with_owner(
            child_node,
            Arc::clone(self.tree.owner()),
        ));
        let inserted = self.tree.tt_mut().insert(hash, &child);
        debug_assert!(inserted, "fresh child TT insertion must succeed");

        let edge = Box::new(Edge::new(Arc::clone(&child), (i, j), r1, r2));
        self.node_mut(parent).prepend_child(edge);
        self.tree.increment_node_count();
        NodeHandle::new(child)
    }

    /// Advance and prune the root while the tree is exclusively borrowed.
    pub(crate) fn advance_root(&mut self, game: &GameState, p1_action: u8, p2_action: u8) {
        let old_root = self.root();
        let (i, j) = {
            let old_low = self.node(&old_root);
            (
                old_low.p1_action_to_outcome_idx(p1_action),
                old_low.p2_action_to_outcome_idx(p2_action),
            )
        };

        let mut cursor = self.node_mut(&old_root).take_first_child();
        let mut new_root = None;
        while let Some(mut edge) = cursor {
            cursor = edge.take_next_sibling();
            let (edge_i, edge_j) = edge.parent_outcome();
            if edge_i == i && edge_j == j && new_root.is_none() {
                let child = Arc::clone(edge.low_node());
                self.assert_shared_node_owned(&child);
                new_root = Some(child);
                // Dropping the selected parent edge updates num_parents.
                drop(edge);
            } else {
                crate::gc::queue(edge);
            }
        }

        crate::gc::flush();
        let root = match new_root {
            Some(root) => root,
            None => self.tree.create_root_for_owner(game),
        };
        self.tree.install_root(root);
        // Do not let this operation's branded liveness handle keep the pruned
        // root visible to the TT recount.
        drop(old_root);
        self.tree.recount_nodes();
    }

    /// Allocate an owner-consistent detached node for unit-test fixtures.
    ///
    /// The node is deliberately not inserted into the TT or reflected in the
    /// production node count. Tests must connect it explicitly when topology
    /// ownership is part of the fixture.
    #[cfg(test)]
    pub(crate) fn test_node(&mut self, low: LowNode) -> NodeHandle<'session> {
        NodeHandle::new(Arc::new(SharedNode::with_owner(
            low,
            Arc::clone(self.tree.owner()),
        )))
    }

    /// Connect two owner-checked fixture nodes through the real edge topology.
    #[cfg(test)]
    pub(crate) fn test_connect(
        &mut self,
        parent: &NodeHandle<'session>,
        child: &NodeHandle<'session>,
        parent_outcome: (u8, u8),
        r1: f32,
        r2: f32,
    ) {
        self.assert_owned(parent);
        self.assert_owned(child);
        let edge = Box::new(Edge::new(Arc::clone(child.arc()), parent_outcome, r1, r2));
        self.node_mut(parent).prepend_child(edge);
    }

    /// Detach a fixture node's full child chain for GC ownership tests.
    #[cfg(test)]
    pub(crate) fn test_detach_children(
        &mut self,
        parent: &NodeHandle<'session>,
    ) -> Option<Box<Edge>> {
        self.node_mut(parent).take_first_child()
    }

    /// Insert an owner-checked fixture node into this tree's TT.
    #[cfg(test)]
    pub(crate) fn test_insert_tt(&mut self, hash: u64, handle: &NodeHandle<'session>) -> bool {
        self.assert_owned(handle);
        self.tree.tt_mut().insert(hash, handle.arc())
    }

    /// Resolve a fixture TT entry into this session after checking ownership.
    #[cfg(test)]
    pub(crate) fn test_tt_lookup(&self, hash: u64) -> Option<NodeHandle<'session>> {
        let node = self.tree.tt().lookup(hash)?;
        self.assert_shared_node_owned(&node);
        Some(NodeHandle::new(node))
    }

    /// Replace the fixture root with an owner-checked node.
    #[cfg(test)]
    pub(crate) fn test_install_root(&mut self, handle: &NodeHandle<'session>) {
        self.assert_owned(handle);
        self.tree.install_root(Arc::clone(handle.arc()));
    }

    /// Clone all child identities for capability-backed DAG invariant tests.
    #[cfg(test)]
    pub(crate) fn test_children(&self, parent: &NodeHandle<'session>) -> Vec<NodeHandle<'session>> {
        let mut children = Vec::new();
        let mut edge = self.node(parent).first_child();
        while let Some(current) = edge {
            let child = NodeHandle::new(Arc::clone(current.low_node()));
            self.assert_owned(&child);
            children.push(child);
            edge = current.next_sibling();
        }
        children
    }

    /// Stable allocation identity for deduplicating a test-only DAG walk.
    #[cfg(test)]
    pub(crate) fn test_node_id(&self, handle: &NodeHandle<'session>) -> usize {
        self.assert_owned(handle);
        Arc::as_ptr(handle.arc()) as usize
    }

    fn assert_owned(&self, handle: &NodeHandle<'session>) {
        self.assert_shared_node_owned(handle.inner());
    }

    fn assert_shared_node_owned(&self, node: &SharedNode) {
        assert!(
            Arc::ptr_eq(&self.owner, node.owner()),
            "node handle belongs to a different MCGS tree"
        );
    }
}

impl MCGSTree {
    /// Owner-checked read seam used by the public observer capability.
    ///
    /// `&self` prevents mutable tree access for the duration, and the HRTB
    /// callback prevents the borrowed payload from escaping the observation
    /// epoch.
    pub(crate) fn with_observed_node<'session, R>(
        &self,
        handle: &NodeHandle<'session>,
        inspect: impl for<'node> FnOnce(&'node LowNode) -> R,
    ) -> R {
        assert!(
            Arc::ptr_eq(self.owner(), handle.inner().owner()),
            "node handle belongs to a different MCGS tree"
        );
        // SAFETY: `&self` excludes an exclusive search/root mutation, the owner
        // check ties the live handle to this tree, and the callback cannot let
        // the payload reference escape its observation epoch.
        let node = unsafe { &*handle.inner().low_node_ptr() };
        inspect(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::OwnerToken;
    use pyrat::{Coordinates, GameBuilder};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn game() -> GameState {
        GameBuilder::new(3, 3)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 2))
            .with_custom_cheese(vec![Coordinates::new(1, 1)])
            .with_max_turns(10)
            .build()
            .create(None)
            .unwrap()
    }

    #[test]
    fn foreign_owner_is_rejected_before_read_or_write() {
        let mut tree = MCGSTree::new(&game());
        tree.with_exclusive(|mut access| {
            let foreign = NodeHandle::new(Arc::new(SharedNode::with_owner(
                LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
                Arc::new(OwnerToken),
            )));

            let read = catch_unwind(AssertUnwindSafe(|| {
                let _ = access.node(&foreign).total_visits();
            }));
            assert!(read.is_err(), "foreign-owner read must panic");

            let write = catch_unwind(AssertUnwindSafe(|| {
                access.node_mut(&foreign).set_terminal();
            }));
            assert!(write.is_err(), "foreign-owner write must panic");
        });
    }
}
