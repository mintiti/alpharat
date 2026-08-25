//! Compiling prototype for a persistent MCGS tree with branded access sessions.
//!
//! The tree itself is deliberately unbranded so it can persist across turns.
//! Each access callback instead receives a fresh invariant session brand:
//!
//! - opaque `NodeHandle<'session>` values own an `Arc` and cross lock epochs;
//! - borrowed `NodeView` and `EdgeView` values remain guard-bound;
//! - sequential access is a zero-lock `&mut Tree` capability;
//! - parallel search adds one search-scoped coarse `RwLock`;
//! - observers return owned summaries and traverse PVs in short view epochs;
//! - exclusive atomics use `get_mut`, while shared access uses CAS/RMW;
//! - productive-budget leases return unused capacity without overshooting.

#![allow(unexpected_cfgs)]

mod model {
    use std::cell::UnsafeCell;
    use std::marker::PhantomData;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

    /// Invariance prevents a handle from being shortened or retagged into a
    /// nested access session's lifetime.
    type Invariant<'session> = PhantomData<fn(&'session mut ()) -> &'session mut ()>;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct OutcomeSummary {
        pub(super) terminal: bool,
        pub(super) visits: u32,
        pub(super) value_sum: i32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct Rewards {
        pub(super) player_one: i16,
        pub(super) player_two: i16,
    }

    struct Edge {
        rewards: Rewards,
        child: Arc<NodeInner>,
    }

    /// Opaque tree identity. Production can validate ownership with one pointer
    /// comparison instead of retaining or scanning a node registry.
    struct OwnerToken;

    struct NodeData {
        label: u32,
        outcome: OutcomeSummary,
        children: Vec<Edge>,
        reservations: AtomicU32,
    }

    /// Node payloads are behind `UnsafeCell` because owned `Arc` handles must
    /// remain alive while a later write guard mutates the payload. The only raw
    /// dereference functions are private to this module.
    struct NodeInner {
        owner: Arc<OwnerToken>,
        data: UnsafeCell<NodeData>,
    }

    impl NodeInner {
        fn new(
            owner: Arc<OwnerToken>,
            label: u32,
            outcome: OutcomeSummary,
            children: Vec<Edge>,
        ) -> Self {
            Self {
                owner,
                data: UnsafeCell::new(NodeData {
                    label,
                    outcome,
                    children,
                    reservations: AtomicU32::new(0),
                }),
            }
        }

        /// Producing the pointer is safe; capability call sites validate the
        /// owner and establish the read/write epoch before dereferencing it.
        fn data_ptr(&self) -> *mut NodeData {
            self.data.get()
        }
    }

    // SAFETY: payload dereferences are sealed behind this module's branded
    // capabilities. Shared views exist only under a search read guard (or an
    // immutable whole-tree observer), and mutable views exist only under the
    // corresponding write guard or an exclusive `&mut PersistentTree`.
    unsafe impl Sync for NodeInner {}

    /// An owned liveness token. Its constructor and raw `Arc` are intentionally
    /// private: safe callers can receive handles only from a capability carrying
    /// the same fresh invariant session brand.
    #[derive(Clone)]
    pub(super) struct NodeHandle<'session> {
        node: Arc<NodeInner>,
        brand: Invariant<'session>,
    }

    impl<'session> NodeHandle<'session> {
        fn new(node: Arc<NodeInner>) -> Self {
            Self {
                node,
                brand: PhantomData,
            }
        }

        fn inner(&self) -> &NodeInner {
            &self.node
        }
    }

    #[derive(Clone, Copy)]
    pub(super) struct NodeView<'guard, 'session> {
        data: &'guard NodeData,
        brand: Invariant<'session>,
    }

    impl<'guard, 'session> NodeView<'guard, 'session> {
        pub(super) fn label(self) -> u32 {
            self.data.label
        }

        pub(super) fn outcome(self) -> OutcomeSummary {
            self.data.outcome
        }

        pub(super) fn reservations(self) -> u32 {
            self.data.reservations.load(Ordering::Relaxed)
        }

        pub(super) fn edge(self, index: usize) -> Option<EdgeView<'guard, 'session>> {
            let data: &'guard NodeData = self.data;
            data.children.get(index).map(|edge| EdgeView {
                edge,
                brand: PhantomData,
            })
        }

        /// Child lookup copies rewards and clones an owned handle. Neither the
        /// borrowed node view nor edge view escapes the current guard epoch.
        pub(super) fn child(self, index: usize) -> Option<OwnedChild<'session>> {
            self.edge(index).map(EdgeView::to_owned)
        }
    }

    #[derive(Clone, Copy)]
    pub(super) struct EdgeView<'guard, 'session> {
        edge: &'guard Edge,
        brand: Invariant<'session>,
    }

    impl<'guard, 'session> EdgeView<'guard, 'session> {
        fn to_owned(self) -> OwnedChild<'session> {
            OwnedChild {
                rewards: self.edge.rewards,
                child: NodeHandle::new(Arc::clone(&self.edge.child)),
            }
        }
    }

    pub(super) struct OwnedChild<'session> {
        pub(super) rewards: Rewards,
        pub(super) child: NodeHandle<'session>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct PvStep {
        pub(super) label: u32,
        pub(super) outcome: OutcomeSummary,
        pub(super) incoming_rewards: Option<Rewards>,
    }

    /// The persistent object has no generative lifetime parameter.
    pub(super) struct PersistentTree {
        owner: Arc<OwnerToken>,
        root: Arc<NodeInner>,
    }

    impl PersistentTree {
        pub(super) fn fixture() -> Self {
            let owner = Arc::new(OwnerToken);
            let leaf = Arc::new(NodeInner::new(
                Arc::clone(&owner),
                30,
                OutcomeSummary {
                    terminal: true,
                    visits: 3,
                    value_sum: 9,
                },
                Vec::new(),
            ));
            let middle = Arc::new(NodeInner::new(
                Arc::clone(&owner),
                20,
                OutcomeSummary {
                    terminal: false,
                    visits: 5,
                    value_sum: 8,
                },
                vec![Edge {
                    rewards: Rewards {
                        player_one: 2,
                        player_two: -2,
                    },
                    child: Arc::clone(&leaf),
                }],
            ));
            let root = Arc::new(NodeInner::new(
                Arc::clone(&owner),
                10,
                OutcomeSummary {
                    terminal: false,
                    visits: 8,
                    value_sum: 11,
                },
                vec![Edge {
                    rewards: Rewards {
                        player_one: 1,
                        player_two: -1,
                    },
                    child: Arc::clone(&middle),
                }],
            ));

            Self { owner, root }
        }

        fn owns(&self, node: &NodeInner) -> bool {
            Arc::ptr_eq(&self.owner, &node.owner)
        }

        pub(super) fn with_exclusive<'tree, R>(
            &'tree mut self,
            access: impl for<'session> FnOnce(ExclusiveAccess<'tree, 'session>) -> R,
        ) -> R {
            access(ExclusiveAccess {
                tree: self,
                brand: PhantomData,
            })
        }

        pub(super) fn with_search_session<'tree, R>(
            &'tree mut self,
            search: impl for<'session> FnOnce(SearchSession<'tree, 'session>) -> R,
        ) -> R {
            search(SearchSession {
                gate: RwLock::new(self),
                brand: PhantomData,
            })
        }

        /// The observer gets its own fresh handle brand. The higher-ranked
        /// callback prevents handles from escaping, while owned snapshots can.
        pub(super) fn observe<'tree, R>(
            &'tree self,
            observer: impl for<'session> FnOnce(Observer<'tree, 'session>) -> R,
        ) -> R {
            observer(Observer {
                tree: self,
                brand: PhantomData,
            })
        }
    }

    /// Zero-lock capability for the normal sequential path.
    pub(super) struct ExclusiveAccess<'tree, 'session> {
        tree: &'tree mut PersistentTree,
        brand: Invariant<'session>,
    }

    impl<'tree, 'session> ExclusiveAccess<'tree, 'session> {
        pub(super) fn root_handle(&self) -> NodeHandle<'session> {
            NodeHandle::new(Arc::clone(&self.tree.root))
        }

        fn data<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> &'access NodeData {
            assert!(
                self.tree.owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: the immutable access borrow covers the returned view, and
            // this capability owns the unique borrow of the whole tree.
            unsafe { &*handle.inner().data_ptr() }
        }

        fn data_mut<'access>(
            &'access mut self,
            handle: &'access NodeHandle<'session>,
        ) -> &'access mut NodeData {
            assert!(
                self.tree.owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: `&mut self` serializes all payload access through this
            // unique whole-tree capability for the returned lifetime.
            unsafe { &mut *handle.inner().data_ptr() }
        }

        pub(super) fn view<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> NodeView<'access, 'session> {
            NodeView {
                data: self.data(handle),
                brand: PhantomData,
            }
        }

        pub(super) fn set_label(&mut self, handle: &NodeHandle<'session>, label: u32) {
            self.data_mut(handle).label = label;
        }

        pub(super) fn reserve(&mut self, handle: &NodeHandle<'session>, count: u32) {
            let reservations = self.data_mut(handle).reservations.get_mut();
            *reservations = reservations
                .checked_add(count)
                .expect("exclusive reservation overflow");
        }

        pub(super) fn release(&mut self, handle: &NodeHandle<'session>, count: u32) {
            let reservations = self.data_mut(handle).reservations.get_mut();
            *reservations = reservations
                .checked_sub(count)
                .expect("exclusive reservation underflow");
        }
    }

    /// Coarse gate that exists only while one parallel search call is active.
    pub(super) struct SearchSession<'tree, 'session> {
        gate: RwLock<&'tree mut PersistentTree>,
        brand: Invariant<'session>,
    }

    impl<'tree, 'session> SearchSession<'tree, 'session> {
        pub(super) fn read(&self) -> ReadAccess<'_, 'tree, 'session> {
            ReadAccess {
                guard: self.gate.read().expect("prototype read lock poisoned"),
                brand: PhantomData,
            }
        }

        pub(super) fn write(&self) -> WriteAccess<'_, 'tree, 'session> {
            WriteAccess {
                guard: self.gate.write().expect("prototype write lock poisoned"),
                brand: PhantomData,
            }
        }

        /// Each loop iteration owns only scalars and the next `Arc` handle when
        /// its read guard drops. Writers may run between these short epochs.
        pub(super) fn principal_variation(&self, max_depth: usize) -> Vec<PvStep> {
            let mut current = {
                let read = self.read();
                read.root_handle()
            };
            let mut incoming_rewards = None;
            let mut pv = Vec::new();

            for _ in 0..max_depth {
                let (label, outcome, next) = {
                    let read = self.read();
                    let view = read.view(&current);
                    (view.label(), view.outcome(), view.child(0))
                };
                pv.push(PvStep {
                    label,
                    outcome,
                    incoming_rewards,
                });
                let Some(next) = next else {
                    break;
                };
                incoming_rewards = Some(next.rewards);
                current = next.child;
            }
            pv
        }
    }

    pub(super) struct ReadAccess<'guard, 'tree, 'session> {
        guard: RwLockReadGuard<'guard, &'tree mut PersistentTree>,
        brand: Invariant<'session>,
    }

    impl<'guard, 'tree, 'session> ReadAccess<'guard, 'tree, 'session> {
        fn tree(&self) -> &PersistentTree {
            &self.guard
        }

        pub(super) fn root_handle(&self) -> NodeHandle<'session> {
            NodeHandle::new(Arc::clone(&self.tree().root))
        }

        fn data<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> &'access NodeData {
            assert!(
                self.tree().owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: the returned shared payload reference borrows this read
            // access, so the write guard cannot be acquired while it is live.
            unsafe { &*handle.inner().data_ptr() }
        }

        pub(super) fn view<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> NodeView<'access, 'session> {
            NodeView {
                data: self.data(handle),
                brand: PhantomData,
            }
        }

        pub(super) fn try_claim_fresh(&self, handle: &NodeHandle<'session>) -> bool {
            self.data(handle)
                .reservations
                .compare_exchange(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        }

        pub(super) fn reserve(&self, handle: &NodeHandle<'session>, count: u32) -> u32 {
            self.data(handle)
                .reservations
                .fetch_add(count, Ordering::Relaxed)
        }

        pub(super) fn release(&self, handle: &NodeHandle<'session>, count: u32) -> u32 {
            self.data(handle)
                .reservations
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_sub(count)
                })
                .expect("shared reservation underflow")
        }
    }

    pub(super) struct WriteAccess<'guard, 'tree, 'session> {
        guard: RwLockWriteGuard<'guard, &'tree mut PersistentTree>,
        brand: Invariant<'session>,
    }

    impl<'guard, 'tree, 'session> WriteAccess<'guard, 'tree, 'session> {
        fn tree(&self) -> &PersistentTree {
            &self.guard
        }

        fn data<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> &'access NodeData {
            assert!(
                self.tree().owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: a write guard may also produce shared views, tied to its
            // borrow; Rust prevents mutable reborrowing while such a view lives.
            unsafe { &*handle.inner().data_ptr() }
        }

        fn data_mut<'access>(
            &'access mut self,
            handle: &'access NodeHandle<'session>,
        ) -> &'access mut NodeData {
            assert!(
                self.tree().owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: the write guard excludes every read/write guard, and the
            // returned reference borrows the guard mutably for its lifetime.
            unsafe { &mut *handle.inner().data_ptr() }
        }

        pub(super) fn view<'access>(
            &'access self,
            handle: &'access NodeHandle<'session>,
        ) -> NodeView<'access, 'session> {
            NodeView {
                data: self.data(handle),
                brand: PhantomData,
            }
        }

        pub(super) fn set_label(&mut self, handle: &NodeHandle<'session>, label: u32) {
            self.data_mut(handle).label = label;
        }

        pub(super) fn set_outcome(
            &mut self,
            handle: &NodeHandle<'session>,
            outcome: OutcomeSummary,
        ) {
            self.data_mut(handle).outcome = outcome;
        }

        pub(super) fn reset_reservations(&mut self, handle: &NodeHandle<'session>) {
            *self.data_mut(handle).reservations.get_mut() = 0;
        }
    }

    pub(super) struct Observer<'tree, 'session> {
        tree: &'tree PersistentTree,
        brand: Invariant<'session>,
    }

    impl<'tree, 'session> Observer<'tree, 'session> {
        pub(super) fn root_handle(&self) -> NodeHandle<'session> {
            NodeHandle::new(Arc::clone(&self.tree.root))
        }

        fn data<'view>(&'view self, handle: &'view NodeHandle<'session>) -> &'view NodeData {
            assert!(
                self.tree.owns(handle.inner()),
                "handle belongs to this tree"
            );
            // SAFETY: the whole persistent tree is immutably borrowed for this
            // observer callback, excluding exclusive/search-session creation.
            unsafe { &*handle.inner().data_ptr() }
        }

        /// The higher-ranked view callback makes each borrow epoch
        /// non-escaping, while `R` may contain owned handles of this session.
        pub(super) fn with_view<R>(
            &self,
            handle: &NodeHandle<'session>,
            inspect: impl for<'view> FnOnce(NodeView<'view, 'session>) -> R,
        ) -> R {
            inspect(NodeView {
                data: self.data(handle),
                brand: PhantomData,
            })
        }

        pub(super) fn principal_variation(&self, max_depth: usize) -> Vec<PvStep> {
            let mut current = self.root_handle();
            let mut incoming_rewards = None;
            let mut pv = Vec::new();

            for _ in 0..max_depth {
                let (label, outcome, next) = self.with_view(&current, |view| {
                    (view.label(), view.outcome(), view.child(0))
                });
                pv.push(PvStep {
                    label,
                    outcome,
                    incoming_rewards,
                });
                let Some(next) = next else {
                    break;
                };
                incoming_rewards = Some(next.rewards);
                current = next.child;
            }
            pv
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct BudgetSnapshot {
        pub(super) initial: u32,
        pub(super) remaining: u32,
        pub(super) outstanding: u32,
    }

    #[derive(Debug)]
    struct BudgetState {
        initial: u32,
        remaining: u32,
        outstanding: u32,
    }

    /// Batch-level coordinator. The graph gate is not involved in leasing.
    pub(super) struct ProductiveBudget {
        state: Mutex<BudgetState>,
    }

    impl ProductiveBudget {
        pub(super) fn new(target: u32) -> Self {
            assert!(target > 0, "prototype budget target must be positive");
            Self {
                state: Mutex::new(BudgetState {
                    initial: target,
                    remaining: target,
                    outstanding: 0,
                }),
            }
        }

        pub(super) fn lease(&self, max: u32) -> Option<BudgetLease<'_>> {
            assert!(max > 0, "lease size must be positive");
            let mut state = self.state.lock().expect("prototype budget poisoned");
            let available = state.remaining - state.outstanding;
            let reserved = available.min(max);
            if reserved == 0 {
                return None;
            }
            state.outstanding += reserved;
            debug_assert!(state.outstanding <= state.remaining);
            Some(BudgetLease {
                budget: self,
                reserved,
                finished: false,
            })
        }

        pub(super) fn snapshot(&self) -> BudgetSnapshot {
            let state = self.state.lock().expect("prototype budget poisoned");
            BudgetSnapshot {
                initial: state.initial,
                remaining: state.remaining,
                outstanding: state.outstanding,
            }
        }
    }

    pub(super) struct BudgetLease<'budget> {
        budget: &'budget ProductiveBudget,
        reserved: u32,
        finished: bool,
    }

    impl BudgetLease<'_> {
        pub(super) fn reserved(&self) -> u32 {
            self.reserved
        }

        pub(super) fn complete(mut self, produced: u32) -> BudgetCompletion {
            assert!(produced <= self.reserved, "batch exceeded its lease");
            let charged = produced;

            let mut state = self.budget.state.lock().expect("prototype budget poisoned");
            state.outstanding -= self.reserved;
            state.remaining -= charged;
            debug_assert!(state.outstanding <= state.remaining);
            debug_assert!(state.remaining <= state.initial);

            self.finished = true;
            BudgetCompletion {
                reserved: self.reserved,
                charged,
                returned: self.reserved - charged,
            }
        }
    }

    impl Drop for BudgetLease<'_> {
        fn drop(&mut self) {
            if self.finished {
                return;
            }
            let mut state = self.budget.state.lock().expect("prototype budget poisoned");
            state.outstanding -= self.reserved;
            debug_assert!(state.outstanding <= state.remaining);
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct BudgetCompletion {
        pub(super) reserved: u32,
        pub(super) charged: u32,
        pub(super) returned: u32,
    }
}

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::thread;
use std::time::Duration;

use model::*;

#[test]
fn persistent_tree_and_session_handle_cross_read_write_epochs() {
    let mut tree = PersistentTree::fixture();
    let before = tree.observe(|observer| observer.principal_variation(8));
    assert_eq!(
        before.iter().map(|step| step.label).collect::<Vec<_>>(),
        [10, 20, 30]
    );

    tree.with_search_session(|session| {
        let child = {
            let read = session.read();
            let root = read.root_handle();
            let root_view = read.view(&root);
            assert_eq!(root_view.outcome().visits, 8);
            let child = root_view.child(0).expect("fixture root child");
            assert_eq!(child.rewards.player_one, 1);
            child.child
        };

        // Deliberate no-guard interval: only the Arc-backed branded handle is
        // retained across the gather/inference boundary.
        thread::yield_now();

        {
            let mut write = session.write();
            write.set_label(&child, 21);
            write.set_outcome(
                &child,
                OutcomeSummary {
                    terminal: false,
                    visits: 6,
                    value_sum: 13,
                },
            );
            assert_eq!(write.view(&child).label(), 21);
        }

        {
            let read = session.read();
            let child_view = read.view(&child);
            assert_eq!(child_view.label(), 21);
            assert_eq!(child_view.outcome().value_sum, 13);
        }

        assert_eq!(
            session
                .principal_variation(8)
                .iter()
                .map(|step| step.label)
                .collect::<Vec<_>>(),
            [10, 21, 30]
        );
    });

    // The unbranded tree remains usable in a later turn/session; only the old
    // session handles were confined to the callback.
    let after = tree.observe(|observer| observer.principal_variation(8));
    assert_eq!(
        after.iter().map(|step| step.label).collect::<Vec<_>>(),
        [10, 21, 30]
    );
    assert_eq!(before[1].label, 20);
    assert_eq!(after[1].outcome.value_sum, 13);
    assert_eq!(after[1].incoming_rewards.unwrap().player_one, 1);
    assert_eq!(after[2].incoming_rewards.unwrap().player_two, -2);
}

#[test]
fn exclusive_access_is_lock_free_and_uses_atomic_get_mut() {
    let mut tree = PersistentTree::fixture();
    tree.with_exclusive(|mut access| {
        assert_eq!(
            std::mem::size_of_val(&access),
            std::mem::size_of::<&mut PersistentTree>()
        );

        let root = access.root_handle();
        assert_eq!(access.view(&root).label(), 10);
        access.set_label(&root, 11);
        access.reserve(&root, 3);
        assert_eq!(access.view(&root).reservations(), 3);
        access.release(&root, 2);
        assert_eq!(access.view(&root).reservations(), 1);
    });

    assert_eq!(
        tree.observe(|observer| observer.principal_variation(1))[0].label,
        11
    );
}

#[test]
fn observer_returns_owned_outcomes_rewards_and_pv() {
    let mut tree = PersistentTree::fixture();
    let snapshot = tree.observe(|observer| observer.principal_variation(8));
    assert_eq!(snapshot.len(), 3);
    assert_eq!(
        snapshot[2].outcome,
        OutcomeSummary {
            terminal: true,
            visits: 3,
            value_sum: 9,
        }
    );
    assert_eq!(
        snapshot[2].incoming_rewards,
        Some(Rewards {
            player_one: 2,
            player_two: -2,
        })
    );

    tree.with_exclusive(|mut access| {
        let root = access.root_handle();
        access.set_label(&root, 99);
    });
    assert_eq!(snapshot[0].label, 10);
    assert_eq!(
        tree.observe(|observer| observer.principal_variation(1))[0].label,
        99
    );
}

#[test]
fn shared_claim_and_rmw_are_atomic() {
    let mut tree = PersistentTree::fixture();
    tree.with_search_session(|session| {
        let node = {
            let read = session.read();
            read.root_handle()
        };
        let start = Arc::new(Barrier::new(5));
        let winners = AtomicU32::new(0);

        thread::scope(|scope| {
            for _ in 0..4 {
                let start = Arc::clone(&start);
                let session = &session;
                let node = node.clone();
                let winners = &winners;
                scope.spawn(move || {
                    let read = session.read();
                    start.wait();
                    if read.try_claim_fresh(&node) {
                        winners.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            start.wait();
        });

        assert_eq!(winners.load(Ordering::Relaxed), 1);
        {
            let read = session.read();
            assert_eq!(read.view(&node).reservations(), 1);
            assert_eq!(read.reserve(&node, 4), 1);
            assert_eq!(read.release(&node, 3), 5);
            assert_eq!(read.view(&node).reservations(), 2);
        }
        {
            let mut write = session.write();
            assert_eq!(write.view(&node).reservations(), 2);
            write.reset_reservations(&node);
        }
        assert_eq!(session.read().view(&node).reservations(), 0);
    });
}

#[test]
fn scoped_readers_overlap_and_writer_is_serialized() {
    let mut tree = PersistentTree::fixture();
    tree.with_search_session(|session| {
        let node = session.read().root_handle();
        let readers_ready = Arc::new(Barrier::new(3));
        let release_readers = Arc::new(Barrier::new(3));
        let (writer_started_tx, writer_started_rx) = mpsc::channel();
        let (writer_done_tx, writer_done_rx) = mpsc::channel();

        thread::scope(|scope| {
            for _ in 0..2 {
                let session = &session;
                let node = node.clone();
                let readers_ready = Arc::clone(&readers_ready);
                let release_readers = Arc::clone(&release_readers);
                scope.spawn(move || {
                    let read = session.read();
                    assert_eq!(read.view(&node).label(), 10);
                    readers_ready.wait();
                    release_readers.wait();
                    assert_eq!(read.view(&node).label(), 10);
                });
            }

            readers_ready.wait();
            let session = &session;
            let node = node.clone();
            scope.spawn(move || {
                writer_started_tx.send(()).unwrap();
                let mut write = session.write();
                write.set_label(&node, 18);
                writer_done_tx.send(()).unwrap();
            });

            writer_started_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("writer thread did not start");
            assert!(
                writer_done_rx
                    .recv_timeout(Duration::from_millis(25))
                    .is_err(),
                "writer acquired while read guards were live"
            );

            release_readers.wait();
            writer_done_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("writer did not acquire after readers exited");
        });

        assert_eq!(session.read().view(&node).label(), 18);
    });
}

#[test]
fn productive_budget_returns_unused_capacity_without_overshoot() {
    let budget = ProductiveBudget::new(10);
    let first = budget.lease(6).unwrap();
    let second = budget.lease(6).unwrap();
    assert_eq!(first.reserved(), 6);
    assert_eq!(second.reserved(), 4);
    assert!(budget.lease(1).is_none());

    let first_done = first.complete(2);
    assert_eq!(
        first_done,
        BudgetCompletion {
            reserved: 6,
            charged: 2,
            returned: 4,
        }
    );
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            initial: 10,
            remaining: 8,
            outstanding: 4,
        }
    );

    let returned = budget.lease(8).unwrap();
    assert_eq!(returned.reserved(), 4);
    assert!(budget.lease(1).is_none());

    assert_eq!(second.complete(4).charged, 4);
    assert_eq!(returned.complete(4).charged, 4);
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            initial: 10,
            remaining: 0,
            outstanding: 0,
        }
    );
}

#[test]
fn concurrent_budget_leases_return_capacity_and_never_overshoot() {
    let budget = ProductiveBudget::new(1_000);
    let charged = AtomicU32::new(0);

    thread::scope(|scope| {
        for _ in 0..8 {
            let budget = &budget;
            let charged = &charged;
            scope.spawn(move || loop {
                match budget.lease(7) {
                    Some(lease) => {
                        let produced = lease.reserved().saturating_sub(1).max(1);
                        let completion = lease.complete(produced);
                        charged.fetch_add(completion.charged, Ordering::Relaxed);
                    }
                    None => {
                        let snapshot = budget.snapshot();
                        if snapshot.remaining == 0 && snapshot.outstanding == 0 {
                            break;
                        }
                        thread::yield_now();
                    }
                }
            });
        }
    });

    assert_eq!(charged.load(Ordering::Relaxed), 1_000);
    assert_eq!(
        budget.snapshot(),
        BudgetSnapshot {
            initial: 1_000,
            remaining: 0,
            outstanding: 0,
        }
    );
}

#[test]
fn abandoned_budget_lease_returns_all_capacity() {
    let budget = ProductiveBudget::new(5);
    let lease = budget.lease(5).unwrap();
    assert_eq!(lease.reserved(), 5);
    drop(lease);

    assert_eq!(budget.snapshot().outstanding, 0);
    assert_eq!(budget.lease(5).unwrap().complete(5).charged, 5);
    assert_eq!(budget.snapshot().remaining, 0);
}

#[test]
fn zero_productive_work_returns_the_entire_lease() {
    let budget = ProductiveBudget::new(3);
    let completion = budget.lease(3).unwrap().complete(0);

    assert_eq!(
        completion,
        BudgetCompletion {
            reserved: 3,
            charged: 0,
            returned: 3,
        }
    );
    assert_eq!(budget.snapshot().remaining, 3);
    assert_eq!(budget.snapshot().outstanding, 0);
    assert_eq!(budget.lease(3).unwrap().complete(3).charged, 3);
}

// These cases are compiled separately by `access_model_compile_fail.rs`.
// Keeping them beside the positive model ensures the negative probes exercise
// the exact same API and lifetime signatures.

#[cfg(access_model_fail_view_escape)]
fn view_cannot_escape_its_read_guard() {
    let mut tree = PersistentTree::fixture();
    tree.with_search_session(|session| {
        let handle = session.read().root_handle();
        let escaped = {
            let read = session.read();
            read.view(&handle)
        };
        let _ = escaped.label();
    });
}

#[cfg(access_model_fail_handle_escape)]
fn handle_cannot_escape_its_generative_session() {
    let mut tree = PersistentTree::fixture();
    let _escaped = tree.with_search_session(|session| session.read().root_handle());
}

#[cfg(access_model_fail_wrong_tree)]
fn handle_cannot_mix_across_nested_tree_sessions() {
    let mut first = PersistentTree::fixture();
    let mut second = PersistentTree::fixture();

    first.with_search_session(|first_session| {
        let handle = first_session.read().root_handle();
        second.with_search_session(|second_session| {
            let read = second_session.read();
            let _ = read.view(&handle);
        });
    });
}

#[cfg(access_model_fail_exclusive_handle_escape)]
fn exclusive_handle_cannot_escape_its_generative_session() {
    let mut tree = PersistentTree::fixture();
    let _escaped = tree.with_exclusive(|access| access.root_handle());
}

#[cfg(access_model_fail_exclusive_view_escape)]
fn exclusive_view_cannot_escape_its_access_borrow() {
    let mut tree = PersistentTree::fixture();
    let _escaped = tree.with_exclusive(|access| {
        let handle = access.root_handle();
        access.view(&handle)
    });
}

#[cfg(access_model_fail_exclusive_alias)]
fn exclusive_read_prevents_overlapping_mutation() {
    let mut tree = PersistentTree::fixture();
    tree.with_exclusive(|mut access| {
        let handle = access.root_handle();
        let read = access.view(&handle);
        access.set_label(&handle, 41);
        let _ = read.label();
    });
}

#[cfg(access_model_fail_exclusive_wrong_tree)]
fn exclusive_handle_cannot_mix_across_nested_tree_sessions() {
    let mut first = PersistentTree::fixture();
    let mut second = PersistentTree::fixture();

    first.with_exclusive(|first_access| {
        let handle = first_access.root_handle();
        second.with_exclusive(|second_access| {
            let _ = second_access.view(&handle);
        });
    });
}
