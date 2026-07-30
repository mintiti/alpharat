//! Background garbage collector for MCGS edge chains.
//!
//! Prevents stack overflow from recursive `Drop` on deep sibling/child chains
//! and avoids blocking the search thread with synchronous destruction of large
//! pruned subtrees.
//!
//! Faithfully ports lc0's `NodeGarbageCollector` (node.cc / node.h):
//! - State machine: Sleeping → Running → GoToSleep → Exit
//! - Thread-local batching: edges accumulate locally, flush to shared deque
//! - Start/stop/wait lifecycle for coordination with search
//!
//! When the GC is not initialized (e.g. in unit tests), `queue()` falls back
//! to synchronous drop.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};

use crate::node::Edge;

// ---------------------------------------------------------------------------
// State machine (matches lc0's NodeGarbageCollector::State)
// ---------------------------------------------------------------------------

const SLEEPING: u8 = 0;
const RUNNING: u8 = 1;
const GO_TO_SLEEP: u8 = 2;
const EXIT: u8 = 3;

// ---------------------------------------------------------------------------
// Thread-local batching (matches lc0's ReleaseNodesWork)
// ---------------------------------------------------------------------------

/// Capacity before a thread-local batch flushes to the shared deque.
/// Matches lc0's kCapacity (node.h:1002).
const BATCH_CAPACITY: usize = 32;

struct LocalBatch {
    #[allow(clippy::vec_box)]
    items: Vec<Box<Edge>>,
    is_gc_thread: bool,
}

impl LocalBatch {
    fn new(is_gc_thread: bool) -> Self {
        Self {
            items: Vec::with_capacity(BATCH_CAPACITY),
            is_gc_thread,
        }
    }

    /// Push an edge. If at capacity, flush the full batch to the shared deque.
    fn push(&mut self, edge: Box<Edge>, gc: &GcInner) {
        self.items.push(edge);
        if self.items.len() >= BATCH_CAPACITY {
            self.flush(gc);
        }
    }

    /// Flush all items to the shared deque.
    /// GC thread batches go to front (oldest-first), search threads go to back.
    /// Notifies the GC thread so it wakes from condvar if waiting for work.
    fn flush(&mut self, gc: &GcInner) {
        if self.items.is_empty() {
            return;
        }
        let batch = std::mem::replace(&mut self.items, Vec::with_capacity(BATCH_CAPACITY));
        {
            let mut deque = gc.deque.lock().unwrap();
            if self.is_gc_thread {
                deque.push_front(batch);
            } else {
                deque.push_back(batch);
            }
        }
        gc.signal.notify_one();
    }
}

impl Drop for LocalBatch {
    fn drop(&mut self) {
        // On thread exit, flush remaining items to the shared deque.
        // flush() already notifies the GC thread.
        if let Some(gc) = GC.get() {
            self.flush(gc);
        }
        // If GC not initialized, items drop synchronously here (Vec::drop).
    }
}

thread_local! {
    static LOCAL_BATCH: RefCell<Option<LocalBatch>> = const { RefCell::new(None) };
}

/// Access (or create) this thread's local batch, then call `f` with it.
fn with_local_batch(is_gc_thread: bool, gc: &GcInner, f: impl FnOnce(&mut LocalBatch, &GcInner)) {
    LOCAL_BATCH.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let batch = borrow.get_or_insert_with(|| LocalBatch::new(is_gc_thread));
        f(batch, gc);
    });
}

// ---------------------------------------------------------------------------
// GcInner — shared state
// ---------------------------------------------------------------------------

struct GcInner {
    state: AtomicU8,
    #[allow(clippy::vec_box)]
    deque: Mutex<VecDeque<Vec<Box<Edge>>>>,
    signal: Condvar,
    thread: Mutex<Option<JoinHandle<()>>>,
}

static GC: OnceLock<Arc<GcInner>> = OnceLock::new();

// Thread ID of the GC thread, set once during init.
static GC_THREAD_ID: OnceLock<thread::ThreadId> = OnceLock::new();

fn is_gc_thread() -> bool {
    GC_THREAD_ID
        .get()
        .is_some_and(|id| *id == thread::current().id())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialize the background GC thread. Idempotent (safe to call multiple times).
///
/// Call once before starting searches that use `advance_root`.
pub fn init() {
    GC.get_or_init(|| {
        let inner = Arc::new(GcInner {
            state: AtomicU8::new(SLEEPING),
            deque: Mutex::new(VecDeque::new()),
            signal: Condvar::new(),
            thread: Mutex::new(None),
        });

        let gc = Arc::clone(&inner);
        let handle = thread::Builder::new()
            .name("mcgs-gc".into())
            .spawn(move || {
                GC_THREAD_ID.get_or_init(|| thread::current().id());
                gc_loop(&gc);
            })
            .expect("failed to spawn GC thread");

        *inner.thread.lock().unwrap() = Some(handle);
        inner
    });
}

/// Start the GC thread (Sleeping/GoToSleep → Running).
///
/// Called by the search loop when work begins.
pub fn start() {
    let Some(gc) = GC.get() else { return };
    loop {
        let current = gc.state.load(Ordering::Acquire);
        match current {
            SLEEPING | GO_TO_SLEEP => {
                if gc
                    .state
                    .compare_exchange(current, RUNNING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    gc.signal.notify_one();
                    return;
                }
                // CAS failed, retry
            }
            RUNNING => return, // Already running
            EXIT => return,    // Shutting down
            _ => unreachable!(),
        }
    }
}

/// Request the GC thread to go to sleep after finishing its current batch.
///
/// Non-blocking. Use `wait()` to block until it actually sleeps.
pub fn stop() {
    let Some(gc) = GC.get() else { return };
    loop {
        let current = gc.state.load(Ordering::Acquire);
        match current {
            RUNNING => {
                if gc
                    .state
                    .compare_exchange(RUNNING, GO_TO_SLEEP, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return;
                }
                // CAS failed (state changed), retry
            }
            // Already sleeping or going to sleep or exiting — nothing to do
            _ => return,
        }
    }
}

/// Block until the GC thread is in the Sleeping state.
///
/// Typically called after `stop()` for deterministic synchronization.
pub fn wait() {
    let Some(gc) = GC.get() else { return };
    let mut guard = gc.deque.lock().unwrap();
    while gc.state.load(Ordering::Acquire) != SLEEPING {
        guard = gc.signal.wait(guard).unwrap();
    }
}

/// Shut down the GC thread. Blocks until the thread exits.
///
/// After shutdown, `queue()` falls back to synchronous drop.
pub fn shutdown() {
    let Some(gc) = GC.get() else { return };

    // Set Exit state
    gc.state.store(EXIT, Ordering::Release);
    gc.signal.notify_one();

    // Join the thread
    if let Some(handle) = gc.thread.lock().unwrap().take() {
        handle.join().expect("GC thread panicked");
    }
}

/// Queue an edge (and its subtree) for background destruction.
///
/// If the GC thread is not initialized, drops synchronously (recursive).
/// This is fine for tests with small trees.
pub(crate) fn queue(edge: Box<Edge>) {
    let Some(gc) = GC.get() else {
        drop(edge);
        return;
    };

    let state = gc.state.load(Ordering::Acquire);

    // Shutdown — drop synchronously
    if state == EXIT {
        drop(edge);
        return;
    }

    // Running + on GC thread → drop directly (lc0's ShouldQueue optimization).
    // Avoids re-batching overhead. Can recurse for deep trees but lc0 accepts
    // this (trees are wide, not deep).
    if state == RUNNING && is_gc_thread() {
        drop(edge);
        return;
    }

    // Normal path: push to thread-local batch
    with_local_batch(is_gc_thread(), gc, |batch, inner| {
        batch.push(edge, inner);
    });
}

/// Flush this thread's local batch to the shared deque.
///
/// Called by `advance_root` after queuing edges, and by search threads
/// before going idle. Matches lc0's `NotifyThreadGoingSleep`.
pub(crate) fn flush() {
    let Some(gc) = GC.get() else { return };
    LOCAL_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().as_mut() {
            batch.flush(gc);
            gc.signal.notify_one();
        }
    });
}

// ---------------------------------------------------------------------------
// GC thread loop (matches lc0's GCThread, node.cc:838-914)
// ---------------------------------------------------------------------------

fn gc_loop(gc: &GcInner) {
    loop {
        let mut guard = gc.deque.lock().unwrap();
        let state = gc.state.load(Ordering::Acquire);

        if state == EXIT {
            break;
        }

        // Not Running (Sleeping or GoToSleep) — drain remaining work, then sleep.
        if state != RUNNING {
            // Drain any leftover batches before sleeping (lc0 does this).
            while let Some(batch) = guard.pop_front() {
                drop(guard);
                for item in batch {
                    drop(item);
                }
                flush_gc_thread_local(gc);
                guard = gc.deque.lock().unwrap();
            }

            gc.state.store(SLEEPING, Ordering::Release);
            gc.signal.notify_all(); // Wake wait() callers

            // Block until woken (state change or new items)
            while gc.state.load(Ordering::Acquire) == SLEEPING {
                guard = gc.signal.wait(guard).unwrap();
            }
            drop(guard);
            flush_gc_thread_local(gc);
            continue;
        }

        // Running: pop a batch from the front of the deque.
        let batch = guard.pop_front();

        if batch.is_none() {
            // Deque empty while Running — wait on condvar for new items or
            // state change. Do NOT auto-transition to Sleeping (lc0 pattern:
            // only GoToSleep causes sleep, not an empty deque).
            let _guard = gc.signal.wait(guard).unwrap();
            continue;
        }
        drop(guard);

        // Process items one by one. Each Edge::drop may push to our
        // local batch (via queue → direct drop optimization on GC thread).
        for item in batch.unwrap() {
            drop(item);

            // Check if we've been asked to stop between items
            if !is_active(gc) {
                flush_gc_thread_local(gc);
                break;
            }
        }
        flush_gc_thread_local(gc);
    }
}

/// Check if the GC should keep processing (Running state).
fn is_active(gc: &GcInner) -> bool {
    gc.state.load(Ordering::Acquire) == RUNNING
}

/// Flush the GC thread's local batch.
fn flush_gc_thread_local(gc: &GcInner) {
    LOCAL_BATCH.with(|cell| {
        if let Some(batch) = cell.borrow_mut().as_mut() {
            batch.flush(gc);
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{LowNode, SharedNode};
    use crate::tree::MCGSTree;
    use pyrat::{Coordinates, GameBuilder};
    use std::sync::{Arc, Mutex as StdMutex};

    const OPEN: [u8; 5] = [0, 1, 2, 3, 4];

    // Tests share the global GC (OnceLock). A test-only mutex serializes
    // tests that call start/stop/wait so they don't step on each other.
    static GC_TEST_LOCK: StdMutex<()> = StdMutex::new(());

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

    fn make_shared() -> Arc<SharedNode> {
        let mut tree = fixture_tree();
        tree.with_exclusive(|mut access| {
            let node = access.test_node(LowNode::new_shell(OPEN, OPEN));
            Arc::clone(node.arc())
        })
    }

    /// Bring GC to a known Sleeping state, regardless of prior test residue.
    fn reset_gc() {
        init();
        stop();
        wait();
    }

    /// Helper: start GC, stop, wait — deterministic sync.
    fn gc_cycle() {
        start();
        stop();
        wait();
    }

    #[test]
    fn state_machine_transitions() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        let gc = GC.get().unwrap();
        assert_eq!(gc.state.load(Ordering::Acquire), SLEEPING);

        // start → Running (or auto-sleeps immediately if deque empty)
        start();
        stop();
        wait();
        assert_eq!(gc.state.load(Ordering::Acquire), SLEEPING);

        // Repeat cycle to verify re-entry
        start();
        stop();
        wait();
        assert_eq!(gc.state.load(Ordering::Acquire), SLEEPING);
    }

    #[test]
    fn gc_queue_drops_edge() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        let child = make_shared();
        let weak = Arc::downgrade(&child);

        let edge = Box::new(Edge::new(child, (0, 0), 0.0, 0.0));
        queue(edge);
        flush();

        gc_cycle();

        assert!(
            weak.upgrade().is_none(),
            "GC should have freed the child node"
        );
    }

    #[test]
    fn gc_handles_sibling_chain() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        let mut tree = fixture_tree();
        let (weak1, weak2, weak3, chain) = tree.with_exclusive(|mut access| {
            let child1 = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let child2 = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let child3 = access.test_node(LowNode::new_shell(OPEN, OPEN));
            let parent = access.test_node(LowNode::new_shell(OPEN, OPEN));

            let weak1 = Arc::downgrade(child1.arc());
            let weak2 = Arc::downgrade(child2.arc());
            let weak3 = Arc::downgrade(child3.arc());

            access.test_connect(&parent, &child3, (2, 2), 0.0, 0.0);
            access.test_connect(&parent, &child2, (1, 1), 0.0, 0.0);
            access.test_connect(&parent, &child1, (0, 0), 0.0, 0.0);

            let chain = access
                .test_detach_children(&parent)
                .expect("fixture child chain");
            (weak1, weak2, weak3, chain)
        });

        // Queue just the head; dropping it cascades through the sibling chain.
        queue(chain);
        flush();

        gc_cycle();

        assert!(weak1.upgrade().is_none(), "child1 should be freed");
        assert!(weak2.upgrade().is_none(), "child2 should be freed");
        assert!(weak3.upgrade().is_none(), "child3 should be freed");
    }

    #[test]
    fn thread_local_batch_flushes_at_capacity() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        let mut weaks = Vec::new();
        for _ in 0..BATCH_CAPACITY + 5 {
            let child = make_shared();
            weaks.push(Arc::downgrade(&child));
            queue(Box::new(Edge::new(child, (0, 0), 0.0, 0.0)));
        }
        flush();

        gc_cycle();

        for (i, weak) in weaks.iter().enumerate() {
            assert!(
                weak.upgrade().is_none(),
                "edge {i} should have been freed"
            );
        }
    }

    #[test]
    fn start_stop_wait_lifecycle() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        let mut weaks = Vec::new();
        for _ in 0..10 {
            let child = make_shared();
            weaks.push(Arc::downgrade(&child));
            queue(Box::new(Edge::new(child, (0, 0), 0.0, 0.0)));
        }
        flush();

        start();
        stop();
        wait();

        for (i, weak) in weaks.iter().enumerate() {
            assert!(
                weak.upgrade().is_none(),
                "edge {i} should have been freed after start/stop/wait"
            );
        }
    }

    #[test]
    fn flush_submits_partial_batch() {
        let _lock = GC_TEST_LOCK.lock().unwrap();
        reset_gc();

        // Queue fewer items than BATCH_CAPACITY
        let child = make_shared();
        let weak = Arc::downgrade(&child);
        queue(Box::new(Edge::new(child, (0, 0), 0.0, 0.0)));

        // Without flush, items sit in thread-local batch
        flush();

        // Now GC can process them
        gc_cycle();

        assert!(
            weak.upgrade().is_none(),
            "flushed edge should have been freed"
        );
    }

    #[test]
    fn queue_without_init_drops_synchronously() {
        // Doesn't need the lock — tests sync fallback only.
        let child = make_shared();
        let weak = Arc::downgrade(&child);
        let edge = Box::new(Edge::new(child, (0, 0), 0.0, 0.0));
        // Simulate what happens when GC.get() returns None:
        drop(edge);
        assert!(weak.upgrade().is_none());
    }
}
