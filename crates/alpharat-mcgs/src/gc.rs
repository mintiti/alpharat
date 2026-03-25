//! Background garbage collector for MCGS edge chains.
//!
//! Prevents stack overflow from recursive `Drop` on deep sibling/child chains
//! and avoids blocking the search thread with synchronous destruction of large
//! pruned subtrees.
//!
//! Adapted from lc0's `NodeGarbageCollector`: a global queue + background thread
//! that drops edges one at a time. Each edge's `Drop` may queue more items
//! (siblings), which the GC thread picks up in the next drain cycle.
//!
//! When the GC is not initialized (e.g. in unit tests), `queue()` falls back
//! to synchronous drop.

use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::node::Edge;

struct GcInner {
    #[allow(clippy::vec_box)] // Edges are always boxed (linked-list nodes); unboxing would move them.
    queue: Mutex<Vec<Box<Edge>>>,
    signal: Condvar,
}

static GC: OnceLock<Arc<GcInner>> = OnceLock::new();

/// Queue an edge (and its subtree) for background destruction.
///
/// If the GC thread is not initialized, drops synchronously (recursive).
/// This is fine for tests with small trees.
pub(crate) fn queue(edge: Box<Edge>) {
    match GC.get() {
        Some(gc) => {
            gc.queue.lock().unwrap().push(edge);
            gc.signal.notify_one();
        }
        None => drop(edge),
    }
}

/// Initialize the background GC thread. Idempotent (safe to call multiple times).
///
/// Call once before starting searches that use `advance_root`.
pub fn init() {
    GC.get_or_init(|| {
        let inner = Arc::new(GcInner {
            queue: Mutex::new(Vec::new()),
            signal: Condvar::new(),
        });

        let gc = Arc::clone(&inner);
        std::thread::Builder::new()
            .name("mcgs-gc".into())
            .spawn(move || gc_loop(&gc))
            .expect("failed to spawn GC thread");

        inner
    });
}

fn gc_loop(gc: &GcInner) {
    let mut local = Vec::new();
    loop {
        // Wait for items
        {
            let mut queue = gc.queue.lock().unwrap();
            while queue.is_empty() {
                queue = gc.signal.wait(queue).unwrap();
            }
            std::mem::swap(&mut local, &mut *queue);
        }

        // Drop one at a time. Each drop may re-queue siblings via Edge::drop.
        for edge in local.drain(..) {
            drop(edge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{LowNode, SharedNode};
    use std::sync::Arc;

    const OPEN: [u8; 5] = [0, 1, 2, 3, 4];

    fn make_shared() -> Arc<SharedNode> {
        Arc::new(SharedNode::new(LowNode::new_shell(OPEN, OPEN)))
    }

    /// Helper: wait for a Weak to expire, with timeout.
    fn wait_for_drop(weak: &std::sync::Weak<SharedNode>, label: &str) {
        for _ in 0..100 {
            if weak.upgrade().is_none() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("GC did not free {label} within timeout");
    }

    #[test]
    fn gc_queue_drops_edge() {
        // Works whether GC is initialized or not (sync fallback or background).
        init();

        let child = make_shared();
        let weak = Arc::downgrade(&child);

        let edge = Box::new(Edge::new(child, (0, 0), 0.0, 0.0));
        queue(edge);

        wait_for_drop(&weak, "child");
    }

    #[test]
    fn gc_handles_sibling_chain() {
        init();

        let child1 = make_shared();
        let child2 = make_shared();
        let child3 = make_shared();
        let weak1 = Arc::downgrade(&child1);
        let weak2 = Arc::downgrade(&child2);
        let weak3 = Arc::downgrade(&child3);

        // Build a sibling chain via parent's child list
        let edge3 = Box::new(Edge::new(child3, (2, 2), 0.0, 0.0));
        let edge2 = Box::new(Edge::new(child2, (1, 1), 0.0, 0.0));
        let parent = SharedNode::new(LowNode::new_shell(OPEN, OPEN));
        parent.get_mut().prepend_child(edge3);
        parent.get_mut().prepend_child(edge2);
        parent.get_mut().prepend_child(Box::new(Edge::new(child1, (0, 0), 0.0, 0.0)));

        // Take the whole chain and queue just the head
        let chain = parent.get_mut().take_first_child().unwrap();
        queue(chain);

        // All should eventually be freed
        wait_for_drop(&weak1, "child1");
        wait_for_drop(&weak2, "child2");
        wait_for_drop(&weak3, "child3");
    }
}
