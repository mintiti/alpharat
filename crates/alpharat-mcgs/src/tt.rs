use std::sync::{Arc, Weak};

use rustc_hash::FxHashMap;

use crate::node::SharedNode;

/// Hash-based transposition table mapping position hashes to shared LowNodes.
///
/// Follows lc0's pattern: Edges hold `Arc<SharedNode>` (strong ownership), the TT
/// holds `Weak<SharedNode>` (observer). When all Edges pointing to a LowNode are
/// dropped, the Weak expires and lookup returns None.
///
/// Single-threaded. No capacity limits. Uses FxHash (not SipHash) since keys
/// are pre-hashed u64 Zobrist values, not untrusted input.
#[derive(Default)]
pub struct TranspositionTable {
    map: FxHashMap<u64, Weak<SharedNode>>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    /// Look up a position by hash. Returns `None` if absent or expired.
    pub fn lookup(&self, hash: u64) -> Option<Arc<SharedNode>> {
        self.map.get(&hash).and_then(|w| w.upgrade())
    }

    /// Insert a node at `hash`. Returns `true` if the entry was written.
    ///
    /// Three cases:
    /// - Vacant: insert, return `true`
    /// - Occupied + expired: replace, return `true`
    /// - Occupied + live: leave existing, return `false`
    ///
    /// Correctness assumes the hash is collision-free for the game state space:
    /// if two different positions produce the same hash, the second insert
    /// silently fails and a subsequent lookup returns the wrong node.
    pub fn insert(&mut self, hash: u64, node: &Arc<SharedNode>) -> bool {
        use std::collections::hash_map::Entry;
        match self.map.entry(hash) {
            Entry::Vacant(slot) => {
                slot.insert(Arc::downgrade(node));
                true
            }
            Entry::Occupied(mut slot) => {
                if slot.get().strong_count() == 0 {
                    slot.insert(Arc::downgrade(node));
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Remove all expired entries. Call after root advancement.
    pub fn evict_expired(&mut self) {
        self.map.retain(|_, w| w.strong_count() > 0);
    }

    /// Raw entry count, including stale (expired) entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Count of entries whose LowNode is still alive. O(n) scan.
    pub fn live_count(&self) -> usize {
        self.map.values().filter(|w| w.strong_count() > 0).count()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Edge, LowNode};

    /// All actions open — simplest effective-action mapping.
    const OPEN: [u8; 5] = [0, 1, 2, 3, 4];

    fn make_node() -> Arc<SharedNode> {
        Arc::new(SharedNode::new(LowNode::new_shell(OPEN, OPEN)))
    }

    // ---- Basic operations ----

    #[test]
    fn insert_and_lookup() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        assert!(tt.insert(42, &node));

        let found = tt.lookup(42).unwrap();
        assert!(Arc::ptr_eq(&found, &node));
    }

    #[test]
    fn lookup_miss() {
        let tt = TranspositionTable::new();
        assert!(tt.lookup(99).is_none());
    }

    #[test]
    fn insert_returns_true_on_empty() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        assert!(tt.insert(0, &node));
    }

    #[test]
    fn multiple_entries() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        let b = make_node();
        let c = make_node();

        tt.insert(1, &a);
        tt.insert(2, &b);
        tt.insert(3, &c);

        assert!(Arc::ptr_eq(&tt.lookup(1).unwrap(), &a));
        assert!(Arc::ptr_eq(&tt.lookup(2).unwrap(), &b));
        assert!(Arc::ptr_eq(&tt.lookup(3).unwrap(), &c));
    }

    // ---- Weak expiry ----

    #[test]
    fn expired_returns_none() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        tt.insert(42, &node);
        drop(node);

        assert!(tt.lookup(42).is_none());
    }

    #[test]
    fn expired_stays_in_map() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        tt.insert(42, &node);
        drop(node);

        assert_eq!(tt.len(), 1);
        assert_eq!(tt.live_count(), 0);
    }

    #[test]
    fn insert_replaces_expired() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        tt.insert(42, &a);
        drop(a);

        let b = make_node();
        assert!(tt.insert(42, &b));
        assert!(Arc::ptr_eq(&tt.lookup(42).unwrap(), &b));
    }

    #[test]
    fn insert_skips_live() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        tt.insert(42, &a);

        let b = make_node();
        assert!(!tt.insert(42, &b));
        assert!(Arc::ptr_eq(&tt.lookup(42).unwrap(), &a));
    }

    // ---- Eviction ----

    #[test]
    fn evict_removes_stale() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        let b = make_node();
        let c = make_node();
        tt.insert(1, &a);
        tt.insert(2, &b);
        tt.insert(3, &c);
        drop(b);

        tt.evict_expired();
        assert_eq!(tt.len(), 2);
        assert!(tt.lookup(1).is_some());
        assert!(tt.lookup(2).is_none());
        assert!(tt.lookup(3).is_some());
    }

    #[test]
    fn evict_noop_all_live() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        let b = make_node();
        tt.insert(1, &a);
        tt.insert(2, &b);

        tt.evict_expired();
        assert_eq!(tt.len(), 2);
    }

    #[test]
    fn evict_clears_all_dead() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        let b = make_node();
        tt.insert(1, &a);
        tt.insert(2, &b);
        drop(a);
        drop(b);

        tt.evict_expired();
        assert!(tt.is_empty());
    }

    // ---- Clear ----

    #[test]
    fn clear_empties() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        tt.insert(1, &a);
        tt.insert(2, &a);

        tt.clear();
        assert!(tt.is_empty());
        assert!(tt.lookup(1).is_none());
        assert!(tt.lookup(2).is_none());
    }

    #[test]
    fn clear_doesnt_drop_live_nodes() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        let weak = Arc::downgrade(&node);
        tt.insert(42, &node);

        tt.clear();
        // Node still alive — our local Arc holds it
        assert!(weak.upgrade().is_some());
    }

    // ---- Integration with Edge lifecycle ----

    #[test]
    fn edge_keeps_node_alive() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        let edge = Edge::new(Arc::clone(&node), (0, 0), 0.0, 0.0);
        tt.insert(42, &node);

        tt.clear();
        // Edge's Arc keeps the node alive even after TT is cleared
        assert!(Arc::ptr_eq(&edge.low_node(), &node));
        assert_eq!(node.get().num_parents(), 1);
    }

    #[test]
    fn edge_drop_expires_weak() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        let edge = Edge::new(Arc::clone(&node), (0, 0), 0.0, 0.0);
        tt.insert(42, &node);

        drop(node); // TT weak + edge Arc remain
        assert!(tt.lookup(42).is_some()); // edge keeps it alive

        drop(edge); // last strong ref gone
        assert!(tt.lookup(42).is_none());
    }

    #[test]
    fn transposition_two_edges() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        let edge1 = Edge::new(Arc::clone(&node), (0, 0), 0.0, 0.0);
        let edge2 = Edge::new(Arc::clone(&node), (1, 1), 0.0, 0.0);
        tt.insert(42, &node);

        assert_eq!(node.get().num_parents(), 2);

        drop(node); // TT + 2 edges remain
        drop(edge1);
        assert!(tt.lookup(42).is_some()); // edge2 still holds it

        drop(edge2);
        assert!(tt.lookup(42).is_none()); // all strong refs gone
    }

    // ---- Edge cases ----

    #[test]
    fn hash_zero() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        assert!(tt.insert(0, &node));
        assert!(Arc::ptr_eq(&tt.lookup(0).unwrap(), &node));
    }

    #[test]
    fn hash_max() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        assert!(tt.insert(u64::MAX, &node));
        assert!(Arc::ptr_eq(&tt.lookup(u64::MAX).unwrap(), &node));
    }

    #[test]
    fn same_node_two_hashes() {
        let mut tt = TranspositionTable::new();
        let node = make_node();
        tt.insert(10, &node);
        tt.insert(20, &node);

        assert!(Arc::ptr_eq(&tt.lookup(10).unwrap(), &tt.lookup(20).unwrap()));
    }

    #[test]
    fn evict_then_reinsert() {
        let mut tt = TranspositionTable::new();
        let a = make_node();
        tt.insert(42, &a);
        drop(a);
        tt.evict_expired();
        assert!(tt.is_empty());

        let b = make_node();
        assert!(tt.insert(42, &b));
        assert!(Arc::ptr_eq(&tt.lookup(42).unwrap(), &b));
    }
}
