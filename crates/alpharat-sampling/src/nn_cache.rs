//! Open-addressing hash table with FIFO eviction for NN evaluation results.
//!
//! Port of LC0's `HashKeyedCache` from `utils/cache.h`, simplified:
//! - No pin/unpin (values are `Copy`)
//! - No mutex (thread-local access only)
//! - No runtime resize (capacity fixed at construction)

use alpharat_mcts::EvalResult;
use std::collections::VecDeque;

const LOAD_FACTOR: f64 = 1.9;

struct Entry {
    key: u64,
    value: EvalResult,
    in_use: bool,
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            key: 0,
            value: EvalResult {
                policy_p1: [0.0; 5],
                policy_p2: [0.0; 5],
                value_p1: 0.0,
                value_p2: 0.0,
            },
            in_use: false,
        }
    }
}

/// Open-addressing hash table with FIFO eviction for `EvalResult`.
///
/// Matches LC0's `HashKeyedCache` semantics:
/// - Linear probing for collision resolution
/// - FIFO eviction (oldest inserted key evicted first)
/// - Insert to existing key is silently ignored (same position = same eval)
/// - Robin Hood rehashing on eviction to maintain probe chain integrity
pub struct NNCache {
    table: Vec<Entry>,
    insertion_order: VecDeque<u64>,
    capacity: usize,
    size: usize,
}

impl NNCache {
    /// Create a new cache with the given capacity (max entries before eviction).
    ///
    /// The hash table is sized at `capacity * 1.9 + 1` for low collision rates.
    pub fn new(capacity: usize) -> Self {
        let table_size = (capacity as f64 * LOAD_FACTOR) as usize + 1;
        let mut table = Vec::with_capacity(table_size);
        table.resize_with(table_size, Entry::default);
        Self {
            table,
            insertion_order: VecDeque::with_capacity(capacity),
            capacity,
            size: 0,
        }
    }

    /// Look up a key in the cache. Returns a copy of the value if found.
    #[inline]
    pub fn lookup(&self, key: u64) -> Option<EvalResult> {
        if self.capacity == 0 {
            return None;
        }
        let table_size = self.table.len();
        let mut idx = (key % table_size as u64) as usize;
        loop {
            if !self.table[idx].in_use {
                return None;
            }
            if self.table[idx].key == key {
                return Some(self.table[idx].value);
            }
            idx += 1;
            if idx >= table_size {
                idx -= table_size;
            }
        }
    }

    /// Insert a key-value pair. If the key already exists, does nothing (LC0 semantics).
    /// If the cache is at capacity, evicts the oldest entry first.
    pub fn insert(&mut self, key: u64, value: EvalResult) {
        if self.capacity == 0 {
            return;
        }

        let table_size = self.table.len();
        let mut idx = (key % table_size as u64) as usize;

        // Linear probe: find empty slot or existing key.
        loop {
            if !self.table[idx].in_use {
                break;
            }
            if self.table[idx].key == key {
                // Already exists — silently ignore.
                return;
            }
            idx += 1;
            if idx >= table_size {
                idx -= table_size;
            }
        }

        // Insert into the empty slot.
        self.table[idx].key = key;
        self.table[idx].value = value;
        self.table[idx].in_use = true;
        self.insertion_order.push_back(key);
        self.size += 1;

        // Evict if over capacity.
        while self.size > self.capacity {
            self.evict_item();
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Evict the oldest entry (front of insertion_order).
    ///
    /// After clearing the slot, performs Robin Hood rehashing: walks the probe
    /// chain forward and moves entries backward if they can be placed closer to
    /// their ideal slot. This prevents probe chain breaks.
    fn evict_item(&mut self) {
        let key = self.insertion_order.pop_front().expect("evict from non-empty");
        let table_size = self.table.len();

        // Find the entry in the table.
        let mut idx = (key % table_size as u64) as usize;
        loop {
            if self.table[idx].in_use && self.table[idx].key == key {
                break;
            }
            idx += 1;
            if idx >= table_size {
                idx -= table_size;
            }
        }

        // Clear the slot.
        self.table[idx].in_use = false;
        self.size -= 1;

        // Robin Hood rehash: compact the probe chain after the cleared slot.
        // Walk forward through consecutive in_use entries. For each, check if
        // it could be placed at `idx` (the hole). If so, move it there and
        // the hole advances.
        let mut next = idx + 1;
        if next >= table_size {
            next -= table_size;
        }
        loop {
            if !self.table[next].in_use {
                break;
            }
            let target = (self.table[next].key % table_size as u64) as usize;
            if !in_range(target, idx + 1, next, table_size) {
                // This entry's ideal slot is at or before `idx`, so moving it
                // to `idx` is an improvement (or neutral). Swap it in.
                self.table.swap(idx, next);
                idx = next;
            }
            next += 1;
            if next >= table_size {
                next -= table_size;
            }
        }
    }
}

/// Check if `target` is in the circular range `[start, end]` (inclusive).
/// Handles wrap-around in the hash table.
fn in_range(target: usize, start: usize, end: usize, table_size: usize) -> bool {
    let start = if start >= table_size {
        start - table_size
    } else {
        start
    };
    if start <= end {
        target >= start && target <= end
    } else {
        target >= start || target <= end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(v: f32) -> EvalResult {
        EvalResult {
            policy_p1: [v; 5],
            policy_p2: [v; 5],
            value_p1: v,
            value_p2: v,
        }
    }

    #[test]
    fn basic_insert_and_lookup() {
        let mut cache = NNCache::new(16);
        let val = make_result(1.0);
        cache.insert(42, val);

        let got = cache.lookup(42);
        assert!(got.is_some());
        assert_eq!(got.unwrap().value_p1, 1.0);
    }

    #[test]
    fn miss_returns_none() {
        let cache = NNCache::new(16);
        assert!(cache.lookup(99).is_none());
    }

    #[test]
    fn insert_ignores_existing() {
        let mut cache = NNCache::new(16);
        cache.insert(42, make_result(1.0));
        cache.insert(42, make_result(2.0)); // should be ignored

        let got = cache.lookup(42).unwrap();
        assert_eq!(got.value_p1, 1.0, "original value should be retained");
    }

    #[test]
    fn fifo_eviction_order() {
        let mut cache = NNCache::new(3);

        cache.insert(1, make_result(1.0));
        cache.insert(2, make_result(2.0));
        cache.insert(3, make_result(3.0));
        assert_eq!(cache.len(), 3);

        // Inserting a 4th should evict key=1 (oldest).
        cache.insert(4, make_result(4.0));
        assert_eq!(cache.len(), 3);
        assert!(cache.lookup(1).is_none(), "key 1 should be evicted");
        assert!(cache.lookup(2).is_some(), "key 2 should remain");
        assert!(cache.lookup(3).is_some(), "key 3 should remain");
        assert!(cache.lookup(4).is_some(), "key 4 should be present");
    }

    #[test]
    fn probe_chain_integrity_after_eviction() {
        // Force two keys to collide by choosing keys that map to the same slot.
        // Table size for capacity=4 is (4 * 1.9 + 1) = 8.
        // Keys 0 and 8 both map to slot 0.
        let mut cache = NNCache::new(4);

        cache.insert(0, make_result(0.0));
        cache.insert(8, make_result(8.0)); // collides with 0, probes to slot 1

        // Evict key 0 (oldest). Robin Hood should compact so key 8 is still findable.
        cache.insert(100, make_result(100.0));
        cache.insert(200, make_result(200.0));
        cache.insert(300, make_result(300.0)); // evicts key 0

        assert!(cache.lookup(0).is_none(), "key 0 should be evicted");
        assert!(cache.lookup(8).is_some(), "key 8 should survive and be findable");
    }

    #[test]
    fn zero_capacity_is_noop() {
        let mut cache = NNCache::new(0);
        cache.insert(1, make_result(1.0));
        assert!(cache.lookup(1).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn many_inserts_and_evictions() {
        let mut cache = NNCache::new(32);

        // Insert 1000 entries — should evict gracefully.
        for i in 0..1000u64 {
            cache.insert(i, make_result(i as f32));
        }

        assert_eq!(cache.len(), 32);

        // The last 32 entries should be present.
        for i in 968..1000u64 {
            assert!(
                cache.lookup(i).is_some(),
                "key {i} should be in cache (recent)"
            );
        }
        // Early entries should be evicted.
        for i in 0..900u64 {
            assert!(
                cache.lookup(i).is_none(),
                "key {i} should be evicted (old)"
            );
        }
    }

    #[test]
    fn different_values_for_different_keys() {
        let mut cache = NNCache::new(16);
        cache.insert(1, make_result(10.0));
        cache.insert(2, make_result(20.0));
        cache.insert(3, make_result(30.0));

        assert_eq!(cache.lookup(1).unwrap().value_p1, 10.0);
        assert_eq!(cache.lookup(2).unwrap().value_p1, 20.0);
        assert_eq!(cache.lookup(3).unwrap().value_p1, 30.0);
    }

    #[test]
    fn capacity_one() {
        let mut cache = NNCache::new(1);

        cache.insert(1, make_result(1.0));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1).is_some());

        cache.insert(2, make_result(2.0));
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1).is_none());
        assert!(cache.lookup(2).is_some());
    }
}
