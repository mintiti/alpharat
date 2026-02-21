//! Caching decorator for any `Backend`, matching LC0's `MemCache` pattern.
//!
//! Sits before the inner backend — cache hits skip NN evaluation entirely.
//! Thread-local caches: each self-play thread gets its own (no contention,
//! no pollution from other games' positions).

use alpharat_mcts::{Backend, EvalResult};
use pyrat::{Coordinates, GameState};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use thread_local::ThreadLocal;

use crate::nn_cache::NNCache;

/// Aggregate cache statistics (atomic — readable from any thread).
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl CacheStats {
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Relaxed);
        let m = self.misses.load(Relaxed);
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }
}

/// Backend decorator that caches NN evaluation results per thread.
///
/// The `evaluate_batch` flow:
/// 1. Hash each game state
/// 2. Check thread-local cache: hits go straight to results, misses collect
/// 3. Forward misses to inner backend
/// 4. Cache miss results, assemble final result vector
pub struct CachedBackend {
    inner: Box<dyn Backend>,
    caches: ThreadLocal<RefCell<NNCache>>,
    capacity: usize,
    pub stats: CacheStats,
}

impl CachedBackend {
    pub fn new(inner: Box<dyn Backend>, capacity: usize) -> Self {
        Self {
            inner,
            caches: ThreadLocal::new(),
            capacity,
            stats: CacheStats::new(),
        }
    }
}

impl Backend for CachedBackend {
    fn evaluate(&self, game: &GameState) -> EvalResult {
        self.evaluate_batch(&[game])[0]
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Vec<EvalResult> {
        if games.is_empty() {
            return Vec::new();
        }

        let capacity = self.capacity;
        let cache_cell = self
            .caches
            .get_or(|| RefCell::new(NNCache::new(capacity)));
        let mut cache = cache_cell.borrow_mut();

        let n = games.len();
        let mut results: Vec<Option<EvalResult>> = vec![None; n];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_games: Vec<&GameState> = Vec::new();

        // Phase 1: check cache for each game state.
        for (i, game) in games.iter().enumerate() {
            let hash = position_hash(game);
            if let Some(cached) = cache.lookup(hash) {
                results[i] = Some(cached);
            } else {
                miss_indices.push(i);
                miss_games.push(game);
            }
        }

        let hits = (n - miss_indices.len()) as u64;
        let misses = miss_indices.len() as u64;
        self.stats.hits.fetch_add(hits, Relaxed);
        self.stats.misses.fetch_add(misses, Relaxed);

        // Phase 2: evaluate misses through the inner backend.
        if !miss_games.is_empty() {
            let miss_results = self.inner.evaluate_batch(&miss_games);

            // Phase 3: cache results and place into output.
            for (j, &orig_idx) in miss_indices.iter().enumerate() {
                let hash = position_hash(games[orig_idx]);
                cache.insert(hash, miss_results[j]);
                results[orig_idx] = Some(miss_results[j]);
            }
        }

        // Unwrap all — every slot should be filled.
        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

/// Compute a hash for a game state that captures everything the NN sees.
///
/// Fields hashed: player positions, scores, mud timers, turn, max_turns,
/// board dimensions, and cheese positions.
///
/// Within a single game (fixed maze), transpositions that reach the same
/// dynamic state will produce the same hash. Cross-game collisions are
/// unlikely due to different cheese configurations.
fn position_hash(game: &GameState) -> u64 {
    // FNV-1a: fast, simple, good distribution for small keys.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut h = FNV_OFFSET;

    macro_rules! mix {
        ($val:expr) => {
            h ^= $val as u64;
            h = h.wrapping_mul(FNV_PRIME);
        };
    }

    // Player positions
    mix!(game.player1.current_pos.x);
    mix!(game.player1.current_pos.y);
    mix!(game.player2.current_pos.x);
    mix!(game.player2.current_pos.y);

    // Scores (as raw bits for exact matching)
    mix!(game.player1.score.to_bits());
    mix!(game.player2.score.to_bits());

    // Mud timers
    mix!(game.player1.mud_timer);
    mix!(game.player2.mud_timer);

    // Turn progress
    mix!(game.turn);
    mix!(game.max_turns);

    // Board dimensions (differentiates cross-game if sizes differ)
    mix!(game.width);
    mix!(game.height);

    // Cheese state: iterate all cells and hash the bitboard.
    // This is O(cells) but cells are small (25 for 5x5, 49 for 7x7)
    // and vastly cheaper than an NN eval.
    let w = game.width;
    let h_board = game.height;
    for y in 0..h_board {
        for x in 0..w {
            if game.cheese.has_cheese(Coordinates::new(x, y)) {
                // Mix in the cell index so different cheese patterns hash differently.
                mix!(y as u64 * w as u64 + x as u64 + 1);
            }
        }
    }

    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use alpharat_mcts::SmartUniformBackend;
    use pyrat::game::types::MudMap;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;

    fn open_5x5(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
        GameState::new_with_config(5, 5, HashMap::new(), MudMap::new(), cheese, p1, p2, 100)
    }

    #[test]
    fn cache_hit_returns_same_result() {
        let backend = CachedBackend::new(Box::new(SmartUniformBackend), 64);
        let game = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );

        let r1 = backend.evaluate(&game);
        let r2 = backend.evaluate(&game);

        assert_eq!(r1.policy_p1, r2.policy_p1);
        assert_eq!(r1.value_p1, r2.value_p1);

        // First call = miss, second = hit.
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 1);
        assert_eq!(backend.stats.misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn batch_with_mixed_hits_and_misses() {
        let backend = CachedBackend::new(Box::new(SmartUniformBackend), 64);

        let g1 = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let g2 = open_5x5(
            Coordinates::new(1, 1),
            Coordinates::new(3, 3),
            &[Coordinates::new(2, 2)],
        );

        // Prime g1 into cache.
        let _ = backend.evaluate(&g1);

        // Batch with g1 (hit) and g2 (miss).
        let results = backend.evaluate_batch(&[&g1, &g2]);
        assert_eq!(results.len(), 2);

        // Verify correctness against direct evaluation.
        let direct1 = SmartUniformBackend.evaluate(&g1);
        let direct2 = SmartUniformBackend.evaluate(&g2);
        assert_eq!(results[0].policy_p1, direct1.policy_p1);
        assert_eq!(results[1].policy_p1, direct2.policy_p1);

        // Stats: 1 miss (prime) + 1 hit + 1 miss (batch) = 2 misses, 1 hit
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 1);
        assert_eq!(backend.stats.misses.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn cache_disabled_at_zero_capacity() {
        let backend = CachedBackend::new(Box::new(SmartUniformBackend), 0);
        let game = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );

        let _ = backend.evaluate(&game);
        let _ = backend.evaluate(&game);

        // All misses — cache is disabled.
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 0);
        assert_eq!(backend.stats.misses.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn thread_isolation() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(CachedBackend::new(Box::new(SmartUniformBackend), 64));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let backend = backend.clone();
                thread::spawn(move || {
                    let x = (i % 5) as u8;
                    let game = open_5x5(
                        Coordinates::new(x, 0),
                        Coordinates::new(4, 4),
                        &[Coordinates::new(2, 2)],
                    );
                    // Each thread evaluates the same game twice.
                    let r1 = backend.evaluate(&game);
                    let r2 = backend.evaluate(&game);
                    assert_eq!(r1.policy_p1, r2.policy_p1);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // 4 threads × 2 evals = 4 misses + 4 hits.
        let total = backend.stats.hits.load(Ordering::Relaxed)
            + backend.stats.misses.load(Ordering::Relaxed);
        assert_eq!(total, 8);
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn empty_batch() {
        let backend = CachedBackend::new(Box::new(SmartUniformBackend), 64);
        let results = backend.evaluate_batch(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn different_cheese_different_hash() {
        let g1 = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 1)],
        );
        let g2 = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );

        let h1 = position_hash(&g1);
        let h2 = position_hash(&g2);
        assert_ne!(h1, h2, "different cheese positions should produce different hashes");
    }

    #[test]
    fn same_state_same_hash() {
        let g1 = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let g2 = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );

        assert_eq!(position_hash(&g1), position_hash(&g2));
    }
}
