//! Caching decorator for any `Backend`, matching LC0's `MemCache` pattern.
//!
//! Sits before the inner backend — cache hits skip NN evaluation entirely.
//! Thread-local caches: each self-play thread gets its own (no contention,
//! no pollution from other games' positions).

use alpharat_mcts::{Backend, BackendError, EvalResult};
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
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        Ok(self.evaluate_batch(&[game])?.into_iter().next().unwrap())
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        if games.is_empty() {
            return Ok(Vec::new());
        }

        let capacity = self.capacity;
        let cache_cell = self
            .caches
            .get_or(|| RefCell::new(NNCache::new(capacity)));
        let mut cache = cache_cell.borrow_mut();

        let n = games.len();
        let mut results: Vec<Option<EvalResult>> = vec![None; n];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_hashes: Vec<u64> = Vec::new();
        let mut miss_games: Vec<&GameState> = Vec::new();

        // Phase 1: check cache for each game state.
        for (i, game) in games.iter().enumerate() {
            let hash = position_hash(game);
            if let Some(cached) = cache.lookup(hash) {
                results[i] = Some(cached);
            } else {
                miss_indices.push(i);
                miss_hashes.push(hash);
                miss_games.push(game);
            }
        }

        let hits = (n - miss_indices.len()) as u64;
        let misses = miss_indices.len() as u64;
        self.stats.hits.fetch_add(hits, Relaxed);
        self.stats.misses.fetch_add(misses, Relaxed);

        // Phase 2: evaluate misses through the inner backend.
        if !miss_games.is_empty() {
            let miss_results = self.inner.evaluate_batch(&miss_games)?;

            // Phase 3: cache results and place into output (reuse hashes from Phase 1).
            for (j, &orig_idx) in miss_indices.iter().enumerate() {
                cache.insert(miss_hashes[j], miss_results[j]);
                results[orig_idx] = Some(miss_results[j]);
            }
        }

        // Unwrap all — every slot should be filled.
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
}

/// Compute a hash for a game state that captures everything the NN sees.
///
/// Fields hashed: player positions, scores, mud timers, turn, max_turns,
/// board dimensions, cheese positions, and maze topology (walls + mud).
///
/// Maze topology is static per game but varies across games (random mazes).
/// Including it prevents cross-game cache collisions when the thread-local
/// cache persists across games within a thread.
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

    let w = game.width;
    let h_board = game.height;

    for y in 0..h_board {
        for x in 0..w {
            let pos = Coordinates::new(x, y);

            // Cheese state
            if game.cheese.has_cheese(pos) {
                mix!(y as u64 * w as u64 + x as u64 + 1);
            }

            // Wall topology: pre-computed 4-bit bitmask per cell.
            mix!(game.move_table.get_valid_moves(pos) as u64);
        }
    }

    // Mud topology: sorted for deterministic order (MudMap is HashMap-backed).
    // iter() returns deduped entries (pos1 < pos2). Typically 0-10 entries.
    let mut mud_entries: Vec<_> = game.mud.iter().collect();
    mud_entries.sort_unstable_by_key(|((p1, p2), _)| (p1.x, p1.y, p2.x, p2.y));
    for ((p1, p2), cost) in &mud_entries {
        mix!(p1.x as u64);
        mix!(p1.y as u64);
        mix!(p2.x as u64);
        mix!(p2.y as u64);
        mix!(*cost as u64);
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

        let r1 = backend.evaluate(&game).unwrap();
        let r2 = backend.evaluate(&game).unwrap();

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
        let _ = backend.evaluate(&g1).unwrap();

        // Batch with g1 (hit) and g2 (miss).
        let results = backend.evaluate_batch(&[&g1, &g2]).unwrap();
        assert_eq!(results.len(), 2);

        // Verify correctness against direct evaluation.
        let direct1 = SmartUniformBackend.evaluate(&g1).unwrap();
        let direct2 = SmartUniformBackend.evaluate(&g2).unwrap();
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

        let _ = backend.evaluate(&game).unwrap();
        let _ = backend.evaluate(&game).unwrap();

        // All misses — cache is disabled.
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 0);
        assert_eq!(backend.stats.misses.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn thread_isolation() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let backend = Arc::new(CachedBackend::new(Box::new(SmartUniformBackend), 64));
        // Barrier ensures both threads' first evals overlap — neither finishes
        // and populates a shared cache before the other starts.
        let barrier = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let backend = backend.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let game = open_5x5(
                        Coordinates::new(0, 0),
                        Coordinates::new(4, 4),
                        &[Coordinates::new(2, 2)],
                    );
                    barrier.wait();
                    let _ = backend.evaluate(&game).unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Thread-local caches: both threads miss (2 misses, 0 hits).
        // A shared cache would give 1 miss + 1 hit.
        assert_eq!(backend.stats.misses.load(Ordering::Relaxed), 2);
        assert_eq!(backend.stats.hits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn empty_batch() {
        let backend = CachedBackend::new(Box::new(SmartUniformBackend), 64);
        let results = backend.evaluate_batch(&[]).unwrap();
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

    #[test]
    fn different_walls_different_hash() {
        let cheese = [Coordinates::new(2, 2)];
        let p1 = Coordinates::new(0, 0);
        let p2 = Coordinates::new(4, 4);

        // Open maze
        let g1 = open_5x5(p1, p2, &cheese);

        // Same positions/cheese, but with a wall between (1,1) and (1,2)
        let mut walls = HashMap::new();
        walls.insert(Coordinates::new(1, 1), vec![Coordinates::new(1, 2)]);
        walls.insert(Coordinates::new(1, 2), vec![Coordinates::new(1, 1)]);
        let g2 = GameState::new_with_config(5, 5, walls, MudMap::new(), &cheese, p1, p2, 100);

        assert_ne!(
            position_hash(&g1),
            position_hash(&g2),
            "different wall layouts should produce different hashes"
        );
    }

    #[test]
    fn different_mud_different_hash() {
        let cheese = [Coordinates::new(2, 2)];
        let p1 = Coordinates::new(0, 0);
        let p2 = Coordinates::new(4, 4);

        // No mud
        let g1 = open_5x5(p1, p2, &cheese);

        // Same positions/cheese, but with mud between (2,2) and (2,3)
        let mut mud = MudMap::new();
        mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);
        let g2 =
            GameState::new_with_config(5, 5, HashMap::new(), mud, &cheese, p1, p2, 100);

        assert_ne!(
            position_hash(&g1),
            position_hash(&g2),
            "different mud layouts should produce different hashes"
        );
    }

    #[test]
    fn same_maze_same_hash() {
        let cheese = [Coordinates::new(2, 2)];
        let p1 = Coordinates::new(0, 0);
        let p2 = Coordinates::new(4, 4);

        // Build identical games with walls + mud independently
        let make_game = || {
            let mut walls = HashMap::new();
            walls.insert(Coordinates::new(1, 1), vec![Coordinates::new(1, 2)]);
            walls.insert(Coordinates::new(1, 2), vec![Coordinates::new(1, 1)]);
            let mut mud = MudMap::new();
            mud.insert(Coordinates::new(3, 3), Coordinates::new(3, 4), 4);
            GameState::new_with_config(5, 5, walls, mud, &cheese, p1, p2, 100)
        };

        let g1 = make_game();
        let g2 = make_game();
        assert_eq!(
            position_hash(&g1),
            position_hash(&g2),
            "identical mazes should produce identical hashes"
        );
    }
}
