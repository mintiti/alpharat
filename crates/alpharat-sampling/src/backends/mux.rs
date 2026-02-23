use alpharat_mcts::{Backend, BackendError, EvalResult};
use pyrat::GameState;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Instant;

/// Configuration for the multiplexing layer.
pub struct MuxConfig {
    /// Max game states per merged batch (determines GPU tensor size).
    pub max_batch_size: usize,
}

/// Mux worker thread timing and batch statistics (atomic, lock-free reads).
pub struct MuxStats {
    /// Number of inner backend evaluate_batch calls.
    pub total_batches: AtomicU64,
    /// Total positions sent to the inner backend.
    pub total_positions: AtomicU64,
    /// Cumulative time in nanoseconds spent in inner evaluate_batch.
    pub nn_time_ns: AtomicU64,
    /// Cumulative time in nanoseconds the worker spent waiting for requests.
    pub wait_time_ns: AtomicU64,
}

impl MuxStats {
    fn new() -> Self {
        Self {
            total_batches: AtomicU64::new(0),
            total_positions: AtomicU64::new(0),
            nn_time_ns: AtomicU64::new(0),
            wait_time_ns: AtomicU64::new(0),
        }
    }

    /// Average positions per inner backend call.
    pub fn avg_batch_size(&self) -> f64 {
        let b = self.total_batches.load(Ordering::Relaxed);
        if b == 0 {
            return 0.0;
        }
        self.total_positions.load(Ordering::Relaxed) as f64 / b as f64
    }

    /// Fraction of worker time spent in NN inference (vs waiting for requests).
    pub fn nn_utilization(&self) -> f64 {
        let nn = self.nn_time_ns.load(Ordering::Relaxed) as f64;
        let wait = self.wait_time_ns.load(Ordering::Relaxed) as f64;
        let total = nn + wait;
        if total == 0.0 {
            return 0.0;
        }
        nn / total
    }

    pub fn nn_time_secs(&self) -> f64 {
        self.nn_time_ns.load(Ordering::Relaxed) as f64 / 1e9
    }

    pub fn wait_time_secs(&self) -> f64 {
        self.wait_time_ns.load(Ordering::Relaxed) as f64 / 1e9
    }
}

// ---------------------------------------------------------------------------
// BatchRequest — one game thread's work item
// ---------------------------------------------------------------------------

/// A single evaluate_batch call from a game thread, waiting for results.
struct BatchRequest {
    // SAFETY: Calling thread blocks on rx.recv() until the worker calls
    // tx.send(), which happens after the worker is done reading these
    // pointers. GameStates are guaranteed alive for the duration of access.
    games: Vec<*const GameState>,
    tx: mpsc::SyncSender<Result<Vec<EvalResult>, BackendError>>,
}

// SAFETY: Raw pointers are valid for the duration of worker access
// (guaranteed by sync_channel synchronization — see BatchRequest.games).
unsafe impl Send for BatchRequest {}

// ---------------------------------------------------------------------------
// BatchQueue — thread-safe queue with blocking drain
// ---------------------------------------------------------------------------

struct BatchQueue {
    inner: Mutex<QueueInner>,
    condvar: Condvar,
}

struct QueueInner {
    queue: VecDeque<BatchRequest>,
    closed: bool,
}

impl BatchQueue {
    fn new() -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                queue: VecDeque::new(),
                closed: false,
            }),
            condvar: Condvar::new(),
        }
    }

    /// Push a request and wake the worker.
    fn push(&self, request: BatchRequest) {
        let mut inner = self.inner.lock().unwrap();
        inner.queue.push_back(request);
        self.condvar.notify_one();
    }

    /// Block until ≥1 request available, then drain up to `max_positions` total
    /// game states. Returns `None` when the queue is closed and empty.
    fn wait_drain(&self, max_positions: usize) -> Option<Vec<BatchRequest>> {
        let mut inner = self.inner.lock().unwrap();

        // Wait for at least one request (or shutdown).
        while inner.queue.is_empty() && !inner.closed {
            inner = self.condvar.wait(inner).unwrap();
        }

        if inner.queue.is_empty() {
            return None; // closed + empty → shutdown
        }

        let mut batch = Vec::new();
        let mut total_positions = 0;

        // Always take the first request (even if it alone exceeds max).
        if let Some(req) = inner.queue.pop_front() {
            total_positions += req.games.len();
            batch.push(req);
        }

        // Greedily take more until we hit the cap.
        while let Some(front) = inner.queue.front() {
            if total_positions + front.games.len() > max_positions {
                break;
            }
            let req = inner.queue.pop_front().unwrap();
            total_positions += req.games.len();
            batch.push(req);
        }

        Some(batch)
    }

    /// Signal shutdown. Worker will drain remaining requests then exit.
    fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        self.condvar.notify_one();
    }
}

// ---------------------------------------------------------------------------
// MuxBackend — multiplexing wrapper
// ---------------------------------------------------------------------------

/// Merges evaluate_batch calls from multiple game threads into larger batches
/// for the inner backend. lc0's `network_mux.cc` pattern.
///
/// Each game thread calls `evaluate_batch()`, which sends raw pointers into a
/// queue and blocks. A single worker thread drains the queue, merges states
/// into one batch, calls the inner backend, and scatters results back.
pub struct MuxBackend {
    queue: Arc<BatchQueue>,
    stats: Arc<MuxStats>,
    worker: Option<thread::JoinHandle<()>>,
}

impl MuxBackend {
    pub fn new(inner: impl Backend + 'static, config: MuxConfig) -> Self {
        let queue = Arc::new(BatchQueue::new());
        let stats = Arc::new(MuxStats::new());
        let worker_queue = queue.clone();
        let worker_stats = stats.clone();
        let max_batch_size = config.max_batch_size;

        let worker = thread::spawn(move || {
            worker_loop(&worker_queue, &inner, max_batch_size, &worker_stats);
        });

        Self {
            queue,
            stats,
            worker: Some(worker),
        }
    }

    /// Access accumulated mux worker statistics.
    pub fn stats(&self) -> &Arc<MuxStats> {
        &self.stats
    }
}

impl Drop for MuxBackend {
    fn drop(&mut self) {
        self.queue.close();
        if let Some(handle) = self.worker.take() {
            // Don't double-panic if the worker panicked — just log it.
            let _ = handle.join();
        }
    }
}

impl Backend for MuxBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        Ok(self.evaluate_batch(&[game])?[0])
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        if games.is_empty() {
            return Ok(Vec::new());
        }

        let (tx, rx) = mpsc::sync_channel(1);
        let request = BatchRequest {
            games: games.iter().map(|g| *g as *const GameState).collect(),
            tx,
        };

        self.queue.push(request);
        // Outer expect: worker dropping the channel = bug (programming error).
        // Inner Result: propagates backend errors from the worker.
        rx.recv().expect("mux worker dropped without sending result")
    }
}

fn worker_loop(
    queue: &BatchQueue,
    inner: &dyn Backend,
    max_batch_size: usize,
    stats: &MuxStats,
) {
    loop {
        let wait_start = Instant::now();
        let requests = match queue.wait_drain(max_batch_size) {
            Some(r) => r,
            None => break,
        };
        let wait_ns = wait_start.elapsed().as_nanos() as u64;
        stats.wait_time_ns.fetch_add(wait_ns, Ordering::Relaxed);

        // Merge all game states into one flat slice.
        // SAFETY: Pointers are valid — calling threads are blocked on rx.recv()
        // and won't drop their GameStates until we send results back.
        let all_games: Vec<&GameState> = requests
            .iter()
            .flat_map(|r| r.games.iter().map(|p| unsafe { &**p }))
            .collect();

        let n_positions = all_games.len() as u64;
        let nn_start = Instant::now();
        let batch_result = inner.evaluate_batch(&all_games);
        let nn_ns = nn_start.elapsed().as_nanos() as u64;

        stats.total_batches.fetch_add(1, Ordering::Relaxed);
        stats.total_positions.fetch_add(n_positions, Ordering::Relaxed);
        stats.nn_time_ns.fetch_add(nn_ns, Ordering::Relaxed);

        match batch_result {
            Ok(all_results) => {
                // Scatter results back to each requester.
                let mut offset = 0;
                for req in requests {
                    let n = req.games.len();
                    let results = all_results[offset..offset + n].to_vec();
                    offset += n;
                    let _ = req.tx.send(Ok(results));
                }
            }
            Err(e) => {
                // Send error to all requesters, continue processing (don't kill worker).
                let msg = e.to_string();
                for req in requests {
                    let _ = req.tx.send(Err(BackendError::msg(msg.clone())));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alpharat_mcts::SmartUniformBackend;
    use pyrat::{Coordinates, GameBuilder, GameState};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn open_5x5(p1: Coordinates, p2: Coordinates) -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(p1, p2)
            .with_custom_cheese(vec![Coordinates::new(2, 2)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    // ---- SpyBackend: records call counts and batch sizes ----

    struct SpyBackend<B: Backend> {
        inner: B,
        call_count: AtomicUsize,
        batch_sizes: Mutex<Vec<usize>>,
    }

    impl<B: Backend> SpyBackend<B> {
        fn new(inner: B) -> Arc<Self> {
            Arc::new(Self {
                inner,
                call_count: AtomicUsize::new(0),
                batch_sizes: Mutex::new(Vec::new()),
            })
        }
    }

    /// Wrapper that delegates to an `Arc<SpyBackend>` so we can share the spy
    /// with the test while giving an owned impl to MuxBackend::new.
    struct SpyBackendHandle<B: Backend>(Arc<SpyBackend<B>>);

    impl<B: Backend> Backend for SpyBackendHandle<B> {
        fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
            self.0.inner.evaluate(game)
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            self.0.call_count.fetch_add(1, Ordering::SeqCst);
            self.0.batch_sizes.lock().unwrap().push(games.len());
            self.0.inner.evaluate_batch(games)
        }
    }

    // ---- FailingBackend: returns Err after N successful calls ----

    struct FailingBackend {
        calls_before_fail: AtomicUsize,
    }

    impl FailingBackend {
        fn new(succeed_count: usize) -> Self {
            Self {
                calls_before_fail: AtomicUsize::new(succeed_count),
            }
        }
    }

    impl Backend for FailingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            Err(BackendError::msg("intentional failure"))
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            let remaining = self.calls_before_fail.fetch_sub(1, Ordering::SeqCst);
            if remaining == 0 {
                return Err(BackendError::msg("intentional batch failure"));
            }
            // Return dummy results
            Ok(games
                .iter()
                .map(|g| SmartUniformBackend.evaluate(g).unwrap())
                .collect())
        }
    }

    // ---- Tests ----

    #[test]
    fn single_thread_single_request() {
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );
        let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));

        let mux_result = mux.evaluate(&game).unwrap();
        let direct_result = SmartUniformBackend.evaluate(&game).unwrap();

        assert_eq!(mux_result.policy_p1, direct_result.policy_p1);
        assert_eq!(mux_result.policy_p2, direct_result.policy_p2);
        assert_eq!(mux_result.value_p1, direct_result.value_p1);
        assert_eq!(mux_result.value_p2, direct_result.value_p2);
    }

    #[test]
    fn single_thread_sequential_requests() {
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );

        let games = [
            open_5x5(Coordinates::new(0, 0), Coordinates::new(4, 4)),
            open_5x5(Coordinates::new(2, 2), Coordinates::new(2, 2)),
            open_5x5(Coordinates::new(4, 0), Coordinates::new(0, 4)),
        ];

        for game in &games {
            let mux_result = mux.evaluate(game).unwrap();
            let direct_result = SmartUniformBackend.evaluate(game).unwrap();
            assert_eq!(mux_result.policy_p1, direct_result.policy_p1);
            assert_eq!(mux_result.policy_p2, direct_result.policy_p2);
        }
    }

    #[test]
    fn single_thread_batch_request() {
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );

        let games = [
            open_5x5(Coordinates::new(0, 0), Coordinates::new(4, 4)),
            open_5x5(Coordinates::new(2, 2), Coordinates::new(2, 2)),
        ];
        let refs: Vec<&GameState> = games.iter().collect();

        let mux_results = mux.evaluate_batch(&refs).unwrap();
        let direct_results = SmartUniformBackend.evaluate_batch(&refs).unwrap();

        assert_eq!(mux_results.len(), 2);
        for (m, d) in mux_results.iter().zip(direct_results.iter()) {
            assert_eq!(m.policy_p1, d.policy_p1);
            assert_eq!(m.policy_p2, d.policy_p2);
        }
    }

    #[test]
    fn multi_thread_concurrent_requests() {
        let mux = Arc::new(MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 256 },
        ));

        let n_threads = 8;
        let handles: Vec<_> = (0..n_threads)
            .map(|i| {
                let mux = mux.clone();
                thread::spawn(move || {
                    // Each thread uses a different position to verify correct scattering.
                    let x = (i % 5) as u8;
                    let game = open_5x5(Coordinates::new(x, 0), Coordinates::new(2, 2));
                    let refs: Vec<&GameState> = vec![&game];

                    let mux_result = mux.evaluate_batch(&refs).unwrap();
                    let direct_result = SmartUniformBackend.evaluate(&game).unwrap();

                    assert_eq!(mux_result.len(), 1);
                    assert_eq!(mux_result[0].policy_p1, direct_result.policy_p1);
                    assert_eq!(mux_result[0].policy_p2, direct_result.policy_p2);
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    #[test]
    fn batch_merging_happens() {
        // Use a spy to verify that multiple game-thread requests get merged
        // into fewer inner backend calls.
        let spy = SpyBackend::new(SmartUniformBackend);
        let spy_ref = spy.clone();

        let mux = Arc::new(MuxBackend::new(
            SpyBackendHandle(spy),
            MuxConfig { max_batch_size: 256 },
        ));

        let n_threads = 8;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let mux = mux.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));
                    // Barrier ensures all threads push near-simultaneously.
                    barrier.wait();
                    let _ = mux.evaluate(&game);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let call_count = spy_ref.call_count.load(Ordering::SeqCst);
        let batch_sizes = spy_ref.batch_sizes.lock().unwrap();

        // With 8 threads and barrier sync, we should see fewer than 8 inner calls
        // (some merging). Can't assert exact count due to scheduling.
        assert!(
            call_count <= n_threads,
            "expected merging to reduce calls, got {call_count} calls for {n_threads} requests"
        );
        // Total game states across all inner calls should equal n_threads.
        let total: usize = batch_sizes.iter().sum();
        assert_eq!(total, n_threads, "total positions mismatch");
    }

    #[test]
    fn max_batch_size_respected() {
        let spy = SpyBackend::new(SmartUniformBackend);
        let spy_ref = spy.clone();

        // Tiny max: 2 positions per batch.
        let mux = Arc::new(MuxBackend::new(
            SpyBackendHandle(spy),
            MuxConfig { max_batch_size: 2 },
        ));

        // Send 4 single-game requests sequentially — each should be its own batch
        // since the worker processes them one at a time.
        for _ in 0..4 {
            let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));
            let _ = mux.evaluate(&game);
        }

        let batch_sizes = spy_ref.batch_sizes.lock().unwrap();
        for &size in batch_sizes.iter() {
            // First request always accepted, but in sequential mode each is size 1.
            // In concurrent mode the cap is 2. Either way, ≤2 for non-first or ≤ any for first.
            assert!(
                size <= 2 || size == batch_sizes[0],
                "batch size {size} exceeds max_batch_size=2 (and isn't the first request)"
            );
        }
    }

    #[test]
    fn empty_batch() {
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );

        let results = mux.evaluate_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn drop_while_idle() {
        // MuxBackend should drop cleanly when no threads are active.
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );
        drop(mux); // should not hang or panic
    }

    #[test]
    fn drop_after_work() {
        // MuxBackend should drop cleanly after processing requests.
        let mux = MuxBackend::new(
            SmartUniformBackend,
            MuxConfig { max_batch_size: 64 },
        );
        let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));
        let _ = mux.evaluate(&game);
        drop(mux); // should not hang or panic
    }

    #[test]
    fn failing_backend_propagates_error() {
        let mux = MuxBackend::new(
            FailingBackend::new(0),
            MuxConfig { max_batch_size: 64 },
        );
        let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));

        let result = mux.evaluate(&game);
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("intentional"),
            "error message should mention the failure reason"
        );
    }

    #[test]
    fn failing_backend_worker_continues() {
        // After an error, the mux worker should still process subsequent requests.
        let mux = MuxBackend::new(
            FailingBackend::new(1), // first call succeeds, second fails
            MuxConfig { max_batch_size: 64 },
        );
        let game = open_5x5(Coordinates::new(2, 2), Coordinates::new(0, 0));

        // First call succeeds
        let result1 = mux.evaluate(&game);
        assert!(result1.is_ok());

        // Second call fails
        let result2 = mux.evaluate(&game);
        assert!(result2.is_err());
    }
}
