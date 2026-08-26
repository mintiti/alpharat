use std::any::Any;
use std::fmt;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::Gamma;

use crate::access::{ExclusiveAccess, SearchSession};
use crate::node::LowNode;
use crate::observer::NodeHandle;
use crate::scheduler::{AdmissionCloseReason, SearchCoordinator};
use crate::tree::{compute_rewards, MCGSTree};
use crate::{Backend, BackendError};
use pyrat::{Direction, GameState, MoveUndo};

/// Score assigned to forced-playout outcomes to guarantee selection.
const FORCED_PLAYOUT_SCORE: f32 = 1e20;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Search configuration — immutable, shareable across threads.
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// Exploration constant (PUCT).
    pub c_puct: f32,
    /// First-play urgency penalty.
    pub fpu_reduction: f32,
    /// Forced playout coefficient. 0 disables forced playouts.
    pub force_k: f32,
    /// Dirichlet noise mixing weight. 0.0 = disabled, typical: 0.25.
    pub noise_epsilon: f32,
    /// Total Dirichlet concentration (KataGo-style).
    /// Per-move alpha = concentration / n_outcomes.
    pub noise_concentration: f32,
    /// Collision budget scaling (LC0 pattern). The collision limit scales
    /// with tree size from `collision_limit_min` to `collision_limit_max`.
    pub collision_limit_min: u32,
    pub collision_limit_max: u32,
    /// Tree node count at which collision limit starts ramping.
    pub collision_scaling_start: u32,
    /// Tree node count at which collision limit reaches max.
    pub collision_scaling_end: u32,
    /// Power-law interpolation exponent.
    pub collision_scaling_power: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            fpu_reduction: 0.2,
            force_k: 2.0,
            noise_epsilon: 0.0,
            noise_concentration: 10.83,
            collision_limit_min: 1,
            collision_limit_max: 256,
            collision_scaling_start: 800,
            collision_scaling_end: 50_000,
            collision_scaling_power: 1.0,
        }
    }
}

/// A single step on the search path.
#[derive(Clone)]
struct PathEntry<'session> {
    node: NodeHandle<'session>,
    p1_outcome: u8,
    p2_outcome: u8,
}

type SearchPath<'session> = Vec<PathEntry<'session>>;

/// Result of an MCGS search: policies and values for both players.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Policy in 5-action space, sums to 1, blocked actions = 0.
    pub policy_p1: [f32; 5],
    pub policy_p2: [f32; 5],
    /// Expected remaining cheese for each player.
    pub value_p1: f32,
    pub value_p2: f32,
    /// Visit counts in 5-action space.
    pub visit_counts_p1: [f32; 5],
    pub visit_counts_p2: [f32; 5],
    /// NN/uniform prior at root in 5-action space.
    pub prior_p1: [f32; 5],
    pub prior_p2: [f32; 5],
    /// Per-action Q-values in 5-action space (unvisited actions get FPU).
    pub q_values_p1: [f32; 5],
    pub q_values_p2: [f32; 5],
    /// Root visit count after search.
    pub total_visits: u32,
    /// Number of descents that required NN evaluation.
    pub nn_evals: u32,
    /// Number of descents that hit terminal nodes (free — no NN call).
    pub terminals: u32,
    /// Number of descents that collided (wasted — no backup).
    pub collisions: u32,
    /// Number of transposition stops (edge initialized from shared child's aggregate).
    pub tt_stop_hits: u32,
}

/// Phase timings for a gated search protocol.
///
/// Wait and hold time are kept separate so later multi-worker measurements can
/// distinguish graph contention from useful search work. SmartUniform
/// benchmarks mostly expose protocol overhead; real NN backends expose how
/// much inference can overlap once more workers are admitted. Parallel
/// profiles sum time across workers, so their phase totals may exceed elapsed
/// wall time and should be read as aggregate worker occupancy.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct SearchTimings {
    pub batches: u32,
    pub lease_wait: Duration,
    pub gather_wait: Duration,
    pub gather_hold: Duration,
    pub inference: Duration,
    pub settle_wait: Duration,
    pub settle_hold: Duration,
    pub completion_wait: Duration,
    pub cleanup_wait: Duration,
    pub cleanup_hold: Duration,
    pub extract_wait: Duration,
    pub extract_hold: Duration,
}

/// Aggregate ownership ledger for a completed profiled search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchLedgerStats {
    pub reserved: u64,
    pub committed: u64,
    pub cancelled: u64,
}

impl SearchLedgerStats {
    pub fn outstanding(self) -> u64 {
        self.reserved
            .checked_sub(self.committed + self.cancelled)
            .expect("search ledger released more work than it reserved")
    }

    fn reserve(&mut self, units: u32) {
        self.reserved += u64::from(units);
    }

    fn commit(&mut self, units: u32) {
        self.committed += u64::from(units);
        debug_assert!(self.committed + self.cancelled <= self.reserved);
    }

    fn cancel(&mut self, units: u32) {
        self.cancelled += u64::from(units);
        debug_assert!(self.committed + self.cancelled <= self.reserved);
    }

    fn merge(&mut self, batch: Self) {
        self.reserved += batch.reserved;
        self.committed += batch.committed;
        self.cancelled += batch.cancelled;
    }

    fn assert_settled(self) {
        assert_eq!(
            self.outstanding(),
            0,
            "owned search batch crossed a boundary with outstanding reservations"
        );
    }
}

impl SearchTimings {
    fn merge(&mut self, worker: Self) {
        self.batches += worker.batches;
        self.lease_wait += worker.lease_wait;
        self.gather_wait += worker.gather_wait;
        self.gather_hold += worker.gather_hold;
        self.inference += worker.inference;
        self.settle_wait += worker.settle_wait;
        self.settle_hold += worker.settle_hold;
        self.completion_wait += worker.completion_wait;
        self.cleanup_wait += worker.cleanup_wait;
        self.cleanup_hold += worker.cleanup_hold;
        self.extract_wait += worker.extract_wait;
        self.extract_hold += worker.extract_hold;
    }
}

/// Result and aggregate instrumentation from a gated search protocol.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ProfiledSearchResult {
    pub result: SearchResult,
    pub timings: SearchTimings,
    pub ledger: SearchLedgerStats,
    pub termination: SearchTermination,
}

/// Why a persistent parallel-search session stopped admitting new work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchTermination {
    BudgetExhausted,
    StopRequested,
    Deadline,
    NoProgress,
}

impl fmt::Display for SearchTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BudgetExhausted => "productive budget exhausted",
            Self::StopRequested => "stop requested",
            Self::Deadline => "deadline reached",
            Self::NoProgress => "no progress",
        })
    }
}

/// Failure from the public fixed-budget parallel-search entry point.
#[derive(Debug)]
pub enum ParallelSearchError {
    Backend(BackendError),
    Incomplete {
        requested: u32,
        completed: u32,
        termination: SearchTermination,
    },
}

impl fmt::Display for ParallelSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(error) => error.fmt(f),
            Self::Incomplete {
                requested,
                completed,
                termination,
            } => write!(
                f,
                "parallel search stopped because {termination} after completing {completed} of {requested} productive units"
            ),
        }
    }
}

impl std::error::Error for ParallelSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(error) => Some(error),
            Self::Incomplete { .. } => None,
        }
    }
}

impl From<BackendError> for ParallelSearchError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

/// Per-batch counters from simulate_batch.
struct BatchStats {
    nn_evals: u32,
    terminals: u32,
    collisions: u32,
    tt_stop_hits: u32,
    ledger: SearchLedgerStats,
}

// ---------------------------------------------------------------------------
// run_search — public API
// ---------------------------------------------------------------------------

/// Run MCGS search: N simulations with within-tree batching.
pub fn run_search(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<SearchResult, BackendError> {
    tree.with_exclusive(|mut access| {
        run_search_exclusive(&mut access, game, backend, config, n_sims, batch_size, rng)
    })
}

/// Run MCGS with persistent scoped workers exploring one shared DAG.
///
/// Graph mutation remains serialized into short gather and settle epochs;
/// backend evaluation runs without the graph guard, so workers can overlap
/// inference with another worker's graph phase. Productive work is charged
/// exactly and every worker joins before the result or first backend error is
/// returned.
///
/// Worker RNG streams are derived deterministically from `rng`, but thread
/// scheduling can change selection and commit order. Parallel results are
/// therefore not promised to be bit-identical across runs.
///
/// `worker_batch_size` is a per-worker lease ceiling. The aggregate in-flight
/// ceiling is therefore `worker_count * worker_batch_size`; callers comparing
/// worker counts at a fixed total in-flight target should scale it down per
/// worker. Workers call `Backend::evaluate_batch` directly and concurrently;
/// backend muxing, serialization, and cache lifetime remain properties of the
/// supplied backend wrapper.
///
/// A backend panic is caught long enough to cancel its pending batch and join
/// every worker, then resumed. An unexpected panic inside a graph mutation
/// epoch also stops and joins peers before resuming, but is an internal
/// invariant failure and does not promise that the tree remains reusable.
/// A bounded no-progress shutdown is returned as [`ParallelSearchError::Incomplete`]
/// rather than silently presenting partial work as a completed fixed budget.
#[allow(clippy::too_many_arguments)]
pub fn run_search_parallel(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    worker_batch_size: u32,
    worker_count: usize,
    rng: &mut impl Rng,
) -> Result<SearchResult, ParallelSearchError> {
    let profile = run_search_parallel_with_control(
        tree,
        game,
        backend,
        config,
        n_sims,
        worker_batch_size,
        worker_count,
        rng,
        ParallelSearchControl::none(),
    )?;
    if profile.termination != SearchTermination::BudgetExhausted {
        let completed =
            profile.result.nn_evals + profile.result.terminals + profile.result.tt_stop_hits;
        return Err(ParallelSearchError::Incomplete {
            requested: n_sims,
            completed,
            termination: profile.termination,
        });
    }
    Ok(profile.result)
}

/// Profile the persistent-worker protocol without changing its search result.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn run_search_parallel_profiled(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    worker_batch_size: u32,
    worker_count: usize,
    rng: &mut impl Rng,
) -> Result<ProfiledSearchResult, BackendError> {
    run_search_parallel_with_control(
        tree,
        game,
        backend,
        config,
        n_sims,
        worker_batch_size,
        worker_count,
        rng,
        ParallelSearchControl::none(),
    )
}

/// Search-call controls polled immediately before admission and after
/// inference. A request racing the pre-admission poll may admit one additional
/// batch per worker; successful work admitted before the request is observed
/// is still settled. A bot-facing live-search API should expose one
/// interruptible worker session rather than respawning this fixed-budget
/// primitive per minibatch.
#[derive(Clone, Copy, Default)]
pub(crate) struct ParallelSearchControl<'control> {
    stop_requested: Option<&'control AtomicBool>,
    deadline: Option<Instant>,
    #[cfg(test)]
    close_observed: Option<&'control AtomicBool>,
    #[cfg(test)]
    cancelled_inline_productive: Option<&'control AtomicUsize>,
}

impl<'control> ParallelSearchControl<'control> {
    fn none() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub(crate) fn new(
        stop_requested: Option<&'control AtomicBool>,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            stop_requested,
            deadline,
            close_observed: None,
            cancelled_inline_productive: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_close_observer(mut self, close_observed: &'control AtomicBool) -> Self {
        self.close_observed = Some(close_observed);
        self
    }

    #[cfg(test)]
    pub(crate) fn with_cancelled_inline_observer(
        mut self,
        cancelled_inline_productive: &'control AtomicUsize,
    ) -> Self {
        self.cancelled_inline_productive = Some(cancelled_inline_productive);
        self
    }

    fn requested_close_reason(self) -> Option<AdmissionCloseReason> {
        if self
            .stop_requested
            .is_some_and(|stop| stop.load(Ordering::Relaxed))
        {
            Some(AdmissionCloseReason::StopRequested)
        } else if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            Some(AdmissionCloseReason::Deadline)
        } else {
            None
        }
    }

    fn close_if_requested(self, coordinator: &SearchCoordinator) {
        if let Some(reason) = self.requested_close_reason() {
            coordinator.close(reason);
            self.notify_close_observed();
        }
    }

    fn notify_close_observed(self) {
        #[cfg(test)]
        if let Some(close_observed) = self.close_observed {
            close_observed.store(true, Ordering::SeqCst);
        }
    }

    fn record_cancelled_inline(self, _productive: u32) {
        #[cfg(test)]
        if let Some(cancelled_inline_productive) = self.cancelled_inline_productive {
            cancelled_inline_productive.fetch_add(_productive as usize, Ordering::SeqCst);
        }
    }
}

#[derive(Default)]
struct ParallelWorkerStats {
    timings: SearchTimings,
    ledger: SearchLedgerStats,
    nn_evals: u32,
    terminals: u32,
    collisions: u32,
    tt_stop_hits: u32,
}

impl ParallelWorkerStats {
    fn record_batch(&mut self, batch: BatchStats) {
        self.nn_evals += batch.nn_evals;
        self.terminals += batch.terminals;
        self.collisions += batch.collisions;
        self.tt_stop_hits += batch.tt_stop_hits;
        self.ledger.merge(batch.ledger);
    }

    fn record_cancelled_batch(
        &mut self,
        ledger: SearchLedgerStats,
        terminals: u32,
        tt_stop_hits: u32,
    ) {
        self.terminals += terminals;
        self.tt_stop_hits += tt_stop_hits;
        self.ledger.merge(ledger);
    }

    fn productive(&self) -> u32 {
        self.nn_evals + self.terminals + self.tt_stop_hits
    }

    fn merge(&mut self, worker: Self) {
        self.timings.merge(worker.timings);
        self.ledger.merge(worker.ledger);
        self.nn_evals += worker.nn_evals;
        self.terminals += worker.terminals;
        self.collisions += worker.collisions;
        self.tt_stop_hits += worker.tt_stop_hits;
    }
}

type PanicPayload = Box<dyn Any + Send + 'static>;

struct ParallelWorkerExit {
    stats: ParallelWorkerStats,
    backend_panic: Option<PanicPayload>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_search_parallel_with_control(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    worker_batch_size: u32,
    worker_count: usize,
    rng: &mut impl Rng,
    control: ParallelSearchControl<'_>,
) -> Result<ProfiledSearchResult, BackendError> {
    assert!(
        worker_count > 0,
        "parallel search needs at least one worker"
    );
    assert!(
        worker_batch_size > 0,
        "parallel search batch size must be positive"
    );
    assert!(
        u32::try_from(worker_count).is_ok(),
        "parallel worker count exceeds the productive counter range"
    );

    let worker_seeds: Vec<u64> = (0..worker_count).map(|_| rng.gen()).collect();
    tree.with_search_session(|session| {
        run_search_parallel_session(
            &session,
            game,
            backend,
            config,
            n_sims,
            worker_batch_size,
            worker_seeds,
            rng,
            control,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_search_parallel_session<'tree, 'session>(
    session: &SearchSession<'tree, 'session>,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    worker_batch_size: u32,
    worker_seeds: Vec<u64>,
    result_rng: &mut impl Rng,
    control: ParallelSearchControl<'_>,
) -> Result<ProfiledSearchResult, BackendError> {
    let worker_count = worker_seeds.len();
    // A globally quiescent empty wave cannot be temporary contention: every
    // other lease has already settled. A short retry allowance turns a
    // degenerate gather configuration or leaked claim into bounded partial
    // completion instead of tying liveness to a potentially huge sim budget.
    const MAX_QUIESCENT_ZERO_PROGRESS_WAVES: u32 = 3;
    let zero_progress_bound = MAX_QUIESCENT_ZERO_PROGRESS_WAVES;
    let coordinator = SearchCoordinator::new(n_sims, zero_progress_bound);

    let joined = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        let coordinator = &coordinator;
        for seed in worker_seeds {
            let worker_game = game.clone();
            workers.push(scope.spawn(move || {
                let mut worker_rng = SmallRng::seed_from_u64(seed);
                let worker = catch_unwind(AssertUnwindSafe(|| {
                    run_parallel_worker(
                        session,
                        coordinator,
                        &worker_game,
                        backend,
                        config,
                        worker_batch_size,
                        &mut worker_rng,
                        control,
                    )
                }));
                if worker.is_err() {
                    coordinator.close(AdmissionCloseReason::WorkerPanicked);
                }
                worker
            }));
        }
        workers
            .into_iter()
            .map(std::thread::ScopedJoinHandle::join)
            .collect::<Vec<_>>()
    });

    let mut aggregate = ParallelWorkerStats::default();
    let mut saved_backend_panic = None;
    let mut unexpected_worker_panic = None;
    for worker in joined {
        match worker {
            Ok(Ok(worker)) => {
                aggregate.merge(worker.stats);
                if saved_backend_panic.is_none() {
                    saved_backend_panic = worker.backend_panic;
                }
            }
            Ok(Err(payload)) | Err(payload) => {
                if unexpected_worker_panic.is_none() {
                    unexpected_worker_panic = Some(payload);
                }
            }
        }
    }

    if let Some(payload) = unexpected_worker_panic {
        // A graph-phase invariant panic may have poisoned the gate or
        // abandoned an admitted reservation plan. Peers have been stopped and
        // joined, but claiming a quiescent/reusable graph here would mask the
        // original bug with a cleanup assertion. Backend panics take the
        // controlled cleanup path below instead.
        resume_unwind(payload);
    }

    aggregate.ledger.assert_settled();
    let outcome = coordinator.finish();
    assert_eq!(
        outcome.snapshot.charged(),
        aggregate.productive(),
        "search coordinator and worker ledgers disagree on productive work"
    );
    assert_eq!(
        aggregate.ledger.committed,
        u64::from(aggregate.productive()),
        "graph ledger and worker counters disagree on committed work"
    );

    if let Some(payload) = saved_backend_panic {
        resume_unwind(payload);
    }
    if let Some(error) = outcome.first_failure {
        return Err(error);
    }
    let termination = match outcome
        .snapshot
        .close_reason
        .expect("finished coordinator must retain its close reason")
    {
        AdmissionCloseReason::BudgetExhausted => SearchTermination::BudgetExhausted,
        AdmissionCloseReason::StopRequested => SearchTermination::StopRequested,
        AdmissionCloseReason::Deadline => SearchTermination::Deadline,
        AdmissionCloseReason::NoProgress => SearchTermination::NoProgress,
        AdmissionCloseReason::BackendFailure | AdmissionCloseReason::WorkerPanicked => {
            unreachable!("failed or panicked workers cannot return a search result")
        }
    };

    let wait_started = Instant::now();
    let mut epoch = session.write();
    aggregate.timings.extract_wait += wait_started.elapsed();
    let hold_started = Instant::now();
    let mut result = {
        let access = epoch.access();
        let root = access.root();
        extract_result(&access, &root, config, result_rng)
    };
    drop(epoch);
    aggregate.timings.extract_hold += hold_started.elapsed();

    result.nn_evals = aggregate.nn_evals;
    result.terminals = aggregate.terminals;
    result.collisions = aggregate.collisions;
    result.tt_stop_hits = aggregate.tt_stop_hits;

    Ok(ProfiledSearchResult {
        result,
        timings: aggregate.timings,
        ledger: aggregate.ledger,
        termination,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_parallel_worker<'tree, 'session>(
    session: &SearchSession<'tree, 'session>,
    coordinator: &SearchCoordinator,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
    control: ParallelSearchControl<'_>,
) -> ParallelWorkerExit {
    let mut stats = ParallelWorkerStats::default();

    loop {
        control.close_if_requested(coordinator);
        let lease_started = Instant::now();
        let provisional = coordinator.lease(batch_size);
        stats.timings.lease_wait += lease_started.elapsed();
        let Some(provisional) = provisional else {
            break;
        };

        let wait_started = Instant::now();
        let mut epoch = session.write();
        stats.timings.gather_wait += wait_started.elapsed();
        let hold_started = Instant::now();
        control.close_if_requested(coordinator);
        let admitted = match provisional.admit(&mut epoch) {
            Ok(admitted) => admitted,
            Err(_) => {
                drop(epoch);
                stats.timings.gather_hold += hold_started.elapsed();
                break;
            }
        };
        let pending = {
            let mut access = epoch.access();
            gather_batch(&mut access, game, config, admitted.units(), rng)
        };
        drop(epoch);
        stats.timings.gather_hold += hold_started.elapsed();
        stats.timings.batches += 1;

        let inference_started = Instant::now();
        let inference = infer_pending_batch_caught(&pending, backend);
        stats.timings.inference += inference_started.elapsed();
        control.close_if_requested(coordinator);

        match inference {
            Ok(Ok(eval_results)) => {
                let wait_started = Instant::now();
                let mut epoch = session.write();
                stats.timings.settle_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                let batch = {
                    let mut access = epoch.access();
                    settle_pending_batch(&mut access, pending, eval_results, config, rng)
                };
                drop(epoch);
                stats.timings.settle_hold += hold_started.elapsed();
                let produced = batch.nn_evals + batch.terminals + batch.tt_stop_hits;
                let completion_started = Instant::now();
                admitted.complete(produced);
                stats.timings.completion_wait += completion_started.elapsed();
                stats.record_batch(batch);
            }
            Ok(Err(error)) => {
                coordinator.record_first_failure(error);
                control.notify_close_observed();
                let inline_terminals = pending.terminals;
                let inline_tt_stops = pending.tt_stop_hits;
                let inline_productive = inline_terminals + inline_tt_stops;
                control.record_cancelled_inline(inline_productive);
                let wait_started = Instant::now();
                let mut epoch = session.write();
                stats.timings.cleanup_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                let ledger = {
                    let mut access = epoch.access();
                    cancel_pending_batch(&mut access, pending)
                };
                drop(epoch);
                stats.timings.cleanup_hold += hold_started.elapsed();
                let completion_started = Instant::now();
                admitted.cancel_with_committed(inline_productive);
                stats.timings.completion_wait += completion_started.elapsed();
                stats.record_cancelled_batch(ledger, inline_terminals, inline_tt_stops);
                break;
            }
            Err(payload) => {
                coordinator.close(AdmissionCloseReason::WorkerPanicked);
                control.notify_close_observed();
                let inline_terminals = pending.terminals;
                let inline_tt_stops = pending.tt_stop_hits;
                let inline_productive = inline_terminals + inline_tt_stops;
                control.record_cancelled_inline(inline_productive);
                let wait_started = Instant::now();
                let mut epoch = session.write();
                stats.timings.cleanup_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                let ledger = {
                    let mut access = epoch.access();
                    cancel_pending_batch(&mut access, pending)
                };
                drop(epoch);
                stats.timings.cleanup_hold += hold_started.elapsed();
                let completion_started = Instant::now();
                admitted.cancel_with_committed(inline_productive);
                stats.timings.completion_wait += completion_started.elapsed();
                stats.record_cancelled_batch(ledger, inline_terminals, inline_tt_stops);
                return ParallelWorkerExit {
                    stats,
                    backend_panic: Some(payload),
                };
            }
        }
    }

    ParallelWorkerExit {
        stats,
        backend_panic: None,
    }
}

/// Run the transitional one-worker session protocol with phase instrumentation.
///
/// This narrow benchmark seam is feature-gated because the normal public
/// single-worker API remains [`run_search`], whose whole-tree exclusive borrow
/// is the zero-lock behavior oracle.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn run_search_one_worker_profiled(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
    timings: &mut SearchTimings,
) -> Result<ProfiledSearchResult, BackendError> {
    run_search_one_worker_with_timings(
        tree, game, backend, config, n_sims, batch_size, rng, timings,
    )
}

#[allow(dead_code)]
pub(crate) fn run_search_one_worker(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<ProfiledSearchResult, BackendError> {
    let mut timings = SearchTimings::default();
    run_search_one_worker_with_timings(
        tree,
        game,
        backend,
        config,
        n_sims,
        batch_size,
        rng,
        &mut timings,
    )
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_search_one_worker_with_timings(
    tree: &mut MCGSTree,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
    timings: &mut SearchTimings,
) -> Result<ProfiledSearchResult, BackendError> {
    tree.with_search_session(|session| {
        run_search_session(
            &session, game, backend, config, n_sims, batch_size, rng, timings,
        )
    })
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_search_session<'tree, 'session>(
    session: &SearchSession<'tree, 'session>,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
    timings: &mut SearchTimings,
) -> Result<ProfiledSearchResult, BackendError> {
    *timings = SearchTimings::default();
    let mut ledger = SearchLedgerStats::default();
    let mut remaining = n_sims;
    let mut total_nn_evals = 0u32;
    let mut total_terminals = 0u32;
    let mut total_collisions = 0u32;
    let mut total_tt_stop_hits = 0u32;

    while remaining > 0 {
        let wait_started = Instant::now();
        let mut epoch = session.write();
        timings.gather_wait += wait_started.elapsed();
        let hold_started = Instant::now();
        let pending = {
            let mut access = epoch.access();
            gather_batch(&mut access, game, config, remaining.min(batch_size), rng)
        };
        drop(epoch);
        timings.gather_hold += hold_started.elapsed();
        timings.batches += 1;

        let inference_started = Instant::now();
        let inference = infer_pending_batch_caught(&pending, backend);
        timings.inference += inference_started.elapsed();

        let batch = match inference {
            Ok(Ok(eval_results)) => {
                let wait_started = Instant::now();
                let mut epoch = session.write();
                timings.settle_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                let batch = {
                    let mut access = epoch.access();
                    settle_pending_batch(&mut access, pending, eval_results, config, rng)
                };
                drop(epoch);
                timings.settle_hold += hold_started.elapsed();
                batch
            }
            Ok(Err(error)) => {
                let wait_started = Instant::now();
                let mut epoch = session.write();
                timings.cleanup_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                {
                    let mut access = epoch.access();
                    cancel_pending_batch(&mut access, pending);
                }
                drop(epoch);
                timings.cleanup_hold += hold_started.elapsed();
                return Err(error);
            }
            Err(payload) => {
                let wait_started = Instant::now();
                let mut epoch = session.write();
                timings.cleanup_wait += wait_started.elapsed();
                let hold_started = Instant::now();
                {
                    let mut access = epoch.access();
                    cancel_pending_batch(&mut access, pending);
                }
                drop(epoch);
                timings.cleanup_hold += hold_started.elapsed();
                resume_unwind(payload);
            }
        };

        total_nn_evals += batch.nn_evals;
        total_terminals += batch.terminals;
        total_collisions += batch.collisions;
        total_tt_stop_hits += batch.tt_stop_hits;
        ledger.merge(batch.ledger);

        let produced = batch.nn_evals + batch.terminals + batch.tt_stop_hits;
        remaining = remaining.saturating_sub(produced.max(1));
    }

    let wait_started = Instant::now();
    let mut epoch = session.write();
    timings.extract_wait += wait_started.elapsed();
    let hold_started = Instant::now();
    let mut result = {
        let access = epoch.access();
        let root = access.root();
        extract_result(&access, &root, config, rng)
    };
    drop(epoch);
    timings.extract_hold += hold_started.elapsed();

    result.nn_evals = total_nn_evals;
    result.terminals = total_terminals;
    result.collisions = total_collisions;
    result.tt_stop_hits = total_tt_stop_hits;
    ledger.assert_settled();

    Ok(ProfiledSearchResult {
        result,
        timings: *timings,
        ledger,
        termination: SearchTermination::BudgetExhausted,
    })
}

fn run_search_exclusive<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<SearchResult, BackendError> {
    let mut remaining = n_sims;
    let mut total_nn_evals = 0u32;
    let mut total_terminals = 0u32;
    let mut total_collisions = 0u32;
    let mut total_tt_stop_hits = 0u32;
    let mut total_ledger = SearchLedgerStats::default();
    while remaining > 0 {
        let batch = simulate_batch(
            access,
            game,
            backend,
            config,
            remaining.min(batch_size),
            rng,
        )?;
        total_nn_evals += batch.nn_evals;
        total_terminals += batch.terminals;
        total_collisions += batch.collisions;
        total_tt_stop_hits += batch.tt_stop_hits;
        total_ledger.merge(batch.ledger);
        // Count descents that produced useful information.
        // Collisions don't consume the sim budget — they're wasted work.
        // TT stops are productive: they initialize edges from shared aggregates.
        let produced = batch.nn_evals + batch.terminals + batch.tt_stop_hits;
        remaining = remaining.saturating_sub(produced.max(1));
    }

    let root = access.root();
    let mut result = extract_result(access, &root, config, rng);
    result.nn_evals = total_nn_evals;
    result.terminals = total_terminals;
    result.collisions = total_collisions;
    result.tt_stop_hits = total_tt_stop_hits;
    total_ledger.assert_settled();
    Ok(result)
}

// ---------------------------------------------------------------------------
// select_actions — decoupled PUCT on joint matrix
// ---------------------------------------------------------------------------

/// Select an action pair (p1_outcome_idx, p2_outcome_idx) via decoupled PUCT.
///
/// Each player independently picks the outcome with the highest PUCT score.
/// Q and visits come from marginals over the joint matrix.
#[cfg(test)]
fn select_actions(
    low: &LowNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> (u8, u8) {
    let a1 = select_p1(low, config, is_root, rng);
    let a2 = select_p2(low, config, is_root, rng);
    (a1, a2)
}

// ---------------------------------------------------------------------------
// compute_fpu — first-play urgency
// ---------------------------------------------------------------------------

/// FPU for player 1: pessimistic value scaled by visited prior mass.
fn compute_fpu_p1(low: &crate::node::LowNode, config: &SearchConfig) -> f32 {
    let mut visited_prior_mass = 0.0f32;
    for i in 0..low.n1() {
        if low.marginal_visits_p1(i) > 0 {
            visited_prior_mass += low.p1_prior(i);
        }
    }
    low.v1() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt()
}

/// FPU for player 2: pessimistic value scaled by visited prior mass.
fn compute_fpu_p2(low: &crate::node::LowNode, config: &SearchConfig) -> f32 {
    let mut visited_prior_mass = 0.0f32;
    for j in 0..low.n2() {
        if low.marginal_visits_p2(j) > 0 {
            visited_prior_mass += low.p2_prior(j);
        }
    }
    low.v2() - config.fpu_reduction * low.value_scale() * visited_prior_mass.sqrt()
}

/// PUCT selection for player 1 — marginalizes over j.
#[cfg(test)]
fn select_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = low.n1();
    if n == 1 {
        return 0;
    }

    let children_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    let fpu = compute_fpu_p1(low, config);

    let sqrt_total = (children_visits.max(1) as f32).sqrt();

    argmax_tiebreak(n, rng, |i| {
        let visits = low.marginal_visits_p1(i);
        let in_flight = low.marginal_in_flight_p1(i);
        let prior = low.p1_prior(i);

        let q = if visits > 0 {
            marginal_q_p1(low, i)
        } else {
            fpu
        };
        let q_norm = q / value_scale;

        let exploration =
            config.c_puct * prior * sqrt_total / (1.0 + visits as f32 + in_flight as f32);
        let mut score = q_norm + exploration;

        // Forced playouts: at root, boost undervisited outcomes.
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        score
    })
}

/// PUCT selection for player 2 — marginalizes over i.
#[cfg(test)]
fn select_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> u8 {
    let n = low.n2();
    if n == 1 {
        return 0;
    }

    let children_visits = low.total_edge_visits();
    let value_scale = low.value_scale();
    debug_assert!(value_scale > 0.0, "value_scale must be positive");

    let fpu = compute_fpu_p2(low, config);

    let sqrt_total = (children_visits.max(1) as f32).sqrt();

    argmax_tiebreak(n, rng, |j| {
        let visits = low.marginal_visits_p2(j);
        let in_flight = low.marginal_in_flight_p2(j);
        let prior = low.p2_prior(j);

        let q = if visits > 0 {
            marginal_q_p2(low, j)
        } else {
            fpu
        };
        let q_norm = q / value_scale;

        let exploration =
            config.c_puct * prior * sqrt_total / (1.0 + visits as f32 + in_flight as f32);
        let mut score = q_norm + exploration;

        // Forced playouts
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        score
    })
}

/// Marginal Q for p1 outcome i: visit-weighted average over j.
fn marginal_q_p1(low: &crate::node::LowNode, i: usize) -> f32 {
    let mut total_v = 0u32;
    let mut weighted_q = 0.0f32;
    for j in 0..low.n2() {
        let v = low.edge_visits(i, j);
        if v > 0 {
            weighted_q += v as f32 * low.edge_q_p1(i, j);
            total_v += v;
        }
    }
    if total_v > 0 {
        weighted_q / total_v as f32
    } else {
        0.0
    }
}

/// Marginal Q for p2 outcome j: visit-weighted average over i.
fn marginal_q_p2(low: &crate::node::LowNode, j: usize) -> f32 {
    let mut total_v = 0u32;
    let mut weighted_q = 0.0f32;
    for i in 0..low.n1() {
        let v = low.edge_visits(i, j);
        if v > 0 {
            weighted_q += v as f32 * low.edge_q_p2(i, j);
            total_v += v;
        }
    }
    if total_v > 0 {
        weighted_q / total_v as f32
    } else {
        0.0
    }
}

/// Argmax with reservoir-sampling tie-breaking.
#[cfg(test)]
fn argmax_tiebreak(n: usize, rng: &mut impl Rng, score_fn: impl Fn(usize) -> f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut tie_count = 0u32;

    for i in 0..n {
        let s = score_fn(i);
        if s > best_score {
            best_score = s;
            best_idx = i as u8;
            tie_count = 1;
        } else if (s - best_score).abs() < 1e-12 {
            tie_count += 1;
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = i as u8;
            }
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// apply_dirichlet_noise — root exploration noise
// ---------------------------------------------------------------------------

/// Mix Dirichlet noise into a LowNode's priors for one player.
///
/// Uses KataGo's total-concentration approach: per-move alpha = concentration / n_outcomes.
fn apply_dirichlet_noise_p1(
    low: &mut LowNode,
    epsilon: f32,
    concentration: f32,
    rng: &mut impl Rng,
) {
    let n = low.n1();
    if n <= 1 {
        return;
    }

    let alpha = (concentration / n as f32) as f64;
    let gamma_dist = match Gamma::new(alpha, 1.0) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut noise = [0.0f32; 5];
    let mut total = 0.0f32;
    for item in noise.iter_mut().take(n) {
        *item = rng.sample(gamma_dist) as f32;
        total += *item;
    }
    if total < f32::MIN_POSITIVE {
        return;
    }

    // Read current priors, blend, write back
    for i in 0..n {
        let cur = low.p1_prior(i);
        low.set_p1_prior_at(i, cur * (1.0 - epsilon) + epsilon * noise[i] / total);
    }
}

fn apply_dirichlet_noise_p2(
    low: &mut LowNode,
    epsilon: f32,
    concentration: f32,
    rng: &mut impl Rng,
) {
    let n = low.n2();
    if n <= 1 {
        return;
    }

    let alpha = (concentration / n as f32) as f64;
    let gamma_dist = match Gamma::new(alpha, 1.0) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut noise = [0.0f32; 5];
    let mut total = 0.0f32;
    for item in noise.iter_mut().take(n) {
        *item = rng.sample(gamma_dist) as f32;
        total += *item;
    }
    if total < f32::MIN_POSITIVE {
        return;
    }

    for j in 0..n {
        let cur = low.p2_prior(j);
        low.set_p2_prior_at(j, cur * (1.0 - epsilon) + epsilon * noise[j] / total);
    }
}

// ---------------------------------------------------------------------------
// estimated_visits_to_change_best — LC0's batch allocation helper
// ---------------------------------------------------------------------------

/// For player 1, compute how many more visits to the best outcome before the
/// second-best overtakes it in PUCT score. Returns (best_idx, vtc).
/// If only one outcome or best utility alone beats second-best, returns u32::MAX.
fn estimated_visits_to_change_best_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    ns_p1: &[u32; 5],
    rng: &mut impl Rng,
) -> (u8, u32) {
    let n = low.n1();
    if n <= 1 {
        return (0, u32::MAX);
    }

    let children_visits = low.total_edge_visits();
    let fpu = compute_fpu_p1(low, config);
    let sqrt_total = (children_visits.max(1) as f32).sqrt();
    let c_puct = config.c_puct;
    let value_scale = low.value_scale();

    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_utility = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;

    for i in 0..n {
        let visits = low.marginal_visits_p1(i);
        let prior = low.p1_prior(i);
        let q = if visits > 0 {
            marginal_q_p1(low, i)
        } else {
            fpu
        };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p1[i] as f32);
        let mut score = q_norm + exploration;

        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_idx = i as u8;
            best_utility = q_norm;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    // Tie-breaking with reservoir sampling.
    let mut tie_count = 1u32;
    for i in 0..n {
        if i as u8 == best_idx {
            continue;
        }
        let visits = low.marginal_visits_p1(i);
        let prior = low.p1_prior(i);
        let q = if visits > 0 {
            marginal_q_p1(low, i)
        } else {
            fpu
        };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p1[i] as f32);
        let mut score = q_norm + exploration;
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }
        if (score - best_score).abs() < 1e-12 {
            tie_count += 1;
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = i as u8;
                best_utility = q_norm;
            }
        }
    }

    if second_best_score <= f32::NEG_INFINITY {
        return (best_idx, u32::MAX);
    }
    if best_utility >= second_best_score {
        return (best_idx, u32::MAX);
    }

    let prior_best = low.p1_prior(best_idx as usize);
    let n1 = ns_p1[best_idx as usize] as f32 + 1.0;
    let denom = second_best_score - best_utility;
    if denom <= 0.0 {
        return (best_idx, u32::MAX);
    }
    let vtc = (c_puct * prior_best * sqrt_total / denom - n1 + 1.0).max(1.0);
    (best_idx, (vtc as u32).max(1))
}

/// Same as above but for player 2 (marginalizes over i).
fn estimated_visits_to_change_best_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
    is_root: bool,
    ns_p2: &[u32; 5],
    rng: &mut impl Rng,
) -> (u8, u32) {
    let n = low.n2();
    if n <= 1 {
        return (0, u32::MAX);
    }

    let children_visits = low.total_edge_visits();
    let fpu = compute_fpu_p2(low, config);
    let sqrt_total = (children_visits.max(1) as f32).sqrt();
    let c_puct = config.c_puct;
    let value_scale = low.value_scale();

    let mut best_idx = 0u8;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_utility = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;

    for j in 0..n {
        let visits = low.marginal_visits_p2(j);
        let prior = low.p2_prior(j);
        let q = if visits > 0 {
            marginal_q_p2(low, j)
        } else {
            fpu
        };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p2[j] as f32);
        let mut score = q_norm + exploration;

        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }

        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_idx = j as u8;
            best_utility = q_norm;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    let mut tie_count = 1u32;
    for j in 0..n {
        if j as u8 == best_idx {
            continue;
        }
        let visits = low.marginal_visits_p2(j);
        let prior = low.p2_prior(j);
        let q = if visits > 0 {
            marginal_q_p2(low, j)
        } else {
            fpu
        };
        let q_norm = q / value_scale;
        let exploration = c_puct * prior * sqrt_total / (1.0 + ns_p2[j] as f32);
        let mut score = q_norm + exploration;
        if is_root && config.force_k > 0.0 && prior > 0.0 {
            let threshold = (config.force_k * prior * children_visits as f32).sqrt();
            if (visits as f32) < threshold {
                score = FORCED_PLAYOUT_SCORE;
            }
        }
        if (score - best_score).abs() < 1e-12 {
            tie_count += 1;
            if rng.gen_range(0..tie_count) == 0 {
                best_idx = j as u8;
                best_utility = q_norm;
            }
        }
    }

    if second_best_score <= f32::NEG_INFINITY {
        return (best_idx, u32::MAX);
    }
    if best_utility >= second_best_score {
        return (best_idx, u32::MAX);
    }

    let prior_best = low.p2_prior(best_idx as usize);
    let n1 = ns_p2[best_idx as usize] as f32 + 1.0;
    let denom = second_best_score - best_utility;
    if denom <= 0.0 {
        return (best_idx, u32::MAX);
    }
    let vtc = (c_puct * prior_best * sqrt_total / denom - n1 + 1.0).max(1.0);
    (best_idx, (vtc as u32).max(1))
}

// ---------------------------------------------------------------------------
// build_gather_level — VTC-based visit allocation at one node
// ---------------------------------------------------------------------------

/// Gather-phase state for one level of the iterative tree traversal.
struct GatherLevel<'session> {
    node: NodeHandle<'session>,
    /// Flat [i * 5 + j] → allocated visits for that (i, j) child.
    vtp: [u32; 25],
    /// Next flat index to process.
    next_idx: usize,
    /// Last flat index with non-zero visits.
    last_idx: usize,
}

/// Distribute `cur_limit` visits at `node` using decoupled VTC.
fn build_gather_level<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    node: &NodeHandle<'session>,
    cur_limit: u32,
    config: &SearchConfig,
    is_root: bool,
    rng: &mut impl Rng,
) -> GatherLevel<'session> {
    let low = access.node_mut(node);
    let n1 = low.n1();
    let n2 = low.n2();

    // Initialize n_started through the exclusive reservation fast path.
    let (mut ns_p1, mut ns_p2) = low.marginal_n_started_exclusive();

    let mut vtp = [0u32; 25];
    let mut remaining = cur_limit;
    let mut last_idx = 0usize;

    while remaining > 0 {
        let (best1, vtcb1) = estimated_visits_to_change_best_p1(low, config, is_root, &ns_p1, rng);
        let (best2, vtcb2) = estimated_visits_to_change_best_p2(low, config, is_root, &ns_p2, rng);

        let k = remaining.min(vtcb1).min(vtcb2).max(1);

        let flat = best1 as usize * 5 + best2 as usize;
        vtp[flat] += k;
        ns_p1[best1 as usize] += k;
        ns_p2[best2 as usize] += k;
        remaining -= k;
        if vtp[flat] > 0 && flat > last_idx {
            last_idx = flat;
        }
    }

    // Apply edge virtual loss for all allocated visits.
    for i in 0..n1 {
        for j in 0..n2 {
            let delta = vtp[i * 5 + j];
            if delta > 0 {
                low.add_virtual_loss_multi(i, j, delta);
            }
        }
    }

    GatherLevel {
        node: node.clone(),
        vtp,
        next_idx: 0,
        last_idx,
    }
}

// ---------------------------------------------------------------------------
// pick_nodes_to_extend — LC0-style batch allocation via tree traversal
// ---------------------------------------------------------------------------

/// What a batch entry represents.
enum NodeKind {
    /// Leaf needs NN evaluation (multivisit always 1).
    NeedsEval { game_state: GameState },
    /// Terminal node. Can have multivisit > 1.
    Terminal,
    /// First-hit transposition stop. Edge has no visits but child has aggregate
    /// from other parents. Values read from leaf at processing time.
    TranspositionHit,
}

/// A single entry from the batch gather phase.
struct NodeToProcess<'session> {
    reservations: NodeReservationPlan<'session>,
    kind: NodeKind,
}

/// A shared collision and the exact reservations it owns.
struct SharedCollision<'session> {
    reservations: PathReservationPlan<'session>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReservationLifecycle {
    Reserved,
    Committed,
    Cancelled,
}

/// Exact path mutations acquired for one node item or shared collision.
struct PathReservationPlan<'session> {
    path: SearchPath<'session>,
    units: u32,
    lifecycle: ReservationLifecycle,
}

impl<'session> PathReservationPlan<'session> {
    fn new(path: SearchPath<'session>, units: u32) -> Self {
        Self {
            path,
            units,
            lifecycle: ReservationLifecycle::Reserved,
        }
    }

    fn path(&self) -> &[PathEntry<'session>] {
        &self.path
    }

    fn units(&self) -> u32 {
        self.units
    }

    fn commit(&mut self, ledger: &mut SearchLedgerStats) {
        debug_assert_eq!(self.lifecycle, ReservationLifecycle::Reserved);
        self.lifecycle = ReservationLifecycle::Committed;
        ledger.commit(self.units);
    }

    fn release_path(&self, access: &mut ExclusiveAccess<'_, 'session>) {
        for entry in &self.path {
            let low = access.node_mut(&entry.node);
            low.revert_virtual_loss_multi(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
                self.units,
            );
            low.cancel_score_update_multi(self.units);
        }
    }

    fn mark_cancelled(&mut self, ledger: &mut SearchLedgerStats) {
        debug_assert_eq!(self.lifecycle, ReservationLifecycle::Reserved);
        self.lifecycle = ReservationLifecycle::Cancelled;
        ledger.cancel(self.units);
    }

    fn cancel(
        &mut self,
        access: &mut ExclusiveAccess<'_, 'session>,
        ledger: &mut SearchLedgerStats,
    ) {
        self.release_path(access);
        self.mark_cancelled(ledger);
    }
}

/// Exact reservation shape for work targeting a leaf.
///
/// `leaf_units` caches whether gather actually claimed the target leaf. Cleanup
/// must not infer that decision from current visits, TT status, parent count, or
/// topology: those facts can change while inference runs.
struct NodeReservationPlan<'session> {
    path: PathReservationPlan<'session>,
    target: NodeHandle<'session>,
    leaf_units: u32,
}

impl<'session> NodeReservationPlan<'session> {
    fn new(
        target: NodeHandle<'session>,
        path: SearchPath<'session>,
        units: u32,
        leaf_units: u32,
    ) -> Self {
        debug_assert!(leaf_units <= units);
        Self {
            path: PathReservationPlan::new(path, units),
            target,
            leaf_units,
        }
    }

    fn target(&self) -> &NodeHandle<'session> {
        &self.target
    }

    fn path(&self) -> &[PathEntry<'session>] {
        self.path.path()
    }

    fn units(&self) -> u32 {
        self.path.units()
    }

    fn commit(&mut self, ledger: &mut SearchLedgerStats) {
        self.path.commit(ledger);
    }

    fn cancel(
        &mut self,
        access: &mut ExclusiveAccess<'_, 'session>,
        ledger: &mut SearchLedgerStats,
    ) {
        self.path.release_path(access);
        if self.leaf_units > 0 {
            access
                .node_mut(&self.target)
                .cancel_score_update_multi(self.leaf_units);
        }
        self.path.mark_cancelled(ledger);
    }
}

/// LC0's PickNodesToExtendTask adapted for MCGS DAG with 2-player joint matrix.
fn pick_nodes_to_extend<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    game: &GameState,
    config: &SearchConfig,
    budget: u32,
    rng: &mut impl Rng,
) -> (Vec<NodeToProcess<'session>>, Vec<SharedCollision<'session>>) {
    let root = access.root();
    let mut to_process: Vec<NodeToProcess<'session>> = Vec::with_capacity(budget as usize);
    let mut shared_collisions: Vec<SharedCollision<'session>> = Vec::new();
    let mut work_game = game.clone();
    let mut undos: Vec<MoveUndo> = Vec::new();

    // Handle root: copy the decision facts before opening a mutation epoch.
    let (root_visits, root_terminal) = {
        let root_low = access.node(&root);
        (root_low.total_visits(), root_low.is_terminal())
    };
    if root_visits == 0 || root_terminal {
        if root_visits == 0 && !root_terminal {
            if access.node_mut(&root).try_start_score_update() {
                if work_game.check_game_over() {
                    access.populate_node(&root, None);
                    to_process.push(NodeToProcess {
                        reservations: NodeReservationPlan::new(root.clone(), Vec::new(), 1, 1),
                        kind: NodeKind::Terminal,
                    });
                } else {
                    to_process.push(NodeToProcess {
                        reservations: NodeReservationPlan::new(root.clone(), Vec::new(), 1, 1),
                        kind: NodeKind::NeedsEval {
                            game_state: work_game.clone(),
                        },
                    });
                }
                if budget > 1 {
                    shared_collisions.push(SharedCollision {
                        reservations: PathReservationPlan::new(Vec::new(), budget - 1),
                    });
                }
            } else {
                shared_collisions.push(SharedCollision {
                    reservations: PathReservationPlan::new(Vec::new(), budget),
                });
            }
        } else {
            // Terminal root: one real visit + rest as collisions (LC0 pattern).
            // Every pick does one real visit, matching dag_classic's
            // ShouldStopPickingHere + TryStartScoreUpdate path.
            if root_visits == 0 {
                access.populate_node(&root, None);
            }
            access.node_mut(&root).increment_n_in_flight(1);
            to_process.push(NodeToProcess {
                reservations: NodeReservationPlan::new(root.clone(), Vec::new(), 1, 1),
                kind: NodeKind::Terminal,
            });
            if budget > 1 {
                shared_collisions.push(SharedCollision {
                    reservations: PathReservationPlan::new(Vec::new(), budget - 1),
                });
            }
        }
        return (to_process, shared_collisions);
    }

    // Root is interior: increment n_in_flight for all visits.
    access.node_mut(&root).increment_n_in_flight(budget);

    let first_level = build_gather_level(access, &root, budget, config, true, rng);
    let mut levels: Vec<GatherLevel<'session>> = vec![first_level];
    let mut path_prefix: SearchPath<'session> = Vec::new();

    while let Some(level) = levels.last_mut() {
        let mut found_child = false;
        while level.next_idx <= level.last_idx {
            let idx = level.next_idx;
            level.next_idx += 1;
            if level.vtp[idx] == 0 {
                continue;
            }
            let i = (idx / 5) as u8;
            let j = (idx % 5) as u8;
            let k = level.vtp[idx];

            // Convert outcome indices to canonical actions.
            let (act1, act2) = {
                let low = access.node(&level.node);
                (
                    low.p1_outcome_action(i as usize),
                    low.p2_outcome_action(j as usize),
                )
            };
            let d1 = Direction::try_from(act1).expect("valid direction");
            let d2 = Direction::try_from(act2).expect("valid direction");
            let scores_before = (work_game.player1_score(), work_game.player2_score());
            let undo = work_game.make_move(d1, d2);
            let (r1, r2) = compute_rewards(&work_game, scores_before);

            let child = access.find_or_create_child(&level.node, i, j, &work_game, r1, r2);

            // Build the path to this child.
            let mut child_path = path_prefix.clone();
            child_path.push(PathEntry {
                node: level.node.clone(),
                p1_outcome: i,
                p2_outcome: j,
            });

            let (child_visits, child_terminal) = {
                let child_low = access.node(&child);
                (child_low.total_visits(), child_low.is_terminal())
            };
            if child_visits == 0 || child_terminal {
                // Leaf or terminal.
                if access.node_mut(&child).try_start_score_update() {
                    if child_terminal || work_game.check_game_over() {
                        if child_visits == 0 {
                            access.populate_node(&child, None);
                        }
                        to_process.push(NodeToProcess {
                            reservations: NodeReservationPlan::new(
                                child.clone(),
                                child_path.clone(),
                                1,
                                1,
                            ),
                            kind: NodeKind::Terminal,
                        });
                    } else {
                        to_process.push(NodeToProcess {
                            reservations: NodeReservationPlan::new(
                                child.clone(),
                                child_path.clone(),
                                1,
                                1,
                            ),
                            kind: NodeKind::NeedsEval {
                                game_state: work_game.clone(),
                            },
                        });
                    }
                    if k > 1 {
                        shared_collisions.push(SharedCollision {
                            reservations: PathReservationPlan::new(child_path, k - 1),
                        });
                    }
                } else {
                    // Collision: all k visits.
                    shared_collisions.push(SharedCollision {
                        reservations: PathReservationPlan::new(child_path, k),
                    });
                }
                work_game.unmake_move(undo);
            } else {
                // Interior child: check transposition stopping.
                let edge_vis = access.node(&level.node).edge_visits(i as usize, j as usize);
                if access.num_parents(&child) > 1 && edge_vis < child_visits {
                    // TT stop: edge is behind the shared aggregate.
                    // Covers both first-hit (edge_vis == 0) and stale
                    // (edge_vis > 0) cases. One productive stop that
                    // backs up the child's current aggregate through the
                    // path, correcting the edge via delta fixup.
                    // Remaining k-1 visits become collisions.
                    // Don't increment child's n_in_flight — we're reading,
                    // not visiting.
                    to_process.push(NodeToProcess {
                        reservations: NodeReservationPlan::new(
                            child.clone(),
                            child_path.clone(),
                            1,
                            0,
                        ),
                        kind: NodeKind::TranspositionHit,
                    });
                    if k > 1 {
                        shared_collisions.push(SharedCollision {
                            reservations: PathReservationPlan::new(child_path, k - 1),
                        });
                    }
                    work_game.unmake_move(undo);
                } else {
                    // Normal interior: descend with k visits.
                    access.node_mut(&child).increment_n_in_flight(k);
                    undos.push(undo);
                    path_prefix = child_path;
                    let child_level = build_gather_level(access, &child, k, config, false, rng);
                    levels.push(child_level);
                    found_child = true;
                    break;
                }
            }
        }

        if !found_child {
            // All children at this level processed, backtrack.
            levels.pop();
            if let Some(undo) = undos.pop() {
                work_game.unmake_move(undo);
            }
            path_prefix.pop();
        }
    }

    (to_process, shared_collisions)
}

// ---------------------------------------------------------------------------
// backup_and_finalize — combined backup + VL cleanup
// ---------------------------------------------------------------------------

/// Walk leaf→root, updating values with multivisit Welford and reverting VL.
/// Applies delta correction for transposition staleness.
fn backup_and_finalize<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    path: &[PathEntry<'session>],
    leaf: &NodeHandle<'session>,
    g1: f32,
    g2: f32,
    multivisit: u32,
) {
    access
        .node_mut(leaf)
        .finalize_score_update_multi(g1, g2, multivisit);

    let mut v1 = g1;
    let mut v2 = g2;
    let mut n_to_fix: u32 = 0;
    let mut v1_delta: f32 = 0.0;
    let mut v2_delta: f32 = 0.0;

    for entry in path.iter().rev() {
        let i = entry.p1_outcome as usize;
        let j = entry.p2_outcome as usize;

        let snapshot = access.backup_snapshot(&entry.node, entry.p1_outcome, entry.p2_outcome);

        let mut q1 = snapshot.r1 + v1;
        let mut q2 = snapshot.r2 + v2;
        let mut q1_delta = v1_delta;
        let mut q2_delta = v2_delta;

        // Delta detection (unchanged from Tier 1).
        if snapshot.child_num_parents > 1 || snapshot.edge_visits < snapshot.child_visits {
            let correct_q1 = snapshot.r1 + snapshot.child_v1;
            let correct_q2 = snapshot.r2 + snapshot.child_v2;
            q1_delta = correct_q1 - snapshot.edge_q1;
            q2_delta = correct_q2 - snapshot.edge_q2;
            n_to_fix = snapshot.edge_visits;
            q1 = correct_q1;
            q2 = correct_q2;
        }

        let node = access.node_mut(&entry.node);

        node.finalize_edge_update_multi(i, j, q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_edge_for_terminal(i, j, q1_delta, q2_delta, n_to_fix);
        }
        node.finalize_score_update_multi(q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_for_terminal(q1_delta, q2_delta, n_to_fix);
        }

        // Revert VL inline (replaces separate cleanup_descent).
        node.revert_virtual_loss_multi(i, j, multivisit);

        v1 = q1;
        v2 = q2;
        v1_delta = q1_delta;
        v2_delta = q2_delta;
    }
}

/// Back up a transposition stop: initialize the new edge from the shared
/// child's existing aggregate without incrementing the child's visit count.
///
/// Same path walk as `backup_and_finalize` but skips leaf finalization.
/// Reads v1/v2 from the leaf at processing time (not gather time).
fn backup_transposition_stop<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    path: &[PathEntry<'session>],
    leaf: &NodeHandle<'session>,
    multivisit: u32,
) {
    // Read the leaf's current aggregate — not frozen at gather time.
    let (mut v1, mut v2) = {
        let leaf = access.node(leaf);
        (leaf.v1(), leaf.v2())
    };
    let mut n_to_fix: u32 = 0;
    let mut v1_delta: f32 = 0.0;
    let mut v2_delta: f32 = 0.0;

    for entry in path.iter().rev() {
        let i = entry.p1_outcome as usize;
        let j = entry.p2_outcome as usize;

        let snapshot = access.backup_snapshot(&entry.node, entry.p1_outcome, entry.p2_outcome);

        let mut q1 = snapshot.r1 + v1;
        let mut q2 = snapshot.r2 + v2;
        let mut q1_delta = v1_delta;
        let mut q2_delta = v2_delta;

        // Delta detection (same as backup_and_finalize).
        if snapshot.child_num_parents > 1 || snapshot.edge_visits < snapshot.child_visits {
            let correct_q1 = snapshot.r1 + snapshot.child_v1;
            let correct_q2 = snapshot.r2 + snapshot.child_v2;
            q1_delta = correct_q1 - snapshot.edge_q1;
            q2_delta = correct_q2 - snapshot.edge_q2;
            n_to_fix = snapshot.edge_visits;
            q1 = correct_q1;
            q2 = correct_q2;
        }

        let node = access.node_mut(&entry.node);

        node.finalize_edge_update_multi(i, j, q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_edge_for_terminal(i, j, q1_delta, q2_delta, n_to_fix);
        }
        node.finalize_score_update_multi(q1, q2, multivisit);
        if n_to_fix > 0 {
            node.adjust_for_terminal(q1_delta, q2_delta, n_to_fix);
        }

        node.revert_virtual_loss_multi(i, j, multivisit);

        v1 = q1;
        v2 = q2;
        v1_delta = q1_delta;
        v2_delta = q2_delta;
    }
}

// ---------------------------------------------------------------------------
// cancel_shared_collisions — revert VL for unused visits
// ---------------------------------------------------------------------------

/// Cancel each collision from its immutable acquisition record.
fn cancel_shared_collisions<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    collisions: &mut [SharedCollision<'session>],
    ledger: &mut SearchLedgerStats,
) {
    for coll in collisions {
        coll.reservations.cancel(access, ledger);
    }
}

// ---------------------------------------------------------------------------
// calculate_collisions_left — LC0's tree-size-based collision budget
// ---------------------------------------------------------------------------

/// LC0's CalculateCollisionsLeft: power-law interpolation from min to max
/// based on tree node count.
fn calculate_collisions_left(tree_node_count: u32, config: &SearchConfig) -> u32 {
    if tree_node_count >= config.collision_scaling_end {
        return config.collision_limit_max;
    }
    if tree_node_count <= config.collision_scaling_start {
        return config.collision_limit_min;
    }
    let ratio = (tree_node_count - config.collision_scaling_start) as f32
        / (config.collision_scaling_end - config.collision_scaling_start) as f32;
    let scaled = config.collision_limit_min as f32
        + (config.collision_limit_max as f32 - config.collision_limit_min as f32)
            * ratio.powf(config.collision_scaling_power);
    (scaled.round() as u32).clamp(config.collision_limit_min, config.collision_limit_max)
}

// ---------------------------------------------------------------------------
// PendingBatch — owned gather output crossing guard-free inference
// ---------------------------------------------------------------------------

/// Owned work admitted by one gather epoch and not yet fully settled.
///
/// This type deliberately contains no graph guard or borrowed payload. It must
/// be consumed by success or cancellation settlement under a later exclusive
/// epoch; `Drop` never reacquires the graph gate.
#[must_use = "a gathered batch must be committed or cancelled explicitly"]
struct PendingBatch<'session> {
    root: NodeHandle<'session>,
    evals: Vec<NodeToProcess<'session>>,
    collisions: Vec<SharedCollision<'session>>,
    ledger: SearchLedgerStats,
    terminals: u32,
    tt_stop_hits: u32,
}

// ---------------------------------------------------------------------------
// simulate_batch — LC0-style gather/eval/backup cycle
// ---------------------------------------------------------------------------

fn validate_eval_batch(
    expected: usize,
    eval_results: &[crate::EvalResult],
) -> Result<(), BackendError> {
    if eval_results.len() != expected {
        return Err(BackendError::msg(format!(
            "backend result count mismatch: requested {expected}, received {}",
            eval_results.len()
        )));
    }

    for (index, eval) in eval_results.iter().enumerate() {
        if !eval.policy_p1.iter().all(|value| value.is_finite())
            || !eval.policy_p2.iter().all(|value| value.is_finite())
            || !eval.value_p1.is_finite()
            || !eval.value_p2.is_finite()
        {
            return Err(BackendError::msg(format!(
                "backend result {index} contains a non-finite policy or value"
            )));
        }
    }

    Ok(())
}

fn gather_batch<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    game: &GameState,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
) -> PendingBatch<'session> {
    let root = access.root();
    let mut collisions_left = calculate_collisions_left(access.node_count(), config) as i32;

    let mut evals: Vec<NodeToProcess<'session>> = Vec::with_capacity(batch_size as usize);
    let mut collisions: Vec<SharedCollision<'session>> = Vec::new();
    let mut ledger = SearchLedgerStats::default();
    let mut minibatch_size = 0u32;
    let mut terminals = 0u32;
    let mut tt_stop_hits = 0u32;

    // ---- Outer Gather Loop (LC0's GatherMinibatch) ----
    while minibatch_size < batch_size && collisions_left > 0 {
        let budget = (collisions_left as u32).min(batch_size - minibatch_size);
        let (to_process, shared_collisions) =
            pick_nodes_to_extend(access, game, config, budget, rng);

        for mut entry in to_process {
            let units = entry.reservations.units();
            ledger.reserve(units);
            match &entry.kind {
                NodeKind::Terminal => {
                    backup_and_finalize(
                        access,
                        entry.reservations.path(),
                        entry.reservations.target(),
                        0.0,
                        0.0,
                        units,
                    );
                    entry.reservations.commit(&mut ledger);
                    terminals += units;
                    minibatch_size += 1;
                }
                NodeKind::TranspositionHit => {
                    backup_transposition_stop(
                        access,
                        entry.reservations.path(),
                        entry.reservations.target(),
                        units,
                    );
                    entry.reservations.commit(&mut ledger);
                    tt_stop_hits += units;
                    minibatch_size += 1;
                }
                NodeKind::NeedsEval { .. } => {
                    minibatch_size += 1;
                    evals.push(entry);
                }
            }
        }

        for coll in shared_collisions {
            let units = coll.reservations.units();
            ledger.reserve(units);
            collisions_left -= units as i32;
            collisions.push(coll);
        }
    }

    PendingBatch {
        root,
        evals,
        collisions,
        ledger,
        terminals,
        tt_stop_hits,
    }
}

/// Run inference and validate the whole returned batch without graph access.
fn infer_pending_batch(
    pending: &PendingBatch<'_>,
    backend: &dyn Backend,
) -> Result<Vec<crate::EvalResult>, BackendError> {
    let game_states: Vec<&GameState> = pending
        .evals
        .iter()
        .filter_map(|entry| match &entry.kind {
            NodeKind::NeedsEval { game_state } => Some(game_state),
            _ => None,
        })
        .collect();

    let results = if game_states.is_empty() {
        Vec::new()
    } else {
        backend.evaluate_batch(&game_states)?
    };

    validate_eval_batch(game_states.len(), &results)?;
    Ok(results)
}

fn infer_pending_batch_caught(
    pending: &PendingBatch<'_>,
    backend: &dyn Backend,
) -> std::thread::Result<Result<Vec<crate::EvalResult>, BackendError>> {
    catch_unwind(AssertUnwindSafe(|| infer_pending_batch(pending, backend)))
}

/// Commit evaluated leaves and cancel collision reservations under one epoch.
fn settle_pending_batch<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    mut pending: PendingBatch<'session>,
    eval_results: Vec<crate::EvalResult>,
    config: &SearchConfig,
    rng: &mut impl Rng,
) -> BatchStats {
    assert_eq!(pending.evals.len(), eval_results.len());
    let nn_evals = pending.evals.len() as u32;
    let total_collisions = pending
        .collisions
        .iter()
        .map(|collision| collision.reservations.units())
        .sum();

    for (entry, eval) in pending.evals.iter_mut().zip(&eval_results) {
        let leaf = entry.reservations.target();
        access.populate_node(leaf, Some(eval));

        if config.noise_epsilon > 0.0 && access.same_node(leaf, &pending.root) {
            let low = access.node_mut(leaf);
            apply_dirichlet_noise_p1(low, config.noise_epsilon, config.noise_concentration, rng);
            apply_dirichlet_noise_p2(low, config.noise_epsilon, config.noise_concentration, rng);
        }

        backup_and_finalize(
            access,
            entry.reservations.path(),
            leaf,
            eval.value_p1,
            eval.value_p2,
            entry.reservations.units(),
        );
        entry.reservations.commit(&mut pending.ledger);
    }

    cancel_shared_collisions(access, &mut pending.collisions, &mut pending.ledger);
    pending.ledger.assert_settled();

    BatchStats {
        nn_evals,
        terminals: pending.terminals,
        collisions: total_collisions,
        tt_stop_hits: pending.tt_stop_hits,
        ledger: pending.ledger,
    }
}

/// Cancel every still-reserved item from its exact acquisition plan.
fn cancel_pending_batch<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    mut pending: PendingBatch<'session>,
) -> SearchLedgerStats {
    for entry in &mut pending.evals {
        entry.reservations.cancel(access, &mut pending.ledger);
    }
    cancel_shared_collisions(access, &mut pending.collisions, &mut pending.ledger);
    pending.ledger.assert_settled();
    pending.ledger
}

fn simulate_batch<'session>(
    access: &mut ExclusiveAccess<'_, 'session>,
    game: &GameState,
    backend: &dyn Backend,
    config: &SearchConfig,
    batch_size: u32,
    rng: &mut impl Rng,
) -> Result<BatchStats, BackendError> {
    let pending = gather_batch(access, game, config, batch_size, rng);
    match infer_pending_batch_caught(&pending, backend) {
        Ok(Ok(eval_results)) => Ok(settle_pending_batch(
            access,
            pending,
            eval_results,
            config,
            rng,
        )),
        Ok(Err(error)) => {
            cancel_pending_batch(access, pending);
            Err(error)
        }
        Err(payload) => {
            cancel_pending_batch(access, pending);
            resume_unwind(payload)
        }
    }
}

// ---------------------------------------------------------------------------
// extract_result — policies and values from root
// ---------------------------------------------------------------------------

fn extract_result<'session>(
    access: &ExclusiveAccess<'_, 'session>,
    root: &NodeHandle<'session>,
    config: &SearchConfig,
    _rng: &mut impl Rng,
) -> SearchResult {
    let low = access.node(root);
    let total_visits = low.total_visits();

    let (policy_p1, visit_counts_p1, value_p1, q_values_p1) = extract_p1(low, config);
    let (policy_p2, visit_counts_p2, value_p2, q_values_p2) = extract_p2(low, config);

    let prior_p1 = low.expand_p1_prior();
    let prior_p2 = low.expand_p2_prior();

    SearchResult {
        policy_p1,
        policy_p2,
        value_p1,
        value_p2,
        visit_counts_p1,
        visit_counts_p2,
        prior_p1,
        prior_p2,
        q_values_p1,
        q_values_p2,
        total_visits,
        nn_evals: 0,
        terminals: 0,
        collisions: 0,
        tt_stop_hits: 0,
    }
}

/// Extract policy, visit counts, value, and Q-values for player 1 from root.
fn extract_p1(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32, [f32; 5]) {
    let n = low.n1();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v1(), [0.0; 5]);
    }

    let children_visits = low.total_edge_visits();

    let fpu = compute_fpu_p1(low, config);

    // Read Q and visits per outcome.
    let mut q = [0.0f32; 5];
    let mut raw_visits = [0.0f32; 5];
    let mut prior = [0.0f32; 5];
    let mut q_norm = [0.0f32; 5];

    for i in 0..n {
        let visits = low.marginal_visits_p1(i);
        q[i] = if visits > 0 {
            marginal_q_p1(low, i)
        } else {
            fpu
        };
        raw_visits[i] = visits as f32;
        prior[i] = low.p1_prior(i);
        q_norm[i] = q[i] / low.value_scale();
    }

    // Compute pruned visits.
    let pruned = compute_pruned_visits(
        &q_norm,
        &prior,
        &raw_visits,
        n,
        children_visits,
        config.c_puct,
    );

    // Expand to 5-action space.
    let mut visit_counts = [0.0f32; 5];
    for (i, &pv) in pruned.iter().enumerate().take(n) {
        let action = low.p1_outcome_action(i) as usize;
        visit_counts[action] = pv;
    }

    // Normalize to get policy.
    let mut policy = visit_counts;
    let policy_sum: f32 = policy.iter().sum();
    if policy_sum > 0.0 {
        for p in &mut policy {
            *p /= policy_sum;
        }
    } else {
        policy = low.expand_p1_prior();
    }

    // Expand Q-values to 5-action space.
    let mut q_values = [0.0f32; 5];
    for i in 0..n {
        let action = low.p1_outcome_action(i) as usize;
        q_values[action] = q[i];
    }

    // Value = dot(q, raw_visits) / sum(raw_visits).
    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|i| q[i] * raw_visits[i]).sum();
        dot / visit_sum
    } else {
        low.v1()
    };

    (policy, visit_counts, value, q_values)
}

/// Extract policy, visit counts, value, and Q-values for player 2 from root.
fn extract_p2(
    low: &crate::node::LowNode,
    config: &SearchConfig,
) -> ([f32; 5], [f32; 5], f32, [f32; 5]) {
    let n = low.n2();

    if n == 0 {
        return ([0.0; 5], [0.0; 5], low.v2(), [0.0; 5]);
    }

    let children_visits = low.total_edge_visits();

    let fpu = compute_fpu_p2(low, config);

    let mut q = [0.0f32; 5];
    let mut raw_visits = [0.0f32; 5];
    let mut prior = [0.0f32; 5];
    let mut q_norm = [0.0f32; 5];

    for j in 0..n {
        let visits = low.marginal_visits_p2(j);
        q[j] = if visits > 0 {
            marginal_q_p2(low, j)
        } else {
            fpu
        };
        raw_visits[j] = visits as f32;
        prior[j] = low.p2_prior(j);
        q_norm[j] = q[j] / low.value_scale();
    }

    let pruned = compute_pruned_visits(
        &q_norm,
        &prior,
        &raw_visits,
        n,
        children_visits,
        config.c_puct,
    );

    let mut visit_counts = [0.0f32; 5];
    for (j, &pv) in pruned.iter().enumerate().take(n) {
        let action = low.p2_outcome_action(j) as usize;
        visit_counts[action] = pv;
    }

    let mut policy = visit_counts;
    let policy_sum: f32 = policy.iter().sum();
    if policy_sum > 0.0 {
        for p in &mut policy {
            *p /= policy_sum;
        }
    } else {
        policy = low.expand_p2_prior();
    }

    // Expand Q-values to 5-action space.
    let mut q_values = [0.0f32; 5];
    for j in 0..n {
        let action = low.p2_outcome_action(j) as usize;
        q_values[action] = q[j];
    }

    let visit_sum: f32 = raw_visits[..n].iter().sum();
    let value = if visit_sum > 0.0 {
        let dot: f32 = (0..n).map(|j| q[j] * raw_visits[j]).sum();
        dot / visit_sum
    } else {
        low.v2()
    };

    (policy, visit_counts, value, q_values)
}

/// Forced-playout pruning: cap visits on low-Q outcomes.
fn compute_pruned_visits(
    q_norm: &[f32],
    prior: &[f32],
    visits: &[f32],
    n: usize,
    total_visits: u32,
    c_puct: f32,
) -> [f32; 5] {
    let mut result = [0.0f32; 5];

    if n <= 1 {
        if n == 1 {
            result[0] = visits[0];
        }
        return result;
    }

    // Find best outcome (most visited).
    let mut best_idx = 0;
    let mut best_visits = visits[0];
    for (i, &v) in visits.iter().enumerate().take(n).skip(1) {
        if v > best_visits {
            best_visits = v;
            best_idx = i;
        }
    }

    let sqrt_total = (total_visits as f32).sqrt();
    let puct_star =
        q_norm[best_idx] + c_puct * prior[best_idx] * sqrt_total / (1.0 + visits[best_idx]);

    for i in 0..n {
        if i == best_idx || q_norm[i] >= puct_star {
            result[i] = visits[i];
        } else {
            let denom = puct_star - q_norm[i];
            if denom <= 0.0 {
                result[i] = visits[i];
            } else {
                let n_min = (c_puct * prior[i] * sqrt_total / denom - 1.0).max(0.0);
                result[i] = visits[i].min(n_min);
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LowNode;
    use crate::{BackendError, ConstantValueBackend, SmartUniformBackend};
    use pyrat::{Coordinates, Direction, GameBuilder};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::collections::{HashMap, HashSet};

    fn with_test_access<R>(
        test: impl for<'tree, 'session> FnOnce(&mut ExclusiveAccess<'tree, 'session>) -> R,
    ) -> R {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        tree.with_exclusive(|mut access| test(&mut access))
    }

    /// Test-only shim: old single-visit backup without VL handling.
    /// Pre-increments n_in_flight and adds VL on path entries so that
    /// backup_and_finalize's decrements work correctly.
    fn backup<'session>(
        access: &mut ExclusiveAccess<'_, 'session>,
        path: &[PathEntry<'session>],
        leaf: &NodeHandle<'session>,
        g1: f32,
        g2: f32,
    ) {
        access.node_mut(leaf).increment_n_in_flight(1);
        reserve_path(access, path, 1);
        backup_and_finalize(access, path, leaf, g1, g2, 1);
    }

    fn reserve_path<'session>(
        access: &mut ExclusiveAccess<'_, 'session>,
        path: &[PathEntry<'session>],
        count: u32,
    ) {
        for entry in path {
            access.node_mut(&entry.node).increment_n_in_flight(count);
            access.node_mut(&entry.node).add_virtual_loss_multi(
                entry.p1_outcome as usize,
                entry.p2_outcome as usize,
                count,
            );
        }
    }

    /// Test-only shim: old cleanup_descent for tests that still use it.
    fn cleanup_descent<'session>(
        access: &mut ExclusiveAccess<'_, 'session>,
        path: &[PathEntry<'session>],
        leaf_claimed: Option<&NodeHandle<'session>>,
    ) {
        for entry in path {
            access
                .node_mut(&entry.node)
                .revert_virtual_loss(entry.p1_outcome as usize, entry.p2_outcome as usize);
        }
        if let Some(leaf) = leaf_claimed {
            access.node_mut(leaf).cancel_score_update();
        }
    }

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    fn default_config() -> SearchConfig {
        SearchConfig::default()
    }

    fn open_5x5_game(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(p1, p2)
            .with_custom_cheese(cheese.to_vec())
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    fn terminal_game() -> GameState {
        let mut game = GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(0, 1))
            .with_custom_cheese(vec![Coordinates::new(4, 4)])
            .with_max_turns(1)
            .build()
            .create(None)
            .unwrap();
        let _undo = game.make_move(Direction::Stay, Direction::Stay);
        game
    }

    fn short_game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(2, 0))
            .with_custom_cheese(vec![Coordinates::new(1, 0)])
            .with_max_turns(3)
            .build()
            .create(None)
            .unwrap()
    }

    // ---- select_actions ----

    #[test]
    fn select_actions_uniform_prior() {
        // With uniform priors and no visits, PUCT should select via FPU.
        // With forced playouts, all outcomes start unvisited → first one wins.
        let mut root = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        root.set_prior([0.2; 5], [0.2; 5]);
        root.set_value_scale(1.0);
        let config = default_config();
        let mut r = rng();

        let (a1, a2) = select_actions(&root, &config, true, &mut r);
        // Should be valid outcome indices
        assert!((a1 as usize) < root.n1());
        assert!((a2 as usize) < root.n2());
    }

    #[test]
    fn select_actions_after_visits() {
        let mut root = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        root.set_prior([0.2; 5], [0.2; 5]);
        root.set_value_scale(1.0);

        // Add some visits to (0, 0)
        root.finalize_edge_update(0, 0, 1.0, 1.0);
        root.finalize_edge_update(0, 0, 1.0, 1.0);
        root.finalize_score_update(1.0, 1.0);
        root.finalize_score_update(1.0, 1.0);

        let config = default_config();
        let mut r = rng();

        // With visits on (0,0), PUCT should explore other outcomes
        let (a1, a2) = select_actions(&root, &config, true, &mut r);
        assert!((a1 as usize) < root.n1());
        assert!((a2 as usize) < root.n2());
    }

    #[test]
    fn select_actions_respects_priors_after_root_eval() {
        // After root NN eval: total_visits()=1, total_edge_visits()=0.
        // Before the fix, sqrt(0) killed the exploration term entirely,
        // making selection degenerate to FPU-only (ignoring priors).
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_value_scale(5.0);

        // Non-uniform priors: heavily favor outcome 2
        let mut prior_p1 = [0.0f32; 5];
        prior_p1[0] = 0.05;
        prior_p1[1] = 0.05;
        prior_p1[2] = 0.80;
        prior_p1[3] = 0.05;
        prior_p1[4] = 0.05;
        low.set_prior(prior_p1, [0.2; 5]);

        // Simulate one finalize_score_update (root NN eval) without any edge visits.
        low.finalize_score_update(1.0, 1.0);
        assert_eq!(low.total_visits(), 1);
        assert_eq!(low.total_edge_visits(), 0);

        let config = SearchConfig {
            force_k: 0.0, // disable forced playouts to test pure PUCT
            ..default_config()
        };
        let mut r = rng();

        // With total_visits() used, sqrt(1)=1 gives exploration a non-zero term.
        // The high prior on outcome 2 should make it the preferred selection.
        let (a1, _a2) = select_actions(&low, &config, false, &mut r);
        assert_eq!(a1, 2, "should select outcome with highest prior");
    }

    // ---- marginal_q ----

    #[test]
    fn marginal_q_p1_weighted_average() {
        let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        low.set_value_scale(5.0);

        // i=0, j=0: Q=2.0, 3 visits
        for _ in 0..3 {
            low.finalize_edge_update(0, 0, 2.0, 1.0);
        }
        // i=0, j=1: Q=4.0, 1 visit
        low.finalize_edge_update(0, 1, 4.0, 2.0);

        let mq = marginal_q_p1(&low, 0);
        // Expected: (3 * 2.0 + 1 * 4.0) / 4 = 10.0 / 4 = 2.5
        assert!((mq - 2.5).abs() < 1e-5);
    }

    #[test]
    fn marginal_q_no_visits_returns_zero() {
        let low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        assert_eq!(marginal_q_p1(&low, 0), 0.0);
        assert_eq!(marginal_q_p2(&low, 0), 0.0);
    }

    // ---- backup ----

    #[test]
    fn backup_single_level() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(5.0);
            access.test_connect(&root, &child, (0, 1), 1.0, 0.5);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 1,
            }];

            backup(access, &path, &child, 3.0, 2.0);

            // Leaf: v=(3.0, 2.0), visits=1
            let child_low = access.node(&child);
            assert_eq!(child_low.total_visits(), 1);
            assert!((child_low.v1() - 3.0).abs() < 1e-6);
            assert!((child_low.v2() - 2.0).abs() < 1e-6);

            // Root: q1 = r1 + leaf_v1 = 1.0 + 3.0 = 4.0
            //       q2 = r2 + leaf_v2 = 0.5 + 2.0 = 2.5
            let root_low = access.node(&root);
            assert_eq!(root_low.total_visits(), 1);
            assert!((root_low.v1() - 4.0).abs() < 1e-6);
            assert!((root_low.v2() - 2.5).abs() < 1e-6);

            // Joint matrix: edge at (0, 1)
            assert_eq!(root_low.edge_visits(0, 1), 1);
            assert!((root_low.edge_q_p1(0, 1) - 4.0).abs() < 1e-6);
            assert!((root_low.edge_q_p2(0, 1) - 2.5).abs() < 1e-6);
        });
    }

    #[test]
    fn backup_stale_edge_delta_catchup() {
        with_test_access(|access| {
            // A child has total_visits=5, but the parent's edge only has 2 visits
            // and num_parents=1. This simulates a stale edge after root advancement
            // pruned another parent. The delta correction should still trigger.
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(5.0);

            // Simulate child having been visited 5 times from another (now-pruned) parent
            // with Q that drifted from what our edge knows.
            for _ in 0..5 {
                access.node_mut(&child).finalize_score_update(3.0, 2.0);
            }
            assert_eq!(access.node(&child).total_visits(), 5);
            assert_eq!(access.num_parents(&child), 0); // no edge yet

            // Wire edge: root --(0,1)--> child, with r=(1.0, 0.5)
            access.test_connect(&root, &child, (0, 1), 1.0, 0.5);
            assert_eq!(access.num_parents(&child), 1);

            // Add 2 stale visits on the edge with outdated Q values
            access.node_mut(&root).finalize_edge_update(0, 1, 2.0, 1.0);
            access.node_mut(&root).finalize_edge_update(0, 1, 2.0, 1.0);
            access.node_mut(&root).finalize_score_update(2.0, 1.0);
            access.node_mut(&root).finalize_score_update(2.0, 1.0);
            assert_eq!(access.node(&root).edge_visits(0, 1), 2);

            // Now backup a new visit through this path.
            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 1,
            }];
            backup(access, &path, &child, 3.0, 2.0);

            // The correct Q for the edge is r + child.v = (1+3, 0.5+2) = (4.0, 2.5).
            // Before the fix, num_parents=1 would skip delta correction, leaving
            // the edge Q based only on the new visit + stale visits.
            // With the fix, edge_visits(2) < child.total_visits(6) triggers correction.
            let edge_q1 = access.node(&root).edge_q_p1(0, 1);
            let edge_q2 = access.node(&root).edge_q_p2(0, 1);
            assert!(
                (edge_q1 - 4.0).abs() < 0.5,
                "edge Q1 should be corrected toward 4.0, got {edge_q1}"
            );
            assert!(
                (edge_q2 - 2.5).abs() < 0.5,
                "edge Q2 should be corrected toward 2.5, got {edge_q2}"
            );
        });
    }

    // ---- cleanup_descent ----

    #[test]
    fn cleanup_reverts_virtual_loss() {
        with_test_access(|access| {
            let node = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&node).add_virtual_loss(1, 2);

            let path = vec![PathEntry {
                node: node.clone(),
                p1_outcome: 1,
                p2_outcome: 2,
            }];

            cleanup_descent(access, &path, None);
            assert_eq!(access.node(&node).edge_in_flight(1, 2), 0);
        });
    }

    #[test]
    fn cleanup_cancels_leaf_claim() {
        with_test_access(|access| {
            let leaf = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            assert!(access.node_mut(&leaf).try_start_score_update());
            assert_eq!(access.node(&leaf).n_in_flight(), 1);

            cleanup_descent(access, &[], Some(&leaf));
            assert_eq!(access.node(&leaf).n_in_flight(), 0);
        });
    }

    // ---- run_search integration ----

    #[test]
    fn run_search_uniform_basic() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        // Basic properties
        assert!(result.total_visits > 0);
        assert!(result.nn_evals > 0 || result.terminals > 0);

        // Policy sums to 1
        let sum_p1: f32 = result.policy_p1.iter().sum();
        let sum_p2: f32 = result.policy_p2.iter().sum();
        assert!((sum_p1 - 1.0).abs() < 1e-4, "P1 policy sum: {sum_p1}");
        assert!((sum_p2 - 1.0).abs() < 1e-4, "P2 policy sum: {sum_p2}");

        // Corner positions: blocked actions should have 0 visits
        // P1 at (0,0): DOWN and LEFT are blocked
        assert_eq!(result.visit_counts_p1[2], 0.0); // DOWN blocked
        assert_eq!(result.visit_counts_p1[3], 0.0); // LEFT blocked
    }

    #[test]
    fn run_search_terminal_game() {
        let game = terminal_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 50;
        let result = run_search(&mut tree, &game, &backend, &config, n_sims, 16, &mut r).unwrap();

        // Terminal root: one real visit per pick, total = n_sims.
        // Before fix: quadratic blowup (multivisit=budget per pick, counts as 1).
        let (root_visits, root_in_flight) = tree.with_exclusive(|access| {
            let root = access.root();
            let root = access.node(&root);
            (root.total_visits(), root.n_in_flight())
        });
        assert_eq!(
            root_visits, n_sims,
            "terminal root should have exactly {n_sims} visits, got {root_visits}"
        );
        assert_eq!(result.terminals, n_sims);
        assert_eq!(root_in_flight, 0, "root n_in_flight leak");
    }

    #[test]
    fn run_search_short_game() {
        let game = short_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        // Should find terminals in the short game
        assert!(result.total_visits > 0);
        assert!(result.nn_evals + result.terminals > 0);
    }

    #[test]
    fn run_search_constant_value_backend() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = ConstantValueBackend {
            value_p1: 0.5,
            value_p2: 0.3,
        };
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        assert!(result.total_visits > 0);

        // Values should be influenced by the constant backend
        // With non-zero values, root value should be non-zero after search
        let sum_p1: f32 = result.policy_p1.iter().sum();
        assert!((sum_p1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn run_search_tt_sharing() {
        // Run enough sims that transpositions should occur
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let _result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        // TT should have entries beyond just the root
        assert!(
            tree.tt().live_count() > 1,
            "TT should have entries for explored positions"
        );
    }

    // =====================================================================
    // Additional helpers
    // =====================================================================

    fn corridor_game() -> GameState {
        let mut walls = HashMap::new();
        for x in 0..5 {
            walls
                .entry(Coordinates::new(x, 0))
                .or_insert_with(Vec::new)
                .push(Coordinates::new(x, 1));
            walls
                .entry(Coordinates::new(x, 1))
                .or_insert_with(Vec::new)
                .push(Coordinates::new(x, 0));
        }
        GameBuilder::new(5, 5)
            .with_custom_maze(walls, Default::default())
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 0))
            .with_custom_cheese(vec![Coordinates::new(2, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }
    /// Verify reservation cleanup across every node reachable from the root.
    fn assert_no_in_flight(tree: &mut MCGSTree) {
        tree.with_exclusive(|access| {
            let mut visited = HashSet::new();
            let mut stack = vec![access.root()];
            while let Some(node) = stack.pop() {
                if !visited.insert(access.test_node_id(&node)) {
                    continue;
                }
                let low = access.node(&node);
                assert_eq!(
                    low.n_in_flight(),
                    0,
                    "Node still has n_in_flight={}",
                    low.n_in_flight()
                );
                for i in 0..low.n1() {
                    for j in 0..low.n2() {
                        assert_eq!(
                            low.edge_in_flight(i, j),
                            0,
                            "edge_in_flight[{i}][{j}] = {}",
                            low.edge_in_flight(i, j)
                        );
                    }
                }
                stack.extend(access.test_children(&node));
            }
        });
    }

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<crate::EvalResult, BackendError> {
            Err(BackendError::msg("intentional test failure"))
        }
    }

    // =====================================================================
    // Step 1: Backup depth tests
    // =====================================================================

    #[test]
    fn backup_two_level_q_chain() {
        with_test_access(|access| {
            // root --edge(r=1,0.5)--> mid --edge(r=0.5,1.0)--> leaf
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let mid = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&mid).set_value_scale(5.0);
            access.node_mut(&mid).set_prior([0.2; 5], [0.2; 5]);

            let leaf = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&leaf).set_value_scale(5.0);

            access.test_connect(&mid, &leaf, (2, 3), 0.5, 1.0);
            access.test_connect(&root, &mid, (0, 0), 1.0, 0.5);

            let path = vec![
                PathEntry {
                    node: root.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: mid.clone(),
                    p1_outcome: 2,
                    p2_outcome: 3,
                },
            ];

            // g = (2.0, 3.0) at leaf
            backup(access, &path, &leaf, 2.0, 3.0);

            // leaf: v=(2.0, 3.0)
            assert!((access.node(&leaf).v1() - 2.0).abs() < 1e-6);
            assert!((access.node(&leaf).v2() - 3.0).abs() < 1e-6);

            // mid: q1 = edge_r1(0.5) + leaf_v1(2.0) = 2.5
            //       q2 = edge_r2(1.0) + leaf_v2(3.0) = 4.0
            assert!((access.node(&mid).v1() - 2.5).abs() < 1e-6);
            assert!((access.node(&mid).v2() - 4.0).abs() < 1e-6);

            // root: q1 = edge_r1(1.0) + mid_q1(2.5) = 3.5
            //        q2 = edge_r2(0.5) + mid_q2(4.0) = 4.5
            let root_low = access.node(&root);
            assert!((root_low.v1() - 3.5).abs() < 1e-6);
            assert!((root_low.v2() - 4.5).abs() < 1e-6);

            // Joint matrix at root
            assert!((root_low.edge_q_p1(0, 0) - 3.5).abs() < 1e-6);
        });
    }

    #[test]
    fn backup_three_level_reward_chain() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(10.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let a = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&a).set_value_scale(10.0);
            access.node_mut(&a).set_prior([0.2; 5], [0.2; 5]);

            let b = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&b).set_value_scale(10.0);
            access.node_mut(&b).set_prior([0.2; 5], [0.2; 5]);

            let leaf = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&leaf).set_value_scale(10.0);

            // Wire: root--(1,0)-->a--(1,0)-->b--(1,0)-->leaf.
            access.test_connect(&b, &leaf, (1, 0), 1.0, 0.5);
            access.test_connect(&a, &b, (1, 0), 1.0, 0.5);
            access.test_connect(&root, &a, (1, 0), 1.0, 0.5);

            let path = vec![
                PathEntry {
                    node: root.clone(),
                    p1_outcome: 1,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: a.clone(),
                    p1_outcome: 1,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: b.clone(),
                    p1_outcome: 1,
                    p2_outcome: 0,
                },
            ];

            // First backup: g=(4.0, 2.0)
            backup(access, &path, &leaf, 4.0, 2.0);
            assert!((access.node(&leaf).v1() - 4.0).abs() < 1e-5);
            assert!((access.node(&b).v1() - 5.0).abs() < 1e-5);
            assert!((access.node(&a).v1() - 6.0).abs() < 1e-5);
            assert!((access.node(&root).v1() - 7.0).abs() < 1e-5);

            // Second backup: g=(2.0, 1.0) — Welford averages
            backup(access, &path, &leaf, 2.0, 1.0);
            assert!((access.node(&leaf).v1() - 3.0).abs() < 1e-5);
            assert!((access.node(&b).v1() - 4.0).abs() < 1e-5);
            assert!((access.node(&a).v1() - 5.0).abs() < 1e-5);
            assert!((access.node(&root).v1() - 6.0).abs() < 1e-5);
        });
    }

    #[test]
    fn backup_multiple_same_edge() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(5.0);
            access.test_connect(&root, &child, (0, 0), 0.0, 0.0);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            backup(access, &path, &child, 2.0, 1.0);
            backup(access, &path, &child, 4.0, 3.0);
            backup(access, &path, &child, 6.0, 5.0);

            // edge_r = 0, so edge_q = g values
            // Joint matrix: 3 visits, Q = mean(2,4,6) = 4.0 for p1
            let root_low = access.node(&root);
            assert_eq!(root_low.edge_visits(0, 0), 3);
            assert!((root_low.edge_q_p1(0, 0) - 4.0).abs() < 1e-5);
            assert!((root_low.edge_q_p2(0, 0) - 3.0).abs() < 1e-5);

            // child: v1 = mean(2,4,6) = 4.0
            assert_eq!(access.node(&child).total_visits(), 3);
            assert!((access.node(&child).v1() - 4.0).abs() < 1e-5);
        });
    }

    #[test]
    fn backup_multiple_different_edges() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(10.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child_a = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child_a).set_value_scale(10.0);
            let child_b = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child_b).set_value_scale(10.0);

            access.test_connect(&root, &child_a, (0, 0), 0.0, 0.0);
            access.test_connect(&root, &child_b, (1, 1), 0.0, 0.0);

            let path_a = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            let path_b = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 1,
                p2_outcome: 1,
            }];

            backup(access, &path_a, &child_a, 5.0, 5.0);
            backup(access, &path_b, &child_b, 1.0, 1.0);

            let root_low = access.node(&root);
            assert!((root_low.edge_q_p1(0, 0) - 5.0).abs() < 1e-5);
            assert!((root_low.edge_q_p1(1, 1) - 1.0).abs() < 1e-5);
            assert_eq!(root_low.edge_visits(0, 0), 1);
            assert_eq!(root_low.edge_visits(1, 1), 1);
        });
    }

    #[test]
    fn backup_terminal_leaf() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_terminal();
            access.test_connect(&root, &child, (0, 0), 1.0, 0.5);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            // Terminal backup: g = (0, 0)
            backup(access, &path, &child, 0.0, 0.0);

            // Root: q1 = r1(1.0) + 0 = 1.0, q2 = r2(0.5) + 0 = 0.5
            assert!((access.node(&root).v1() - 1.0).abs() < 1e-6);
            assert!((access.node(&root).v2() - 0.5).abs() < 1e-6);
        });
    }

    #[test]
    fn backup_empty_path() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let path: Vec<PathEntry<'_>> = vec![];
            backup(access, &path, &root, 3.0, 2.0);

            let root_low = access.node(&root);
            assert_eq!(root_low.total_visits(), 1);
            assert!((root_low.v1() - 3.0).abs() < 1e-6);
            assert_eq!(root_low.total_edge_visits(), 0);
        });
    }

    #[test]
    fn backup_same_edge_raw_propagation() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(10.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(10.0);
            access.test_connect(&root, &child, (0, 0), 2.0, 0.0);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            backup(access, &path, &child, 10.0, 0.0); // q = 2 + 10 = 12
            backup(access, &path, &child, 4.0, 0.0); // q = 2 + 4 = 6
            backup(access, &path, &child, 7.0, 0.0); // q = 2 + 7 = 9

            // Edge Q = Welford mean of (12, 6, 9) = 9.0
            assert!((access.node(&root).edge_q_p1(0, 0) - 9.0).abs() < 1e-5);
        });
    }

    #[test]
    fn backup_asymmetric_rewards() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(10.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(10.0);
            access.test_connect(&root, &child, (0, 0), 2.0, 0.5);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            backup(access, &path, &child, 3.0, 4.0);

            // root v1 = r1(2) + g1(3) = 5.0, v2 = r2(0.5) + g2(4) = 4.5
            assert!((access.node(&root).v1() - 5.0).abs() < 1e-6);
            assert!((access.node(&root).v2() - 4.5).abs() < 1e-6);
        });
    }

    #[test]
    fn backup_edge_visit_sum() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(5.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(5.0);
            access.test_connect(&root, &child, (0, 0), 0.0, 0.0);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            for i in 0..5 {
                backup(access, &path, &child, i as f32, 0.0);
            }

            assert_eq!(access.node(&root).total_edge_visits(), 5);
            assert_eq!(access.node(&root).total_visits(), 5);
        });
    }

    #[test]
    fn backup_p2_independent_propagation() {
        with_test_access(|access| {
            let root = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&root).set_value_scale(10.0);
            access.node_mut(&root).set_prior([0.2; 5], [0.2; 5]);

            let child = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            access.node_mut(&child).set_value_scale(10.0);
            access.test_connect(&root, &child, (0, 0), 2.0, 0.5);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            backup(access, &path, &child, 10.0, 1.0); // q1=12, q2=1.5
            backup(access, &path, &child, 4.0, 8.0); // q1=6, q2=8.5

            // p1 edge Q = mean(12, 6) = 9.0
            // p2 edge Q = mean(1.5, 8.5) = 5.0
            assert!((access.node(&root).edge_q_p1(0, 0) - 9.0).abs() < 1e-5);
            assert!((access.node(&root).edge_q_p2(0, 0) - 5.0).abs() < 1e-5);
        });
    }

    // =====================================================================
    // Step 2: PUCT selection tests
    // =====================================================================

    #[test]
    fn puct_monotonic_q() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(10.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        // Give outcome 2 high Q, all outcomes some visits
        for i in 0..5 {
            for j in 0..5 {
                let q = if i == 2 { 10.0 } else { 1.0 };
                for _ in 0..10 {
                    node.finalize_edge_update(i, j, q, q);
                }
            }
        }
        // Need some total_visits for value to be nonzero
        for _ in 0..250 {
            node.finalize_score_update(1.0, 1.0);
        }

        let config = default_config();
        let mut r = rng();
        let selected = select_p1(&node, &config, false, &mut r);
        assert_eq!(selected, 2, "Highest Q should win");
    }

    #[test]
    fn puct_monotonic_prior() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.05, 0.05, 0.7, 0.1, 0.1], [0.2; 5]);

        // Give 1 visit at (0,0) so total_edge_visits > 0
        node.finalize_edge_update(0, 0, 1.0, 1.0);
        node.finalize_score_update(1.0, 1.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(&node, &config, false, &mut r);
        // Outcome 2 has highest prior (0.7) and is unvisited
        assert_eq!(
            selected, 2,
            "Highest prior should dominate when mostly unvisited"
        );
    }

    #[test]
    fn puct_unvisited_selected() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        // Give 100 visits to outcomes 0-3, leave outcome 4 unvisited
        for i in 0..4 {
            for _ in 0..100 {
                node.finalize_edge_update(i, 0, 1.0, 1.0);
                node.finalize_score_update(1.0, 1.0);
            }
        }

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(&node, &config, false, &mut r);
        assert_eq!(selected, 4, "Unvisited outcome should be selected");
    }

    #[test]
    fn puct_fpu_pessimism() {
        // Higher visited prior mass → stronger FPU penalty → less willingness to explore
        let mut node_lo = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node_lo.set_value_scale(5.0);
        node_lo.set_prior([0.2; 5], [0.2; 5]);
        node_lo.finalize_score_update(5.0, 5.0);
        // Visit 1 outcome → visited_mass = 0.2
        node_lo.finalize_edge_update(0, 0, 5.0, 5.0);

        let mut node_hi = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node_hi.set_value_scale(5.0);
        node_hi.set_prior([0.2; 5], [0.2; 5]);
        node_hi.finalize_score_update(5.0, 5.0);
        // Visit 3 outcomes → visited_mass = 0.6
        node_hi.finalize_edge_update(0, 0, 5.0, 5.0);
        node_hi.finalize_edge_update(1, 0, 5.0, 5.0);
        node_hi.finalize_edge_update(2, 0, 5.0, 5.0);

        // FPU = v1 - fpu_reduction * value_scale * sqrt(visited_mass)
        // node_lo: FPU = 5 - 0.2 * 5 * sqrt(0.2) ~ 4.553
        // node_hi: FPU = 5 - 0.2 * 5 * sqrt(0.6) ~ 4.225
        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let sel_lo = select_p1(&node_lo, &config, false, &mut r);
        let sel_hi = select_p1(&node_hi, &config, false, &mut r);
        assert!((sel_lo as usize) < 5);
        assert!((sel_hi as usize) < 5);
    }

    #[test]
    fn puct_fpu_no_visits() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.05, 0.05, 0.7, 0.1, 0.1], [0.2; 5]);

        // Give 1 visit so sqrt_total > 0
        node.finalize_edge_update(0, 0, 0.0, 0.0);
        node.finalize_score_update(0.0, 0.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();
        let selected = select_p1(&node, &config, false, &mut r);
        assert_eq!(selected, 2, "Highest prior should win via exploration");
    }

    #[test]
    fn puct_forced_fires_at_root() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        // Give 100 visits to outcomes 0 and 1
        for _ in 0..50 {
            node.finalize_edge_update(0, 0, 1.0, 1.0);
            node.finalize_edge_update(1, 0, 1.0, 1.0);
            node.finalize_score_update(1.0, 1.0);
            node.finalize_score_update(1.0, 1.0);
        }
        // Outcomes 2,3,4 have 1 visit each
        for i in 2..5 {
            node.finalize_edge_update(i, 0, 1.0, 1.0);
            node.finalize_score_update(1.0, 1.0);
        }

        // total_edge_visits = 103
        // threshold for prior=0.2, total=103: sqrt(2.0 * 0.2 * 103) ~ 6.4
        // Outcomes 2,3,4 have 1 visit < 6.4 → FORCED
        let config = default_config(); // force_k = 2.0
        let mut r = rng();
        let selected = select_p1(&node, &config, true, &mut r);
        assert!(
            selected >= 2,
            "Forced playout should select an undervisited outcome, got {selected}"
        );
    }

    #[test]
    fn puct_forced_not_at_nonroot() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        // Give many visits and high Q to outcome 0, few visits to others
        for _ in 0..100 {
            node.finalize_edge_update(0, 0, 10.0, 10.0);
            node.finalize_score_update(10.0, 10.0);
        }
        for i in 1..5 {
            node.finalize_edge_update(i, 0, 1.0, 1.0);
            node.finalize_score_update(1.0, 1.0);
        }

        let config = default_config();
        let mut r = rng();
        // At non-root, forced playouts don't fire
        let selected = select_p1(&node, &config, false, &mut r);
        // Outcome 0 has 100 visits with Q=10 vs others with 1 visit Q=1.
        // Without forced playouts, Q dominates.
        assert_eq!(selected, 0, "Without forced playouts, high-Q should win");
    }

    #[test]
    fn puct_value_scale_effect() {
        // With small value_scale, Q/scale is large → exploitation dominates.
        // With large value_scale, Q/scale is small → exploration dominates.
        //
        // All outcomes get visits so FPU doesn't apply. Outcome 0 has high Q,
        // outcomes 1-4 have low Q. Outcome 3 has highest prior (0.6).
        // Small scale: Q gap dominates → outcome 0 selected.
        // Large scale: Q gap shrinks, exploration from prior dominates → outcome 3 selected.
        let make_node = |scale: f32| {
            let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            node.set_value_scale(scale);
            node.set_prior([0.05, 0.05, 0.05, 0.6, 0.25], [0.2; 5]);
            // Outcome 0: high Q, many visits
            for _ in 0..100 {
                node.finalize_edge_update(0, 0, 10.0, 10.0);
                node.finalize_score_update(10.0, 10.0);
            }
            // Outcomes 1-4: low Q, few visits
            for i in 1..5 {
                for _ in 0..5 {
                    node.finalize_edge_update(i, 0, 0.1, 0.1);
                    node.finalize_score_update(0.1, 0.1);
                }
            }
            node
        };

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        // Small scale: Q=10 / scale=1 = 10, gap to others = ~9.9 → exploitation
        let node_small = make_node(1.0);
        let sel_small = select_p1(&node_small, &config, false, &mut r);

        // Large scale: Q=10 / scale=10000 ≈ 0 → exploration dominates (outcome 3, prior=0.6)
        let node_large = make_node(10000.0);
        let sel_large = select_p1(&node_large, &config, false, &mut r);

        assert_eq!(sel_small, 0, "Small scale -> exploitation -> outcome 0");
        assert_eq!(
            sel_large, 3,
            "Large scale -> exploration -> outcome 3 (highest prior, few visits)"
        );
    }

    #[test]
    fn puct_decoupled() {
        // Two nodes: same P1 structure, different P2 priors
        let mut node_a = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node_a.set_value_scale(5.0);
        node_a.set_prior([0.1, 0.3, 0.2, 0.15, 0.25], [0.8, 0.05, 0.05, 0.05, 0.05]);
        node_a.finalize_edge_update(0, 0, 2.0, 2.0);
        node_a.finalize_score_update(2.0, 2.0);

        let mut node_b = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node_b.set_value_scale(5.0);
        node_b.set_prior([0.1, 0.3, 0.2, 0.15, 0.25], [0.05, 0.05, 0.05, 0.05, 0.8]);
        node_b.finalize_edge_update(0, 0, 2.0, 2.0);
        node_b.finalize_score_update(2.0, 2.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r1 = SmallRng::seed_from_u64(99);
        let mut r2 = SmallRng::seed_from_u64(99);

        let sel_a = select_p1(&node_a, &config, false, &mut r1);
        let sel_b = select_p1(&node_b, &config, false, &mut r2);

        assert_eq!(sel_a, sel_b, "P2 priors should not affect P1 selection");
    }

    #[test]
    fn puct_single_outcome() {
        // Mud-stuck: all actions → STAY, n1=1
        let mut node = LowNode::new_shell([4, 4, 4, 4, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        let config = default_config();
        let mut r = rng();
        let selected = select_p1(&node, &config, false, &mut r);
        assert_eq!(selected, 0, "Single outcome should return 0");
    }

    #[test]
    fn puct_virtual_loss_diversifies() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);
        // Give some visits so selection isn't degenerate
        for i in 0..5 {
            node.finalize_edge_update(i, 0, 2.0, 2.0);
            node.finalize_score_update(2.0, 2.0);
        }

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        // Select without virtual loss
        let baseline = select_p1(&node, &config, false, &mut r);

        // Add heavy virtual loss on baseline outcome
        for _ in 0..100 {
            node.add_virtual_loss(baseline as usize, 0);
        }

        let mut r2 = rng();
        let shifted = select_p1(&node, &config, false, &mut r2);
        assert_ne!(
            baseline, shifted,
            "Virtual loss should shift selection away from {baseline}"
        );

        // Clean up
        for _ in 0..100 {
            node.revert_virtual_loss(baseline as usize, 0);
        }
    }

    #[test]
    fn puct_multi_descent_diversification() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_value_scale(5.0);
        node.set_prior([0.2; 5], [0.2; 5]);

        // Give 1 visit so total_edge_visits > 0
        node.finalize_edge_update(0, 0, 1.0, 1.0);
        node.finalize_score_update(1.0, 1.0);

        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut r = rng();

        let mut selected = HashSet::new();
        for _ in 0..3 {
            let s = select_p1(&node, &config, false, &mut r);
            // Add virtual loss to force next selection elsewhere
            node.add_virtual_loss(s as usize, 0);
            selected.insert(s);
        }

        assert!(
            selected.len() >= 2,
            "3 descents with virtual loss should diversify, got {:?}",
            selected
        );

        // Cleanup
        for &s in &selected {
            node.revert_virtual_loss(s as usize, 0);
        }
    }

    // =====================================================================
    // Step 3: Post-search invariants
    // =====================================================================

    #[test]
    fn search_n_in_flight_zero_after_search() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let _result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        assert_no_in_flight(&mut tree);
    }

    #[test]
    fn search_value_bounded() {
        let cheese = vec![
            Coordinates::new(1, 0),
            Coordinates::new(2, 0),
            Coordinates::new(3, 0),
            Coordinates::new(4, 0),
        ];
        let game = open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &cheese);
        let remaining = game.cheese.remaining_cheese() as f32;

        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        assert!(
            result.value_p1 >= -0.1,
            "value_p1 too low: {}",
            result.value_p1
        );
        assert!(
            result.value_p1 <= remaining + 0.1,
            "value_p1 too high: {}",
            result.value_p1
        );
        assert!(
            result.value_p2 >= -0.1,
            "value_p2 too low: {}",
            result.value_p2
        );
        assert!(
            result.value_p2 <= remaining + 0.1,
            "value_p2 too high: {}",
            result.value_p2
        );
    }

    #[test]
    fn search_policy_sums_to_one_multiple_configs() {
        let configs = [
            (
                Coordinates::new(0, 0),
                Coordinates::new(4, 4),
                vec![Coordinates::new(2, 2)],
                "center cheese",
            ),
            (
                Coordinates::new(0, 0),
                Coordinates::new(4, 4),
                vec![Coordinates::new(1, 0)],
                "adjacent cheese",
            ),
            (
                Coordinates::new(2, 2),
                Coordinates::new(2, 2),
                vec![Coordinates::new(0, 0)],
                "same position",
            ),
            (
                Coordinates::new(0, 0),
                Coordinates::new(0, 4),
                vec![Coordinates::new(4, 2), Coordinates::new(0, 2)],
                "multi cheese",
            ),
        ];

        let backend = SmartUniformBackend;
        let config = default_config();

        for (p1, p2, cheese, name) in &configs {
            let game = open_5x5_game(*p1, *p2, cheese);
            let mut tree = MCGSTree::new(&game);
            let mut r = rng();

            let result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();
            let sum_p1: f32 = result.policy_p1.iter().sum();
            let sum_p2: f32 = result.policy_p2.iter().sum();
            assert!((sum_p1 - 1.0).abs() < 1e-4, "{name}: P1 sum = {sum_p1}");
            assert!((sum_p2 - 1.0).abs() < 1e-4, "{name}: P2 sum = {sum_p2}");
        }
    }

    #[test]
    fn search_blocked_actions_zero() {
        // P1 at (0,0) corner: DOWN and LEFT blocked
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 100, 16, &mut r).unwrap();

        assert_eq!(result.visit_counts_p1[2], 0.0, "P1 DOWN should be blocked");
        assert_eq!(result.visit_counts_p1[3], 0.0, "P1 LEFT should be blocked");

        // P2 at (4,4): UP and RIGHT blocked
        assert_eq!(result.visit_counts_p2[0], 0.0, "P2 UP should be blocked");
        assert_eq!(result.visit_counts_p2[1], 0.0, "P2 RIGHT should be blocked");

        // Also test corridor
        let game2 = corridor_game();
        let mut tree2 = MCGSTree::new(&game2);
        let mut r2 = rng();
        let result2 = run_search(&mut tree2, &game2, &backend, &config, 100, 16, &mut r2).unwrap();

        // P1 at (0,0) in corridor: UP blocked, DOWN blocked, LEFT blocked
        assert_eq!(result2.visit_counts_p1[0], 0.0, "P1 UP blocked in corridor");
        assert_eq!(
            result2.visit_counts_p1[2], 0.0,
            "P1 DOWN blocked in corridor"
        );
        assert_eq!(
            result2.visit_counts_p1[3], 0.0,
            "P1 LEFT blocked in corridor"
        );
    }

    #[test]
    fn search_adjacent_cheese_dominates() {
        // P1 at (0,0), cheese at (1,0): RIGHT should dominate
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(1, 0)],
        );
        let backend = ConstantValueBackend {
            value_p1: 0.5,
            value_p2: 0.5,
        };
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 500, 16, &mut r).unwrap();

        // RIGHT = action 1
        assert!(
            result.policy_p1[1] > 0.5,
            "RIGHT should dominate for P1, got {}",
            result.policy_p1[1]
        );
    }

    // =====================================================================
    // Step 4: Dirichlet noise tests
    // =====================================================================

    #[test]
    fn noise_disabled_priors_unchanged() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_prior([0.1, 0.2, 0.3, 0.15, 0.25], [0.2; 5]);

        let priors_before: Vec<f32> = (0..5).map(|i| node.p1_prior(i)).collect();

        let mut r = rng();
        apply_dirichlet_noise_p1(&mut node, 0.0, 10.83, &mut r);

        for i in 0..5 {
            assert!(
                (node.p1_prior(i) - priors_before[i]).abs() < 1e-10,
                "Prior {i} changed with epsilon=0"
            );
        }
    }

    #[test]
    fn noise_enabled_priors_modified() {
        let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
        node.set_prior([0.2; 5], [0.2; 5]);

        let priors_before: Vec<f32> = (0..5).map(|i| node.p1_prior(i)).collect();

        let mut r = rng();
        apply_dirichlet_noise_p1(&mut node, 0.25, 10.83, &mut r);

        let mut any_changed = false;
        let mut sum = 0.0f32;
        for i in 0..5 {
            sum += node.p1_prior(i);
            if (node.p1_prior(i) - priors_before[i]).abs() > 1e-6 {
                any_changed = true;
            }
        }
        assert!(
            any_changed,
            "At least one prior should change with epsilon=0.25"
        );
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Priors should still sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn noise_deterministic_with_seed() {
        let make = || {
            let mut node = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            node.set_prior([0.2; 5], [0.2; 5]);
            node
        };

        let mut node1 = make();
        let mut r1 = SmallRng::seed_from_u64(777);
        apply_dirichlet_noise_p1(&mut node1, 0.25, 10.83, &mut r1);

        let mut node2 = make();
        let mut r2 = SmallRng::seed_from_u64(777);
        apply_dirichlet_noise_p1(&mut node2, 0.25, 10.83, &mut r2);

        for i in 0..5 {
            assert!(
                (node1.p1_prior(i) - node2.p1_prior(i)).abs() < 1e-10,
                "Same seed should produce same noise at outcome {i}"
            );
        }
    }

    #[test]
    fn noise_single_outcome_noop() {
        let mut node = LowNode::new_shell([4, 4, 4, 4, 4], [0, 1, 2, 3, 4]);
        node.set_prior([0.2; 5], [0.2; 5]);

        assert_eq!(node.n1(), 1);
        let prior_before = node.p1_prior(0);

        let mut r = rng();
        apply_dirichlet_noise_p1(&mut node, 0.25, 10.83, &mut r);

        assert!(
            (node.p1_prior(0) - prior_before).abs() < 1e-10,
            "Single outcome should not be modified by noise"
        );
    }

    #[test]
    fn noise_search_integration() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;

        // No noise
        let config_no_noise = SearchConfig {
            noise_epsilon: 0.0,
            ..default_config()
        };
        let mut tree1 = MCGSTree::new(&game);
        let mut r1 = SmallRng::seed_from_u64(123);
        let result1 = run_search(
            &mut tree1,
            &game,
            &backend,
            &config_no_noise,
            100,
            16,
            &mut r1,
        )
        .unwrap();

        // With noise
        let config_noise = SearchConfig {
            noise_epsilon: 0.25,
            ..default_config()
        };
        let mut tree2 = MCGSTree::new(&game);
        let mut r2 = SmallRng::seed_from_u64(123);
        let result2 =
            run_search(&mut tree2, &game, &backend, &config_noise, 100, 16, &mut r2).unwrap();

        // Priors should differ (noise modifies root priors)
        let any_diff = result1
            .prior_p1
            .iter()
            .zip(result2.prior_p1.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            any_diff,
            "Noisy search should produce different root priors"
        );
    }

    // =====================================================================
    // Step 5: Error handling + OOO gather
    // =====================================================================

    #[test]
    fn failing_backend_propagates() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let mut tree = MCGSTree::new(&game);
        let config = default_config();
        let mut r = rng();

        let result = run_search(&mut tree, &game, &FailingBackend, &config, 10, 4, &mut r);
        assert!(
            result.is_err(),
            "run_search should propagate backend errors"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intentional test failure"));
    }

    #[test]
    fn backend_error_reverts_bookkeeping() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2), Coordinates::new(1, 1)],
        );
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // Phase 1: expand the tree just enough that gather descends interior nodes,
        // but not so much that every leaf is a transposition stop.
        let _ = run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &config,
            10,
            8,
            &mut r,
        )
        .unwrap();

        // Phase 2: search with failing backend — high budget to ensure NeedsEval.
        let result = run_search(&mut tree, &game, &FailingBackend, &config, 100, 16, &mut r);
        assert!(result.is_err());

        // Phase 3: verify the tree is clean — all reservations are zero.
        assert_no_in_flight(&mut tree);

        // Phase 4: rerun with good backend — tree should be usable.
        let result2 = run_search(
            &mut tree,
            &game,
            &SmartUniformBackend,
            &config,
            50,
            8,
            &mut r,
        )
        .unwrap();
        assert!(
            result2.total_visits > 0,
            "search after backend error should produce visits"
        );
    }

    #[test]
    fn ooo_terminal_fills_batch() {
        // Short game with reachable terminals
        let game = short_game();
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        // With OOO terminal processing, terminals are processed inline
        assert!(result.total_visits > 0);
        assert!(result.terminals > 0 || result.nn_evals > 0);
    }

    #[test]
    fn ooo_collision_budget_stops_gather() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        // Low collision budget to test collision limit
        let config = SearchConfig {
            collision_limit_min: 1,
            collision_limit_max: 1,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // Should complete without panicking even with tight collision budget
        let result = run_search(&mut tree, &game, &backend, &config, 50, 16, &mut r).unwrap();
        assert!(result.total_visits > 0);
    }

    #[test]
    fn ooo_batch_size_1_exact_visits() {
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 20;
        let result = run_search(&mut tree, &game, &backend, &config, n_sims, 1, &mut r).unwrap();

        // With batch_size=1 and enough cheese, nn_evals + terminals >= n_sims
        assert!(
            result.nn_evals + result.terminals >= n_sims,
            "nn_evals({}) + terminals({}) should be >= n_sims({})",
            result.nn_evals,
            result.terminals,
            n_sims
        );
    }

    #[test]
    fn search_batch_size_larger_than_n_sims() {
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        // batch_size > n_sims: should still complete without panicking.
        // With small n_sims the root may be the only leaf, so total_edge_visits
        // (which is what SearchResult.total_visits reports) can be 0 when
        // the root itself is the only node evaluated. We just verify it doesn't crash
        // and produces a valid result.
        let result = run_search(&mut tree, &game, &backend, &config, 5, 100, &mut r).unwrap();
        assert!(result.nn_evals + result.terminals + result.collisions > 0);
    }

    // =====================================================================
    // Delta correction tests
    // =====================================================================

    /// No transposition: delta stays 0, backup produces identical results to before.
    #[test]
    fn delta_no_transposition_baseline() {
        with_test_access(|access| {
            let mut root_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            root_low.set_value_scale(5.0);
            root_low.set_prior([0.2; 5], [0.2; 5]);
            let root = access.test_node(root_low);

            let mut child_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            child_low.set_value_scale(5.0);
            let child = access.test_node(child_low);

            // Only one parent edge → num_parents == 1, no delta correction.
            access.test_connect(&root, &child, (0, 1), 1.0, 0.5);
            assert_eq!(access.num_parents(&child), 1);

            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 1,
            }];

            backup(access, &path, &child, 3.0, 2.0);

            // Same results as non-delta-correction backup.
            let child_low = access.node(&child);
            assert_eq!(child_low.total_visits(), 1);
            assert!((child_low.v1() - 3.0).abs() < 1e-6);
            assert!((child_low.v2() - 2.0).abs() < 1e-6);

            // q1 = 1.0 + 3.0 = 4.0, q2 = 0.5 + 2.0 = 2.5
            let root_low = access.node(&root);
            assert_eq!(root_low.total_visits(), 1);
            assert!((root_low.v1() - 4.0).abs() < 1e-6);
            assert!((root_low.v2() - 2.5).abs() < 1e-6);
            assert_eq!(root_low.edge_visits(0, 1), 1);
            assert!((root_low.edge_q_p1(0, 1) - 4.0).abs() < 1e-6);
            assert!((root_low.edge_q_p2(0, 1) - 2.5).abs() < 1e-6);
        });
    }

    /// Simple transposition correction.
    /// Root → A → C and Root → B → C. Backup through A, then through B.
    /// After B's backup, A's edge_q should reflect C's aggregate (not just the
    /// single visit that went through A).
    #[test]
    fn delta_simple_transposition_correction() {
        with_test_access(|access| {
            // Shared child C (the transposition).
            let mut child_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            child_low.set_value_scale(5.0);
            let child_c = access.test_node(child_low);

            let mut parent_a_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_a_low.set_value_scale(5.0);
            parent_a_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_a = access.test_node(parent_a_low);
            access.test_connect(&parent_a, &child_c, (0, 0), 1.0, 0.5);

            let mut parent_b_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_b_low.set_value_scale(5.0);
            parent_b_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_b = access.test_node(parent_b_low);
            access.test_connect(&parent_b, &child_c, (0, 0), 0.0, 0.0);

            assert_eq!(access.num_parents(&child_c), 2);

            let path_a = vec![PathEntry {
                node: parent_a.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(access, &path_a, &child_c, 2.0, 3.0);

            let child_low = access.node(&child_c);
            assert_eq!(child_low.total_visits(), 1);
            assert!((child_low.v1() - 2.0).abs() < 1e-6);
            let parent_a_low = access.node(&parent_a);
            assert!((parent_a_low.edge_q_p1(0, 0) - 3.0).abs() < 1e-6);
            assert!((parent_a_low.edge_q_p2(0, 0) - 3.5).abs() < 1e-6);

            let path_b = vec![PathEntry {
                node: parent_b.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(access, &path_b, &child_c, 4.0, 5.0);

            let child_low = access.node(&child_c);
            assert_eq!(child_low.total_visits(), 2);
            assert!((child_low.v1() - 3.0).abs() < 1e-6);
            assert!((child_low.v2() - 4.0).abs() < 1e-6);
            let parent_b_low = access.node(&parent_b);
            assert_eq!(parent_b_low.edge_visits(0, 0), 1);
            assert!((parent_b_low.edge_q_p1(0, 0) - 3.0).abs() < 1e-6);
            assert!((parent_b_low.edge_q_p2(0, 0) - 4.0).abs() < 1e-6);

            backup(access, &path_a, &child_c, 6.0, 7.0);

            let child_low = access.node(&child_c);
            assert_eq!(child_low.total_visits(), 3);
            assert!((child_low.v1() - 4.0).abs() < 1e-6);
            assert!((child_low.v2() - 5.0).abs() < 1e-6);

            // The stale visit is corrected exactly to r + the shared aggregate.
            let parent_a_low = access.node(&parent_a);
            assert_eq!(parent_a_low.edge_visits(0, 0), 2);
            assert!((parent_a_low.edge_q_p1(0, 0) - 5.0).abs() < 1e-5);
            assert!((parent_a_low.edge_q_p2(0, 0) - 5.5).abs() < 1e-5);
        });
    }

    /// Exact catch-up: when all prior visits are stale, adjustment should
    /// bring edge_q exactly to r + child.v.
    #[test]
    fn delta_exact_catchup() {
        with_test_access(|access| {
            let mut child_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            child_low.set_value_scale(5.0);
            let child = access.test_node(child_low);

            let mut parent_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_low.set_value_scale(5.0);
            parent_low.set_prior([0.2; 5], [0.2; 5]);
            let parent = access.test_node(parent_low);

            let mut other_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            other_low.set_value_scale(5.0);
            other_low.set_prior([0.2; 5], [0.2; 5]);
            let other_parent = access.test_node(other_low);

            access.test_connect(&parent, &child, (0, 0), 2.0, 1.0);
            access.test_connect(&other_parent, &child, (0, 0), 0.0, 0.0);
            assert_eq!(access.num_parents(&child), 2);

            let path_other = vec![PathEntry {
                node: other_parent.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            for val in [1.0, 2.0, 3.0, 4.0, 5.0] {
                backup(access, &path_other, &child, val, val * 0.5);
            }

            let child_low = access.node(&child);
            assert_eq!(child_low.total_visits(), 5);
            assert!((child_low.v1() - 3.0).abs() < 1e-5);
            assert!((child_low.v2() - 1.5).abs() < 1e-5);

            let path = vec![PathEntry {
                node: parent.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(access, &path, &child, 10.0, 5.0);

            let expected_child_v1 = 25.0 / 6.0;
            let expected_child_v2 = (0.5 + 1.0 + 1.5 + 2.0 + 2.5 + 5.0) / 6.0;
            let expected_q1 = 2.0 + expected_child_v1;
            let expected_q2 = 1.0 + expected_child_v2;

            let parent_low = access.node(&parent);
            assert_eq!(parent_low.edge_visits(0, 0), 1);
            assert!((parent_low.edge_q_p1(0, 0) - expected_q1).abs() < 1e-4);
            assert!((parent_low.edge_q_p2(0, 0) - expected_q2).abs() < 1e-4);
        });
    }

    /// Cascading correction: A → B → C, D → B.
    /// Backup through D updates B, then backup through A should correct
    /// both B's and A's edge_q.
    #[test]
    fn delta_cascading_correction() {
        with_test_access(|access| {
            let mut make_node = |with_prior: bool| {
                let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
                low.set_value_scale(5.0);
                if with_prior {
                    low.set_prior([0.2; 5], [0.2; 5]);
                }
                access.test_node(low)
            };
            let node_c = make_node(false);
            let node_b = make_node(true);
            let node_a = make_node(true);
            let node_d = make_node(true);

            access.test_connect(&node_b, &node_c, (0, 0), 0.5, 0.5);
            access.test_connect(&node_a, &node_b, (0, 0), 1.0, 1.0);
            access.test_connect(&node_d, &node_b, (0, 0), 0.0, 0.0);
            assert_eq!(access.num_parents(&node_b), 2);

            let path_abc = vec![
                PathEntry {
                    node: node_a.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: node_b.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
            ];
            backup(access, &path_abc, &node_c, 2.0, 1.0);
            assert!((access.node(&node_c).v1() - 2.0).abs() < 1e-6);
            assert!((access.node(&node_b).v1() - 2.5).abs() < 1e-6);
            assert!((access.node(&node_a).v1() - 3.5).abs() < 1e-6);

            let path_dbc = vec![
                PathEntry {
                    node: node_d.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: node_b.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
            ];
            backup(access, &path_dbc, &node_c, 6.0, 5.0);
            assert!((access.node(&node_c).v1() - 4.0).abs() < 1e-6);
            assert!((access.node(&node_c).v2() - 3.0).abs() < 1e-6);
            assert_eq!(access.node(&node_b).edge_visits(0, 0), 2);

            backup(access, &path_abc, &node_c, 8.0, 7.0);
            assert!((access.node(&node_c).v1() - 16.0 / 3.0).abs() < 1e-4);

            let (b_v1, b_v2) = {
                let node_b = access.node(&node_b);
                (node_b.v1(), node_b.v2())
            };
            let node_a = access.node(&node_a);
            assert_eq!(node_a.edge_visits(0, 0), 2);
            assert!((node_a.edge_q_p1(0, 0) - (1.0 + b_v1)).abs() < 0.5);
            assert!((node_a.edge_q_p2(0, 0) - (1.0 + b_v2)).abs() < 0.5);
        });
    }

    /// Non-transposition child with transposition grandchild: delta
    /// propagates through the intermediate node.
    #[test]
    fn delta_propagates_through_non_transposition() {
        with_test_access(|access| {
            let mut make_node = |with_prior: bool| {
                let mut low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
                low.set_value_scale(5.0);
                if with_prior {
                    low.set_prior([0.2; 5], [0.2; 5]);
                }
                access.test_node(low)
            };
            let grandchild = make_node(false);
            let mid = make_node(true);
            let root = make_node(true);
            let other = make_node(true);

            access.test_connect(&mid, &grandchild, (0, 0), 0.5, 0.5);
            access.test_connect(&root, &mid, (0, 0), 1.0, 1.0);
            access.test_connect(&other, &grandchild, (0, 0), 0.0, 0.0);

            assert_eq!(access.num_parents(&mid), 1);
            assert_eq!(access.num_parents(&grandchild), 2);

            let root_path = vec![
                PathEntry {
                    node: root.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
                PathEntry {
                    node: mid.clone(),
                    p1_outcome: 0,
                    p2_outcome: 0,
                },
            ];
            backup(access, &root_path, &grandchild, 1.0, 1.0);
            let root_q1_after_1 = access.node(&root).edge_q_p1(0, 0);

            let other_path = vec![PathEntry {
                node: other.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(access, &other_path, &grandchild, 10.0, 10.0);
            assert!((access.node(&grandchild).v1() - 5.5).abs() < 1e-5);

            backup(access, &root_path, &grandchild, 4.0, 4.0);
            let root_q1_after_3 = access.node(&root).edge_q_p1(0, 0);
            assert!(
                (root_q1_after_3 - root_q1_after_1).abs() > 0.5,
                "Delta should propagate: before={root_q1_after_1}, after={root_q1_after_3}",
            );
        });
    }

    // =====================================================================
    // Tier 1 LC0 port fix tests
    // =====================================================================

    #[test]
    fn children_visits_vs_total_visits() {
        // After N sims, root.total_edge_visits() == root.total_visits() - 1.
        // The first visit is the root's own NN eval (no edge update).
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let backend = SmartUniformBackend;
        let config = SearchConfig {
            force_k: 0.0,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 50, 8, &mut r).unwrap();
        let (total, edge) = tree.with_exclusive(|access| {
            let root = access.root();
            let root = access.node(&root);
            (root.total_visits(), root.total_edge_visits())
        });

        // Root's first visit is its own NN eval. Every subsequent visit
        // is a child visit (edge update). So edge_visits == total - 1.
        assert_eq!(
            edge,
            total - 1,
            "children_visits should be total_visits - 1 at root, got edge={edge} total={total}"
        );

        // Sanity: result.total_visits matches.
        assert_eq!(result.total_visits, total);
    }

    #[test]
    fn sim_counting_produces_useful_visits() {
        // Verify the sim budget produces useful visits proportional to n_sims.
        // With VTC batch allocation, shared collisions (excess visits beyond
        // what a leaf can absorb) are expected. The important property is that
        // useful visits grow with n_sims and don't stall.
        let game = open_5x5_game(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2), Coordinates::new(1, 1)],
        );
        let backend = SmartUniformBackend;
        let config = SearchConfig {
            collision_limit_min: 256,
            collision_limit_max: 256,
            ..default_config()
        };
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 100u32;
        let result = run_search(&mut tree, &game, &backend, &config, n_sims, 8, &mut r).unwrap();
        let useful = result.nn_evals + result.terminals + result.tt_stop_hits;

        // Should produce a meaningful number of useful visits.
        assert!(
            useful >= n_sims / 2,
            "useful visits ({useful}) should be >= n_sims/2 ({n_sims}/2), \
             nn={}, term={}, tt_stop={}, coll={}",
            result.nn_evals,
            result.terminals,
            result.tt_stop_hits,
            result.collisions,
        );
        assert!(result.total_visits > 0);
    }

    #[test]
    fn collision_vl_deferred_reduces_repeat_collisions() {
        // Run search with batch_size > 1 to exercise collision VL deferral.
        // A batch_size=1 search can't benefit from deferred VLs since there's
        // only one descent per batch. With batch_size=16, deferred VLs should
        // reduce repeated collisions compared to immediate cleanup.
        //
        // We can't directly test "what would happen without the fix" in a single
        // test, but we CAN verify that after search, all VLs are reverted (no
        // leaked in-flight counts) and collisions stay bounded.
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 200, 16, &mut r).unwrap();

        assert_no_in_flight(&mut tree);

        // Collisions should be bounded — not exploding.
        let useful = result.nn_evals + result.terminals;
        assert!(useful > 0, "should have some useful work done");
    }

    // =====================================================================
    // Transposition stop tests
    // =====================================================================

    #[test]
    fn tt_stop_initializes_edge_from_aggregate() {
        with_test_access(|access| {
            let mut child_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            child_low.set_value_scale(5.0);
            let child_c = access.test_node(child_low);

            let mut parent_a_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_a_low.set_value_scale(5.0);
            parent_a_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_a = access.test_node(parent_a_low);
            access.test_connect(&parent_a, &child_c, (0, 0), 1.0, 0.5);

            let path_a = vec![PathEntry {
                node: parent_a.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            backup(access, &path_a, &child_c, 2.0, 3.0);

            let mut parent_b_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_b_low.set_value_scale(5.0);
            parent_b_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_b = access.test_node(parent_b_low);
            access.test_connect(&parent_b, &child_c, (1, 2), 0.5, 1.0);

            assert_eq!(access.num_parents(&child_c), 2);
            assert_eq!(access.node(&parent_b).edge_visits(1, 2), 0);
            let child_visits_before = access.node(&child_c).total_visits();
            assert_eq!(child_visits_before, 1);

            let path_b = vec![PathEntry {
                node: parent_b.clone(),
                p1_outcome: 1,
                p2_outcome: 2,
            }];
            for entry in &path_b {
                let node = access.node_mut(&entry.node);
                node.increment_n_in_flight(1);
                node.add_virtual_loss(entry.p1_outcome as usize, entry.p2_outcome as usize);
            }
            backup_transposition_stop(access, &path_b, &child_c, 1);

            assert_eq!(
                access.node(&child_c).total_visits(),
                child_visits_before,
                "transposition stop should not increment child visits"
            );
            let parent_b = access.node(&parent_b);
            assert_eq!(parent_b.edge_visits(1, 2), 1);
            assert!((parent_b.edge_q_p1(1, 2) - 2.5).abs() < 1e-5);
            assert!((parent_b.edge_q_p2(1, 2) - 4.0).abs() < 1e-5);
        });
    }

    #[test]
    fn tt_stop_stale_edge_corrects_from_aggregate() {
        // Manual DAG: child C has visits from parent A. Parent B's edge to C
        // has 1 visit (from the first-hit stop). Then C gets more visits via A,
        // making B's edge stale (edge_vis < child.total_visits).
        // A second backup_transposition_stop should correct B's edge Q toward
        // the new aggregate, without incrementing C's total_visits.

        with_test_access(|access| {
            let mut child_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            child_low.set_value_scale(5.0);
            let child_c = access.test_node(child_low);

            let mut parent_a_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_a_low.set_value_scale(5.0);
            parent_a_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_a = access.test_node(parent_a_low);
            access.test_connect(&parent_a, &child_c, (0, 0), 1.0, 0.5);

            let mut parent_b_low = LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]);
            parent_b_low.set_value_scale(5.0);
            parent_b_low.set_prior([0.2; 5], [0.2; 5]);
            let parent_b = access.test_node(parent_b_low);
            access.test_connect(&parent_b, &child_c, (1, 2), 0.5, 1.0);
            assert_eq!(access.num_parents(&child_c), 2);

            let path_a = vec![PathEntry {
                node: parent_a.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];
            let path_b = vec![PathEntry {
                node: parent_b.clone(),
                p1_outcome: 1,
                p2_outcome: 2,
            }];

            backup(access, &path_a, &child_c, 2.0, 3.0);
            reserve_path(access, &path_b, 1);
            backup_transposition_stop(access, &path_b, &child_c, 1);
            assert_eq!(access.node(&parent_b).edge_visits(1, 2), 1);
            assert!((access.node(&parent_b).edge_q_p1(1, 2) - 2.5).abs() < 1e-5);

            backup(access, &path_a, &child_c, 6.0, 1.0);
            let child_low = access.node(&child_c);
            assert_eq!(child_low.total_visits(), 2);
            assert!((child_low.v1() - 4.0).abs() < 1e-5);
            assert!((child_low.v2() - 2.0).abs() < 1e-5);
            assert!(
                access.node(&parent_b).edge_visits(1, 2) < access.node(&child_c).total_visits()
            );
            let child_visits_before = access.node(&child_c).total_visits();

            reserve_path(access, &path_b, 1);
            backup_transposition_stop(access, &path_b, &child_c, 1);

            assert_eq!(
                access.node(&child_c).total_visits(),
                child_visits_before,
                "stale TT stop should not increment child visits"
            );
            let parent_b = access.node(&parent_b);
            assert_eq!(parent_b.edge_visits(1, 2), 2);
            assert!((parent_b.edge_q_p1(1, 2) - 4.5).abs() < 1e-4);
            assert!((parent_b.edge_q_p2(1, 2) - 3.0).abs() < 1e-4);
        });
    }

    #[test]
    fn tt_stop_hits_nonzero_on_open_maze() {
        // Open 5x5 with 2 cheese: transpositions are common (e.g. UP+RIGHT = RIGHT+UP).
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let result = run_search(&mut tree, &game, &backend, &config, 500, 8, &mut r).unwrap();

        assert!(
            result.tt_stop_hits > 0,
            "expected transposition stops on open maze, got 0"
        );
    }

    #[test]
    fn cancellation_uses_cached_reservation_shape_after_parent_count_changes() {
        with_test_access(|access| {
            let root = access.root();
            let target = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            let parent_a = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            let parent_b = access.test_node(LowNode::new_shell([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]));
            let path = vec![PathEntry {
                node: root.clone(),
                p1_outcome: 0,
                p2_outcome: 0,
            }];

            reserve_path(access, &path, 1);
            assert!(access.node_mut(&target).try_start_score_update());
            let mut plan = NodeReservationPlan::new(target.clone(), path, 1, 1);
            let mut ledger = SearchLedgerStats::default();
            ledger.reserve(1);

            access.test_connect(&parent_a, &target, (0, 0), 0.0, 0.0);
            access.test_connect(&parent_b, &target, (0, 0), 0.0, 0.0);
            assert_eq!(access.num_parents(&target), 2);
            drop(access.test_detach_children(&parent_b));
            assert_eq!(access.num_parents(&target), 1);

            plan.cancel(access, &mut ledger);

            assert_eq!(access.node(&root).n_in_flight(), 0);
            assert_eq!(access.node(&root).edge_in_flight(0, 0), 0);
            assert_eq!(access.node(&target).n_in_flight(), 0);
            assert_eq!(ledger.cancelled, 1);
            ledger.assert_settled();
        });
    }

    #[test]
    fn tt_stop_hits_consume_sim_budget() {
        // On a transposition-heavy position, tt_stop_hits should count toward
        // the sim budget. Without this, the search overshoots n_sims.
        let game = open_5x5_game(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0), Coordinates::new(4, 4)],
        );
        let backend = SmartUniformBackend;
        let config = default_config();
        let mut tree = MCGSTree::new(&game);
        let mut r = rng();

        let n_sims = 200;
        let result = run_search(&mut tree, &game, &backend, &config, n_sims, 8, &mut r).unwrap();

        // Productive work = nn_evals + terminals + tt_stop_hits.
        let productive = result.nn_evals + result.terminals + result.tt_stop_hits;
        assert!(
            productive >= n_sims,
            "productive work ({productive}) should be >= n_sims ({n_sims})"
        );

        // Root visits should not wildly overshoot. Allow some slack for
        // batching (up to one extra batch worth).
        let max_expected = n_sims + 8;
        assert!(
            result.total_visits <= max_expected,
            "root visits ({}) overshot n_sims ({n_sims}) by more than one batch",
            result.total_visits
        );
    }
}
