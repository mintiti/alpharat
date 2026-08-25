//! Fixed-work MCGS calibration through the production search boundary.
//!
//! A workload file freezes the deterministic game, search configuration, and trial seed schedule.
//! Each planned [`SearchRequest`] then varies only the backend and concurrency coordinate. The
//! harness returns protocol-v1 [`SearchTrial`] rows; run-folder orchestration belongs to the later
//! calibration runner.

use std::any::Any;
use std::collections::BTreeSet;
use std::io::Write;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use alpharat_eval_core::{Backend, BackendError, EvalResult};
use alpharat_mcgs::{
    run_search_parallel_profiled, MCGSTree, ProfiledSearchResult, SearchConfig, SearchTermination,
    SearchTimings,
};
use pyrat::{Coordinates, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::calibration::{
    ResolvedBackend, SearchRequest, SearchResolved, SearchTrial, TrialPhase, TrialStatus,
    SEARCH_HEADERS,
};

pub const SEARCH_WORKLOAD_VERSION: u32 = 1;
pub const DEFAULT_SEARCH_WORKLOAD_JSON: &str = include_str!("../fixtures/mcgs-fixed-work-v1.json");

/// The stable fixture, configuration, and seed schedule shared by a search run's cases.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchWorkload {
    pub schema_version: u32,
    pub game: SearchGame,
    pub search: SearchConfiguration,
    pub seeds: SearchSeeds,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchGame {
    pub width: u8,
    pub height: u8,
    pub player_1: [u8; 2],
    pub player_2: [u8; 2],
    pub cheese: Vec<[u8; 2]>,
    pub max_turns: u16,
    pub creation_seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfiguration {
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub force_k: f32,
    pub noise_epsilon: f32,
    pub noise_concentration: f32,
    pub collision_limit_min: u32,
    pub collision_limit_max: u32,
    pub collision_scaling_start: u32,
    pub collision_scaling_end: u32,
    pub collision_scaling_power: f32,
}

impl From<&SearchConfiguration> for SearchConfig {
    fn from(value: &SearchConfiguration) -> Self {
        Self {
            c_puct: value.c_puct,
            fpu_reduction: value.fpu_reduction,
            force_k: value.force_k,
            noise_epsilon: value.noise_epsilon,
            noise_concentration: value.noise_concentration,
            collision_limit_min: value.collision_limit_min,
            collision_limit_max: value.collision_limit_max,
            collision_scaling_start: value.collision_scaling_start,
            collision_scaling_end: value.collision_scaling_end,
            collision_scaling_power: value.collision_scaling_power,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSeeds {
    pub warmup_base: u64,
    pub measured_base: u64,
}

/// Inner device-call facts when a wrapper, such as the inference mux, changes the outer calls.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceSnapshot {
    pub calls: u64,
    pub positions: u64,
    pub inference: Duration,
}

/// Optional instrumentation supplied by a backend wrapper for its inner device calls.
pub trait DeviceStats: Send + Sync {
    fn reset(&self);
    fn snapshot(&self) -> DeviceSnapshot;
}

/// A production backend plus the outer- and optional inner-call instrumentation used by search.
pub struct SearchBackendProbe {
    inner: Box<dyn Backend>,
    device_stats: Option<Box<dyn DeviceStats>>,
    active_callers: AtomicUsize,
    peak_callers: AtomicUsize,
    state: Mutex<ProbeState>,
}

impl SearchBackendProbe {
    pub fn direct(inner: Box<dyn Backend>) -> Self {
        Self::new(inner, None)
    }

    pub fn with_device_stats(inner: Box<dyn Backend>, device_stats: Box<dyn DeviceStats>) -> Self {
        Self::new(inner, Some(device_stats))
    }

    fn new(inner: Box<dyn Backend>, device_stats: Option<Box<dyn DeviceStats>>) -> Self {
        Self {
            inner,
            device_stats,
            active_callers: AtomicUsize::new(0),
            peak_callers: AtomicUsize::new(0),
            state: Mutex::new(ProbeState {
                epoch: Instant::now(),
                intervals: Vec::new(),
            }),
        }
    }

    fn reset(&self) {
        assert_eq!(
            self.active_callers.load(Ordering::Acquire),
            0,
            "cannot reset backend probe while calls are active"
        );
        self.peak_callers.store(0, Ordering::Release);
        if let Some(device_stats) = &self.device_stats {
            device_stats.reset();
        }
        let mut state = self.state.lock().expect("backend probe mutex poisoned");
        state.epoch = Instant::now();
        state.intervals.clear();
    }

    fn record<T>(
        &self,
        positions: usize,
        call: impl FnOnce(&dyn Backend) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        let started = Instant::now();
        let active = self.active_callers.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak_callers.fetch_max(active, Ordering::AcqRel);
        let active_guard = ActiveCaller(&self.active_callers);

        let result = catch_unwind(AssertUnwindSafe(|| call(self.inner.as_ref())));
        let ended = Instant::now();
        drop(active_guard);

        let mut state = self.state.lock().expect("backend probe mutex poisoned");
        let epoch = state.epoch;
        state.intervals.push(CallInterval {
            start: started.duration_since(epoch),
            end: ended.duration_since(epoch),
            positions,
        });
        drop(state);
        match result {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }

    fn snapshot(&self) -> BackendSnapshot {
        assert_eq!(
            self.active_callers.load(Ordering::Acquire),
            0,
            "cannot snapshot backend probe while calls are active"
        );
        let state = self.state.lock().expect("backend probe mutex poisoned");
        let mut intervals = state.intervals.clone();
        intervals.sort_by_key(|interval| interval.start);

        let caller_time = intervals
            .iter()
            .map(|interval| interval.end - interval.start)
            .sum();
        let union_time = union_duration(&intervals);
        let direct_device = DeviceSnapshot {
            calls: intervals.len() as u64,
            positions: intervals
                .iter()
                .map(|interval| interval.positions as u64)
                .sum(),
            inference: union_time,
        };

        BackendSnapshot {
            calls: intervals.len() as u64,
            positions: intervals
                .iter()
                .map(|interval| interval.positions as u64)
                .sum(),
            caller_time,
            union_time,
            peak_callers: self.peak_callers.load(Ordering::Acquire) as u32,
            device: self
                .device_stats
                .as_ref()
                .map_or(direct_device, |stats| stats.snapshot()),
        }
    }
}

impl Backend for SearchBackendProbe {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        self.record(1, |backend| backend.evaluate(game))
    }

    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        self.record(games.len(), |backend| backend.evaluate_batch(games))
    }
}

/// The resolved setup and observed trials produced for one fixed-work search case.
#[derive(Debug)]
pub struct SearchCaseResult {
    pub resolved: SearchResolved,
    pub trials: Vec<SearchTrial>,
}

#[derive(Debug, Error)]
pub enum SearchBenchmarkError {
    #[error("invalid fixed-work search case: {0}")]
    Invalid(String),
    #[error("failed to read search workload '{}': {source}", path.display())]
    ReadWorkload {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error("failed to decode search workload '{}': {source}", path.display())]
    DecodeWorkload {
        path: std::path::PathBuf,
        source: serde_json::Error,
    },
}

/// Decode the checked-in fixed-work workload used when no custom artifact is supplied.
pub fn default_search_workload() -> Result<SearchWorkload, SearchBenchmarkError> {
    let workload: SearchWorkload =
        serde_json::from_str(DEFAULT_SEARCH_WORKLOAD_JSON).map_err(|source| {
            SearchBenchmarkError::DecodeWorkload {
                path: Path::new("<built-in mcgs-fixed-work-v1.json>").to_path_buf(),
                source,
            }
        })?;
    validate_workload(&workload)?;
    Ok(workload)
}

/// Load a content-addressable workload artifact for use by planned search cases.
pub fn load_search_workload(path: &Path) -> Result<SearchWorkload, SearchBenchmarkError> {
    let bytes = std::fs::read(path).map_err(|source| SearchBenchmarkError::ReadWorkload {
        path: path.to_path_buf(),
        source,
    })?;
    let workload =
        serde_json::from_slice(&bytes).map_err(|source| SearchBenchmarkError::DecodeWorkload {
            path: path.to_path_buf(),
            source,
        })?;
    validate_workload(&workload)?;
    Ok(workload)
}

/// Run every warmup and measured trial requested for one fixed-work search case.
///
/// Every trial starts from a fresh tree and its phase-specific deterministic seed. Backend errors
/// and panics become failed rows after the parallel search has joined and cleaned up its workers.
pub fn run_search_case(
    case_id: &str,
    request: &SearchRequest,
    resolved_backend: ResolvedBackend,
    backend: &SearchBackendProbe,
    workload: &SearchWorkload,
) -> Result<SearchCaseResult, SearchBenchmarkError> {
    validate_case(case_id, request, &resolved_backend, workload)?;
    let game = build_game(&workload.game)?;
    let config = SearchConfig::from(&workload.search);
    let workers = usize::try_from(request.workers).map_err(|_| {
        SearchBenchmarkError::Invalid(format!(
            "worker count {} does not fit this platform",
            request.workers
        ))
    })?;
    let worker_batch = request.total_in_flight / request.workers;

    let planned_trials = u64::from(request.warmup_trials) + u64::from(request.measured_trials);
    let mut trials = Vec::with_capacity(usize::try_from(planned_trials).unwrap_or(0));
    for (phase, count, seed_base) in [
        (
            TrialPhase::Warmup,
            request.warmup_trials,
            workload.seeds.warmup_base,
        ),
        (
            TrialPhase::Measured,
            request.measured_trials,
            workload.seeds.measured_base,
        ),
    ] {
        for trial in 1..=count {
            let seed = seed_base + u64::from(trial - 1);
            trials.push(run_trial(
                case_id,
                phase,
                trial,
                seed,
                request,
                worker_batch,
                workers,
                backend,
                &game,
                &config,
            ));
        }
    }

    Ok(SearchCaseResult {
        resolved: SearchResolved {
            backend: resolved_backend,
            workers: request.workers,
            worker_batch,
            total_in_flight: request.total_in_flight,
            mux_max_batch: request.mux_max_batch,
            productive_work: request.productive_work,
        },
        trials,
    })
}

/// Write protocol-v1 search rows, including the exact header for an empty result set.
pub fn write_search_trials(writer: impl Write, trials: &[SearchTrial]) -> Result<(), csv::Error> {
    let mut output = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(writer);
    output.write_record(SEARCH_HEADERS)?;
    for trial in trials {
        output.serialize(trial)?;
    }
    output.flush().map_err(csv::Error::from)
}

fn validate_case(
    case_id: &str,
    request: &SearchRequest,
    resolved: &ResolvedBackend,
    workload: &SearchWorkload,
) -> Result<(), SearchBenchmarkError> {
    if case_id.trim().is_empty() {
        return Err(SearchBenchmarkError::Invalid(
            "case id cannot be empty".to_owned(),
        ));
    }
    if request.workers == 0
        || request.total_in_flight == 0
        || request.productive_work == 0
        || request.measured_trials == 0
    {
        return Err(SearchBenchmarkError::Invalid(
            "workers, total in-flight capacity, productive work, and measured trials must be positive"
                .to_owned(),
        ));
    }
    if !request.total_in_flight.is_multiple_of(request.workers) {
        return Err(SearchBenchmarkError::Invalid(format!(
            "total in-flight capacity {} is not divisible by {} workers",
            request.total_in_flight, request.workers
        )));
    }
    if request.mux_max_batch == Some(0) {
        return Err(SearchBenchmarkError::Invalid(
            "mux max batch must be positive".to_owned(),
        ));
    }
    if resolved.backend != request.backend {
        return Err(SearchBenchmarkError::Invalid(format!(
            "resolved backend {:?} does not match requested backend {:?}",
            resolved.backend, request.backend
        )));
    }
    let _ = usize::try_from(request.workers).map_err(|_| {
        SearchBenchmarkError::Invalid(format!(
            "worker count {} does not fit this platform",
            request.workers
        ))
    })?;
    for (label, base, count) in [
        ("warmup", workload.seeds.warmup_base, request.warmup_trials),
        (
            "measured",
            workload.seeds.measured_base,
            request.measured_trials,
        ),
    ] {
        if count > 0 && base.checked_add(u64::from(count - 1)).is_none() {
            return Err(SearchBenchmarkError::Invalid(format!(
                "{label} seed range exceeds u64"
            )));
        }
    }
    validate_workload(workload)
}

fn validate_workload(workload: &SearchWorkload) -> Result<(), SearchBenchmarkError> {
    if workload.schema_version != SEARCH_WORKLOAD_VERSION {
        return Err(SearchBenchmarkError::Invalid(format!(
            "unsupported search workload version {}; expected {}",
            workload.schema_version, SEARCH_WORKLOAD_VERSION
        )));
    }
    let game = &workload.game;
    if game.width < 2 || game.height < 2 || game.max_turns == 0 {
        return Err(SearchBenchmarkError::Invalid(
            "workload width and height must be at least 2 and max turns must be positive"
                .to_owned(),
        ));
    }
    let in_bounds = |position: [u8; 2]| position[0] < game.width && position[1] < game.height;
    if !in_bounds(game.player_1)
        || !in_bounds(game.player_2)
        || game.player_1 == game.player_2
        || game.cheese.iter().copied().any(|position| {
            !in_bounds(position) || position == game.player_1 || position == game.player_2
        })
    {
        return Err(SearchBenchmarkError::Invalid(
            "workload positions must be in bounds and players, cheese, and each other cannot overlap"
                .to_owned(),
        ));
    }
    if game.cheese.iter().copied().collect::<BTreeSet<_>>().len() != game.cheese.len() {
        return Err(SearchBenchmarkError::Invalid(
            "workload cheese positions must be unique".to_owned(),
        ));
    }

    let config = &workload.search;
    let finite = [
        config.c_puct,
        config.fpu_reduction,
        config.force_k,
        config.noise_epsilon,
        config.noise_concentration,
        config.collision_scaling_power,
    ]
    .into_iter()
    .all(f32::is_finite);
    if !finite
        || config.c_puct < 0.0
        || config.fpu_reduction < 0.0
        || config.force_k < 0.0
        || !(0.0..=1.0).contains(&config.noise_epsilon)
        || config.noise_concentration <= 0.0
        || config.collision_limit_min == 0
        || config.collision_limit_min > config.collision_limit_max
        || config.collision_scaling_start >= config.collision_scaling_end
        || config.collision_scaling_power <= 0.0
    {
        return Err(SearchBenchmarkError::Invalid(
            "workload search configuration is outside its valid range".to_owned(),
        ));
    }
    Ok(())
}

fn build_game(game: &SearchGame) -> Result<GameState, SearchBenchmarkError> {
    GameBuilder::new(game.width, game.height)
        .with_open_maze()
        .with_custom_positions(
            Coordinates::new(game.player_1[0], game.player_1[1]),
            Coordinates::new(game.player_2[0], game.player_2[1]),
        )
        .with_custom_cheese(
            game.cheese
                .iter()
                .map(|position| Coordinates::new(position[0], position[1]))
                .collect(),
        )
        .with_max_turns(game.max_turns)
        .build()
        .create(game.creation_seed)
        .map_err(|error| {
            SearchBenchmarkError::Invalid(format!("failed to build workload game: {error}"))
        })
}

#[allow(clippy::too_many_arguments)]
fn run_trial(
    case_id: &str,
    phase: TrialPhase,
    trial: u32,
    seed: u64,
    request: &SearchRequest,
    worker_batch: u32,
    workers: usize,
    backend: &SearchBackendProbe,
    game: &GameState,
    config: &SearchConfig,
) -> SearchTrial {
    backend.reset();
    let mut tree = MCGSTree::new(game);
    let mut rng = SmallRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        run_search_parallel_profiled(
            &mut tree,
            game,
            backend,
            config,
            request.productive_work,
            worker_batch,
            workers,
            &mut rng,
        )
    }));
    let wall = started.elapsed();
    let observed = backend.snapshot();

    match outcome {
        Ok(Ok(profile)) => {
            match completed_trial(case_id, phase, trial, request, wall, &profile, observed) {
                Ok(row) => row,
                Err(error) => failed_trial(case_id, phase, trial, wall, observed, error),
            }
        }
        Ok(Err(error)) => failed_trial(
            case_id,
            phase,
            trial,
            wall,
            observed,
            format!("search backend failed: {error}"),
        ),
        Err(payload) => failed_trial(
            case_id,
            phase,
            trial,
            wall,
            observed,
            format!("search panicked: {}", panic_message(payload)),
        ),
    }
}

fn completed_trial(
    case_id: &str,
    phase: TrialPhase,
    trial: u32,
    request: &SearchRequest,
    wall: Duration,
    profile: &ProfiledSearchResult,
    backend: BackendSnapshot,
) -> Result<SearchTrial, String> {
    let result = &profile.result;
    let productive = result.nn_evals + result.terminals + result.tt_stop_hits;
    if profile.termination != SearchTermination::BudgetExhausted {
        return Err(format!(
            "search stopped because {} after completing {productive} of {} productive units",
            profile.termination, request.productive_work
        ));
    }
    if productive != request.productive_work {
        return Err(format!(
            "search completed {productive} productive units; expected {}",
            request.productive_work
        ));
    }
    if profile.ledger.outstanding() != 0
        || profile.ledger.committed != u64::from(request.productive_work)
    {
        return Err(format!(
            "search ledger did not close exactly: reserved={}, committed={}, cancelled={}",
            profile.ledger.reserved, profile.ledger.committed, profile.ledger.cancelled
        ));
    }
    if backend.positions != u64::from(result.nn_evals)
        || backend.device.positions != backend.positions
    {
        return Err(format!(
            "search/backend work disagrees: nn_evals={}, outer_positions={}, device_positions={}",
            result.nn_evals, backend.positions, backend.device.positions
        ));
    }

    let ledger_reserved = u32::try_from(profile.ledger.reserved)
        .map_err(|_| "ledger reserved count exceeds protocol-v1 u32 range".to_owned())?;
    let ledger_committed = u32::try_from(profile.ledger.committed)
        .map_err(|_| "ledger committed count exceeds protocol-v1 u32 range".to_owned())?;
    let ledger_cancelled = u32::try_from(profile.ledger.cancelled)
        .map_err(|_| "ledger cancelled count exceeds protocol-v1 u32 range".to_owned())?;
    let occupancy = phase_occupancy(profile.timings);
    let caller_per_union = duration_ratio(backend.caller_time, backend.union_time);
    let device_avg_batch = if backend.device.calls == 0 {
        0.0
    } else {
        backend.device.positions as f64 / backend.device.calls as f64
    };

    Ok(SearchTrial {
        case_id: case_id.to_owned(),
        phase,
        trial,
        status: TrialStatus::Completed,
        error: None,
        wall_ms: Some(milliseconds(wall)),
        productive_per_s: Some(rate(productive, wall)),
        nn_evals: Some(result.nn_evals),
        terminals: Some(result.terminals),
        tt_stops: Some(result.tt_stop_hits),
        collisions: Some(result.collisions),
        batches: Some(profile.timings.batches),
        phase_occupancy_ms: Some(milliseconds(occupancy)),
        phase_occupancy_per_wall: Some(duration_ratio(occupancy, wall)),
        lease_wait_ms: Some(milliseconds(profile.timings.lease_wait)),
        gate_wait_ms: Some(milliseconds(gate_wait(profile.timings))),
        gate_hold_ms: Some(milliseconds(gate_hold(profile.timings))),
        inference_caller_ms: Some(milliseconds(profile.timings.inference)),
        completion_wait_ms: Some(milliseconds(profile.timings.completion_wait)),
        backend_calls: Some(backend.calls),
        backend_positions: Some(backend.positions),
        backend_caller_ms: Some(milliseconds(backend.caller_time)),
        backend_union_ms: Some(milliseconds(backend.union_time)),
        backend_caller_per_union: Some(caller_per_union),
        backend_peak_callers: Some(backend.peak_callers),
        device_calls: Some(backend.device.calls),
        device_positions: Some(backend.device.positions),
        device_avg_batch: Some(device_avg_batch),
        device_inference_ms: Some(milliseconds(backend.device.inference)),
        ledger_reserved: Some(ledger_reserved),
        ledger_committed: Some(ledger_committed),
        ledger_cancelled: Some(ledger_cancelled),
        policy_p1_0: Some(result.policy_p1[0]),
        policy_p1_1: Some(result.policy_p1[1]),
        policy_p1_2: Some(result.policy_p1[2]),
        policy_p1_3: Some(result.policy_p1[3]),
        policy_p1_4: Some(result.policy_p1[4]),
        policy_p2_0: Some(result.policy_p2[0]),
        policy_p2_1: Some(result.policy_p2[1]),
        policy_p2_2: Some(result.policy_p2[2]),
        policy_p2_3: Some(result.policy_p2[3]),
        policy_p2_4: Some(result.policy_p2[4]),
    })
}

fn failed_trial(
    case_id: &str,
    phase: TrialPhase,
    trial: u32,
    wall: Duration,
    backend: BackendSnapshot,
    error: String,
) -> SearchTrial {
    let caller_per_union = duration_ratio(backend.caller_time, backend.union_time);
    let device_avg_batch = if backend.device.calls == 0 {
        0.0
    } else {
        backend.device.positions as f64 / backend.device.calls as f64
    };
    SearchTrial {
        case_id: case_id.to_owned(),
        phase,
        trial,
        status: TrialStatus::Failed,
        error: Some(error),
        wall_ms: Some(milliseconds(wall)),
        productive_per_s: None,
        nn_evals: None,
        terminals: None,
        tt_stops: None,
        collisions: None,
        batches: None,
        phase_occupancy_ms: None,
        phase_occupancy_per_wall: None,
        lease_wait_ms: None,
        gate_wait_ms: None,
        gate_hold_ms: None,
        inference_caller_ms: None,
        completion_wait_ms: None,
        backend_calls: Some(backend.calls),
        backend_positions: Some(backend.positions),
        backend_caller_ms: Some(milliseconds(backend.caller_time)),
        backend_union_ms: Some(milliseconds(backend.union_time)),
        backend_caller_per_union: Some(caller_per_union),
        backend_peak_callers: Some(backend.peak_callers),
        device_calls: Some(backend.device.calls),
        device_positions: Some(backend.device.positions),
        device_avg_batch: Some(device_avg_batch),
        device_inference_ms: Some(milliseconds(backend.device.inference)),
        ledger_reserved: None,
        ledger_committed: None,
        ledger_cancelled: None,
        policy_p1_0: None,
        policy_p1_1: None,
        policy_p1_2: None,
        policy_p1_3: None,
        policy_p1_4: None,
        policy_p2_0: None,
        policy_p2_1: None,
        policy_p2_2: None,
        policy_p2_3: None,
        policy_p2_4: None,
    }
}

#[derive(Clone, Copy)]
struct CallInterval {
    start: Duration,
    end: Duration,
    positions: usize,
}

struct ProbeState {
    epoch: Instant,
    intervals: Vec<CallInterval>,
}

struct ActiveCaller<'probe>(&'probe AtomicUsize);

impl Drop for ActiveCaller<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Copy)]
struct BackendSnapshot {
    calls: u64,
    positions: u64,
    caller_time: Duration,
    union_time: Duration,
    peak_callers: u32,
    device: DeviceSnapshot,
}

fn union_duration(intervals: &[CallInterval]) -> Duration {
    let Some(first) = intervals.first() else {
        return Duration::ZERO;
    };
    let mut total = Duration::ZERO;
    let mut range_start = first.start;
    let mut range_end = first.end;
    for interval in &intervals[1..] {
        if interval.start <= range_end {
            range_end = range_end.max(interval.end);
        } else {
            total += range_end - range_start;
            range_start = interval.start;
            range_end = interval.end;
        }
    }
    total + (range_end - range_start)
}

fn phase_occupancy(timings: SearchTimings) -> Duration {
    timings.lease_wait
        + timings.gather_wait
        + timings.gather_hold
        + timings.inference
        + timings.settle_wait
        + timings.settle_hold
        + timings.completion_wait
        + timings.cleanup_wait
        + timings.cleanup_hold
        + timings.extract_wait
        + timings.extract_hold
}

fn gate_wait(timings: SearchTimings) -> Duration {
    timings.gather_wait + timings.settle_wait + timings.cleanup_wait + timings.extract_wait
}

fn gate_hold(timings: SearchTimings) -> Duration {
    timings.gather_hold + timings.settle_hold + timings.cleanup_hold + timings.extract_hold
}

fn duration_ratio(numerator: Duration, denominator: Duration) -> f64 {
    if denominator.is_zero() {
        0.0
    } else {
        numerator.as_secs_f64() / denominator.as_secs_f64()
    }
}

fn rate(units: u32, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        f64::from(units) / duration.as_secs_f64()
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    use alpharat_eval_core::{BackendError, SmartUniformBackend};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::calibration::{
        BackendRequest, BackendSerialization, OnnxProvider, TrialStatus, CAPACITY_WORKLOAD_FILE,
        SEARCH_WORKLOAD_FILE,
    };

    fn request(workers: u32, total_in_flight: u32) -> SearchRequest {
        SearchRequest {
            backend: BackendRequest::SmartUniform,
            workers,
            total_in_flight,
            mux_max_batch: None,
            productive_work: 32,
            warmup_trials: 1,
            measured_trials: 2,
        }
    }

    fn resolved(backend: BackendRequest) -> ResolvedBackend {
        ResolvedBackend {
            backend,
            serialization: BackendSerialization::None,
        }
    }

    #[test]
    fn records_planned_phases_exact_work_and_resolved_worker_batch() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(SmartUniformBackend));
        let result = run_search_case(
            "smart-w2-fixed16",
            &request(2, 16),
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        assert_eq!(result.resolved.worker_batch, 8);
        assert_eq!(result.resolved.total_in_flight, 16);
        assert_eq!(result.trials.len(), 3);
        assert_eq!(
            result
                .trials
                .iter()
                .map(|trial| (trial.phase, trial.trial))
                .collect::<Vec<_>>(),
            vec![
                (TrialPhase::Warmup, 1),
                (TrialPhase::Measured, 1),
                (TrialPhase::Measured, 2),
            ]
        );
        for trial in &result.trials {
            assert_eq!(trial.status, TrialStatus::Completed);
            assert_eq!(
                trial.nn_evals.unwrap() + trial.terminals.unwrap() + trial.tt_stops.unwrap(),
                32
            );
            assert_eq!(trial.ledger_committed, Some(32));
            assert_eq!(
                trial.ledger_reserved,
                Some(trial.ledger_committed.unwrap() + trial.ledger_cancelled.unwrap())
            );
        }
    }

    #[test]
    fn fixed_total_and_scaled_capacity_resolve_to_distinct_coordinates() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(SmartUniformBackend));
        let mut fixed = request(4, 256);
        fixed.warmup_trials = 0;
        fixed.measured_trials = 1;
        fixed.productive_work = 8;
        let mut scaled = fixed.clone();
        scaled.total_in_flight = 4 * 1_024;

        let fixed = run_search_case(
            "fixed",
            &fixed,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();
        let scaled = run_search_case(
            "scaled",
            &scaled,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        assert_eq!(fixed.resolved.worker_batch, 64);
        assert_eq!(fixed.resolved.total_in_flight, 256);
        assert_eq!(scaled.resolved.worker_batch, 1_024);
        assert_eq!(scaled.resolved.total_in_flight, 4_096);
    }

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            Err(BackendError::msg("controlled search failure"))
        }
    }

    #[test]
    fn backend_failure_becomes_a_failed_trial_row() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(FailingBackend));
        let mut request = request(1, 8);
        request.warmup_trials = 0;
        request.measured_trials = 1;
        let result = run_search_case(
            "failure",
            &request,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Failed);
        assert!(trial
            .error
            .as_deref()
            .unwrap()
            .contains("controlled search failure"));
        assert_eq!(trial.productive_per_s, None);
    }

    struct PanickingBackend;

    impl Backend for PanickingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            panic!("controlled search panic")
        }
    }

    #[test]
    fn backend_panic_is_preserved_after_worker_cleanup() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(PanickingBackend));
        let mut request = request(1, 8);
        request.warmup_trials = 0;
        request.measured_trials = 1;
        let result = run_search_case(
            "panic",
            &request,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Failed);
        assert!(trial
            .error
            .as_deref()
            .unwrap()
            .contains("controlled search panic"));
        assert_eq!(trial.backend_calls, Some(1));
        assert!(trial
            .backend_positions
            .is_some_and(|positions| positions > 0));
        assert_eq!(trial.device_calls, trial.backend_calls);
        assert_eq!(trial.device_positions, trial.backend_positions);
    }

    #[derive(Default)]
    struct DeviceCounters {
        calls: AtomicU64,
        positions: AtomicU64,
    }

    struct CountingDeviceBackend {
        counters: Arc<DeviceCounters>,
    }

    impl Backend for CountingDeviceBackend {
        fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
            self.counters.calls.fetch_add(1, Ordering::AcqRel);
            self.counters.positions.fetch_add(1, Ordering::AcqRel);
            SmartUniformBackend.evaluate(game)
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            self.counters.calls.fetch_add(1, Ordering::AcqRel);
            self.counters
                .positions
                .fetch_add(games.len() as u64, Ordering::AcqRel);
            SmartUniformBackend.evaluate_batch(games)
        }
    }

    struct CoalescedDeviceStats {
        counters: Arc<DeviceCounters>,
    }

    impl DeviceStats for CoalescedDeviceStats {
        fn reset(&self) {
            self.counters.calls.store(0, Ordering::Release);
            self.counters.positions.store(0, Ordering::Release);
        }

        fn snapshot(&self) -> DeviceSnapshot {
            let outer_calls = self.counters.calls.load(Ordering::Acquire);
            DeviceSnapshot {
                // Model a wrapper that combines pairs of outer requests into device calls.
                calls: outer_calls.div_ceil(2),
                positions: self.counters.positions.load(Ordering::Acquire),
                inference: Duration::from_millis(1),
            }
        }
    }

    #[test]
    fn inner_device_observations_remain_distinct_from_outer_calls() {
        let workload = default_search_workload().unwrap();
        let counters = Arc::new(DeviceCounters::default());
        let inner = CountingDeviceBackend {
            counters: counters.clone(),
        };
        let device = CoalescedDeviceStats {
            counters: counters.clone(),
        };
        let backend = SearchBackendProbe::with_device_stats(Box::new(inner), Box::new(device));
        let mut request = request(2, 16);
        request.warmup_trials = 0;
        request.measured_trials = 1;
        request.productive_work = 100;
        let result = run_search_case(
            "device",
            &request,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Completed);
        assert_ne!(trial.device_calls, trial.backend_calls);
        assert_eq!(trial.device_positions, trial.backend_positions);
    }

    #[test]
    fn csv_rows_follow_the_frozen_protocol_shape() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(SmartUniformBackend));
        let mut request = request(1, 8);
        request.warmup_trials = 0;
        request.measured_trials = 1;
        let result = run_search_case(
            "shape",
            &request,
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap();

        let mut csv = Vec::new();
        write_search_trials(&mut csv, &result.trials).unwrap();
        let text = String::from_utf8(csv).unwrap();
        assert_eq!(text.lines().next().unwrap(), SEARCH_HEADERS.join(","));
        let decoded = csv::Reader::from_reader(text.as_bytes())
            .deserialize::<SearchTrial>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, result.trials);
    }

    #[test]
    fn produced_rows_are_accepted_by_the_v1_folder_validator() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/calibration-v1/valid");
        let folder = tempfile::tempdir().unwrap();
        fs::copy(fixture.join("run.json"), folder.path().join("run.json")).unwrap();
        for file in [CAPACITY_WORKLOAD_FILE, SEARCH_WORKLOAD_FILE] {
            fs::copy(fixture.join(file), folder.path().join(file)).unwrap();
        }
        fs::copy(
            fixture.join("capacity-trials.csv"),
            folder.path().join("capacity-trials.csv"),
        )
        .unwrap();

        let request = SearchRequest {
            backend: BackendRequest::Onnx {
                provider: OnnxProvider::Cpu,
            },
            workers: 2,
            total_in_flight: 128,
            mux_max_batch: None,
            productive_work: 100,
            warmup_trials: 1,
            measured_trials: 1,
        };
        let resolved = ResolvedBackend {
            backend: request.backend.clone(),
            serialization: BackendSerialization::SessionMutex,
        };
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(SmartUniformBackend));
        let result = run_search_case(
            "search-cpu-w2-cap128",
            &request,
            resolved,
            &backend,
            &workload,
        )
        .unwrap();
        let search_path = folder.path().join("search-trials.csv");
        let mut output = fs::File::create(&search_path).unwrap();
        write_search_trials(&mut output, &result.trials).unwrap();
        drop(output);

        let search_bytes = fs::read(&search_path).unwrap();
        let mut record: crate::calibration::RunRecord =
            serde_json::from_slice(&fs::read(folder.path().join("run.json")).unwrap()).unwrap();
        let trial_file = record.trial_files.search.as_mut().unwrap();
        trial_file.sha256 = format!("{:x}", Sha256::digest(&search_bytes));
        trial_file.rows = result.trials.len() as u32;
        fs::write(
            folder.path().join("run.json"),
            serde_json::to_vec_pretty(&record).unwrap(),
        )
        .unwrap();

        let loaded = crate::calibration::load_run_folder(folder.path()).unwrap();
        assert_eq!(loaded.search_trials, result.trials);
    }

    #[test]
    fn rejects_fractional_worker_capacity_before_search() {
        let workload = default_search_workload().unwrap();
        let backend = SearchBackendProbe::direct(Box::new(SmartUniformBackend));
        let error = run_search_case(
            "fractional",
            &request(3, 8),
            resolved(BackendRequest::SmartUniform),
            &backend,
            &workload,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not divisible"));
    }

    #[test]
    fn checked_in_workload_matches_the_embedded_rerun_artifact() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/mcgs-fixed-work-v1.json");
        let from_file = load_search_workload(&path).unwrap();
        let embedded = default_search_workload().unwrap();

        assert_eq!(from_file, embedded);
        assert_eq!(from_file.seeds.warmup_base, 1_000);
        assert_eq!(from_file.seeds.measured_base, 42);
        assert_eq!(from_file.game.cheese.len(), 10);
        assert_eq!(SearchConfig::from(&from_file.search).c_puct, 1.5);
    }
}
