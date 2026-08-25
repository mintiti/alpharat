//! Production-boundary backend-capacity measurement.
//!
//! The harness calls the same [`Backend`] interface search uses, including any observation
//! encoding and provider serialization inside that implementation. It measures one planned case
//! at a time and returns protocol-v1 trial rows; run-folder orchestration belongs to the later
//! calibration runner.

use std::any::Any;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

use alpharat_eval_core::{Backend, EvalResult};
use pyrat::{Coordinates, GameBuilder, GameState};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::calibration::{
    BackendSerialization, CapacityRequest, CapacityResolved, CapacityTrial, ResolvedBackend,
    TrialPhase, TrialStatus, CAPACITY_HEADERS,
};

pub const CAPACITY_WORKLOAD_VERSION: u32 = 1;

/// Exact deterministic positions used by a production-boundary capacity case.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityWorkload {
    pub schema_version: u32,
    pub width: u8,
    pub height: u8,
    pub player_1: [u8; 2],
    pub player_2: [u8; 2],
    pub cheese: Vec<[u8; 2]>,
    pub max_turns: u16,
    pub creation_seed: Option<u64>,
}

/// Resolve the standard open-grid capacity fixture into explicit, hashable positions.
pub fn standard_capacity_workload(
    width: u8,
    height: u8,
) -> Result<CapacityWorkload, CapacityBenchmarkError> {
    if width < 2 || height < 2 {
        return Err(CapacityBenchmarkError::Invalid(
            "capacity workload width and height must both be at least 2".to_owned(),
        ));
    }
    let mut cheese = Vec::new();
    let target = ((u16::from(width) * u16::from(height)) / 5).max(1) as usize;
    'outer: for y in 0..height {
        for x in 0..width {
            if (x + y) % 2 == 1 && (x, y) != (0, 0) && (x, y) != (width - 1, height - 1) {
                cheese.push([x, y]);
                if cheese.len() == target {
                    break 'outer;
                }
            }
        }
    }
    Ok(CapacityWorkload {
        schema_version: CAPACITY_WORKLOAD_VERSION,
        width,
        height,
        player_1: [0, 0],
        player_2: [width - 1, height - 1],
        cheese,
        max_turns: u16::from(width.max(height)) * 10,
        creation_seed: Some(0),
    })
}

/// Decode an explicit capacity workload artifact.
pub fn load_capacity_workload(
    path: &std::path::Path,
) -> Result<CapacityWorkload, CapacityBenchmarkError> {
    let bytes = std::fs::read(path).map_err(|error| {
        CapacityBenchmarkError::Invalid(format!(
            "failed to read capacity workload '{}': {error}",
            path.display()
        ))
    })?;
    let workload: CapacityWorkload = serde_json::from_slice(&bytes).map_err(|error| {
        CapacityBenchmarkError::Invalid(format!(
            "failed to decode capacity workload '{}': {error}",
            path.display()
        ))
    })?;
    validate_workload(&workload)?;
    Ok(workload)
}

/// Build the exact game positions consumed by the capacity producer.
pub fn build_capacity_workload(
    workload: &CapacityWorkload,
) -> Result<Vec<GameState>, CapacityBenchmarkError> {
    validate_workload(workload)?;
    let cheese = workload
        .cheese
        .iter()
        .map(|position| Coordinates::new(position[0], position[1]))
        .collect();
    let game = GameBuilder::new(workload.width, workload.height)
        .with_open_maze()
        .with_custom_positions(
            Coordinates::new(workload.player_1[0], workload.player_1[1]),
            Coordinates::new(workload.player_2[0], workload.player_2[1]),
        )
        .with_custom_cheese(cheese)
        .with_max_turns(workload.max_turns)
        .build()
        .create(workload.creation_seed)
        .map_err(|error| {
            CapacityBenchmarkError::Invalid(format!(
                "failed to create deterministic capacity workload: {error}"
            ))
        })?;
    Ok(vec![game])
}

/// The resolved setup and observed trials produced for one capacity case.
#[derive(Debug)]
pub struct CapacityCaseResult {
    pub resolved: CapacityResolved,
    pub trials: Vec<CapacityTrial>,
}

#[derive(Debug, Error)]
pub enum CapacityBenchmarkError {
    #[error("invalid backend-capacity case: {0}")]
    Invalid(String),
}

fn validate_workload(workload: &CapacityWorkload) -> Result<(), CapacityBenchmarkError> {
    if workload.schema_version != CAPACITY_WORKLOAD_VERSION {
        return Err(CapacityBenchmarkError::Invalid(format!(
            "unsupported capacity workload version {}; expected {CAPACITY_WORKLOAD_VERSION}",
            workload.schema_version
        )));
    }
    if workload.width < 2 || workload.height < 2 || workload.max_turns == 0 {
        return Err(CapacityBenchmarkError::Invalid(
            "capacity workload dimensions must be at least 2 and max turns must be positive"
                .to_owned(),
        ));
    }
    let inside = |position: [u8; 2]| position[0] < workload.width && position[1] < workload.height;
    if !inside(workload.player_1)
        || !inside(workload.player_2)
        || workload.player_1 == workload.player_2
        || workload.cheese.is_empty()
        || workload
            .cheese
            .iter()
            .copied()
            .any(|position| !inside(position))
    {
        return Err(CapacityBenchmarkError::Invalid(
            "capacity workload positions must be distinct, in bounds, and include cheese"
                .to_owned(),
        ));
    }
    let unique = workload
        .cheese
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if unique.len() != workload.cheese.len()
        || unique.contains(&workload.player_1)
        || unique.contains(&workload.player_2)
    {
        return Err(CapacityBenchmarkError::Invalid(
            "capacity workload cheese must be unique and separate from the players".to_owned(),
        ));
    }
    Ok(())
}

/// Run every warmup and measured trial requested for one backend-capacity case.
///
/// Backend failures and panics become failed trial rows so the record does not silently lose an
/// attempted measurement. An error from this function instead means the case could not be set up.
pub fn run_capacity_case(
    case_id: &str,
    request: &CapacityRequest,
    resolved_backend: ResolvedBackend,
    backend: &dyn Backend,
    workload: &[GameState],
) -> Result<CapacityCaseResult, CapacityBenchmarkError> {
    validate_case(case_id, request, &resolved_backend, workload)?;

    let batch_size = usize::try_from(request.batch_size).map_err(|_| {
        CapacityBenchmarkError::Invalid(format!(
            "batch size {} does not fit this platform",
            request.batch_size
        ))
    })?;
    let batch = workload.iter().cycle().take(batch_size).collect::<Vec<_>>();

    let planned_trials = u64::from(request.warmup_trials) + u64::from(request.measured_trials);
    let mut trials = Vec::with_capacity(usize::try_from(planned_trials).unwrap_or(0));
    for (phase, count) in [
        (TrialPhase::Warmup, request.warmup_trials),
        (TrialPhase::Measured, request.measured_trials),
    ] {
        for trial in 1..=count {
            trials.push(run_trial(case_id, phase, trial, request, backend, &batch));
        }
    }

    Ok(CapacityCaseResult {
        resolved: CapacityResolved {
            backend: resolved_backend,
            batch_size: request.batch_size,
            callers: request.callers,
            calls_per_caller: request.calls_per_caller,
        },
        trials,
    })
}

/// Write protocol-v1 capacity rows, including the exact header for an empty result set.
pub fn write_capacity_trials(
    writer: impl Write,
    trials: &[CapacityTrial],
) -> Result<(), csv::Error> {
    let mut output = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(writer);
    output.write_record(CAPACITY_HEADERS)?;
    for trial in trials {
        output.serialize(trial)?;
    }
    output.flush().map_err(csv::Error::from)
}

fn validate_case(
    case_id: &str,
    request: &CapacityRequest,
    resolved: &ResolvedBackend,
    workload: &[GameState],
) -> Result<(), CapacityBenchmarkError> {
    if case_id.trim().is_empty() {
        return Err(CapacityBenchmarkError::Invalid(
            "case id cannot be empty".to_owned(),
        ));
    }
    if request.batch_size == 0
        || request.callers == 0
        || request.calls_per_caller == 0
        || request.measured_trials == 0
    {
        return Err(CapacityBenchmarkError::Invalid(
            "batch size, callers, calls per caller, and measured trials must be positive"
                .to_owned(),
        ));
    }
    if workload.is_empty() {
        return Err(CapacityBenchmarkError::Invalid(
            "the workload must contain at least one game state".to_owned(),
        ));
    }
    if resolved.backend != request.backend {
        return Err(CapacityBenchmarkError::Invalid(format!(
            "resolved backend {:?} does not match requested backend {:?}",
            resolved.backend, request.backend
        )));
    }
    usize::try_from(request.callers)
        .ok()
        .and_then(|callers| callers.checked_add(1))
        .ok_or_else(|| {
            CapacityBenchmarkError::Invalid(format!(
                "caller count {} does not fit this platform",
                request.callers
            ))
        })?;
    let _ = u64::from(request.callers)
        .checked_mul(u64::from(request.calls_per_caller))
        .and_then(|calls| calls.checked_mul(u64::from(request.batch_size)))
        .ok_or_else(|| {
            CapacityBenchmarkError::Invalid("requested work exceeds protocol counters".to_owned())
        })?;
    Ok(())
}

fn run_trial(
    case_id: &str,
    phase: TrialPhase,
    trial: u32,
    request: &CapacityRequest,
    backend: &dyn Backend,
    batch: &[&GameState],
) -> CapacityTrial {
    let callers = usize::try_from(request.callers).expect("validated caller count");
    let ready = Barrier::new(callers + 1);
    let start = Barrier::new(callers + 1);
    let probe = CallProbe::default();

    let (wall, failures) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(callers);
        for caller in 0..callers {
            let ready = &ready;
            let start = &start;
            let probe = &probe;
            handles.push(scope.spawn(move || {
                ready.wait();
                start.wait();
                for call in 1..=request.calls_per_caller {
                    probe.call(backend, batch).map_err(|message| {
                        format!("caller {}, call {call}: {message}", caller + 1)
                    })?;
                }
                Ok::<_, String>(())
            }));
        }

        // Do not include thread creation in the capacity coordinate. Every caller reaches the
        // ready barrier before the measured start barrier is released.
        ready.wait();
        let started = Instant::now();
        start.wait();

        let failures = handles
            .into_iter()
            .filter_map(|handle| match handle.join() {
                Ok(Ok(())) => None,
                Ok(Err(message)) => Some(message),
                Err(payload) => Some(format!(
                    "caller thread panicked outside the backend call: {}",
                    panic_message(payload)
                )),
            })
            .collect::<Vec<_>>();
        (started.elapsed(), failures)
    });

    let snapshot = probe.snapshot();
    let completed = failures.is_empty();
    let error = if completed {
        None
    } else if failures.len() == 1 {
        failures.into_iter().next()
    } else {
        Some(format!(
            "{}; {} additional caller failure(s)",
            failures[0],
            failures.len() - 1
        ))
    };
    let device_avg_batch = if snapshot.calls == 0 {
        0.0
    } else {
        snapshot.positions as f64 / snapshot.calls as f64
    };

    CapacityTrial {
        case_id: case_id.to_owned(),
        phase,
        trial,
        status: if completed {
            TrialStatus::Completed
        } else {
            TrialStatus::Failed
        },
        error,
        wall_ms: Some(milliseconds(wall)),
        positions_per_s: completed.then(|| {
            let seconds = wall.as_secs_f64();
            if seconds == 0.0 {
                0.0
            } else {
                snapshot.positions as f64 / seconds
            }
        }),
        outer_calls: Some(snapshot.calls),
        outer_positions: Some(snapshot.positions),
        caller_time_ms: Some(milliseconds(snapshot.caller_time)),
        caller_union_ms: Some(milliseconds(snapshot.caller_union)),
        caller_peak: Some(snapshot.peak_callers),
        // Without a mux, one production Backend call is one device-facing call. The duration is
        // the union of those calls, including encoding and provider serialization by design.
        device_calls: Some(snapshot.calls),
        device_positions: Some(snapshot.positions),
        device_avg_batch: Some(device_avg_batch),
        device_inference_ms: Some(milliseconds(snapshot.caller_union)),
    }
}

#[derive(Clone, Copy)]
struct CallInterval {
    started: Instant,
    ended: Instant,
    positions: u64,
}

#[derive(Default)]
struct CallProbe {
    active: AtomicU32,
    peak: AtomicU32,
    intervals: Mutex<Vec<CallInterval>>,
}

impl CallProbe {
    fn call(&self, backend: &dyn Backend, batch: &[&GameState]) -> Result<(), String> {
        let started = Instant::now();
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(active, Ordering::AcqRel);
        let active_guard = ActiveCall(&self.active);

        let result = catch_unwind(AssertUnwindSafe(|| backend.evaluate_batch(batch)));
        let ended = Instant::now();
        drop(active_guard);
        self.intervals
            .lock()
            .expect("capacity call-probe mutex poisoned")
            .push(CallInterval {
                started,
                ended,
                positions: batch.len() as u64,
            });

        let evaluations = match result {
            Ok(Ok(evaluations)) => evaluations,
            Ok(Err(error)) => return Err(format!("backend evaluation failed: {error}")),
            Err(payload) => {
                return Err(format!(
                    "backend evaluation panicked: {}",
                    panic_message(payload)
                ))
            }
        };
        if evaluations.len() != batch.len() {
            return Err(format!(
                "backend returned {} evaluations for {} positions",
                evaluations.len(),
                batch.len()
            ));
        }
        if let Some(index) = evaluations
            .iter()
            .position(|evaluation| !is_finite(evaluation))
        {
            return Err(format!(
                "backend returned a non-finite evaluation at position {index}"
            ));
        }
        Ok(())
    }

    fn snapshot(&self) -> CallSnapshot {
        assert_eq!(
            self.active.load(Ordering::Acquire),
            0,
            "cannot snapshot while backend calls are active"
        );
        let mut intervals = self
            .intervals
            .lock()
            .expect("capacity call-probe mutex poisoned")
            .clone();
        intervals.sort_by_key(|interval| interval.started);

        let caller_time = intervals.iter().fold(Duration::ZERO, |total, interval| {
            total + (interval.ended - interval.started)
        });
        let caller_union = union_duration(&intervals);

        CallSnapshot {
            calls: intervals.len() as u64,
            positions: intervals.iter().map(|interval| interval.positions).sum(),
            caller_time,
            caller_union,
            peak_callers: self.peak.load(Ordering::Acquire),
        }
    }
}

struct ActiveCall<'a>(&'a AtomicU32);

impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct CallSnapshot {
    calls: u64,
    positions: u64,
    caller_time: Duration,
    caller_union: Duration,
    peak_callers: u32,
}

fn union_duration(intervals: &[CallInterval]) -> Duration {
    let Some(first) = intervals.first() else {
        return Duration::ZERO;
    };
    let mut total = Duration::ZERO;
    let mut range_start = first.started;
    let mut range_end = first.ended;
    for interval in &intervals[1..] {
        if interval.started <= range_end {
            range_end = range_end.max(interval.ended);
        } else {
            total += range_end - range_start;
            range_start = interval.started;
            range_end = interval.ended;
        }
    }
    total + (range_end - range_start)
}

fn is_finite(result: &EvalResult) -> bool {
    result.policy_p1.iter().all(|value| value.is_finite())
        && result.policy_p2.iter().all(|value| value.is_finite())
        && result.value_p1.is_finite()
        && result.value_p2.is_finite()
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

/// Standard resolved serialization for AlphaRat's direct production backends.
pub fn direct_serialization(backend: &crate::calibration::BackendRequest) -> BackendSerialization {
    match backend {
        crate::calibration::BackendRequest::SmartUniform => BackendSerialization::None,
        crate::calibration::BackendRequest::Onnx { .. } => BackendSerialization::SessionMutex,
        crate::calibration::BackendRequest::TensorRt => BackendSerialization::ContextMutex,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use alpharat_eval_core::{BackendError, SmartUniformBackend};
    use pyrat::{Coordinates, GameBuilder};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::calibration::{
        BackendRequest, TrialStatus, CAPACITY_WORKLOAD_FILE, SEARCH_WORKLOAD_FILE,
    };

    fn game() -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(2, 2)])
            .with_max_turns(30)
            .build()
            .create(Some(0))
            .unwrap()
    }

    fn request(callers: u32, calls_per_caller: u32) -> CapacityRequest {
        CapacityRequest {
            backend: BackendRequest::SmartUniform,
            batch_size: 4,
            callers,
            calls_per_caller,
            warmup_trials: 0,
            measured_trials: 1,
        }
    }

    fn resolved() -> ResolvedBackend {
        ResolvedBackend {
            backend: BackendRequest::SmartUniform,
            serialization: BackendSerialization::None,
        }
    }

    fn evaluation() -> EvalResult {
        EvalResult {
            policy_p1: [0.2; 5],
            policy_p2: [0.2; 5],
            value_p1: 0.0,
            value_p2: 0.0,
        }
    }

    struct RendezvousBackend {
        first_wave: Barrier,
        calls: AtomicUsize,
    }

    impl Backend for RendezvousBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            Ok(evaluation())
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            let call = self.calls.fetch_add(1, Ordering::AcqRel);
            if call < 3 {
                self.first_wave.wait();
            }
            Ok(vec![evaluation(); games.len()])
        }
    }

    #[test]
    fn records_exact_work_and_concurrent_callers() {
        let backend = RendezvousBackend {
            first_wave: Barrier::new(3),
            calls: AtomicUsize::new(0),
        };
        let result = run_capacity_case(
            "fake-b4-c3",
            &request(3, 2),
            resolved(),
            &backend,
            &[game()],
        )
        .unwrap();

        assert_eq!(result.resolved.batch_size, 4);
        assert_eq!(result.trials.len(), 1);
        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Completed);
        assert_eq!(trial.outer_calls, Some(6));
        assert_eq!(trial.outer_positions, Some(24));
        assert_eq!(trial.caller_peak, Some(3));
        assert_eq!(trial.device_calls, Some(6));
        assert_eq!(trial.device_positions, Some(24));
        assert_eq!(trial.device_avg_batch, Some(4.0));
    }

    struct FailingBackend;

    impl Backend for FailingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            Err(BackendError::msg("controlled failure"))
        }
    }

    #[test]
    fn backend_failure_becomes_a_failed_trial_row() {
        let result = run_capacity_case(
            "failure",
            &request(1, 2),
            resolved(),
            &FailingBackend,
            &[game()],
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Failed);
        assert!(trial
            .error
            .as_deref()
            .unwrap()
            .contains("controlled failure"));
        assert_eq!(trial.outer_calls, Some(1));
        assert_eq!(trial.positions_per_s, None);
    }

    struct ShortBackend;

    impl Backend for ShortBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            Ok(evaluation())
        }

        fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
            Ok(vec![evaluation(); games.len() - 1])
        }
    }

    #[test]
    fn malformed_output_is_recorded_before_throughput_is_claimed() {
        let result = run_capacity_case(
            "short",
            &request(1, 1),
            resolved(),
            &ShortBackend,
            &[game()],
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Failed);
        assert!(trial
            .error
            .as_deref()
            .unwrap()
            .contains("returned 3 evaluations for 4 positions"));
        assert_eq!(trial.positions_per_s, None);
    }

    struct PanickingBackend;

    impl Backend for PanickingBackend {
        fn evaluate(&self, _game: &GameState) -> Result<EvalResult, BackendError> {
            panic!("controlled panic")
        }
    }

    #[test]
    fn backend_panic_is_preserved_as_a_failed_trial() {
        let result = run_capacity_case(
            "panic",
            &request(1, 1),
            resolved(),
            &PanickingBackend,
            &[game()],
        )
        .unwrap();

        let trial = &result.trials[0];
        assert_eq!(trial.status, TrialStatus::Failed);
        assert!(trial
            .error
            .as_deref()
            .unwrap()
            .contains("backend evaluation panicked: controlled panic"));
    }

    #[test]
    fn phases_and_csv_follow_the_frozen_protocol_shape() {
        let mut request = request(1, 1);
        request.warmup_trials = 1;
        request.measured_trials = 2;
        let result = run_capacity_case(
            "shape",
            &request,
            resolved(),
            &SmartUniformBackend,
            &[game()],
        )
        .unwrap();
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

        let mut csv = Vec::new();
        write_capacity_trials(&mut csv, &result.trials).unwrap();
        let text = String::from_utf8(csv).unwrap();
        assert_eq!(text.lines().next().unwrap(), CAPACITY_HEADERS.join(","));
        let decoded = csv::Reader::from_reader(text.as_bytes())
            .deserialize::<CapacityTrial>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, result.trials);
    }

    #[test]
    fn produced_rows_are_accepted_by_the_v1_folder_validator() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/calibration-v1/valid");
        let folder = tempfile::tempdir().unwrap();
        fs::copy(fixture.join("run.json"), folder.path().join("run.json")).unwrap();
        for file in [CAPACITY_WORKLOAD_FILE, SEARCH_WORKLOAD_FILE] {
            fs::copy(fixture.join(file), folder.path().join(file)).unwrap();
        }
        fs::copy(
            fixture.join("search-trials.csv"),
            folder.path().join("search-trials.csv"),
        )
        .unwrap();

        let request = CapacityRequest {
            backend: BackendRequest::Onnx {
                provider: crate::calibration::OnnxProvider::Cpu,
            },
            batch_size: 8,
            callers: 2,
            calls_per_caller: 4,
            warmup_trials: 1,
            measured_trials: 1,
        };
        let resolved = ResolvedBackend {
            backend: request.backend.clone(),
            serialization: BackendSerialization::SessionMutex,
        };
        let result = run_capacity_case(
            "capacity-cpu-b8-c2",
            &request,
            resolved,
            &SmartUniformBackend,
            &[game()],
        )
        .unwrap();
        let capacity_path = folder.path().join("capacity-trials.csv");
        let mut output = fs::File::create(&capacity_path).unwrap();
        write_capacity_trials(&mut output, &result.trials).unwrap();
        drop(output);

        let capacity_bytes = fs::read(&capacity_path).unwrap();
        let mut record: crate::calibration::RunRecord =
            serde_json::from_slice(&fs::read(folder.path().join("run.json")).unwrap()).unwrap();
        let trial_file = record.trial_files.backend_capacity.as_mut().unwrap();
        trial_file.sha256 = format!("{:x}", Sha256::digest(&capacity_bytes));
        trial_file.rows = result.trials.len() as u32;
        fs::write(
            folder.path().join("run.json"),
            serde_json::to_vec_pretty(&record).unwrap(),
        )
        .unwrap();

        let loaded = crate::calibration::load_run_folder(folder.path()).unwrap();
        assert_eq!(loaded.capacity_trials, result.trials);
    }
}
