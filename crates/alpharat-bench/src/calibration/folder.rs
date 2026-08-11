use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::model::{
    ArtifactIdentity, BackendRequest, BenchmarkKind, CaseOutcome, CasePlan, CaseRecord, CaseState,
    ComparisonAxis, RunRecord, SourceState, CAPACITY_TRIALS_FILE, PROTOCOL_VERSION,
    RUN_RECORD_FILE, SEARCH_TRIALS_FILE,
};
use super::trials::{
    CapacityTrial, SearchTrial, TrialPhase, TrialStatus, CAPACITY_HEADERS, SEARCH_HEADERS,
};

#[derive(Debug)]
pub struct LoadedRun {
    pub folder: PathBuf,
    pub record: RunRecord,
    pub capacity_trials: Vec<CapacityTrial>,
    pub search_trials: Vec<SearchTrial>,
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid CSV in {path}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("invalid calibration record: {0}")]
    Invalid(String),
    #[error("SHA-256 mismatch for {path}: expected {expected}, found {actual}")]
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
}

impl ProtocolError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

pub fn load_run_folder(folder: impl AsRef<Path>) -> Result<LoadedRun, ProtocolError> {
    let folder = folder.as_ref();
    let record_path = folder.join(RUN_RECORD_FILE);
    let record_bytes = read_file(&record_path)?;
    let record: RunRecord =
        serde_json::from_slice(&record_bytes).map_err(|source| ProtocolError::Json {
            path: record_path,
            source,
        })?;

    validate_record(folder, &record)?;

    let capacity_trials = match &record.trial_files.backend_capacity {
        Some(file) => read_capacity_trials(folder, file)?,
        None => Vec::new(),
    };
    let search_trials = match &record.trial_files.search {
        Some(file) => read_search_trials(folder, file)?,
        None => Vec::new(),
    };

    validate_capacity_trials(&record, &capacity_trials)?;
    validate_search_trials(&record, &search_trials)?;

    Ok(LoadedRun {
        folder: folder.to_path_buf(),
        record,
        capacity_trials,
        search_trials,
    })
}

fn validate_record(folder: &Path, record: &RunRecord) -> Result<(), ProtocolError> {
    if record.protocol_version != PROTOCOL_VERSION {
        return Err(ProtocolError::invalid(format!(
            "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
            record.protocol_version
        )));
    }
    require_text("run_id", &record.run_id)?;
    require_text("created_at", &record.created_at)?;
    validate_context(folder, record)?;

    if record.plan.cases.is_empty() {
        return Err(ProtocolError::invalid("the run plan has no cases"));
    }

    let mut planned = BTreeMap::new();
    for case in &record.plan.cases {
        require_text("case id", case.id())?;
        validate_artifact("case workload", case.workload())?;
        validate_case_plan(case)?;
        if planned.insert(case.id(), case).is_some() {
            return Err(ProtocolError::invalid(format!(
                "duplicate planned case id '{}'",
                case.id()
            )));
        }
    }
    validate_axes(record)?;
    validate_plan_differences(record)?;

    let mut outcomes = BTreeMap::new();
    for case in &record.cases {
        require_text("case outcome id", case.id())?;
        if outcomes.insert(case.id(), case).is_some() {
            return Err(ProtocolError::invalid(format!(
                "duplicate case outcome id '{}'",
                case.id()
            )));
        }
        let plan = planned.get(case.id()).ok_or_else(|| {
            ProtocolError::invalid(format!("outcome for unplanned case '{}'", case.id()))
        })?;
        if plan.kind() != case.kind() {
            return Err(ProtocolError::invalid(format!(
                "case '{}' changes benchmark kind between plan and record",
                case.id()
            )));
        }
        validate_case_outcome(plan, case)?;
    }
    for case_id in planned.keys() {
        if !outcomes.contains_key(case_id) {
            return Err(ProtocolError::invalid(format!(
                "planned case '{case_id}' has no recorded outcome"
            )));
        }
    }

    let has_capacity = record
        .plan
        .cases
        .iter()
        .any(|case| case.kind() == BenchmarkKind::BackendCapacity);
    let has_search = record
        .plan
        .cases
        .iter()
        .any(|case| case.kind() == BenchmarkKind::Search);
    validate_trial_file_presence(
        "backend-capacity",
        has_capacity,
        record.trial_files.backend_capacity.is_some(),
    )?;
    validate_trial_file_presence("search", has_search, record.trial_files.search.is_some())?;

    Ok(())
}

fn validate_context(folder: &Path, record: &RunRecord) -> Result<(), ProtocolError> {
    let context = &record.context;
    require_text("source revision", &context.source.revision)?;
    match &context.source.state {
        SourceState::Clean => {}
        SourceState::Patched {
            patch_file,
            patch_sha256,
        } => {
            require_relative_file("source patch", patch_file)?;
            require_sha256("source patch", patch_sha256)?;
            verify_file_hash(folder, patch_file, patch_sha256)?;
        }
        SourceState::NonReproducible { reason } => {
            require_text("non-reproducible source reason", reason)?;
        }
    }

    require_text("build profile", &context.build.profile)?;
    require_text("build target", &context.build.target)?;
    require_text("rustc version", &context.build.rustc_version)?;
    require_sha256("benchmark binary", &context.build.binary_sha256)?;
    if context.build.command.is_empty() || context.build.command.iter().any(|part| part.is_empty())
    {
        return Err(ProtocolError::invalid(
            "build command must contain non-empty arguments",
        ));
    }
    if context
        .build
        .features
        .iter()
        .any(|feature| feature.is_empty())
    {
        return Err(ProtocolError::invalid("build features cannot be empty"));
    }

    require_text("operating system", &context.hardware.operating_system)?;
    require_text("hardware architecture", &context.hardware.architecture)?;
    require_text("CPU", &context.hardware.cpu)?;
    if context.hardware.logical_cores == 0 || context.hardware.memory_bytes == 0 {
        return Err(ProtocolError::invalid(
            "hardware core count and memory must be positive",
        ));
    }
    for accelerator in &context.hardware.accelerators {
        require_text("accelerator kind", &accelerator.kind)?;
        require_text("accelerator name", &accelerator.name)?;
    }
    let unique_accelerators = context
        .hardware
        .accelerators
        .iter()
        .collect::<BTreeSet<_>>();
    if unique_accelerators.len() != context.hardware.accelerators.len() {
        return Err(ProtocolError::invalid(
            "hardware accelerators contain duplicate entries",
        ));
    }

    if context.runtime.software.is_empty() {
        return Err(ProtocolError::invalid(
            "runtime software versions cannot be empty",
        ));
    }
    let mut software = BTreeSet::new();
    for version in &context.runtime.software {
        require_text("software name", &version.name)?;
        require_text("software version", &version.version)?;
        if !software.insert(&version.name) {
            return Err(ProtocolError::invalid(format!(
                "duplicate software version entry '{}'",
                version.name
            )));
        }
    }

    validate_artifact("model", &context.model)
}

fn validate_artifact(label: &str, artifact: &ArtifactIdentity) -> Result<(), ProtocolError> {
    require_text(&format!("{label} label"), &artifact.label)?;
    require_sha256(label, &artifact.sha256)?;
    if artifact.path_hint.as_ref().is_some_and(String::is_empty) {
        return Err(ProtocolError::invalid(format!(
            "{label} path hint cannot be empty"
        )));
    }
    Ok(())
}

fn validate_case_plan(case: &CasePlan) -> Result<(), ProtocolError> {
    match case {
        CasePlan::BackendCapacity { requested, .. } => {
            if requested.batch_size == 0
                || requested.callers == 0
                || requested.calls_per_caller == 0
                || requested.measured_trials == 0
            {
                return Err(ProtocolError::invalid(format!(
                    "capacity case '{}' requires positive batch, callers, calls, and measured trials",
                    case.id()
                )));
            }
            if matches!(requested.backend, BackendRequest::SmartUniform) {
                return Err(ProtocolError::invalid(format!(
                    "capacity case '{}' cannot use the non-inference SmartUniform backend",
                    case.id()
                )));
            }
        }
        CasePlan::Search { requested, .. } => {
            if requested.workers == 0
                || requested.total_in_flight == 0
                || requested.productive_work == 0
                || requested.measured_trials == 0
            {
                return Err(ProtocolError::invalid(format!(
                    "search case '{}' requires positive workers, capacity, work, and measured trials",
                    case.id()
                )));
            }
            if requested
                .mux_max_batch
                .is_some_and(|max_batch| max_batch == 0)
            {
                return Err(ProtocolError::invalid(format!(
                    "search case '{}' has a zero mux batch limit",
                    case.id()
                )));
            }
        }
    }
    Ok(())
}

fn validate_axes(record: &RunRecord) -> Result<(), ProtocolError> {
    let has_capacity = record
        .plan
        .cases
        .iter()
        .any(|case| case.kind() == BenchmarkKind::BackendCapacity);
    let has_search = record
        .plan
        .cases
        .iter()
        .any(|case| case.kind() == BenchmarkKind::Search);

    for axis in &record.plan.comparison_axes {
        let applicable = match axis {
            ComparisonAxis::BatchSize | ComparisonAxis::Callers => has_capacity,
            ComparisonAxis::Workers
            | ComparisonAxis::TotalInFlight
            | ComparisonAxis::MuxMaxBatch => has_search,
            ComparisonAxis::Source
            | ComparisonAxis::Build
            | ComparisonAxis::Hardware
            | ComparisonAxis::Runtime
            | ComparisonAxis::Backend => true,
        };
        if !applicable {
            return Err(ProtocolError::invalid(format!(
                "comparison axis {axis:?} has no matching benchmark case"
            )));
        }
    }
    Ok(())
}

fn validate_plan_differences(record: &RunRecord) -> Result<(), ProtocolError> {
    for (index, left) in record.plan.cases.iter().enumerate() {
        for right in &record.plan.cases[index + 1..] {
            if left.kind() != right.kind() {
                continue;
            }
            if !left.workload().has_same_content_as(right.workload()) {
                return Err(ProtocolError::invalid(format!(
                    "cases '{}' and '{}' use different workloads; workload is fixed in v1",
                    left.id(),
                    right.id()
                )));
            }
            let differences = case_plan_differences(left, right)?;
            let undeclared: Vec<_> = differences
                .difference(&record.plan.comparison_axes)
                .copied()
                .collect();
            if !undeclared.is_empty() {
                return Err(ProtocolError::invalid(format!(
                    "cases '{}' and '{}' differ on undeclared axes {undeclared:?}",
                    left.id(),
                    right.id()
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn case_plan_differences(
    left: &CasePlan,
    right: &CasePlan,
) -> Result<BTreeSet<ComparisonAxis>, ProtocolError> {
    let mut differences = BTreeSet::new();
    match (left, right) {
        (
            CasePlan::BackendCapacity {
                requested: left, ..
            },
            CasePlan::BackendCapacity {
                requested: right, ..
            },
        ) => {
            if left.backend != right.backend {
                differences.insert(ComparisonAxis::Backend);
            }
            if left.batch_size != right.batch_size {
                differences.insert(ComparisonAxis::BatchSize);
            }
            if left.callers != right.callers {
                differences.insert(ComparisonAxis::Callers);
            }
            if left.calls_per_caller != right.calls_per_caller
                || left.warmup_trials != right.warmup_trials
                || left.measured_trials != right.measured_trials
            {
                return Err(ProtocolError::invalid(
                    "capacity cases change fixed work or trial counts",
                ));
            }
        }
        (
            CasePlan::Search {
                requested: left, ..
            },
            CasePlan::Search {
                requested: right, ..
            },
        ) => {
            if left.backend != right.backend {
                differences.insert(ComparisonAxis::Backend);
            }
            if left.workers != right.workers {
                differences.insert(ComparisonAxis::Workers);
            }
            if left.total_in_flight != right.total_in_flight {
                differences.insert(ComparisonAxis::TotalInFlight);
            }
            if left.mux_max_batch != right.mux_max_batch {
                differences.insert(ComparisonAxis::MuxMaxBatch);
            }
            if left.productive_work != right.productive_work
                || left.warmup_trials != right.warmup_trials
                || left.measured_trials != right.measured_trials
            {
                return Err(ProtocolError::invalid(
                    "search cases change fixed work or trial counts",
                ));
            }
        }
        _ => {
            return Err(ProtocolError::invalid(
                "cannot compare plans for different benchmark kinds",
            ));
        }
    }
    Ok(differences)
}

fn validate_case_outcome(plan: &CasePlan, record: &CaseRecord) -> Result<(), ProtocolError> {
    match (plan, record) {
        (
            CasePlan::BackendCapacity { requested, .. },
            CaseRecord::BackendCapacity { outcome, .. },
        ) => match outcome {
            CaseOutcome::Completed { resolved } => {
                if resolved.backend.backend != requested.backend
                    || resolved.batch_size != requested.batch_size
                    || resolved.callers != requested.callers
                    || resolved.calls_per_caller != requested.calls_per_caller
                {
                    return Err(ProtocolError::invalid(format!(
                        "capacity case '{}' resolved to a different requested setup",
                        plan.id()
                    )));
                }
            }
            other => validate_non_completed(plan.id(), other)?,
        },
        (CasePlan::Search { requested, .. }, CaseRecord::Search { outcome, .. }) => match outcome {
            CaseOutcome::Completed { resolved } => {
                if resolved.backend.backend != requested.backend
                    || resolved.workers != requested.workers
                    || resolved.total_in_flight != requested.total_in_flight
                    || resolved.mux_max_batch != requested.mux_max_batch
                    || resolved.productive_work != requested.productive_work
                {
                    return Err(ProtocolError::invalid(format!(
                        "search case '{}' resolved to a different requested setup",
                        plan.id()
                    )));
                }
                if resolved.worker_batch == 0
                    || u64::from(resolved.worker_batch) * u64::from(resolved.workers)
                        != u64::from(resolved.total_in_flight)
                {
                    return Err(ProtocolError::invalid(format!(
                        "search case '{}' has inconsistent worker and total capacity",
                        plan.id()
                    )));
                }
            }
            other => validate_non_completed(plan.id(), other)?,
        },
        _ => unreachable!("benchmark kind was checked before outcome validation"),
    }
    Ok(())
}

fn validate_non_completed<T>(case_id: &str, outcome: &CaseOutcome<T>) -> Result<(), ProtocolError> {
    let text = match outcome {
        CaseOutcome::Failed { message, .. } => message,
        CaseOutcome::Unsupported { reason } | CaseOutcome::Interrupted { reason } => reason,
        CaseOutcome::Completed { .. } => return Ok(()),
    };
    require_text(&format!("case '{case_id}' outcome reason"), text)
}

fn validate_trial_file_presence(
    kind: &str,
    planned: bool,
    present: bool,
) -> Result<(), ProtocolError> {
    if planned != present {
        return Err(ProtocolError::invalid(format!(
            "{kind} trial file presence does not match the run plan"
        )));
    }
    Ok(())
}

fn read_capacity_trials(
    folder: &Path,
    file: &super::model::TrialFile,
) -> Result<Vec<CapacityTrial>, ProtocolError> {
    if file.path != CAPACITY_TRIALS_FILE {
        return Err(ProtocolError::invalid(format!(
            "backend-capacity trials must use '{CAPACITY_TRIALS_FILE}'"
        )));
    }
    read_trials(folder, file, CAPACITY_HEADERS)
}

fn read_search_trials(
    folder: &Path,
    file: &super::model::TrialFile,
) -> Result<Vec<SearchTrial>, ProtocolError> {
    if file.path != SEARCH_TRIALS_FILE {
        return Err(ProtocolError::invalid(format!(
            "search trials must use '{SEARCH_TRIALS_FILE}'"
        )));
    }
    read_trials(folder, file, SEARCH_HEADERS)
}

fn read_trials<T>(
    folder: &Path,
    file: &super::model::TrialFile,
    expected_headers: &[&str],
) -> Result<Vec<T>, ProtocolError>
where
    T: for<'de> serde::Deserialize<'de>,
{
    require_relative_file("trial file", &file.path)?;
    require_sha256("trial file", &file.sha256)?;
    let path = folder.join(&file.path);
    let bytes = read_file(&path)?;
    verify_hash(&path, &bytes, &file.sha256)?;

    let mut reader = csv::ReaderBuilder::new().from_reader(bytes.as_slice());
    let headers = reader
        .headers()
        .map_err(|source| ProtocolError::Csv {
            path: path.clone(),
            source,
        })?
        .clone();
    if headers.iter().collect::<Vec<_>>() != expected_headers {
        return Err(ProtocolError::invalid(format!(
            "{} has an unexpected header; protocol v1 requires {}",
            path.display(),
            expected_headers.join(",")
        )));
    }

    let mut rows = Vec::new();
    for result in reader.deserialize() {
        rows.push(result.map_err(|source| ProtocolError::Csv {
            path: path.clone(),
            source,
        })?);
    }
    let declared_rows = usize::try_from(file.rows).map_err(|_| {
        ProtocolError::invalid(format!(
            "{} declares more rows than this platform can load",
            path.display()
        ))
    })?;
    if rows.len() != declared_rows {
        return Err(ProtocolError::invalid(format!(
            "{} declares {} rows but contains {}",
            path.display(),
            file.rows,
            rows.len()
        )));
    }
    Ok(rows)
}

fn validate_capacity_trials(
    record: &RunRecord,
    trials: &[CapacityTrial],
) -> Result<(), ProtocolError> {
    let plans: BTreeMap<_, _> = record
        .plan
        .cases
        .iter()
        .filter_map(|case| match case {
            CasePlan::BackendCapacity { requested, .. } => Some((case.id(), (case, requested))),
            CasePlan::Search { .. } => None,
        })
        .collect();
    let outcomes: BTreeMap<_, _> = record
        .cases
        .iter()
        .filter_map(|case| match case {
            CaseRecord::BackendCapacity { outcome, .. } => Some((case.id(), outcome.state())),
            CaseRecord::Search { .. } => None,
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut counts: BTreeMap<&str, (u32, u32)> = BTreeMap::new();

    for trial in trials {
        let (_, request) = plans.get(trial.case_id.as_str()).ok_or_else(|| {
            ProtocolError::invalid(format!(
                "capacity trial refers to unknown case '{}'",
                trial.case_id
            ))
        })?;
        validate_trial_key(
            &mut seen,
            &trial.case_id,
            trial.phase,
            trial.trial,
            request.warmup_trials,
            request.measured_trials,
        )?;
        let entry = counts.entry(&trial.case_id).or_default();
        match trial.phase {
            TrialPhase::Warmup => entry.0 += 1,
            TrialPhase::Measured => entry.1 += 1,
        }
        validate_trial_matches_case_state(
            &trial.case_id,
            outcomes[trial.case_id.as_str()],
            trial.status,
        )?;
        validate_capacity_trial_metrics(trial, request)?;
    }

    validate_trial_counts(&plans, &outcomes, &counts)
}

fn validate_search_trials(record: &RunRecord, trials: &[SearchTrial]) -> Result<(), ProtocolError> {
    let plans: BTreeMap<_, _> = record
        .plan
        .cases
        .iter()
        .filter_map(|case| match case {
            CasePlan::Search { requested, .. } => Some((case.id(), (case, requested))),
            CasePlan::BackendCapacity { .. } => None,
        })
        .collect();
    let outcomes: BTreeMap<_, _> = record
        .cases
        .iter()
        .filter_map(|case| match case {
            CaseRecord::Search { outcome, .. } => Some((case.id(), outcome.state())),
            CaseRecord::BackendCapacity { .. } => None,
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut counts: BTreeMap<&str, (u32, u32)> = BTreeMap::new();

    for trial in trials {
        let (_, request) = plans.get(trial.case_id.as_str()).ok_or_else(|| {
            ProtocolError::invalid(format!(
                "search trial refers to unknown case '{}'",
                trial.case_id
            ))
        })?;
        validate_trial_key(
            &mut seen,
            &trial.case_id,
            trial.phase,
            trial.trial,
            request.warmup_trials,
            request.measured_trials,
        )?;
        let entry = counts.entry(&trial.case_id).or_default();
        match trial.phase {
            TrialPhase::Warmup => entry.0 += 1,
            TrialPhase::Measured => entry.1 += 1,
        }
        validate_trial_matches_case_state(
            &trial.case_id,
            outcomes[trial.case_id.as_str()],
            trial.status,
        )?;
        validate_search_trial_metrics(trial, request)?;
    }

    validate_trial_counts(&plans, &outcomes, &counts)
}

fn validate_trial_key(
    seen: &mut BTreeSet<(String, u8, u32)>,
    case_id: &str,
    phase: TrialPhase,
    trial: u32,
    warmups: u32,
    measured: u32,
) -> Result<(), ProtocolError> {
    let limit = match phase {
        TrialPhase::Warmup => warmups,
        TrialPhase::Measured => measured,
    };
    if trial == 0 || trial > limit {
        return Err(ProtocolError::invalid(format!(
            "trial {trial} for case '{case_id}' is outside the planned {phase:?} range 1..={limit}"
        )));
    }
    let phase_key = match phase {
        TrialPhase::Warmup => 0,
        TrialPhase::Measured => 1,
    };
    if !seen.insert((case_id.to_owned(), phase_key, trial)) {
        return Err(ProtocolError::invalid(format!(
            "duplicate {phase:?} trial {trial} for case '{case_id}'"
        )));
    }
    Ok(())
}

fn validate_trial_counts<T>(
    plans: &BTreeMap<&str, (&CasePlan, &T)>,
    outcomes: &BTreeMap<&str, CaseState>,
    counts: &BTreeMap<&str, (u32, u32)>,
) -> Result<(), ProtocolError> {
    for (case_id, (plan, _)) in plans {
        let actual = counts.get(case_id).copied().unwrap_or_default();
        match outcomes[case_id] {
            CaseState::Completed => {
                let expected = (plan.warmup_trials(), plan.measured_trials());
                if actual != expected {
                    return Err(ProtocolError::invalid(format!(
                        "completed case '{case_id}' has {actual:?} warmup/measured rows; expected {expected:?}"
                    )));
                }
            }
            CaseState::Unsupported if actual != (0, 0) => {
                return Err(ProtocolError::invalid(format!(
                    "unsupported case '{case_id}' cannot contain trials"
                )));
            }
            CaseState::Failed | CaseState::Interrupted | CaseState::Unsupported => {}
        }
    }
    Ok(())
}

fn validate_trial_matches_case_state(
    case_id: &str,
    case_state: CaseState,
    trial_status: TrialStatus,
) -> Result<(), ProtocolError> {
    if case_state == CaseState::Completed && trial_status != TrialStatus::Completed {
        return Err(ProtocolError::invalid(format!(
            "completed case '{case_id}' contains a {trial_status:?} trial"
        )));
    }
    Ok(())
}

fn validate_capacity_trial_metrics(
    trial: &CapacityTrial,
    request: &super::model::CapacityRequest,
) -> Result<(), ProtocolError> {
    validate_trial_status(&trial.case_id, trial.status, trial.error.as_deref())?;
    let values = [
        trial.wall_ms,
        trial.positions_per_s,
        trial.caller_time_ms,
        trial.caller_union_ms,
        trial.device_avg_batch,
        trial.device_inference_ms,
    ];
    validate_optional_numbers(&trial.case_id, &values)?;
    if trial.status != TrialStatus::Completed {
        return Ok(());
    }
    require_all(
        &trial.case_id,
        &[
            trial.wall_ms.is_some(),
            trial.positions_per_s.is_some(),
            trial.outer_calls.is_some(),
            trial.outer_positions.is_some(),
            trial.caller_time_ms.is_some(),
            trial.caller_union_ms.is_some(),
            trial.caller_peak.is_some(),
            trial.device_calls.is_some(),
            trial.device_positions.is_some(),
            trial.device_avg_batch.is_some(),
            trial.device_inference_ms.is_some(),
        ],
    )?;
    let expected_calls = u128::from(request.callers) * u128::from(request.calls_per_caller);
    let expected_positions = expected_calls * u128::from(request.batch_size);
    if trial.outer_calls.map(u128::from) != Some(expected_calls)
        || trial.outer_positions.map(u128::from) != Some(expected_positions)
        || trial.device_positions.map(u128::from) != Some(expected_positions)
        || trial.caller_peak.is_some_and(|peak| peak > request.callers)
    {
        return Err(ProtocolError::invalid(format!(
            "completed capacity trial for '{}' disagrees with its requested work",
            trial.case_id
        )));
    }
    Ok(())
}

fn validate_search_trial_metrics(
    trial: &SearchTrial,
    request: &super::model::SearchRequest,
) -> Result<(), ProtocolError> {
    validate_trial_status(&trial.case_id, trial.status, trial.error.as_deref())?;
    let values = [
        trial.wall_ms,
        trial.productive_per_s,
        trial.phase_occupancy_ms,
        trial.phase_occupancy_per_wall,
        trial.lease_wait_ms,
        trial.gate_wait_ms,
        trial.gate_hold_ms,
        trial.inference_caller_ms,
        trial.completion_wait_ms,
        trial.backend_caller_ms,
        trial.backend_union_ms,
        trial.backend_caller_per_union,
        trial.device_avg_batch,
        trial.device_inference_ms,
    ];
    validate_optional_numbers(&trial.case_id, &values)?;
    let policies = [
        trial.policy_p1_0,
        trial.policy_p1_1,
        trial.policy_p1_2,
        trial.policy_p1_3,
        trial.policy_p1_4,
        trial.policy_p2_0,
        trial.policy_p2_1,
        trial.policy_p2_2,
        trial.policy_p2_3,
        trial.policy_p2_4,
    ];
    validate_optional_policy(&trial.case_id, &policies)?;
    if trial.status != TrialStatus::Completed {
        return Ok(());
    }
    require_all(
        &trial.case_id,
        &[
            trial.wall_ms.is_some(),
            trial.productive_per_s.is_some(),
            trial.nn_evals.is_some(),
            trial.terminals.is_some(),
            trial.tt_stops.is_some(),
            trial.collisions.is_some(),
            trial.batches.is_some(),
            trial.phase_occupancy_ms.is_some(),
            trial.phase_occupancy_per_wall.is_some(),
            trial.lease_wait_ms.is_some(),
            trial.gate_wait_ms.is_some(),
            trial.gate_hold_ms.is_some(),
            trial.inference_caller_ms.is_some(),
            trial.completion_wait_ms.is_some(),
            trial.backend_calls.is_some(),
            trial.backend_positions.is_some(),
            trial.backend_caller_ms.is_some(),
            trial.backend_union_ms.is_some(),
            trial.backend_caller_per_union.is_some(),
            trial.backend_peak_callers.is_some(),
            trial.device_calls.is_some(),
            trial.device_positions.is_some(),
            trial.device_avg_batch.is_some(),
            trial.device_inference_ms.is_some(),
            trial.ledger_reserved.is_some(),
            trial.ledger_committed.is_some(),
            trial.ledger_cancelled.is_some(),
            policies.iter().all(Option::is_some),
        ],
    )?;
    let productive = u64::from(trial.nn_evals.unwrap())
        + u64::from(trial.terminals.unwrap())
        + u64::from(trial.tt_stops.unwrap());
    let ledger_total =
        u64::from(trial.ledger_committed.unwrap()) + u64::from(trial.ledger_cancelled.unwrap());
    if productive != u64::from(request.productive_work)
        || trial.ledger_committed != Some(request.productive_work)
        || u64::from(trial.ledger_reserved.unwrap()) != ledger_total
        || trial.backend_positions != trial.nn_evals.map(u64::from)
        || trial.device_positions != trial.backend_positions
    {
        return Err(ProtocolError::invalid(format!(
            "completed search trial for '{}' violates work or ledger accounting",
            trial.case_id
        )));
    }
    validate_policy_sum(&trial.case_id, &policies[..5])?;
    validate_policy_sum(&trial.case_id, &policies[5..])
}

fn validate_trial_status(
    case_id: &str,
    status: TrialStatus,
    error: Option<&str>,
) -> Result<(), ProtocolError> {
    match status {
        TrialStatus::Completed if error.is_some() => Err(ProtocolError::invalid(format!(
            "completed trial for '{case_id}' cannot carry an error"
        ))),
        TrialStatus::Failed | TrialStatus::Interrupted
            if error.is_none_or(|message| message.trim().is_empty()) =>
        {
            Err(ProtocolError::invalid(format!(
                "{status:?} trial for '{case_id}' requires a reason"
            )))
        }
        _ => Ok(()),
    }
}

fn validate_optional_numbers(case_id: &str, values: &[Option<f64>]) -> Result<(), ProtocolError> {
    if values
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ProtocolError::invalid(format!(
            "trial for '{case_id}' contains a negative or non-finite metric"
        )));
    }
    Ok(())
}

fn validate_optional_policy(case_id: &str, values: &[Option<f32>]) -> Result<(), ProtocolError> {
    if values
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ProtocolError::invalid(format!(
            "trial for '{case_id}' contains an invalid policy value"
        )));
    }
    Ok(())
}

fn validate_policy_sum(case_id: &str, values: &[Option<f32>]) -> Result<(), ProtocolError> {
    let sum: f32 = values.iter().map(|value| value.unwrap()).sum();
    if (sum - 1.0).abs() > 1e-4 {
        return Err(ProtocolError::invalid(format!(
            "trial for '{case_id}' contains a policy that sums to {sum}"
        )));
    }
    Ok(())
}

fn require_all(case_id: &str, present: &[bool]) -> Result<(), ProtocolError> {
    if present.iter().any(|value| !value) {
        return Err(ProtocolError::invalid(format!(
            "completed trial for '{case_id}' is missing required metrics"
        )));
    }
    Ok(())
}

fn require_text(label: &str, value: &str) -> Result<(), ProtocolError> {
    if value.trim().is_empty() {
        return Err(ProtocolError::invalid(format!("{label} cannot be empty")));
    }
    Ok(())
}

fn require_sha256(label: &str, value: &str) -> Result<(), ProtocolError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProtocolError::invalid(format!(
            "{label} must use a lowercase 64-character SHA-256"
        )));
    }
    Ok(())
}

fn require_relative_file(label: &str, value: &str) -> Result<(), ProtocolError> {
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(ProtocolError::invalid(format!(
            "{label} must be a plain relative path inside the run folder"
        )));
    }
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>, ProtocolError> {
    fs::read(path).map_err(|source| ProtocolError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn verify_file_hash(folder: &Path, relative: &str, expected: &str) -> Result<(), ProtocolError> {
    let path = folder.join(relative);
    let bytes = read_file(&path)?;
    verify_hash(&path, &bytes, expected)
}

fn verify_hash(path: &Path, bytes: &[u8], expected: &str) -> Result<(), ProtocolError> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(ProtocolError::HashMismatch {
            path: path.to_path_buf(),
            expected: expected.to_owned(),
            actual,
        });
    }
    Ok(())
}
